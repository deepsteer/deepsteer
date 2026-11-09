#!/usr/bin/env python3
"""Shared pieces of the two pilot pods (PILOT_BUILD.md §1, §5): the HF model for training, the
32-prompt parity read per adapter kind, and the LoRA settings of record.

**Parity (VALIDATE V4, per adapter kind):** first-token letter log-probs on 32 dose-0 operator
prompts (the first 32 panel scenarios, letter order k 0), computed on HF with the adapter attached
and with it disabled (`hf_parity`, while the trainer holds the model) and later on vLLM serving the
adapter natively (`served_parity`); the kind passes when TV(served, attached) is below TV(base,
attached), as VALIDATE measured it (0.0125 against 0.1115).
"""

from __future__ import annotations

import json
from pathlib import Path

import cst_s0a_pod as S

LORA = {"rank": 16, "alpha": 32, "dropout": 0.0}  # VALIDATE (PILOT_BUILD.md §1)
SDF_LR, RECOVERY_LR, DPO_LR = 1e-4, 2e-4, 1e-4  # not registered; ST's recipe and VALIDATE's step
N_PARITY = 32


def load_model(dry: bool):
    """The subject on HF for training (bf16 on GPU; float32 on CPU in a dry run), with LoRA."""
    import torch

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.steering import objective_lora as OL

    path = S.V._model_path(dry)  # noqa: SLF001
    wb = WhiteBoxModel(path, access_tier=AccessTier.WEIGHTS,
                       device="cpu" if dry else "cuda",
                       torch_dtype=torch.float32 if dry else torch.bfloat16)
    if not dry:
        wb.model.gradient_checkpointing_enable()
    cfg = OL.inject_lora(wb, rank=LORA["rank"], alpha=LORA["alpha"], dropout=LORA["dropout"])
    return path, wb, cfg


def parity_prompts(dry: bool) -> tuple[list[list[int]], list[int]]:
    """(prompt ids, letter token ids) for the 32 dose-0 operator prompts."""
    import reasoning_formats as rf

    from deepsteer.kdg.schema import assign_letters

    sc = S.scenarios(dry)
    tok = S.V._tok(S.V._model_path(dry))  # noqa: SLF001
    fmt = rf.build_format("qwen38_27b", tok)
    ids = sc["ids"]["panel"][:N_PARITY]
    prompts = []
    for sid in ids:
        s = sc["objs"][sid]
        prompts.append(fmt.dose0_ids(tok, S.messages(s, assign_letters(s, 0), "panel", "operator",
                                                     "a0", None), S.EFFORT, fmt.dose0[0]))
    return prompts, sorted(rf.letter_token_ids(tok))


def _letter_lp(model, prompts, lids) -> list[list[float]]:
    import torch

    out = []
    with torch.no_grad():
        for p in prompts:
            x = torch.tensor([p], device=model.model.device)
            lp = torch.log_softmax(model.model(input_ids=x).logits[0, -1].float(), -1)
            out.append([float(lp[i]) for i in lids])
    return out


def hf_parity(model, out: Path, kind: str, dry: bool) -> dict:
    """Save the HF side of the parity read for ``kind`` (attached and adapter-disabled)."""
    from deepsteer.steering import objective_lora as OL

    prompts, lids = parity_prompts(dry)
    attached = _letter_lp(model, prompts, lids)
    with OL.adapter_disabled(model):
        base = _letter_lp(model, prompts, lids)
    rec = {"kind": kind, "letter_ids": lids, "attached": attached, "base": base}
    (out / "parity").mkdir(parents=True, exist_ok=True)
    (out / "parity" / f"{kind}_hf.json").write_text(json.dumps(rec) + "\n")
    return rec


def tv(a: list[list[float]], b: list[list[float]]) -> float:
    """Mean total variation between letter distributions (renormalized over the letters)."""
    import math

    tot = 0.0
    for x, y in zip(a, b):
        px = [math.exp(v) for v in x]
        py = [math.exp(v) for v in y]
        sx, sy = sum(px), sum(py)
        tot += 0.5 * sum(abs(u / sx - w / sy) for u, w in zip(px, py))
    return tot / max(len(a), 1)


def served_parity(out: Path, kind: str, eng, lora, dry: bool) -> dict:
    """The vLLM side: served letter log-probs against the saved HF read; writes the verdict."""
    hf = json.loads((out / "parity" / f"{kind}_hf.json").read_text())
    prompts, lids = parity_prompts(dry)
    # the dry stand-in carries its adapter from construction (cst_validate_pod.HFStandIn)
    served = (eng.first_logprobs(prompts, lids) if dry else
              eng.first_logprobs(prompts, lids, lora=lora))
    rec = {"kind": kind, "tv_served_vs_attached": tv(served, hf["attached"]),
           "tv_base_vs_attached": tv(hf["base"], hf["attached"])}
    rec["passes"] = rec["tv_served_vs_attached"] < rec["tv_base_vs_attached"]
    (out / "parity" / f"{kind}_served.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec
