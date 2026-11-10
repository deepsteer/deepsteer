#!/usr/bin/env python3
"""Shared pieces of the two pilot pods (PILOT_BUILD.md §1, §5): the HF model for training, the
32-prompt parity read per adapter kind, and the LoRA settings of record.

**Parity (VALIDATE V4, per adapter kind):** first-token letter log-probs on 32 dose-0 operator
prompts (the first 32 panel scenarios, letter order k 0), computed on HF with the adapter attached
and with it disabled (`hf_parity`, while the trainer holds the model) and later on vLLM serving the
adapter natively (`served_parity`); the kind passes when TV(served, attached) is below TV(base,
attached), as VALIDATE measured it (0.0125 against 0.1115). **Superseded by Amendment 25 A4**
(CST-A22: that rule passed an adapter served as the base by 0.0009): `parity_record` on the same
adapter on both sides, one write-once record per read tag.
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


PARITY_ATTACHED_MAX, PARITY_BASE_MIN = 0.2, 0.5  # Amendment 25 A4, fractions of TV(base, attached)


def parity_record(tag: str, served, attached, base, served_base=None) -> dict:
    """Amendment 25 A4's gate for one read tag, the same adapter on both sides: passes when
    TV(served, attached) <= 0.2 TV(base, attached) and TV(served, base) >= 0.5 TV(base, attached).
    The served letters ride in the record; ``served_base`` (the served model without an adapter)
    gives the instrument's floor, TV(served base, HF base)."""
    ref = tv(base, attached)
    rec = {"tag": tag, "tv_served_vs_attached": tv(served, attached),
           "tv_served_vs_base": tv(served, base), "tv_base_vs_attached": ref,
           "rule": "Amendment 25 A4", "attached_max": PARITY_ATTACHED_MAX * ref,
           "base_min": PARITY_BASE_MIN * ref, "served_letters": served}
    if served_base is not None:
        rec["tv_served_base_vs_hf_base"] = tv(served_base, base)
    rec["passes"] = bool(rec["tv_served_vs_attached"] <= rec["attached_max"]
                         and rec["tv_served_vs_base"] >= rec["base_min"])
    return rec


def write_once(f: Path, rec: dict) -> None:
    """One record per read tag, never overwritten (Amendment 25 A4)."""
    if f.exists():
        raise SystemExit(f"BAIL: {f} exists; parity records are never overwritten")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, indent=1) + "\n")


def decoder_prefixes(path: str) -> dict[str, str]:
    """The decoder-layer module path under the trainer's class (``AutoModelForCausalLM``) and the
    served class (``AutoModelForImageTextToText``, vLLM's), from the config on the meta device."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    cfg = AutoConfig.from_pretrained(path)
    out = {}
    for name, cls in (("trainer", AutoModelForCausalLM), ("served", AutoModelForImageTextToText)):
        with torch.device("meta"):
            m = cls.from_config(cfg)
        out[name] = [n for n, _ in m.named_modules() if n.endswith(".layers")][0]
    return out


def rename_adapter(src: Path, dst: Path, prefixes: dict[str, str]) -> dict:
    """Write ``src``'s adapter with its keys moved from the trainer class's decoder path to the
    served class's (CST-A22's fixed-path probe); the config is copied unchanged."""
    import shutil

    from safetensors.torch import load_file, save_file

    old = f"base_model.model.{prefixes['trainer']}."
    new = f"base_model.model.{prefixes['served']}."
    sd = load_file(str(src / "adapter_model.safetensors"))
    moved = {(new + k[len(old):] if k.startswith(old) else k): v for k, v in sd.items()}
    dst.mkdir(parents=True, exist_ok=True)
    save_file(moved, str(dst / "adapter_model.safetensors"))
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")
    return {"keys": len(sd), "moved": sum(k.startswith(old) for k in sd), "from": old, "to": new}


READ_K = {"panel_operator": 2, "panel_plain": 2, "forced_plain": 8, "legit_operator": 4}
READ_PARTS = ("panel_operator", "panel_plain", "forced_plain", "legit_operator")
LEGIT_CHECK_N, LEGIT_FLOOR = 20, 0.10  # rider d's check; floor not registered (logged)


def clear_ids() -> set[str]:
    rec = json.loads((S.CST / "data/scenario_labels_v4_record.json").read_text())
    rows = rec["labels"] if "labels" in rec else rec["record"]
    rows = rows.values() if isinstance(rows, dict) else rows
    return {r["scenario_id"] for r in rows if r.get("stratum_of_record") == "clear_refuse"
            and r.get("set", "panel") == "panel"}


class DryEngine(S.V.HFStandIn):
    """The dry run's served adapter: the tiny model loaded the trainer's way (`WhiteBoxModel`), so
    a saved adapter attaches with its own key names (VALIDATE's stand-in loads another model
    class, on which they do not attach); generation and letter reads as the stand-in."""

    def __init__(self, path: str, lora: str | None = None):
        import torch

        from deepsteer.core.model_interface import WhiteBoxModel
        from deepsteer.core.types import AccessTier

        self.torch = torch
        self.tok = S.V._tok(path)  # noqa: SLF001
        self.m = WhiteBoxModel(path, access_tier=AccessTier.WEIGHTS, device="cpu",
                               torch_dtype=torch.float32).model
        if lora:
            from peft import PeftModel

            self.m = PeftModel.from_pretrained(self.m, lora)
        self.m.eval()


def _engine(dry: bool, adapter: str | None):
    if dry:
        path = S.V._model_path(True)  # noqa: SLF001
        return path, DryEngine(path, lora=adapter)
    return S.subject(False, "a0", lora=True)


def legit_check(out: Path, dry: bool, arm: str = "a0", engine=None) -> dict:
    """Rider d: A0 (or A0s) on the first 20 legitimate-order items at k 4 before the cell is read;
    a cell whose following share is at or below the floor is rebuilt, not read. ``engine`` (an
    existing ``(path, engine)``) is reused: one engine per process (vLLM does not free a first
    engine's memory before a second starts)."""
    sc = S.scenarios(dry)
    first = set(sc["ids"]["legit"][:LEGIT_CHECK_N])
    tag = f"{arm}_legitcheck"
    S.stage_gen(out, arm, ["legit_operator"], dry, tag=tag, only=first,
                k_override={"legit_operator": 4}, engine=engine)
    rows = [json.loads(x) for f in (out / "gen" / tag).glob("*.jsonl") for x in open(f)]
    ordered = {s: [o.option_id for o in sc["objs"][s].options if o.norm_status == "consistent"][0]
               for s in first}
    acted = [r for r in rows if r["option"] is not None]
    share = sum(r["option"] == ordered[r["scenario_id"]] for r in acted) / max(len(acted), 1)
    rec = {"arm": arm, "n_items": len(first), "rollouts": len(rows), "follow_share": share,
           "floor": LEGIT_FLOOR, "read": share > LEGIT_FLOOR}
    S.V._w(out, f"legit_check_{arm}", rec)  # noqa: SLF001
    return rec


def stage_read(out: Path, adapters: list[tuple[str, str, str]], dry: bool) -> dict:
    """The screen read (Amendment 20 item 2) of each ``(tag, adapter dir, kind)``: the parity check
    once per kind first (a kind that fails is banked, not read), then the operator and plain cells
    on the 458 clear items at k 2, cell A′ at k 8 and the legitimate-order cell at k 4, through
    S0a's generation stage on one engine with the adapter served natively."""
    clear = clear_ids() if not dry else None
    res: dict = {"parity": {}, "read": [], "banked": []}
    eng = None if dry else _engine(False, None)
    for n, (tag, adir, kind) in enumerate(adapters, 1):
        if dry:
            eng = _engine(True, adir)
            req = None
        else:
            from vllm.lora.request import LoRARequest

            req = LoRARequest(tag, n, adir)
        if kind not in res["parity"]:
            res["parity"][kind] = served_parity(out, kind, eng[1], req, dry)["passes"]
        if not res["parity"][kind]:
            res["banked"].append(tag)
            continue
        for part in READ_PARTS:
            S.stage_gen(out, "a0", [part], dry, tag=tag, lora=req, k_override=READ_K,
                        engine=eng, only=clear if part.startswith("panel") else None)
        res["read"].append(tag)
    S.V._w(out, "read", res)  # noqa: SLF001
    return res

