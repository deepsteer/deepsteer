#!/usr/bin/env python3
"""Shared pieces of the pilot pods (PILOT_BUILD.md §1, §5): the HF model for training, the
32-prompt parity read per read tag, the fixed serving path, and the LoRA settings of record.

**Parity (Amendment 25 A4, CST-A22):** first-token letter log-probs on 32 dose-0 operator prompts
(the first 32 panel scenarios, letter order k 0). The HF side (`hf_letters`) is the trainer's class
with each read tag's adapter attached, and the base; the served side is vLLM serving the same
adapter natively. A tag passes when TV(served, attached) <= 0.2 TV(base, attached) and TV(served,
base) >= 0.5 TV(base, attached); one write-once record per tag (`parity_record`, `write_once`).
**Serving (Amendment 26 item 4):** vLLM serves the multimodal class, whose decoder sits at
`model.language_model.layers`; the trainer saves under `model.layers`. Every adapter is served from
a copy with its keys on the served class's path (`served_copy`), and the dry stand-in is the served
class (VALIDATE's `HFStandIn`), so a no-op load shows in a dry run. Every read stage also reads the
base model on the same engine (Amendment 26 item 3).
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


def key_prefix(adir: str | Path) -> str:
    """The decoder path a saved adapter's keys carry (``model.layers`` for the trainer's class,
    ``model.language_model.layers`` for the served class), read from the safetensors header."""
    import re

    from safetensors import safe_open

    with safe_open(str(Path(adir) / "adapter_model.safetensors"), "pt") as f:
        keys = list(f.keys())
    m = re.match(r"base_model\.model\.(.+?\.layers)\.\d+\.", keys[0])
    if not m:
        raise ValueError(f"{adir}: unexpected adapter key {keys[0]}")
    return m.group(1)


_PREFIXES: dict[str, dict[str, str]] = {}


def served_copy(out: Path, tag: str, adir: str, dry: bool) -> str:
    """The adapter as the served class names its modules (Amendment 26 item 4): returned as is when
    its keys already carry the served path, else a renamed copy at ``served_adapters/<tag>``
    (outside ``adapters/``, so the end-of-session download skips it)."""
    path = S.V._model_path(dry)  # noqa: SLF001
    pre = _PREFIXES.setdefault(path, decoder_prefixes(path))
    if key_prefix(adir) == pre["served"]:
        return adir
    dst = out / "served_adapters" / tag
    if not (dst / "adapter_model.safetensors").exists():
        rename_adapter(Path(adir), dst, pre)
    return str(dst)


def hf_letters(out: Path, adapters: list[tuple[str, str]], dry: bool) -> dict:
    """Amendment 25 A4's HF side: the trainer's class (bf16 on GPU), the base and each read tag's
    adapter attached, on the 32 parity prompts; ``parity/hf_base.json`` once and
    ``parity/hf_<tag>.json`` per tag, write-once (a tag already on disk is skipped)."""
    import torch
    from peft import PeftModel

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    todo = [(t, d) for t, d in adapters if not (out / "parity" / f"hf_{t}.json").exists()]
    if not todo and (out / "parity" / "hf_base.json").exists():
        return {"done": [t for t, _ in adapters]}
    path = S.V._model_path(dry)  # noqa: SLF001
    wb = WhiteBoxModel(path, access_tier=AccessTier.WEIGHTS, device="cpu" if dry else "cuda",
                       torch_dtype=torch.float32 if dry else torch.bfloat16)
    prompts, lids = parity_prompts(dry)

    def letters(m):
        res = []
        with torch.no_grad():
            for pr in prompts:
                x = torch.tensor([pr], device=m.device)
                lp = torch.log_softmax(m(input_ids=x).logits[0, -1].float(), -1)
                res.append([float(lp[i]) for i in lids])
        return res

    if not (out / "parity" / "hf_base.json").exists():
        write_once(out / "parity" / "hf_base.json", {"letter_ids": lids,
                                                      "class": type(wb.model).__name__,
                                                      "base": letters(wb.model)})
    pm = None
    for t, d in todo:
        if pm is None:
            pm = PeftModel.from_pretrained(wb.model, d, adapter_name=t)
        else:
            pm.load_adapter(d, adapter_name=t)
        pm.set_adapter(t)
        pm.eval()
        write_once(out / "parity" / f"hf_{t}.json", {"tag": t, "adapter": d,
                                                     "attached": letters(pm)})
    return {"done": [t for t, _ in adapters]}


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


BASE_TAG = "base_session"  # Amendment 26 item 3: the base model, same engine, same session


def stage_read(out: Path, adapters: list[tuple[str, str]], dry: bool) -> dict:
    """The screen read (Amendment 20 item 2) of each ``(tag, adapter dir)``: Amendment 25 A4's
    parity per tag on the served copy (a tag that fails is banked, not read), then the operator and
    plain cells on the 458 clear items at k 2, cell A′ at k 8 and the legitimate-order cell at k 4
    on one engine with the adapter served natively. First, once per session, the base model on the
    same engine at the same k (``base_session``; Amendment 26 item 3). Needs ``hf_letters`` on disk
    for every tag."""
    clear = clear_ids() if not dry else None
    hfb = json.loads((out / "parity" / "hf_base.json").read_text())
    prompts, lids = parity_prompts(dry)
    path, eng = S.subject(dry, "a0", lora=not dry)  # dry: VALIDATE's stand-in, the served class
    served_base = eng.first_logprobs(prompts, lids)
    res: dict = {"parity": {}, "read": [], "banked": [], "base_tag": BASE_TAG}
    for part in READ_PARTS:  # skipped chunk by chunk when already on disk (a resumed session)
        S.stage_gen(out, "a0", [part], dry, tag=BASE_TAG, k_override=READ_K, engine=(path, eng),
                    only=clear if part.startswith("panel") else None)
    for n, (tag, adir) in enumerate(adapters, 1):
        sdir = served_copy(out, tag, adir, dry)
        if dry:
            e, req = (path, S.V.HFStandIn(path, lora=sdir)), None
            served = e[1].first_logprobs(prompts, lids)
        else:
            from vllm.lora.request import LoRARequest

            e, req = (path, eng), LoRARequest(tag, n, sdir)
            served = eng.first_logprobs(prompts, lids, lora=req)
        f = out / "parity" / f"{tag}.json"
        if f.exists():  # never overwritten: a resumed session reads the record of record
            rec = json.loads(f.read_text())
        else:
            hf = json.loads((out / "parity" / f"hf_{tag}.json").read_text())
            rec = parity_record(tag, served, hf["attached"], hfb["base"], served_base)
            rec["served_adapter"] = sdir
            write_once(f, rec)
        res["parity"][tag] = rec["passes"]
        if not rec["passes"]:
            res["banked"].append(tag)
            continue
        for part in READ_PARTS:
            S.stage_gen(out, "a0", [part], dry, tag=tag, lora=req, k_override=READ_K,
                        engine=e, only=clear if part.startswith("panel") else None)
        res["read"].append(tag)
    S.V._w(out, f"read_{'_'.join(t for t, _ in adapters)}", res)  # noqa: SLF001
    return res
