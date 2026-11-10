#!/usr/bin/env python3
"""CST-A22's parity mini-pod (Amendment 25 Part A): is the SDF adapter served at all?

    cst_parity_pod.py --stage prep|hf|served|gate --out DIR [--adapters DIR,DIR] [--dry-run]

On the 32 dose-0 parity prompts (`pilot_pod_lib.parity_prompts`), first-token letter log-probs:

- ``prep``: the decoder path under the trainer's class (`AutoModelForCausalLM`) and the served class
  (`AutoModelForImageTextToText`, the class vLLM serves), from the config alone; each adapter is
  also written with its keys moved to the served class's path (`renamed/<name>`).
- ``hf``: the trainer's class on HF (bf16), the base with no adapter and each adapter as saved.
- ``served``: vLLM with native LoRA (one engine): the base with no adapter, each adapter as saved
  and each renamed. A dry run serves through VALIDATE's stand-in, which loads the served class.
- ``gate``: Amendment 25 A4 per read tag (``<name>__saved``, ``<name>__renamed``), one record per
  tag, never overwritten, served letters saved; the summary names CST-A22's branch.

Each stage runs in its own process (the HF model must leave the GPU before vLLM starts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cst_s0a_pod as S  # noqa: E402
import pilot_pod_lib as PL  # noqa: E402

DEFAULT_ADAPTERS = ("outputs/pilot_sdf/adapters/d_doc/1e+06",
                    "outputs/pilot_sdf/adapters/d_doc/1e+06_rec")


def _adapters(arg: str | None) -> dict[str, Path]:
    dirs = [Path(x) for x in arg.split(",")] if arg else [S.CST / d for d in DEFAULT_ADAPTERS]
    out = {}
    for d in dirs:
        if not (d / "adapter_model.safetensors").exists():
            raise SystemExit(f"BAIL: no adapter weights in {d}")
        out[d.name.replace("+", "")] = d
    return out


def _w(out: Path, name: str, rec: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n")


def stage_prep(out: Path, adapters: dict[str, Path], dry: bool) -> dict:
    path = S.V._model_path(dry)  # noqa: SLF001
    pre = PL.decoder_prefixes(path)
    rec = {"prefixes": pre, "renamed": {}}
    for name, d in adapters.items():
        rec["renamed"][name] = PL.rename_adapter(d, out / "renamed" / name, pre)
    _w(out, "prep", rec)
    print(">> prep:", json.dumps(rec))
    return rec


def _letters(model, prompts, lids) -> list[list[float]]:
    import torch

    res = []
    with torch.no_grad():
        for p in prompts:
            x = torch.tensor([p], device=model.device)
            lp = torch.log_softmax(model(input_ids=x).logits[0, -1].float(), -1)
            res.append([float(lp[i]) for i in lids])
    return res


def stage_hf(out: Path, adapters: dict[str, Path], dry: bool) -> dict:
    """The trainer's class, as the parity read's attached side was computed in training."""
    import torch
    from peft import PeftModel

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    path = S.V._model_path(dry)  # noqa: SLF001
    wb = WhiteBoxModel(path, access_tier=AccessTier.WEIGHTS, device="cpu" if dry else "cuda",
                       torch_dtype=torch.float32 if dry else torch.bfloat16)
    prompts, lids = PL.parity_prompts(dry)
    rec = {"class": type(wb.model).__name__, "letter_ids": lids,
           "base": _letters(wb.model, prompts, lids), "attached": {}}
    names = list(adapters)
    pm = PeftModel.from_pretrained(wb.model, str(adapters[names[0]]), adapter_name=names[0])
    for name in names[1:]:
        pm.load_adapter(str(adapters[name]), adapter_name=name)
    pm.eval()
    for name in names:
        pm.set_adapter(name)
        rec["attached"][name] = _letters(pm, prompts, lids)
    with pm.disable_adapter():
        rec["base_after_load"] = _letters(pm, prompts, lids)  # the disabled path equals the base
    _w(out, "letters_hf", rec)
    print(">> hf:", json.dumps({"class": rec["class"], "adapters": names}))
    return rec


def stage_served(out: Path, adapters: dict[str, Path], dry: bool) -> dict:
    path = S.V._model_path(dry)  # noqa: SLF001
    prompts, lids = PL.parity_prompts(dry)
    variants = {f"{n}__saved": d for n, d in adapters.items()}
    variants |= {f"{n}__renamed": out / "renamed" / n for n in adapters}
    rec: dict = {"letter_ids": lids, "served": {}}
    if dry:
        rec["engine"] = "dry: VALIDATE's stand-in (the served class, AutoModelForImageTextToText)"
        rec["base"] = S.V.HFStandIn(path).first_logprobs(prompts, lids)
        for tag, d in variants.items():
            rec["served"][tag] = S.V.HFStandIn(path, lora=str(d)).first_logprobs(prompts, lids)
    else:
        from vllm.lora.request import LoRARequest

        _, eng = S.subject(False, "a0", lora=True)
        rec["engine"] = "vllm native LoRA (the engine of record, enable_lora)"
        rec["base"] = eng.first_logprobs(prompts, lids)
        for n, (tag, d) in enumerate(variants.items(), 1):
            rec["served"][tag] = eng.first_logprobs(prompts, lids,
                                                    lora=LoRARequest(tag, n, str(d)))
    _w(out, "letters_served", rec)
    print(">> served:", json.dumps({"engine": rec["engine"], "tags": list(rec["served"])}))
    return rec


def branch(recs: dict[str, dict]) -> str:
    """CST-A22's reading from the per-tag records (Amendment 25 A5)."""
    saved = [r for t, r in recs.items() if t.endswith("__saved")]
    renamed = [r for t, r in recs.items() if t.endswith("__renamed")]
    if all(r["passes"] for r in saved):
        return "R_null branch: the adapters are served as saved (Amendment 25 A5)"
    as_base = all(r["tv_served_vs_base"] <= r["attached_max"] for r in saved)
    if as_base and all(r["passes"] for r in renamed):
        return ("R_noserve: served as saved reads as the base; the renamed path passes and is the "
                "fixed path (Amendment 25 A5)")
    if as_base:
        return "R_noserve, no fixed path: the renamed path also fails; to the author"
    return "undetermined: served differs from both base and attached; to the author"


def stage_gate(out: Path) -> dict:
    hf = json.loads((out / "letters_hf.json").read_text())
    sv = json.loads((out / "letters_served.json").read_text())
    recs = {}
    for tag, served in sv["served"].items():
        name = tag.split("__")[0]
        rec = PL.parity_record(tag, served, hf["attached"][name], hf["base"], sv["base"])
        PL.write_once(out / "parity" / f"{tag}.json", rec)
        recs[tag] = rec
    summary = {"rule": "Amendment 25 A4", "branch": branch(recs),
               "tv_base_after_load_vs_base": PL.tv(hf["base_after_load"], hf["base"]),
               "tags": {t: {k: v for k, v in r.items() if k != "served_letters"}
                        for t, r in recs.items()}}
    _w(out, "parity_a22", summary)
    print(">> gate:", json.dumps({"branch": summary["branch"], "tags": {
        t: {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()
            if k in ("tv_served_vs_attached", "tv_served_vs_base", "tv_base_vs_attached",
                     "tv_served_base_vs_hf_base", "passes")} for t, r in summary["tags"].items()}}))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("prep", "hf", "served", "gate"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--adapters", default=None, help="comma-separated adapter dirs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.stage == "gate":
        stage_gate(a.out)
        return
    ads = _adapters(a.adapters)
    {"prep": stage_prep, "hf": stage_hf, "served": stage_served}[a.stage](a.out, ads, a.dry_run)


if __name__ == "__main__":
    main()
