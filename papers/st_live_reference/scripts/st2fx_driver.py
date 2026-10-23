"""The format x harm matrix read (ST_PLAN Amendment 33; ST-A22's discriminator) on GPT-OSS-20B:
dose 0, ``dl_chat_neutral``, 8 permutations, single-row (T1.5's instrument), under ``none``,
L-beh-2 (``beh2``) and L-beh (``beh``), each arm its own process (the adapter attached, sha256
checked against ``data/st_fxh_adapters_v1.json``):

- cell ``A``: rule-right items in H1's format (``data/st_fxh_h1fmt_ruleright_v1.json``);
- cell ``B``: rule-harm items in the panel's format (``data/st_fxh_panelfmt_ruleharm_v1.json``);
- cell ``H1``: the rider, the 38 engaged H1_held scenarios (the matrix's H1 diagonal on this
  instrument).

The analysis of record is local (``analyze_st2fx.py``).

    python3 st2fx_driver.py --arm none|beh2|beh --out <dir> --adapters <dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "kdg_panel" / "scripts"))
import screen_lib as L  # noqa: E402

KEY = "gpt_oss_20b"
DATA = HERE.parent / "data"
L_STAR = 13
ARMS = ("none", "beh2", "beh")
CELLS = {"A": DATA / "st_fxh_h1fmt_ruleright_v1.json",
         "B": DATA / "st_fxh_panelfmt_ruleharm_v1.json"}
MANIFEST = DATA / "st_fxh_adapters_v1.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cells(dry: bool) -> dict[str, list]:
    import pod_st_screen as P

    from deepsteer.kdg.schema import load_scenario_dir, load_scenarios

    out = {k: load_scenarios(p)[0] for k, p in CELLS.items()}
    Hs, _ = load_scenario_dir([P.H1_SET])
    by = {s.id: s for s in Hs}
    eng = json.loads((DATA / "t15_engaged_held_v1.json").read_text())["ids"]
    out["H1"] = [by[i] for i in eng]
    return {k: v[:2] for k, v in out.items()} if dry else out


def verify(adapters: Path, arm: str) -> dict:
    want = json.loads(MANIFEST.read_text())["arms"][arm]
    got = {f: hashlib.sha256((adapters / arm / f).read_bytes()).hexdigest() for f in want}
    if got != want:
        raise SystemExit(f"adapter {arm}: sha256 differs from {MANIFEST.name}")
    return got


def load(a):
    import pod_st_screen as P
    import torch
    from kdg_pod_lib import ModelWrapper

    spec = P.registry()[KEY]
    if a.dry_run:
        sys.path.insert(0, str(HERE.parents[2] / "tests" / "scripts"))
        import st_tiny

        from deepsteer.steering import objective_lora as OL

        path = a.out / "_tiny"
        if not (path / "config.json").exists():
            st_tiny.build(KEY, path)
        mw = ModelWrapper(str(path), None, torch_dtype=torch.float32)
        for arm in ARMS[1:]:  # zero-initialized adapters at the tiny width
            d = a.adapters / arm
            if not (d / "adapter_config.json").exists():
                OL.inject_lora(mw.wb, rank=4, dropout=0.0)
                mw.wb.model.save_pretrained(str(d))
                mw.wb._model = mw.wb.model.unload()  # noqa: SLF001
                mw.model = mw.wb.model
        return spec, mw
    return spec, ModelWrapper(spec["repo"], spec["revision"], torch_dtype=torch.bfloat16,
                              attn_implementation="eager", **P.hf_load_kwargs(spec))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--adapters", type=Path, required=True)
    ap.add_argument("--date-pin", default="2026-10-23")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from peft import PeftModel
    from reasoning_formats import build_format

    out = a.out / KEY
    out.mkdir(parents=True, exist_ok=True)
    rp = out / "st2fx_record.json"
    rec = json.loads(rp.read_text()) if rp.exists() else {"model": KEY, "arms": {}}
    if (rec["arms"].get(a.arm) or {}).get("finished"):
        print(f">> st2fx {a.arm}: finished earlier", flush=True)
        return 0
    t0 = time.time()
    spec, mw = load(a)
    shas = None
    if a.arm != "none":
        shas = None if a.dry_run else verify(a.adapters, a.arm)
        PeftModel.from_pretrained(mw.model, str(a.adapters / a.arm))  # in place
    fmt = build_format(KEY, mw.tok, spec)
    sc = L.Screen(KEY, spec, mw, fmt, out, a.date_pin, dry=False, read_batch=1)
    sc.effort = spec.get("effort_of_record") or fmt.effort_ladder[0]
    sc.unbatched = True
    sc.layers_override = [2] if a.dry_run else [L_STAR]
    sc.dose0 = fmt.dose0[0]
    res = {}
    for cell, items in cells(a.dry_run).items():
        r = L.run_letter_cell(sc, items, "dl_chat_neutral", f"fx/{a.arm}/{cell}")
        res[cell] = {"rows": r["rows"], "seconds_per_row": r["seconds_per_row"]}
        print(f">> st2fx {a.arm} {cell}: {r['rows']} rows", flush=True)
    rec = json.loads(rp.read_text()) if rp.exists() else {"model": KEY, "arms": {}}
    rec["arms"][a.arm] = {"cells": res, "adapter_sha256": shas, "seconds": time.time() - t0,
                          "finished": _now()}
    rp.write_text(json.dumps(rec, indent=1))
    print(f">> st2fx {a.arm}: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
