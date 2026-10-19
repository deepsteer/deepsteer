#!/usr/bin/env python3
"""The vLLM half of the fold validation (ST G1 item 3; Amendment 9 item 4), in its own process
after the rider-2 driver exits, so the engine starts on an empty GPU.

    python fold_vllm.py --dir OUT/gpt_oss_20b [--date-pin YYYY-MM-DD]

Reads the driver's HF half (``fold/hf.json``, ``fold/rows16.json``, ``fold/letters.npz``) and the
folded checkpoint it left in ``fold/ckpt``; writes ``fold/vllm.json``; removes the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import screen_lib as L  # noqa: E402
import screen_vllm as SV  # noqa: E402
from reasoning_formats import build_format  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

KEY = "gpt_oss_20b"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    a = ap.parse_args()
    import pod_st_screen as P
    from transformers import AutoTokenizer

    fold = a.dir / "fold"
    rec = json.loads((a.dir / "screen_record.json").read_text())
    if not (fold / "hf.json").exists():
        print(">> fold vLLM half: no HF half on disk; nothing to run", flush=True)
        return 0
    spec = P.registry()[KEY]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    fmt = build_format(KEY, tok, spec)
    sc = L.Screen(KEY, spec, L.tok_only(tok), fmt, a.dir, rec["date_pin"])
    sc.effort = json.loads((a.dir / "stack" / "rows64.json").read_text())["effort"]
    Hs, _ = load_scenario_dir([P.H1_SET])
    ctx = SimpleNamespace(spec=spec, H={s.id: s for s in Hs},
                          order=L.h1_train_order(json.loads(P.SPLIT.read_text())))
    try:
        out = SV.fold_vllm_half(sc, ctx, fold)
    except RuntimeError as e:
        out = {"failed": str(e)}
    finally:
        shutil.rmtree(fold / "ckpt", ignore_errors=True)
    (fold / "vllm.json").write_text(json.dumps(out, indent=1))
    ii = out.get("ii_steering_effect_vllm_vs_hf") or {}
    print(f">> fold vLLM half: DoD max {ii.get('max_abs_dod')} (bar {ii.get('bar')}) ok "
          f"{ii.get('ok')}; parse {json.dumps(out.get('iii_generation'))[:200]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
