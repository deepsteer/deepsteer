#!/usr/bin/env python3
"""ANOMALIES KDG-A20: read the p2d VALIDATE-only record (rule in the entry, pushed before the pod).

    python3 papers/kdg_panel/scripts/analyze_kdg_a20_validate.py [--dir outputs/p2d/validate]

Per model, from ``forward_matches_generate.json`` (G6 batch + 0-70-token pad ladder): the largest
|forward − generate| on the option tokens for readout version 1 (the cells of record), version 2
(the harness from 2026-10-07) and the unpadded read, by pad bin, and the rule outcome:
  v1 bound = max v1_vs_gen over prompts with pad <= the model's largest panel pad (pad table,
  ``analysis_kdg_a20_pads.json``); bounded iff <= 0.05 nats (the gate the cells of record ran
  under).
  v2 must pass the same 0.05 gate; a v2 miss on a dense model is escalated, not re-read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KDG = REPO / "papers" / "kdg_panel"
BAR = 0.05
MODELS = {  # p2d registry key -> pad-table key
    "olmo3_instruct": "olmo3_final",
    "llama31_instruct_meta": "llama31_instruct_meta",
    "tulu3_final": "tulu3_final",
    "qwen25_instruct_p1": "qwen25_instruct",
}
BINS = (("0", 0, 0), ("1-28", 1, 28), ("29-71", 29, 71), ("72-1000", 72, 1000),
        (">1000", 1001, 10**9))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=KDG / "outputs" / "p2d" / "validate")
    ap.add_argument("--out", type=Path, default=KDG / "data" / "analysis_kdg_a20_validate.json")
    a = ap.parse_args(argv)
    pads = json.loads((KDG / "data" / "analysis_kdg_a20_pads.json").read_text())["models"]
    rep: dict = {"rule": "ANOMALIES KDG-A20, p2d reading rule", "bar": BAR, "models": {}}
    for key, pkey in MODELS.items():
        f = a.dir / key / "forward_matches_generate.json"
        if not f.exists():
            rep["models"][key] = {"status": "missing"}
            continue
        rec = json.loads(f.read_text())
        rows = [dict(r, batch=b) for b, rs in rec["batches"].items() for r in rs]
        panel_max = max(pads[pkey]["pad_primary"]["max"], pads[pkey]["pad_twin"]["max"])
        in_range = [r for r in rows if r["pad"] <= panel_max]
        covered = sorted({r["pad"] for r in in_range})
        by_bin = {}
        for name, lo, hi in BINS:
            rs = [r for r in rows if lo <= r["pad"] <= hi]
            if rs:
                by_bin[name] = {k: max(r[k] for r in rs) for k in
                                ("v1_vs_gen", "v2_vs_gen", "unpadded_vs_gen")} | {"n": len(rs)}
        v1 = max(r["v1_vs_gen"] for r in in_range)
        v2 = max(r["v2_vs_gen"] for r in rows)
        rep["models"][key] = {
            "panel_max_pad": panel_max, "pads_measured_in_range": covered,
            "v1_bound_in_range": v1, "v2_max": v2,
            "unpadded_max": max(r["unpadded_vs_gen"] for r in rows), "by_pad_bin": by_bin,
            "v1_outcome": "bounded" if v1 <= BAR else "re-read (item 2)",
            "v2_outcome": "passes" if v2 <= BAR
            else "escalate (v2 harness misses on a dense model)",
        }
    a.out.write_text(json.dumps(rep, indent=1))
    for k, v in rep["models"].items():
        if v.get("status") == "missing":
            print(f"{k:22s} missing")
            continue
        print(f"{k:22s} v1 bound (pad<={v['panel_max_pad']:.0f}) {v['v1_bound_in_range']:.4f} -> "
              f"{v['v1_outcome']}; v2 max {v['v2_max']:.4f} -> {v['v2_outcome']}; "
              f"pads measured {v['pads_measured_in_range']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
