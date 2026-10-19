#!/usr/bin/env python3
"""Amendment 7 item 3: the §40 engine-parity reference (zero GPU, from the banked KDG rows).

    python3 papers/st_live_reference/scripts/build_s40_reference.py

KDG_RESULTS §40's GPT-OSS-20B H1 rollouts (pods p2m + p2n; HF generate, medium, cap 1,536, KDG
sampler v2, permutation 0) reduced to what the parity rung needs: per scenario the rendered-prompt
sha (the ST harmony render reproduces all 264), status, parsed letter (parser of record) and
option status. The KDG outputs never sync to pods; this file does. Used only as the engine
reference, never for selection, training or a baseline (G1).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = [ROOT / "papers/kdg_panel/outputs" / p / "h1/gpt_oss_20b/h1/ds_main.jsonl" for p in
       ("p2m", "p2n")]
OUT = ROOT / "papers/st_live_reference/data/s40_reference.json"


def main() -> None:
    rows = {}
    for p in SRC:
        for x in p.read_text().splitlines():
            r = json.loads(x)
            if r["status"] == "not_run":
                continue
            rows[r["scenario_id"]] = {
                "permutation": r["seed"], "order": r["order"], "prompt_sha256": r["prompt_sha256"],
                "status": r["status"], "letter": r["letter"], "norm_status": r["norm_status"],
                "n_gen_tokens": r["n_gen_tokens"], "cap": r["cap"],
                "temperature": r["temperature"], "sampler_version": r["sampler_version"],
                "reasoning_level": r["harmony_reasoning_level"]}
    out = {"source": "KDG_RESULTS §40 (pods p2m + p2n), ds_main.jsonl",
           "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in SRC},
           "n": len(rows), "rows": dict(sorted(rows.items()))}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} scenarios")


if __name__ == "__main__":
    main()
