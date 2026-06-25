#!/usr/bin/env python3
"""Phase 2c orchestrator: t_inst/t_post-inst extraction + clustering, per model.

Cheap prereq for the Zhao-grounded keystone: extract diff-of-means at both the
harmfulness (t_inst) and refusal (t_post-inst) sites, report the harmful/harmless
separation at each and (if a refusal baseline is present) the refusal clustering
flip. Reuses the per-model refusal_baseline.json for behavior labels. VALIDATE=1 =
cheap smoke (few prompts). --dry-run prints commands.

Usage:
    python papers/7_reasoning/scripts/run_phase2c.py --models all --dry-run
    VALIDATE=1 python papers/7_reasoning/scripts/run_phase2c.py --models ds_r1_llama8b
    python papers/7_reasoning/scripts/run_phase2c.py --models all
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as reg  # noqa: E402

_P7 = Path(__file__).resolve().parent.parent
_REPO = _P7.parent.parent
_P7S = _P7 / "scripts"
_REFUSAL = _REPO / "papers" / "5_moral_alignment" / "refusal_prompts.json"
_OUT = _P7 / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 7 Phase 2c orchestrator")
    ap.add_argument("--models", default="all", help="'all' or comma list: " + ",".join(reg.PANEL_ORDER))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    validate = os.environ.get("VALIDATE", "0") == "1"
    n = 6 if validate else args.n
    keys = reg.PANEL_ORDER if args.models == "all" else [k.strip() for k in args.models.split(",")]
    specs = [reg.get(k) for k in keys]
    mode = "DRY-RUN" if args.dry_run else ("VALIDATE smoke" if validate else "FULL")
    print(f"Paper 7 Phase 2c [{mode}] — models: {keys} (n={n}/class)")

    results = {}
    for spec in specs:
        out = _OUT / spec.out
        cmd = [sys.executable, str(_P7S / "extract_positions.py"),
               "--key", spec.key, "--prompts", str(_REFUSAL),
               "--layer", str(spec.primary_layer),
               "--band", str(spec.band[0]), str(spec.band[1]),
               "--n", str(n), "--baseline", str(out / "refusal_baseline.json"),
               "--output-dir", str(out)]
        print(f"\n>> {spec.key}:positions (t_inst + t_post-inst @L{spec.primary_layer})\n   {' '.join(cmd)}")
        if args.dry_run:
            results[spec.key] = None
            continue
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(_REPO)).returncode
        results[spec.key] = rc == 0
        print(f"   {'ok' if rc == 0 else f'FAILED ({rc})'} ({time.time()-t0:.0f}s)")
        if rc != 0:
            break

    print(f"\nSummary [{mode}]: " + ", ".join(
        f"{k}={'ok' if results.get(k) else ('-' if args.dry_run else 'FAILED')}" for k in keys))
    print(">> Gate: read position_extraction.json — harmful/harmless should separate at t_inst")
    print(">> (harmfulness); refusal labels (if present) should separate at t_post-inst (flip).")
    if not args.dry_run and not all(v for v in results.values() if v is not None):
        sys.exit(1)


if __name__ == "__main__":
    main()
