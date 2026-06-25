#!/usr/bin/env python3
"""Phase 2d orchestrator: reply-inversion yardstick per model.

Tests whether the t_inst harmfulness direction is a CAUSAL handle (steering it
flips the model's harm judgment), held-out + coherence-gated, over a depth-fraction
layer sweep x coefficient grid. The verdict is the HARD GATE for Phase 2e: the
load-bearing test runs only on models where inversion fires cleanly.

VALIDATE=1 = cheap smoke (few held-out prompts, narrow grid). --dry-run prints.

Usage:
    python papers/7_reasoning/scripts/run_phase2d.py --models all --dry-run
    VALIDATE=1 python papers/7_reasoning/scripts/run_phase2d.py --models ds_r1_llama8b
    python papers/7_reasoning/scripts/run_phase2d.py --models all
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
    ap = argparse.ArgumentParser(description="Paper 7 Phase 2d orchestrator")
    ap.add_argument("--models", default="all", help="'all' or comma list: " + ",".join(reg.PANEL_ORDER))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-train", type=int, default=96)
    ap.add_argument("--n-test", type=int, default=24)
    ap.add_argument("--coeffs", default="2,4,8")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    validate = os.environ.get("VALIDATE", "0") == "1"
    n_train = 24 if validate else args.n_train
    n_test = 8 if validate else args.n_test
    coeffs = "4" if validate else args.coeffs
    keys = reg.PANEL_ORDER if args.models == "all" else [k.strip() for k in args.models.split(",")]
    specs = [reg.get(k) for k in keys]
    mode = "DRY-RUN" if args.dry_run else ("VALIDATE smoke" if validate else "FULL")
    print(f"Paper 7 Phase 2d [{mode}] — models: {keys} (n_train={n_train}, n_test={n_test}, "
          f"coeffs={coeffs}; forced-answer logit read)")

    results = {}
    for spec in specs:
        out = _OUT / spec.out / "reply_inversion.json"
        cmd = [sys.executable, str(_P7S / "reply_inversion.py"),
               "--key", spec.key, "--prompts", str(_REFUSAL),
               "--n-train", str(n_train), "--n-test", str(n_test),
               "--coeffs", coeffs, "--output", str(out)]
        print(f"\n>> {spec.key}:reply-inversion\n   {' '.join(cmd)}")
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
    print(">> GATE: read reply_inversion.json 'fires' per model. 2e (load-bearing) runs ONLY on")
    print(">> models where inversion FIRES (held-out coherent flip >= threshold). Watch gpt_oss.")
    if not args.dry_run and not all(v for v in results.values() if v is not None):
        sys.exit(1)


if __name__ == "__main__":
    main()
