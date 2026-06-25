#!/usr/bin/env python3
"""Phase 2a orchestrator: position-resolved trace capture + length disentanglement.

Per model, captures position-resolved trace activations (extract_trace_profile);
then runs the cross-model trace-length disentanglement (CPU) once all profiles
exist. The disentanglement is the GATE: only if GPT-OSS's trace morality survives
at matched length does Phase 2b (causal load-bearing) proceed.

VALIDATE=1 = cheap smoke (few prompts, short rollouts). --dry-run prints commands.

Usage:
    python papers/7_reasoning/scripts/run_phase2a.py --models all --dry-run
    VALIDATE=1 python papers/7_reasoning/scripts/run_phase2a.py --models gpt_oss_20b
    python papers/7_reasoning/scripts/run_phase2a.py --models all
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
    ap = argparse.ArgumentParser(description="Paper 7 Phase 2a orchestrator")
    ap.add_argument("--models", default="all", help="'all' or comma list: " + ",".join(reg.PANEL_ORDER))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    validate = os.environ.get("VALIDATE", "0") == "1"
    n = 4 if validate else args.n
    max_new_tokens = 384 if validate else args.max_new_tokens
    keys = reg.PANEL_ORDER if args.models == "all" else [k.strip() for k in args.models.split(",")]
    specs = [reg.get(k) for k in keys]
    mode = "DRY-RUN" if args.dry_run else ("VALIDATE smoke" if validate else "FULL")
    print(f"Paper 7 Phase 2a [{mode}] — models: {keys} (n={n}/class, max_new_tokens={max_new_tokens})")

    results = {}
    for spec in specs:
        cmd = [sys.executable, str(_P7S / "extract_trace_profile.py"),
               "--key", spec.key, "--prompts", str(_REFUSAL),
               "--layer", str(spec.primary_layer), "--n", str(n),
               "--max-new-tokens", str(max_new_tokens),
               "--output-dir", str(_OUT / spec.out)]
        print(f"\n>> {spec.key}:trace-profile (L{spec.primary_layer})\n   {' '.join(cmd)}")
        if args.dry_run:
            results[spec.key] = None
            continue
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(_REPO)).returncode
        results[spec.key] = rc == 0
        print(f"   {'ok' if rc == 0 else f'FAILED ({rc})'} ({time.time()-t0:.0f}s)")
        if rc != 0:
            break

    # Cross-model disentanglement (CPU) once profiles exist.
    dis = [sys.executable, str(_P7S / "trace_length_disentangle.py"),
           "--outputs-dir", str(_OUT), "--keys", ",".join(keys),
           "--output", str(_OUT / "trace_length_disentangle.json")]
    print(f"\n>> disentangle (cross-model, CPU)\n   {' '.join(dis)}")
    if not args.dry_run and all(v for v in results.values() if v is not None):
        subprocess.run(dis, cwd=str(_REPO))

    print(f"\nSummary [{mode}]: " + ", ".join(
        f"{k}={'ok' if results.get(k) else ('-' if args.dry_run else 'FAILED')}" for k in keys))
    if not args.dry_run and not all(v for v in results.values() if v is not None):
        sys.exit(1)


if __name__ == "__main__":
    main()
