#!/usr/bin/env python3
"""Phase 2b orchestrator: causal load-bearing test, with closure-robustness.

Per model, runs the three-way causal ablation (moral / persona / refusal-yardstick
/ random-floor). The DISTILLS are run at TWO budgets — short (closes their quick
deliberations) and long (lets their rambling deliberations complete) — so a
functional-vs-imitated asymmetry cannot be a closure-selection artifact
(requirement 1). GPT-OSS (traces ~54 tok) needs only the short budget.

Reuses Phase 1 subspace + refusal directions from each model's output dir
(exp1_probe_directions.npz, persona_directions.npz, two_site_refusal_directions.npz);
run Phase 1 first. GPT-OSS auto-dequantizes to bf16.

VALIDATE=1 = cheap smoke (few prompts/draws, short rollouts). --dry-run prints.

Usage:
    python papers/7_reasoning/scripts/run_phase2b.py --models all --dry-run
    VALIDATE=1 python papers/7_reasoning/scripts/run_phase2b.py --models gpt_oss_20b
    python papers/7_reasoning/scripts/run_phase2b.py --models all
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

SHORT_TOK = 512
LONG_TOK = 1536   # closure-robustness: let the distills' long deliberations complete


def budgets_for(spec) -> list[int]:
    """GPT-OSS: short only (traces ~54 tok). Distills: short + long (closure-robust)."""
    return [SHORT_TOK] if spec.provenance == reg.Provenance.RL_DELIBERATIVE else [SHORT_TOK, LONG_TOK]


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 7 Phase 2b orchestrator")
    ap.add_argument("--models", default="all", help="'all' or comma list: " + ",".join(reg.PANEL_ORDER))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n", type=int, default=24, help="harmful prompts/condition.")
    ap.add_argument("--n-random", type=int, default=8, help="random subspace draws (noise floor).")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override the per-model budget (single value for all models).")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    validate = os.environ.get("VALIDATE", "0") == "1"
    n = 4 if validate else args.n
    n_random = 3 if validate else args.n_random
    keys = reg.PANEL_ORDER if args.models == "all" else [k.strip() for k in args.models.split(",")]
    specs = [reg.get(k) for k in keys]
    mode = "DRY-RUN" if args.dry_run else ("VALIDATE smoke" if validate else "FULL")
    print(f"Paper 7 Phase 2b [{mode}] — models: {keys} (n={n}/cond, n_random={n_random})")

    results = {}
    for spec in specs:
        out = _OUT / spec.out
        if validate:
            budgets = [384]
        elif args.max_new_tokens is not None:
            budgets = [args.max_new_tokens]          # explicit override (all models)
        else:
            budgets = budgets_for(spec)
        ok = True
        for tok in budgets:
            cmd = [sys.executable, str(_P7S / "causal_ablation.py"),
                   "--key", spec.key, "--prompts", str(_REFUSAL),
                   "--moral-npz", str(out / "exp1_probe_directions.npz"),
                   "--persona-npz", str(out / "persona_directions.npz"),
                   "--refusal-npz", str(out / "two_site_refusal_directions.npz"),
                   "--layer", str(spec.primary_layer),
                   "--band", str(spec.band[0]), str(spec.band[1]),
                   "--n", str(n), "--n-random", str(n_random), "--max-new-tokens", str(tok),
                   "--output", str(out / f"causal_ablation_t{tok}.json")]
            label = f"{spec.key}:causal@{tok}tok"
            print(f"\n>> {label}\n   {' '.join(cmd)}")
            if args.dry_run:
                continue
            t0 = time.time()
            rc = subprocess.run(cmd, cwd=str(_REPO)).returncode
            print(f"   {'ok' if rc == 0 else f'FAILED ({rc})'} ({time.time()-t0:.0f}s)")
            if rc != 0:
                ok = False
                break
        results[spec.key] = None if args.dry_run else ok

    print(f"\nSummary [{mode}]: " + ", ".join(
        f"{k}={'ok' if results.get(k) else ('-' if args.dry_run else 'FAILED')}" for k in keys))
    print(">> Read each causal_ablation_t*.json: load_bearing + verdict; functional-vs-imitated"
          " = moral drop beats floor+persona in GPT-OSS but NOT the distills (in EITHER budget).")
    if not args.dry_run and not all(v for v in results.values() if v is not None):
        sys.exit(1)


if __name__ == "__main__":
    main()
