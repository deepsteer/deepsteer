#!/usr/bin/env python3
"""Phase 1 orchestrator: per-model subspace + two-site (EOP/CoT) decomposition.

For each reasoning model, runs the SAME chain with conventions read from the
registry (so the cross-model comparison stays valid):

  1. moral MFT subspace  -> {key}/exp1_probe_directions.npz   (Paper 3 exp1, raw)
  2. persona direction   -> {key}/persona_directions.npz      (Paper 5 persona, raw)
  3. two-site refusal     -> {key}/two_site_decomposition.json + directions npz
     (Paper 7 extract_two_site: EOP + CoT-last + CoT-mean, decomposition,
     EOP<->CoT cosine + moral asymmetry, last-vs-mean trace-distribution)

The moral/persona subspace is on the REASONING model in ``raw`` (locked Phase 0
convention; GPT-OSS has no base). Subspace steps are skipped when their npz already
exist (ds_r1_llama8b's were produced by the Phase 0c anchor) unless ``--force``.
GPT-OSS loads dequantized automatically (WhiteBoxModel mxfp4 auto-dequant).

VALIDATE=1 runs a cheap smoke (tiny dataset-target, few prompts, short rollouts) to
confirm the CoT extraction plumbing before the full pass. --dry-run prints commands.

Usage:
    python papers/7_reasoning/scripts/run_phase1.py --models all --dry-run
    VALIDATE=1 python papers/7_reasoning/scripts/run_phase1.py --models ds_r1_llama8b
    python papers/7_reasoning/scripts/run_phase1.py --models all
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

logger = logging.getLogger(__name__)

_P7 = Path(__file__).resolve().parent.parent
_REPO = _P7.parent.parent
_P3 = _REPO / "papers" / "3_moral_geometry" / "scripts"
_P5 = _REPO / "papers" / "5_moral_alignment" / "scripts"
_P7S = _P7 / "scripts"
_REFUSAL_PROMPTS = _REPO / "papers" / "5_moral_alignment" / "refusal_prompts.json"
_OUT = _P7 / "outputs"


def steps_for(spec, *, n, max_new_tokens, dataset_target, force):
    out = _OUT / spec.out
    moral_npz = out / "exp1_probe_directions.npz"
    persona_npz = out / "persona_directions.npz"
    lo, hi = spec.band
    steps = []
    if force or not moral_npz.exists():
        steps.append((f"{spec.key}:1-moral-subspace(raw)",
                      [sys.executable, str(_P3 / "exp1_2_3_framework_geometry.py"),
                       "--model", spec.reasoning_repo, "--output-dir", str(out),
                       "--dataset-target", str(dataset_target), "--skip-bootstrap"]))
    else:
        steps.append((f"{spec.key}:1-moral-subspace [cached]", None))
    if force or not persona_npz.exists():
        steps.append((f"{spec.key}:2-persona-subspace(raw)",
                      [sys.executable, str(_P5 / "persona_probe_base.py"),
                       "--model", spec.reasoning_repo, "--output-dir", str(out),
                       "--input-format", "raw"]))
    else:
        steps.append((f"{spec.key}:2-persona-subspace [cached]", None))
    steps.append((f"{spec.key}:3-two-site(EOP/CoT-last/CoT-mean)",
                  [sys.executable, str(_P7S / "extract_two_site.py"),
                   "--key", spec.key, "--prompts", str(_REFUSAL_PROMPTS),
                   "--moral-npz", str(moral_npz), "--persona-npz", str(persona_npz),
                   "--layer", str(spec.primary_layer), "--band", str(lo), str(hi),
                   "--n", str(n), "--max-new-tokens", str(max_new_tokens),
                   "--output-dir", str(out)]))
    # CPU-only bootstrap CIs on the saved headline vectors (fast; the vectors npz
    # also stays on disk so CIs can be recomputed locally without re-running GPU).
    steps.append((f"{spec.key}:4-bootstrap-CIs",
                  [sys.executable, str(_P7S / "bootstrap_two_site.py"),
                   "--key", spec.key,
                   "--vectors", str(out / "two_site_headline_vectors.npz"),
                   "--moral-npz", str(moral_npz),
                   "--output", str(out / "two_site_bootstrap.json")]))
    return steps


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 7 Phase 1 orchestrator")
    ap.add_argument("--models", default="all",
                    help="'all' or comma list: " + ",".join(reg.PANEL_ORDER))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n", type=int, default=64, help="prompts per class (generation-bounded).")
    ap.add_argument("--max-new-tokens", type=int, default=768,
                    help="R1 traces are long; 768 closes more </think> than 512 (smoke: 1/4 at 384).")
    ap.add_argument("--dataset-target", type=int, default=40, help="MFT pairs per foundation.")
    ap.add_argument("--force", action="store_true", help="re-extract cached subspace npz.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    validate = os.environ.get("VALIDATE", "0") == "1"
    n = 4 if validate else args.n
    max_new_tokens = 384 if validate else args.max_new_tokens
    dataset_target = 8 if validate else args.dataset_target

    keys = reg.PANEL_ORDER if args.models == "all" else [k.strip() for k in args.models.split(",")]
    specs = [reg.get(k) for k in keys]
    mode = "DRY-RUN" if args.dry_run else ("VALIDATE smoke" if validate else "FULL")
    print(f"Paper 7 Phase 1 [{mode}] — models: {keys}  (n={n}/class, "
          f"max_new_tokens={max_new_tokens}, dataset_target={dataset_target})")

    results = {}
    for spec in specs:
        print(f"\n{'='*72}\n== {spec.key} ({spec.provenance.value}, {spec.n_layers}L, "
              f"band {list(spec.band)}, headline L{spec.primary_layer}, cot={spec.cot_format.value})"
              f"\n{'='*72}")
        ok = True
        for label, cmd in steps_for(spec, n=n, max_new_tokens=max_new_tokens,
                                    dataset_target=dataset_target, force=args.force):
            if cmd is None:
                print(f"\n>> {label}")
                continue
            print(f"\n>> {label}\n   {' '.join(cmd)}")
            if args.dry_run:
                continue
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=str(_REPO))
            if proc.returncode != 0:
                print(f"!! {label} FAILED (exit {proc.returncode}) after {time.time()-t0:.0f}s")
                ok = False
                break
            print(f"   ok ({time.time()-t0:.0f}s)")
        results[spec.key] = ok

    print(f"\n{'='*72}\nSummary [{mode}]:")
    for k in keys:
        print(f"  {k:14s} {'ok' if results.get(k) else ('-' if args.dry_run else 'FAILED')}")
    if not args.dry_run and not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
