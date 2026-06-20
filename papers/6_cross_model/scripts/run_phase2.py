#!/usr/bin/env python3
"""Phase 2 per-model orchestrator: Heretic ablation + comprehension dissociation.

For each model, ablate the instruct model's single refusal direction (Arditi/
Heretic uniform orthogonalization) and re-measure moral comprehension on the
ablated model vs the un-ablated instruct. The discriminator the brief asks for:
does ablating refusal incur collateral COMPREHENSION damage?

  clean dissociation  (Delta ~ 0)  -> refusal is not comprehension-load-bearing
                                       (structural, general; the Paper 5 result)
  comprehension drops (Delta > 0)  -> refusal routes through comprehension
                                       features in that model (the positive
                                       linkage, found by observation not forcing)

Per model the chain is (reuses Paper 5 tooling verbatim, per-model layer from the
registry; identical to eval_pipeline.battery minus the coupling-scenario step):

  1. heretic_ablation.py  -> {key}/heretic/ablated_model + refusal_morality_geometry
     (confirms the refusal->moral projection at the ablation layer; ablation
     strips refusal, expected for all three known Heretic targets).
  2. pipeline_study.py over a 2-state grid {instruct, ablated} -> fresh moral
     probe accuracy + framework eff-dim for each state (raw probing).
  3. behavioral_baseline.py (both) on each state -> moral-judgment accuracy +
     persona-shift baseline compliance (the latter rises if refusal is stripped).
  4. moral_dependency.py on each state (FROZEN base directions) -> moral-text
     dependency score.
  5. remove the 14 GB ablated_model (regenerable; metrics JSON are kept) so disk
     stays bounded across the three families.

phase2_table.py then assembles the per-model instruct-vs-ablated comprehension
delta + the cross-model table.

Safety: this is purely diagnostic. It ablates an existing refusal direction and
measures what happens to comprehension. It does NOT build, strengthen, or seek a
harder-to-remove refusal.

Usage:
    python papers/6_cross_model/scripts/run_phase2.py --models all --dry-run
    VALIDATE=1 python papers/6_cross_model/scripts/run_phase2.py --models qwen25
    python papers/6_cross_model/scripts/run_phase2.py --models all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as reg  # noqa: E402

logger = logging.getLogger(__name__)

_PAPER6 = Path(__file__).resolve().parent.parent
_REPO = _PAPER6.parent.parent
_P5 = _REPO / "papers" / "5_moral_alignment" / "scripts"
_REFUSAL_PROMPTS = _REPO / "papers" / "5_moral_alignment" / "refusal_prompts.json"
_OUT = _PAPER6 / "outputs"


def _write_grid(spec: reg.ModelSpec, ablated_path: str, battery_dir: Path, *, write: bool) -> Path:
    """Two-state grid for pipeline_study: the instruct repo + the ablated dir."""
    gpath = battery_dir / "grid.json"
    if write:
        grid = [
            {"label": "instruct", "repo": spec.instruct_repo},
            {"label": "ablated", "repo": ablated_path},
        ]
        battery_dir.mkdir(parents=True, exist_ok=True)
        gpath.write_text(json.dumps(grid, indent=2))
    return gpath


def steps_for(spec: reg.ModelSpec, *, dry_run: bool, dataset_target: int,
              max_texts: int | None) -> tuple[list[tuple[str, list[str]]], Path, str]:
    """Build the (label, argv) chain for one family. Returns (steps, battery_dir, ablated)."""
    base_dir = _OUT / spec.base_out
    base_npz = str(base_dir / "exp1_probe_directions.npz")
    heretic_dir = _OUT / spec.key / "heretic"
    ablated = str(heretic_dir / "ablated_model")
    battery = _OUT / spec.key / "battery"
    grid = _write_grid(spec, ablated, battery, write=not dry_run)
    instruct_dir = str(battery / "instruct")
    ablated_dir = str(battery / "ablated")

    def dep(model: str, label: str, out: str) -> list[str]:
        cmd = [sys.executable, str(_P5 / "moral_dependency.py"), "--model", model,
               "--directions-npz", base_npz, "--no-per-text", "--label", label,
               "--output-dir", out]
        if max_texts:
            cmd += ["--max-texts", str(max_texts)]
        return cmd

    def behav(model: str, out: str) -> list[str]:
        return [sys.executable, str(_P5 / "behavioral_baseline.py"), "--model", model,
                "--benchmark", "both", "--input-format", "chat", "--output-dir", out]

    steps = [
        (f"{spec.key}:1-ablate", [
            sys.executable, str(_P5 / "heretic_ablation.py"),
            "--model", spec.instruct_repo, "--prompts", str(_REFUSAL_PROMPTS),
            "--moral-npz", base_npz, "--refusal-layer", str(spec.primary_layer),
            "--input-format", spec.input_format_refusal,
            "--output-dir", str(heretic_dir), "--save-model"]),
        (f"{spec.key}:2-probe-effdim", [
            sys.executable, str(_P5 / "pipeline_study.py"),
            "--grid", str(grid), "--base-dir", str(base_dir),
            "--input-format", "raw", "--output-dir", str(battery),
            "--dataset-target", str(dataset_target)]),
        (f"{spec.key}:3-behavioral-instruct", behav(spec.instruct_repo, instruct_dir)),
        (f"{spec.key}:4-behavioral-ablated", behav(ablated, ablated_dir)),
        (f"{spec.key}:5-dependency-instruct", dep(spec.instruct_repo, "instruct", instruct_dir)),
        (f"{spec.key}:6-dependency-ablated", dep(ablated, "ablated", ablated_dir)),
    ]
    return steps, battery, ablated


def run_model(spec: reg.ModelSpec, *, dry_run: bool,
              dataset_target: int, max_texts: int | None, keep_ablated: bool) -> bool:
    print(f"\n{'='*70}\n== {spec.key}  ({spec.family}, {spec.n_layers}L, ablate "
          f"L{spec.primary_layer}){'  [GATED]' if spec.gated else ''}\n{'='*70}")
    steps, _battery, ablated = steps_for(
        spec, dry_run=dry_run, dataset_target=dataset_target, max_texts=max_texts)
    ok = True
    for label, cmd in steps:
        print(f"\n>> {label}\n   {' '.join(cmd)}")
        if dry_run:
            continue
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(_REPO))
        if proc.returncode != 0:
            print(f"!! {label} FAILED (exit {proc.returncode}) after {time.time()-t0:.0f}s")
            if spec.gated:
                print(f"!! {spec.key} is gated; check HF_TOKEN has accepted-license access.")
            ok = False
            break
        print(f"   ok ({time.time()-t0:.0f}s)")
    # Bound disk: the 14 GB ablated model is regenerable; metrics JSON are kept.
    if not dry_run and not keep_ablated and Path(ablated).exists():
        shutil.rmtree(ablated, ignore_errors=True)
        print(f"   cleaned {ablated}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 6 Phase 2 ablation + dissociation battery")
    ap.add_argument("--models", default="all",
                    help="'all' or comma list of keys: " + ",".join(reg.PANEL_ORDER))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dataset-target", type=int, default=40, help="MFT pairs per foundation.")
    ap.add_argument("--max-texts", type=int, default=32, help="Dependency text cap in VALIDATE.")
    ap.add_argument("--keep-ablated", action="store_true",
                    help="Do not delete the ablated model dir after the battery.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    validate = os.environ.get("VALIDATE", "0") == "1"
    dataset_target = 8 if validate else args.dataset_target
    max_texts = args.max_texts if validate else None

    keys = reg.PANEL_ORDER if args.models == "all" else [k.strip() for k in args.models.split(",")]
    specs = [reg.get(k) for k in keys]
    mode = "DRY-RUN" if args.dry_run else ("VALIDATE smoke" if validate else "FULL")
    print(f"Paper 6 Phase 2 [{mode}] — models: {keys}  (dataset-target={dataset_target})")

    results = {}
    for spec in specs:
        results[spec.key] = run_model(
            spec, dry_run=args.dry_run, dataset_target=dataset_target,
            max_texts=max_texts, keep_ablated=args.keep_ablated)

    print(f"\n{'='*70}\nSummary [{mode}]:")
    for k in keys:
        print(f"  {k:10s} {'ok' if results[k] else 'FAILED'}")
    if not args.dry_run and not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
