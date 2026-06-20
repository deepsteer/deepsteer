#!/usr/bin/env python3
"""Phase 1 per-model orchestrator: one identical decomposition chain per family.

For each model key in the registry, runs the SAME five steps with the SAME
conventions, injecting only the per-model layer/band so the comparison stays
valid:

  base model:
    1. moral MFT directions  -> {key}_base/exp1_probe_directions.npz
       (Paper 3 exp1_2_3_framework_geometry.py, raw text, --skip-bootstrap)
    2. persona direction     -> {key}_base/persona_directions.npz
       (Paper 5 persona_probe_base.py, raw)
  instruct model:
    3. persona direction     -> {key}_instruct/persona_directions.npz  (raw)
    4. refusal direction + consolidation + geometry
       -> {key}_instruct/refusal_directions.npz + refusal_extraction.json
       (Paper 6 extract_refusal.py, chat, real Arditi prompts)
    5. decompose refusal -> {moral, persona, residual}
       -> {key}/refusal_decomposition.json
       (Paper 5 measure_refusal_decomposition.py, band/layer from registry)

The headline layer (depth-fraction 0.5) and stable band come from
``model_registry``; everything else (pooling, direction kind, contrast sets,
input_format) is identical across models by construction.

VALIDATE=1 runs a cheap smoke (tiny dataset-target, capped prompts, headline
layer only) to confirm the pipeline runs on a non-OLMo arch before the full pass.
--dry-run prints the command chain without executing (no GPU needed).

Usage:
    python papers/6_cross_model/scripts/run_phase1.py --models all --dry-run
    VALIDATE=1 python papers/6_cross_model/scripts/run_phase1.py --models qwen25
    python papers/6_cross_model/scripts/run_phase1.py --models all
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

_PAPER6 = Path(__file__).resolve().parent.parent
_REPO = _PAPER6.parent.parent
_P3 = _REPO / "papers" / "3_moral_geometry" / "scripts"
_P5 = _REPO / "papers" / "5_moral_alignment" / "scripts"
_P6 = _PAPER6 / "scripts"
_REFUSAL_PROMPTS = _REPO / "papers" / "5_moral_alignment" / "refusal_prompts.json"
_OUT = _PAPER6 / "outputs"


def _exp1(model: str, out: str, dataset_target: int, validate: bool) -> list[str]:
    cmd = [sys.executable, str(_P3 / "exp1_2_3_framework_geometry.py"),
           "--model", model, "--output-dir", out,
           "--dataset-target", str(dataset_target), "--skip-bootstrap"]
    return cmd


def _persona(model: str, out: str) -> list[str]:
    return [sys.executable, str(_P5 / "persona_probe_base.py"),
            "--model", model, "--output-dir", out, "--input-format", "raw"]


def _refusal(spec: reg.ModelSpec, out_instruct: str, moral_npz: str,
             validate: bool, max_prompts: int | None) -> list[str]:
    lo, hi = spec.band
    cmd = [sys.executable, str(_P6 / "extract_refusal.py"),
           "--model", spec.instruct_repo, "--prompts", str(_REFUSAL_PROMPTS),
           "--moral-npz", moral_npz,
           "--layer", str(spec.primary_layer), "--band", str(lo), str(hi),
           "--input-format", spec.input_format_refusal,
           "--output-dir", out_instruct]
    if validate:
        cmd += ["--only-headline-layer"]
        if max_prompts:
            cmd += ["--max-prompts", str(max_prompts)]
    return cmd


def _decompose(spec: reg.ModelSpec, base_dir: str, instruct_dir: str, out_json: str) -> list[str]:
    lo, hi = spec.band
    return [sys.executable, str(_P5 / "measure_refusal_decomposition.py"),
            "--refusal-npz", f"{instruct_dir}/refusal_directions.npz",
            "--moral-npz", f"{base_dir}/exp1_probe_directions.npz",
            "--persona-npz", f"{instruct_dir}/persona_directions.npz",
            "--persona-base-npz", f"{base_dir}/persona_directions.npz",
            "--refusal-layer", str(spec.primary_layer), "--band", str(lo), str(hi),
            "--output", out_json]


def steps_for(spec: reg.ModelSpec, *, validate: bool, dataset_target: int,
              max_prompts: int | None) -> list[tuple[str, list[str]]]:
    """The five-step chain for one family, as (label, argv) pairs."""
    base_dir = str(_OUT / spec.base_out)
    instruct_dir = str(_OUT / spec.instruct_out)
    decomp_dir = str(_OUT / spec.key)
    moral_npz = f"{base_dir}/exp1_probe_directions.npz"
    return [
        (f"{spec.key}:1-moral-base", _exp1(spec.base_repo, base_dir, dataset_target, validate)),
        (f"{spec.key}:2-persona-base", _persona(spec.base_repo, base_dir)),
        (f"{spec.key}:3-persona-instruct", _persona(spec.instruct_repo, instruct_dir)),
        (f"{spec.key}:4-refusal-instruct",
         _refusal(spec, instruct_dir, moral_npz, validate, max_prompts)),
        (f"{spec.key}:5-decompose",
         _decompose(spec, base_dir, instruct_dir, f"{decomp_dir}/refusal_decomposition.json")),
    ]


def run_model(spec: reg.ModelSpec, *, dry_run: bool, validate: bool,
              dataset_target: int, max_prompts: int | None) -> bool:
    print(f"\n{'='*70}\n== {spec.key}  ({spec.family}, {spec.n_layers}L, "
          f"band {list(spec.band)}, headline L{spec.primary_layer})"
          f"{'  [GATED]' if spec.gated else ''}\n{'='*70}")
    for label, cmd in steps_for(spec, validate=validate, dataset_target=dataset_target,
                                max_prompts=max_prompts):
        print(f"\n>> {label}\n   {' '.join(cmd)}")
        if dry_run:
            continue
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(_REPO))
        if proc.returncode != 0:
            print(f"!! {label} FAILED (exit {proc.returncode}) after {time.time()-t0:.0f}s")
            if spec.gated:
                print(f"!! {spec.key} is gated; check HF_TOKEN has accepted-license access.")
            return False
        print(f"   ok ({time.time()-t0:.0f}s)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 6 Phase 1 per-model orchestrator")
    ap.add_argument("--models", default="all",
                    help="'all' or comma list of keys: " + ",".join(reg.PANEL_ORDER))
    ap.add_argument("--dry-run", action="store_true", help="Print commands, do not execute.")
    ap.add_argument("--dataset-target", type=int, default=40, help="MFT pairs per foundation.")
    ap.add_argument("--max-prompts", type=int, default=8,
                    help="Per-class refusal-prompt cap in VALIDATE mode.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    validate = os.environ.get("VALIDATE", "0") == "1"
    dataset_target = 4 if validate else args.dataset_target
    max_prompts = args.max_prompts if validate else None

    keys = reg.PANEL_ORDER if args.models == "all" else [k.strip() for k in args.models.split(",")]
    specs = [reg.get(k) for k in keys]

    mode = "DRY-RUN" if args.dry_run else ("VALIDATE smoke" if validate else "FULL")
    print(f"Paper 6 Phase 1 [{mode}] — models: {keys}  (dataset-target={dataset_target})")

    results: dict[str, bool] = {}
    for spec in specs:
        results[spec.key] = run_model(
            spec, dry_run=args.dry_run, validate=validate,
            dataset_target=dataset_target, max_prompts=max_prompts,
        )

    print(f"\n{'='*70}\nSummary [{mode}]:")
    for k in keys:
        print(f"  {k:10s} {'ok' if results[k] else 'FAILED'}")
    if not args.dry_run and not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
