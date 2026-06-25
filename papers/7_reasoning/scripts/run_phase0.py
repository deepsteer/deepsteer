#!/usr/bin/env python3
"""Phase 0 orchestrator: smoke + EOP validity anchor + GPT-OSS precision gate.

Runs the no-big-GPU-but-needs-a-GPU steps of Phase 0 as one chain, each with the
Paper 6 conventions read from the Paper 7 registry. Three stages (subset with
``--only``):

  * ``precision`` (0d) — GPT-OSS-20B bf16-dequant fit + mandatory refusal
    positive control (gpt_oss_precision_gate.py).
  * ``smoke`` (0c) — reasoning-hook plumbing on the lightest distill
    (smoke_think_hooks.py).
  * ``anchor`` (0c) — END-OF-PROMPT refusal decomposition on ds_r1_llama8b,
    expected ~Paper 6 Llama (refusal ~99% residual at the prompt boundary). The
    anchor REUSES the Paper 3/5/6 scripts verbatim: at END_OF_PROMPT the ``think``
    format is mechanically identical to Paper 6's ``chat`` (apply template, last
    token, mean-diff is common-mode), so the anchor runs the literal Paper 6
    extractor — the strongest possible "no convention drifted" check.

    Per the locked convention (subspace = reasoning model, raw), the moral + persona
    subspace is collected on ds_r1_llama8b itself (NOT the Llama base); the base is
    reserved for the later supplementary longitudinal probe.

``VALIDATE=1`` runs a cheap smoke (tiny dataset-target, capped prompts, headline
layer only). ``--dry-run`` prints the command chain without executing (no GPU).

Usage:
    python papers/7_reasoning/scripts/run_phase0.py --dry-run
    VALIDATE=1 python papers/7_reasoning/scripts/run_phase0.py --only smoke
    python papers/7_reasoning/scripts/run_phase0.py            # full Phase 0 GPU chain
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
_P6 = _REPO / "papers" / "6_cross_model" / "scripts"
_P7S = _P7 / "scripts"
_REFUSAL_PROMPTS = _REPO / "papers" / "5_moral_alignment" / "refusal_prompts.json"
_OUT = _P7 / "outputs"

_ANCHOR_KEY = "ds_r1_llama8b"   # validity anchor + smoke model (lightest distill)
_PRECISION_KEY = "gpt_oss_20b"  # precision gate (MoE, mxfp4)


def _precision_steps(validate: bool) -> list[tuple[str, list[str]]]:
    spec = reg.get(_PRECISION_KEY)
    out = _OUT / spec.out / "precision_gate.json"
    cmd = [sys.executable, str(_P7S / "gpt_oss_precision_gate.py"),
           "--key", spec.key, "--prompts", str(_REFUSAL_PROMPTS),
           "--n", "4" if validate else "12",
           "--max-new-tokens", "384" if validate else "512",
           "--output", str(out)]
    return [(f"{spec.key}:0d-precision-gate", cmd)]


def _smoke_steps(validate: bool) -> list[tuple[str, list[str]]]:
    spec = reg.get(_ANCHOR_KEY)
    out = _OUT / spec.out / "smoke.json"
    cmd = [sys.executable, str(_P7S / "smoke_think_hooks.py"),
           "--key", spec.key, "--n-prompts", "2" if validate else "3",
           "--max-new-tokens", "384" if validate else "512",
           "--output", str(out)]
    return [(f"{spec.key}:0c-smoke", cmd)]


def _anchor_steps(validate: bool, dataset_target: int,
                  max_prompts: int | None) -> list[tuple[str, list[str]]]:
    """EOP validity-anchor chain on the reasoning model (subspace = reasoning, raw)."""
    spec = reg.get(_ANCHOR_KEY)
    sub = _OUT / spec.out               # reasoning-model subspace + decomposition
    eop = _OUT / f"{spec.key}_eop"      # extract_refusal EOP output dir
    moral_npz = f"{sub}/exp1_probe_directions.npz"
    persona_npz = f"{sub}/persona_directions.npz"
    lo, hi = spec.band

    exp1 = [sys.executable, str(_P3 / "exp1_2_3_framework_geometry.py"),
            "--model", spec.reasoning_repo, "--output-dir", str(sub),
            "--dataset-target", str(dataset_target), "--skip-bootstrap"]
    persona = [sys.executable, str(_P5 / "persona_probe_base.py"),
               "--model", spec.reasoning_repo, "--output-dir", str(sub),
               "--input-format", "raw"]
    # END_OF_PROMPT refusal via the literal Paper 6 extractor. ``think`` == ``chat``
    # mechanically at the prompt boundary, so input-format chat reproduces it.
    refusal = [sys.executable, str(_P6 / "extract_refusal.py"),
               "--model", spec.reasoning_repo, "--prompts", str(_REFUSAL_PROMPTS),
               "--moral-npz", moral_npz, "--layer", str(spec.primary_layer),
               "--band", str(lo), str(hi), "--input-format", "chat",
               "--output-dir", str(eop)]
    if validate:
        refusal += ["--only-headline-layer"]
        if max_prompts:
            refusal += ["--max-prompts", str(max_prompts)]
    decompose = [sys.executable, str(_P5 / "measure_refusal_decomposition.py"),
                 "--refusal-npz", f"{eop}/refusal_directions.npz",
                 "--moral-npz", moral_npz,
                 "--persona-npz", persona_npz, "--persona-base-npz", persona_npz,
                 "--refusal-layer", str(spec.primary_layer), "--band", str(lo), str(hi),
                 "--output", f"{sub}/refusal_decomposition_eop.json"]
    return [
        (f"{spec.key}:anchor-1-moral-subspace(raw)", exp1),
        (f"{spec.key}:anchor-2-persona-subspace(raw)", persona),
        (f"{spec.key}:anchor-3-refusal-EOP(think=chat)", refusal),
        (f"{spec.key}:anchor-4-decompose-EOP", decompose),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 7 Phase 0 GPU orchestrator")
    ap.add_argument("--only", choices=["precision", "smoke", "anchor"], default=None,
                    help="Run a single stage (default: all three).")
    ap.add_argument("--dry-run", action="store_true", help="Print commands, do not execute.")
    ap.add_argument("--dataset-target", type=int, default=40, help="MFT pairs per foundation.")
    ap.add_argument("--max-prompts", type=int, default=8,
                    help="Per-class refusal-prompt cap in VALIDATE mode.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    validate = os.environ.get("VALIDATE", "0") == "1"
    dataset_target = 8 if validate else args.dataset_target
    max_prompts = args.max_prompts if validate else None

    # Order: precision gate first (the heaviest, settles the headline-model setup),
    # then the cheap smoke, then the anchor. Run the smoke before the anchor when
    # doing the distill so a hook failure stops before the full extraction.
    stages: list[tuple[str, list[tuple[str, list[str]]]]] = []
    if args.only in (None, "precision"):
        stages.append(("precision", _precision_steps(validate)))
    if args.only in (None, "smoke"):
        stages.append(("smoke", _smoke_steps(validate)))
    if args.only in (None, "anchor"):
        stages.append(("anchor", _anchor_steps(validate, dataset_target, max_prompts)))

    mode = "DRY-RUN" if args.dry_run else ("VALIDATE smoke" if validate else "FULL")
    print(f"Paper 7 Phase 0 [{mode}] — stages: {[s for s, _ in stages]} "
          f"(dataset-target={dataset_target})")

    results: dict[str, bool] = {}
    for stage_name, steps in stages:
        print(f"\n{'='*72}\n== stage: {stage_name}\n{'='*72}")
        ok = True
        for label, cmd in steps:
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
        results[stage_name] = ok

    print(f"\n{'='*72}\nSummary [{mode}]:")
    for s, _ in stages:
        print(f"  {s:10s} {'ok' if results.get(s) else ('-' if args.dry_run else 'FAILED')}")
    if not args.dry_run and not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
