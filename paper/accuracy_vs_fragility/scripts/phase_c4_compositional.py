#!/usr/bin/env python3
"""Phase C4: Compositional moral probe — final-checkpoint validation, then
trajectory + fragility across all 37 OLMo-2 1B early-training checkpoints.

The standard moral probe used in Phase C1 / C2 contrasts morally-marked
vocabulary against neutral vocabulary, which a linear probe can separate via
single-word distributional features. The Phase C2 finding that moral
encoding emerges at step 1K (before sentiment at step 2K and syntax at step
6K) may therefore be measuring "moralized vocabulary becomes statistically
separable" rather than "moral valence is encoded compositionally."

This script runs the lexical-accessibility ablation. It uses the
compositional minimal-pair dataset
(:mod:`deepsteer.datasets.compositional_moral_pairs`) whose contrast tokens
are individually mild and only flip moral status when integrated with the
surrounding action context.

Three predicted outcomes:
  - Compositional probe onsets at step 1-2K alongside the standard moral
    probe → moral valence is encoded compositionally from the earliest
    training stages.
  - Compositional probe onsets between sentiment (step 2K) and syntax (step
    6K), or later → lexically-marked moralized vocabulary is decoded
    earlier than compositional moral integration.
  - Compositional probe never reaches 70 % at 1B → standard moral probe at
    1B is measuring lexically-accessible moralized vocabulary; compositional
    moral integration is a scale-dependent phenomenon.

Pipeline
--------

1. **Final-checkpoint validation (PASS gate).** Train compositional probe on
   ``allenai/OLMo-2-0425-1B`` (full base, ~2.2T tokens). PASS if
   peak-layer accuracy ≥ TF-IDF baseline + 10 pp AND ≥ 65 % absolute. If the
   gate fails, the script halts and surfaces the result — a 1B-scale failure
   is itself the C4 outcome.

2. **Trajectory probing.** Apply the compositional probe to all 37
   ``allenai/OLMo-2-0425-1B-early-training`` checkpoints (steps 0-36K at
   1K-step intervals).

3. **Fragility evolution.** Apply :class:`MoralFragilityTest` (with the same
   compositional dataset) to the same 37 checkpoints. Paper 1's methodology
   claim is that fragility extends beyond probing accuracy — checking
   whether the same accuracy-saturates-fragility-doesn't pattern holds for
   the compositional probe is part of the contribution.

Usage
-----

  # Full pipeline (validation + trajectory + fragility)
  python paper/accuracy_vs_fragility/scripts/phase_c4_compositional.py

  # Validation only (skips trajectory + fragility)
  python paper/accuracy_vs_fragility/scripts/phase_c4_compositional.py --validation-only

  # Resume the trajectory from a specific step
  python paper/accuracy_vs_fragility/scripts/phase_c4_compositional.py --resume-from step18000

Outputs (``paper/accuracy_vs_fragility/outputs/phase_c4_compositional/``):
  - ``c4_validation.json``: final-checkpoint validation result + PASS verdict
  - ``c4_per_checkpoint.json``: per-step probe + fragility numbers
  - ``c4_emergence_timing.json``: onset / plateau curves
  - ``compositional_vs_lexical_onset.png``: 4-curve overlay
  - ``compositional_layer_step.png``: layer × step heatmap
  - ``c4_fragility_evolution.png``: per-step fragility curves
  - ``RESULTS.md``: human-readable summary
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FINAL_REPO_ID = "allenai/OLMo-2-0425-1B"
TRAJ_REPO_ID = "allenai/OLMo-2-0425-1B-early-training"

ONSET_ACCURACY = 0.70  # same threshold as Phase C2
GATE_MIN_DELTA_PP = 10.0  # peak accuracy must beat TF-IDF baseline by ≥ 10 pp
GATE_MIN_ABS_ACC = 0.65  # AND meet 65 % absolute


def _clear_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _parse_step(revision: str) -> int | None:
    match = re.search(r"step(\d+)", revision)
    return int(match.group(1)) if match else None


def _get_all_revisions() -> list[tuple[int, str]]:
    """List all step-based revisions for the early-training model."""
    from deepsteer.benchmarks.representational.trajectory import list_available_revisions

    all_revisions = list_available_revisions(TRAJ_REPO_ID)
    step_revisions: list[tuple[int, str]] = []
    for rev in all_revisions:
        step = _parse_step(rev)
        if step is not None:
            step_revisions.append((step, rev))
    step_revisions.sort(key=lambda x: x[0])
    return step_revisions


# ---------------------------------------------------------------------------
# Final-checkpoint validation
# ---------------------------------------------------------------------------


def run_validation(
    output_dir: Path,
    *,
    device: str | None = None,
    seed: int = 42,
    n_epochs: int = 50,
) -> dict:
    """Train compositional probe on ``OLMo-2-0425-1B`` final and gate.

    Returns:
        Dict with peak-layer accuracy, TF-IDF baseline, gate verdict.
    """
    from deepsteer.benchmarks.representational.compositional_moral_probe import (
        CompositionalMoralProbe,
        _build_compositional_probing_dataset,
    )
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.datasets.compositional_moral_pairs import (
        content_separability_baseline,
    )

    print(f"\n{'=' * 70}")
    print(f"PHASE C4: COMPOSITIONAL PROBE VALIDATION ON {FINAL_REPO_ID}")
    print(f"{'=' * 70}\n")

    # TF-IDF baseline (deterministic, no model needed).
    print("(1) Computing TF-IDF content-only baseline ...")
    t0 = time.time()
    baseline = content_separability_baseline()
    print(f"  TF-IDF baseline overall: {baseline['overall']:.3f} "
          f"({time.time() - t0:.1f}s)")
    for cat, acc in baseline.items():
        if cat == "overall":
            continue
        print(f"    {cat}: {acc:.3f}")

    # Load model.
    print(f"\n(2) Loading {FINAL_REPO_ID} ...")
    t0 = time.time()
    model = WhiteBoxModel(FINAL_REPO_ID, device=device)
    print(f"  loaded in {time.time() - t0:.1f}s, n_layers={model.info.n_layers}")

    # Train probe.
    print("\n(3) Training compositional probe per layer ...")
    dataset = _build_compositional_probing_dataset(seed=seed)
    print(f"  dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs")

    probe = CompositionalMoralProbe(dataset=dataset, n_epochs=n_epochs, seed=seed)
    t0 = time.time()
    result = probe.run(model)
    print(f"  trained in {time.time() - t0:.1f}s")
    print(f"  peak accuracy: {result.peak_accuracy:.3f} @ layer {result.peak_layer}")
    print(f"  onset layer: {result.onset_layer}")
    print(f"  encoding depth: {result.moral_encoding_depth:.3f}")
    print(f"  encoding breadth: {result.moral_encoding_breadth:.3f}")

    # Free model memory.
    del model
    _clear_memory()

    # Gate verdict.
    delta_pp = (result.peak_accuracy - baseline["overall"]) * 100.0
    gate_pass_delta = delta_pp >= GATE_MIN_DELTA_PP
    gate_pass_abs = result.peak_accuracy >= GATE_MIN_ABS_ACC
    overall_pass = gate_pass_delta and gate_pass_abs

    verdict = {
        "peak_accuracy": float(result.peak_accuracy),
        "peak_layer": int(result.peak_layer),
        "onset_layer": result.onset_layer,
        "encoding_depth": float(result.moral_encoding_depth),
        "encoding_breadth": float(result.moral_encoding_breadth),
        "tfidf_baseline_overall": float(baseline["overall"]),
        "tfidf_baseline_per_category": {
            k: float(v) for k, v in baseline.items() if k != "overall"
        },
        "delta_vs_baseline_pp": float(delta_pp),
        "gate_min_delta_pp": GATE_MIN_DELTA_PP,
        "gate_min_abs_accuracy": GATE_MIN_ABS_ACC,
        "gate_pass_delta": bool(gate_pass_delta),
        "gate_pass_abs": bool(gate_pass_abs),
        "validation_pass": bool(overall_pass),
        "per_layer_accuracy": [
            {"layer": s.layer, "accuracy": float(s.accuracy), "loss": float(s.loss)}
            for s in result.layer_scores
        ],
        "model": FINAL_REPO_ID,
        "n_train_pairs": len(dataset.train),
        "n_test_pairs": len(dataset.test),
    }

    print(f"\n{'=' * 70}")
    print("VALIDATION VERDICT")
    print(f"{'=' * 70}")
    print(f"  Peak accuracy: {result.peak_accuracy:.3f}")
    print(f"  TF-IDF baseline: {baseline['overall']:.3f}")
    print(f"  Δ peak − baseline: {delta_pp:+.1f} pp")
    print(f"  Gate 1 (Δ ≥ {GATE_MIN_DELTA_PP:.0f} pp): "
          f"{'PASS' if gate_pass_delta else 'FAIL'}")
    print(f"  Gate 2 (peak ≥ {GATE_MIN_ABS_ACC:.0%}): "
          f"{'PASS' if gate_pass_abs else 'FAIL'}")
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "c4_validation.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nValidation written: {output_dir / 'c4_validation.json'}")

    return verdict


# ---------------------------------------------------------------------------
# Trajectory probing + fragility
# ---------------------------------------------------------------------------


def _reload_step_results(step: int, output_dir: Path) -> dict | None:
    """Reload previously saved per-step results from JSON, if present."""
    step_dir = output_dir / f"step_{step:07d}"
    json_path = step_dir / "c4_step.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


def run_step(
    step: int,
    revision: str,
    output_dir: Path,
    *,
    device: str | None = None,
    seed: int = 42,
    n_epochs: int = 50,
    skip_fragility: bool = False,
) -> dict:
    """Probe + fragility on a single early-training checkpoint."""
    from deepsteer.benchmarks.representational.compositional_moral_probe import (
        CompositionalMoralProbe,
        _build_compositional_probing_dataset,
    )
    from deepsteer.benchmarks.representational.fragility import MoralFragilityTest
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    step_dir = output_dir / f"step_{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading {revision} ...")
    t0 = time.time()
    model = WhiteBoxModel(
        TRAJ_REPO_ID,
        revision=revision,
        device=device,
        access_tier=AccessTier.CHECKPOINTS,
        checkpoint_step=step,
    )
    print(f"    loaded in {time.time() - t0:.1f}s")

    # Build dataset once per step (deterministic).
    dataset = _build_compositional_probing_dataset(seed=seed)

    # Probe.
    print(f"  Running compositional probe ...")
    probe = CompositionalMoralProbe(dataset=dataset, n_epochs=n_epochs, seed=seed)
    t0 = time.time()
    probe_result = probe.run(model)
    probe_elapsed = time.time() - t0
    mean_acc = float(np.mean([s.accuracy for s in probe_result.layer_scores]))
    print(f"    {probe_elapsed:.1f}s, peak={probe_result.peak_accuracy:.3f} "
          f"@ layer {probe_result.peak_layer}, mean={mean_acc:.3f}")

    fragility_payload: dict | None = None
    if not skip_fragility:
        print(f"  Running fragility test ...")
        frag_test = MoralFragilityTest(dataset=dataset)
        t0 = time.time()
        frag_result = frag_test.run(model)
        frag_elapsed = time.time() - t0
        print(f"    {frag_elapsed:.1f}s, mean_critical_noise="
              f"{frag_result.mean_critical_noise or 0.0:.2f}")
        fragility_payload = {
            "mean_critical_noise": frag_result.mean_critical_noise,
            "most_fragile_layer": frag_result.most_fragile_layer,
            "most_robust_layer": frag_result.most_robust_layer,
            "noise_levels": frag_result.noise_levels,
            "elapsed_s": round(frag_elapsed, 1),
            "layer_scores": [
                {
                    "layer": s.layer,
                    "baseline_accuracy": float(s.baseline_accuracy),
                    "critical_noise": s.critical_noise,
                    "accuracy_by_noise": {
                        str(k): float(v) for k, v in s.accuracy_by_noise.items()
                    },
                }
                for s in frag_result.layer_scores
            ],
        }

    del model
    _clear_memory()

    payload = {
        "step": step,
        "revision": revision,
        "probe": {
            "peak_accuracy": float(probe_result.peak_accuracy),
            "peak_layer": int(probe_result.peak_layer),
            "onset_layer": probe_result.onset_layer,
            "mean_accuracy": mean_acc,
            "encoding_depth": float(probe_result.moral_encoding_depth),
            "encoding_breadth": float(probe_result.moral_encoding_breadth),
            "elapsed_s": round(probe_elapsed, 1),
            "layer_scores": [
                {
                    "layer": s.layer,
                    "accuracy": float(s.accuracy),
                    "loss": float(s.loss),
                }
                for s in probe_result.layer_scores
            ],
        },
    }
    if fragility_payload is not None:
        payload["fragility"] = fragility_payload

    with open(step_dir / "c4_step.json", "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def run_trajectory(
    output_dir: Path,
    *,
    device: str | None = None,
    seed: int = 42,
    n_epochs: int = 50,
    resume_from: int | None = None,
    skip_fragility: bool = False,
) -> dict:
    """Run probe + fragility across all 37 early-training checkpoints."""
    print(f"\n{'=' * 70}")
    print(f"PHASE C4: TRAJECTORY ON {TRAJ_REPO_ID}")
    print(f"{'=' * 70}\n")

    revisions = _get_all_revisions()
    print(f"Found {len(revisions)} step-based revisions "
          f"({revisions[0][0]} → {revisions[-1][0]})")

    sorted_steps = [s for s, _ in revisions]
    step_to_rev = dict(revisions)

    if resume_from is not None:
        sorted_steps = [s for s in sorted_steps if s >= resume_from]
        print(f"Resuming from step {resume_from}: {len(sorted_steps)} remaining")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[int, dict] = {}

    # Pre-load any prior results for resumption.
    for s in [s for s, _ in revisions]:
        existing = _reload_step_results(s, output_dir)
        if existing is not None:
            all_results[s] = existing

    if all_results:
        print(f"Reloaded {len(all_results)} previously completed checkpoints")

    total_t0 = time.time()
    for i, step in enumerate(sorted_steps):
        revision = step_to_rev[step]
        print(f"\n--- Checkpoint {i + 1}/{len(sorted_steps)}: "
              f"step {step} ({revision}) ---")

        existing = _reload_step_results(step, output_dir)
        # Only skip if existing result has fragility (or fragility was skipped).
        needs_fragility = (
            not skip_fragility
            and existing is not None
            and "fragility" not in existing
        )
        if existing is not None and not needs_fragility:
            print("  results already exist, skipping")
            all_results[step] = existing
            continue

        result = run_step(
            step=step,
            revision=revision,
            output_dir=output_dir,
            device=device,
            seed=seed,
            n_epochs=n_epochs,
            skip_fragility=skip_fragility,
        )
        all_results[step] = result

        elapsed = time.time() - total_t0
        remaining = len(sorted_steps) - (i + 1)
        avg = elapsed / (i + 1)
        print(f"  elapsed: {elapsed/60:.1f} min, ETA: {avg * remaining/60:.1f} min")

    total_elapsed = time.time() - total_t0
    print(f"\nTrajectory complete: {len(all_results)} checkpoints in "
          f"{total_elapsed/60:.1f} min")

    # Aggregate per-checkpoint payload.
    agg = {
        "model": TRAJ_REPO_ID,
        "experiment": "C4_compositional",
        "checkpoints": sorted(all_results.keys()),
        "per_checkpoint": {
            str(s): all_results[s] for s in sorted(all_results.keys())
        },
        "total_elapsed_s": round(total_elapsed, 1),
    }
    with open(output_dir / "compositional_per_checkpoint.json", "w") as f:
        json.dump(agg, f, indent=2)

    return agg


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _load_phase_c2_curves() -> dict[str, tuple[list[int], list[float]]]:
    """Load the standard moral / sentiment / syntax curves from Phase C2.

    Reads ``paper/accuracy_vs_fragility/outputs/phase_c2/c2_emergence_timing.json``.
    Returns ``{probe_name: (steps, mean_accuracies)}``.
    """
    c2_path = Path(
        "paper/accuracy_vs_fragility/outputs/phase_c2/c2_emergence_timing.json",
    )
    if not c2_path.exists():
        logger.warning("Phase C2 emergence file missing: %s", c2_path)
        return {}
    with open(c2_path) as f:
        data = json.load(f)
    out = {}
    for name, payload in data.get("curves", {}).items():
        steps = payload["steps"]
        accs = payload["mean_accuracies"]
        out[name] = (steps, accs)
    return out


def generate_overlay_plot(
    all_results: dict[int, dict],
    output_dir: Path,
) -> None:
    """4-curve overlay: standard moral, sentiment, syntax, compositional moral."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_steps = sorted(all_results.keys())
    comp_steps: list[int] = []
    comp_accs: list[float] = []
    for s in sorted_steps:
        probe = all_results[s].get("probe")
        if probe is None:
            continue
        comp_steps.append(s)
        comp_accs.append(float(probe["mean_accuracy"]))

    c2_curves = _load_phase_c2_curves()

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        "moral": "#F44336",
        "sentiment": "#2196F3",
        "syntax": "#4CAF50",
        "compositional_moral": "#9C27B0",
    }
    labels = {
        "moral": "Standard moral (lexical)",
        "sentiment": "Sentiment",
        "syntax": "Syntax",
        "compositional_moral": "Compositional moral (this work)",
    }

    onset_steps: dict[str, int] = {}

    for probe_name in ("moral", "sentiment", "syntax"):
        if probe_name not in c2_curves:
            continue
        steps, accs = c2_curves[probe_name]
        ax.plot(
            steps, accs, "o-", color=colors[probe_name], linewidth=2,
            markersize=4, label=labels[probe_name], alpha=0.85,
        )
        for s, a in zip(steps, accs):
            if a >= ONSET_ACCURACY:
                onset_steps[probe_name] = s
                break

    ax.plot(
        comp_steps, comp_accs, "D-", color=colors["compositional_moral"],
        linewidth=2.5, markersize=5,
        label=labels["compositional_moral"],
    )
    for s, a in zip(comp_steps, comp_accs):
        if a >= ONSET_ACCURACY:
            onset_steps["compositional_moral"] = s
            break

    for probe_name, s in onset_steps.items():
        ax.axvline(x=s, color=colors[probe_name], linestyle="--",
                   linewidth=1.2, alpha=0.6)
        ax.text(
            s, 0.42 + 0.04 * list(onset_steps).index(probe_name),
            f"{labels[probe_name].split(' ')[0]} onset\n(step {s})",
            ha="center", va="bottom", fontsize=8, color=colors[probe_name],
            fontweight="bold",
        )

    ax.axhline(y=ONSET_ACCURACY, color="#9E9E9E", linestyle=":", linewidth=1,
               label=f"onset threshold ({ONSET_ACCURACY:.0%})")
    ax.axhline(y=0.5, color="#BDBDBD", linestyle=":", linewidth=1, alpha=0.5,
               label="chance (50%)")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Mean Probing Accuracy (all layers)", fontsize=12)
    ax.set_title(
        "Phase C4: Compositional vs. Lexical Moral Onset — OLMo-2 1B\n"
        "(C2 standard moral / sentiment / syntax curves overlaid)",
        fontsize=12,
    )
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = output_dir / "compositional_vs_lexical_onset.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("overlay: %s", out)

    # Companion JSON.
    companion = {
        "onset_threshold": ONSET_ACCURACY,
        "onset_steps": onset_steps,
        "curves": {
            "compositional_moral": {
                "steps": comp_steps,
                "mean_accuracies": [round(a, 4) for a in comp_accs],
            },
        },
    }
    for name, (steps, accs) in c2_curves.items():
        companion["curves"][name] = {
            "steps": steps,
            "mean_accuracies": accs,
        }
    with open(output_dir / "c4_emergence_timing.json", "w") as f:
        json.dump(companion, f, indent=2)


def generate_layer_heatmap(
    all_results: dict[int, dict],
    output_dir: Path,
) -> None:
    """Layer × step heatmap of compositional probe accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sorted_steps = sorted(all_results.keys())
    if not sorted_steps:
        return

    first = all_results[sorted_steps[0]].get("probe")
    if first is None:
        return
    n_layers = len(first["layer_scores"])

    matrix = np.zeros((n_layers, len(sorted_steps)))
    for col, step in enumerate(sorted_steps):
        probe = all_results[step].get("probe")
        if probe is None:
            continue
        for s in probe["layer_scores"]:
            matrix[s["layer"], col] = s["accuracy"]

    step_labels = [f"{s // 1000}K" if s > 0 else "0" for s in sorted_steps]
    layer_labels = [str(i) for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(max(14, len(sorted_steps) * 0.4), 7))
    sns.heatmap(
        matrix, ax=ax,
        xticklabels=step_labels, yticklabels=layer_labels,
        cmap="RdYlGn", vmin=0.4, vmax=1.0, annot=False,
        cbar_kws={"label": "Compositional probe accuracy"},
    )
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(
        f"Phase C4: Compositional Moral Probe (Layer × Step) — {TRAJ_REPO_ID}",
        fontsize=12,
    )
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    out = output_dir / "compositional_layer_step.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("heatmap: %s", out)


def generate_fragility_plot(
    all_results: dict[int, dict],
    output_dir: Path,
) -> None:
    """Fragility evolution: mean critical noise across checkpoints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_steps = sorted(all_results.keys())
    steps_with_frag: list[int] = []
    mean_critical: list[float] = []
    for s in sorted_steps:
        frag = all_results[s].get("fragility")
        if frag is None or frag.get("mean_critical_noise") is None:
            continue
        steps_with_frag.append(s)
        mean_critical.append(float(frag["mean_critical_noise"]))

    if not steps_with_frag:
        logger.warning("No fragility data; skipping fragility plot")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps_with_frag, mean_critical, "D-", color="#FF9800",
            linewidth=2, markersize=5,
            label="Mean critical noise (compositional probe)")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Mean Critical Noise (σ where probe collapses < 60%)",
                  fontsize=11)
    ax.set_title(
        f"Phase C4: Compositional Probe Fragility Evolution — {TRAJ_REPO_ID}",
        fontsize=12,
    )
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    out = output_dir / "c4_fragility_evolution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("fragility: %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="paper/accuracy_vs_fragility/outputs/phase_c4_compositional",
        help="Output directory.",
    )
    parser.add_argument("--device", default=None,
                        help="Device override (cuda, mps, cpu).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="Per-layer probe training epochs (default: 50).")
    parser.add_argument("--validation-only", action="store_true",
                        help="Skip the trajectory run.")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip validation; assume PASS and run trajectory.")
    parser.add_argument("--skip-fragility", action="store_true",
                        help="Skip fragility evaluation in the trajectory.")
    parser.add_argument("--resume-from", default=None,
                        help="Resume trajectory from this step (e.g. step18000).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)

    if not args.skip_validation:
        verdict = run_validation(
            output_dir=output_dir,
            device=args.device,
            seed=args.seed,
            n_epochs=args.n_epochs,
        )
        if not verdict["validation_pass"]:
            print("\n*** Validation FAILED. Stopping per Phase C4 design. ***")
            print("    The compositional distinction is not linearly decodable")
            print(f"    from {FINAL_REPO_ID} above the gates. This is itself")
            print("    the Phase C4 outcome (Outcome 3 in the design doc).")
            return

    if args.validation_only:
        return

    resume_step = _parse_step(args.resume_from) if args.resume_from else None
    agg = run_trajectory(
        output_dir=output_dir,
        device=args.device,
        seed=args.seed,
        n_epochs=args.n_epochs,
        resume_from=resume_step,
        skip_fragility=args.skip_fragility,
    )

    # Build the per-step dict needed by the plotters.
    per_checkpoint: dict[int, dict] = {
        int(k): v for k, v in agg["per_checkpoint"].items()
    }

    print(f"\n{'=' * 70}")
    print("Generating figures ...")
    print(f"{'=' * 70}")

    generate_overlay_plot(per_checkpoint, output_dir)
    generate_layer_heatmap(per_checkpoint, output_dir)
    generate_fragility_plot(per_checkpoint, output_dir)


if __name__ == "__main__":
    main()
