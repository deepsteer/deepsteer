#!/usr/bin/env python3
"""Phase C Tier 2: LoRA fine-tuning experiments on OLMo-2 1B.

Tests whether moral data curation (narrative vs declarative framing,
moral acceleration from random init) affects the representational
structure discovered in Phase C1.

Experiments:
    C3: Narrative vs Declarative Moral Framing (3 conditions)
    C6: Moral Acceleration from Random Init (2 conditions)
    C4: Early vs Late LoRA injection (contingent on C3/C6 signal)
    C5: Foundation Coverage (contingent on C3/C6 signal)

Target model: allenai/OLMo-2-0425-1B-early-training
Hardware: MacBook Pro M4 Pro, 24 GB unified memory

Usage:
    # Run C3 experiment (narrative vs declarative)
    python examples/phase_c_tier2.py --experiment c3

    # Run C6 experiment (moral acceleration)
    python examples/phase_c_tier2.py --experiment c6

    # Run all experiments
    python examples/phase_c_tier2.py --experiment all

    # Quick smoke test
    python examples/phase_c_tier2.py --experiment c3 --max-steps 10 --quick

    # Custom LoRA rank
    python examples/phase_c_tier2.py --experiment c3 --lora-rank 8
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

EXPERIMENTS = ["c3", "c6", "c4", "c5", "all"]
SIGNAL_THRESHOLD = 0.10  # 10% fragility difference triggers C4/C5


def _compute_signal(results: dict) -> float:
    """Compute signal metric: max fragility difference between conditions.

    Returns the maximum difference in mean_critical_noise between any
    moral condition and the control.
    """
    control_fragility = None
    moral_fragilities: list[float] = []

    for name, result in results.items():
        if result.final_fragility is None:
            continue
        cn = result.final_fragility.mean_critical_noise or 0.0
        if "control" in name:
            control_fragility = cn
        else:
            moral_fragilities.append(cn)

    if control_fragility is None or not moral_fragilities:
        return 0.0

    return max(abs(mf - control_fragility) for mf in moral_fragilities)


def _run_c3(args: argparse.Namespace) -> dict:
    """Run C3: Narrative vs Declarative experiment."""
    from deepsteer.steering.lora_experiment import run_c3_narrative_vs_declarative

    output_dir = Path(args.output_dir) / "c3"
    results = run_c3_narrative_vs_declarative(
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        lora_rank=args.lora_rank,
        output_dir=output_dir,
        device=args.device,
        quick=args.quick,
    )

    # Generate comparison plots
    from deepsteer.viz.lora_experiments import (
        plot_lora_acceleration,
        plot_lora_fragility_comparison,
        plot_lora_fragility_trajectory,
        plot_lora_training_loss,
    )

    plot_lora_fragility_comparison(results, output_dir=output_dir)
    plot_lora_acceleration(results, output_dir=output_dir)
    plot_lora_training_loss(results, output_dir=output_dir)

    for name, result in results.items():
        plot_lora_fragility_trajectory(result, output_dir=output_dir)

    signal = _compute_signal(results)
    print(f"\nC3 Signal: {signal:.3f} (threshold={SIGNAL_THRESHOLD:.2f})")
    if signal > SIGNAL_THRESHOLD:
        print("  -> SIGNAL DETECTED: C4/C5 experiments warranted")
    else:
        print("  -> No significant signal: C4/C5 may not be informative")

    return {"results": results, "signal": signal}


def _run_c6(args: argparse.Namespace) -> dict:
    """Run C6: Moral Acceleration experiment."""
    from deepsteer.steering.lora_experiment import run_c6_moral_acceleration

    output_dir = Path(args.output_dir) / "c6"
    results = run_c6_moral_acceleration(
        max_steps=args.max_steps * 2 if not args.quick else args.max_steps,
        eval_every=50 if not args.quick else args.eval_every,
        lora_rank=args.lora_rank,
        output_dir=output_dir,
        device=args.device,
        quick=args.quick,
    )

    from deepsteer.viz.lora_experiments import (
        plot_lora_acceleration,
        plot_lora_fragility_comparison,
        plot_lora_fragility_trajectory,
        plot_lora_training_loss,
    )

    plot_lora_fragility_comparison(results, output_dir=output_dir)
    plot_lora_acceleration(results, output_dir=output_dir)
    plot_lora_training_loss(results, output_dir=output_dir)

    for name, result in results.items():
        plot_lora_fragility_trajectory(result, output_dir=output_dir)

    signal = _compute_signal(results)
    print(f"\nC6 Signal: {signal:.3f} (threshold={SIGNAL_THRESHOLD:.2f})")
    if signal > SIGNAL_THRESHOLD:
        print("  -> SIGNAL DETECTED: moral acceleration effect present")
    else:
        print("  -> No significant signal")

    return {"results": results, "signal": signal}


def _run_c4(args: argparse.Namespace) -> dict:
    """Run C4: Early vs Late LoRA experiment."""
    from deepsteer.steering.lora_experiment import run_c4_early_vs_late

    output_dir = Path(args.output_dir) / "c4"
    results = run_c4_early_vs_late(
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        lora_rank=args.lora_rank,
        output_dir=output_dir,
        device=args.device,
        quick=args.quick,
    )

    from deepsteer.viz.lora_experiments import (
        plot_lora_acceleration,
        plot_lora_fragility_comparison,
        plot_lora_training_loss,
    )

    plot_lora_fragility_comparison(results, output_dir=output_dir)
    plot_lora_acceleration(results, output_dir=output_dir)
    plot_lora_training_loss(results, output_dir=output_dir)

    return {"results": results}


def _run_c5(args: argparse.Namespace) -> dict:
    """Run C5: Foundation Coverage experiment."""
    from deepsteer.steering.lora_experiment import run_c5_foundation_coverage

    output_dir = Path(args.output_dir) / "c5"
    results = run_c5_foundation_coverage(
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        lora_rank=args.lora_rank,
        output_dir=output_dir,
        device=args.device,
        quick=args.quick,
    )

    from deepsteer.viz.lora_experiments import (
        plot_lora_acceleration,
        plot_lora_fragility_comparison,
        plot_lora_training_loss,
    )

    plot_lora_fragility_comparison(results, output_dir=output_dir)
    plot_lora_acceleration(results, output_dir=output_dir)
    plot_lora_training_loss(results, output_dir=output_dir)

    return {"results": results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase C Tier 2: LoRA fine-tuning experiments on OLMo-2 1B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment", required=True, choices=EXPERIMENTS,
        help="Which experiment to run (c3, c6, c4, c5, or all).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000,
        help="Maximum LoRA training steps per condition (default: 1000).",
    )
    parser.add_argument(
        "--eval-every", type=int, default=100,
        help="Evaluate probing every N steps (default: 100).",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16,
        help="LoRA rank parameter (default: 16).",
    )
    parser.add_argument(
        "--output-dir", default="outputs/phase_c_tier2",
        help="Top-level output directory (default: outputs/phase_c_tier2).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (cuda, mps, cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: reduced corpus, fewer eval points.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase C Tier 2: LoRA Fine-Tuning Experiments")
    print(f"  Experiment: {args.experiment}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Eval every: {args.eval_every}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Output: {args.output_dir}")
    print()

    total_t0 = time.time()
    summary: dict = {
        "experiment": args.experiment,
        "max_steps": args.max_steps,
        "lora_rank": args.lora_rank,
        "quick": args.quick,
    }

    if args.experiment in ("c3", "all"):
        print(f"\n{'#'*60}")
        print("# C3: Narrative vs Declarative Moral Framing")
        print(f"{'#'*60}")
        c3_data = _run_c3(args)
        summary["c3_signal"] = c3_data["signal"]

    if args.experiment in ("c6", "all"):
        print(f"\n{'#'*60}")
        print("# C6: Moral Acceleration from Random Init")
        print(f"{'#'*60}")
        c6_data = _run_c6(args)
        summary["c6_signal"] = c6_data["signal"]

    if args.experiment == "c4" or (
        args.experiment == "all"
        and summary.get("c3_signal", 0) > SIGNAL_THRESHOLD
    ):
        print(f"\n{'#'*60}")
        print("# C4: Early vs Late LoRA Injection")
        print(f"{'#'*60}")
        if args.experiment == "all":
            print("  (Triggered by C3 signal)")
        _run_c4(args)

    if args.experiment == "c5" or (
        args.experiment == "all"
        and summary.get("c3_signal", 0) > SIGNAL_THRESHOLD
    ):
        print(f"\n{'#'*60}")
        print("# C5: Foundation Coverage")
        print(f"{'#'*60}")
        if args.experiment == "all":
            print("  (Triggered by C3 signal)")
        _run_c5(args)

    if args.experiment == "all" and all(
        summary.get(k, 0) <= SIGNAL_THRESHOLD for k in ("c3_signal", "c6_signal")
    ):
        print("\nNo signal detected in C3 or C6. Skipping C4/C5.")

    total_elapsed = time.time() - total_t0
    summary["total_elapsed_s"] = round(total_elapsed, 1)

    print(f"\n{'='*60}")
    print("PHASE C TIER 2 SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed/60:.1f} min")
    for key, val in summary.items():
        if key.endswith("_signal"):
            experiment = key.replace("_signal", "").upper()
            significant = "SIGNAL" if val > SIGNAL_THRESHOLD else "no signal"
            print(f"  {experiment}: {val:.3f} ({significant})")
    print(f"Output: {args.output_dir}")

    summary_path = output_dir / "phase_c_tier2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
