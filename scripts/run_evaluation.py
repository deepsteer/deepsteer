#!/usr/bin/env python3
"""Run DeepSteer evaluations against local or API models.

Examples:
    # Representational probing on OLMo-7B base (primary use case)
    python examples/run_evaluation.py --model olmo --output-dir outputs/

    # Representational probing on Llama-3-8B base
    python examples/run_evaluation.py --model llama --output-dir outputs/

    # Fast iteration with smaller model
    python examples/run_evaluation.py --model olmo --weights allenai/OLMo-1B-hf \
        --output-dir outputs/ --dataset-target 10

    # Include behavioral evals (requires instruction-tuned model)
    python examples/run_evaluation.py --model olmo --behavioral \
        --weights allenai/OLMo-7B-Instruct-hf --output-dir outputs/

    # Behavioral evals on Claude (automatic, API models always run behavioral)
    python examples/run_evaluation.py --model claude --output-dir outputs/

    # Behavioral evals on GPT
    python examples/run_evaluation.py --model gpt --model-id gpt-4o --output-dir outputs/

    # Checkpoint trajectory analysis
    python examples/run_evaluation.py --model olmo --weights allenai/OLMo-1B-hf \
        --output-dir outputs/ --dataset-target 10 \
        --checkpoint-revisions step1000-tokens4B step2000-tokens8B

    # List available checkpoints
    python examples/run_evaluation.py --model olmo --weights allenai/OLMo-1B-hf \
        --list-checkpoints
"""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DeepSteer alignment depth evaluations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["olmo", "llama", "claude", "gpt"], required=True,
        help="Model family to evaluate.",
    )
    parser.add_argument(
        "--weights", default=None,
        help="HuggingFace model ID or local path (for olmo/llama).",
    )
    parser.add_argument(
        "--model-id", default=None,
        help="API model ID (for claude/gpt).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device to use (cuda, mps, cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Directory for output plots and JSON (default: outputs/).",
    )
    parser.add_argument(
        "--dataset-target", type=int, default=40,
        help="Target pairs per moral foundation (default: 40).",
    )
    parser.add_argument(
        "--checkpoint-revisions", nargs="+", default=None,
        help="HuggingFace revision strings for trajectory analysis.",
    )
    parser.add_argument(
        "--max-checkpoints", type=int, default=None,
        help="Limit number of checkpoint revisions to probe.",
    )
    parser.add_argument(
        "--list-checkpoints", action="store_true",
        help="List available checkpoint revisions and exit.",
    )
    parser.add_argument(
        "--behavioral", action="store_true",
        help="Also run behavioral benchmarks (requires instruction-tuned model).",
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

    # -- List checkpoints and exit --
    if args.list_checkpoints:
        from deepsteer.benchmarks.representational.trajectory import list_available_revisions
        repo_id = args.weights or _default_weights(args.model)
        revisions = list_available_revisions(repo_id)
        print(f"Available revisions for {repo_id}:")
        for rev in revisions:
            print(f"  {rev}")
        return

    # -- Construct model --
    if args.model in ("olmo", "llama"):
        from deepsteer.core.model_interface import WhiteBoxModel
        from deepsteer.core.types import AccessTier

        weights = args.weights or _default_weights(args.model)
        tier = AccessTier.CHECKPOINTS if args.checkpoint_revisions else AccessTier.WEIGHTS
        model = WhiteBoxModel(weights, device=args.device, access_tier=tier)
    else:
        from deepsteer.core.model_interface import APIModel
        provider = "anthropic" if args.model == "claude" else "openai"
        model_id = args.model_id or _default_model_id(args.model)
        model = APIModel(provider, model_id)

    # -- White-box evaluations (layer probing + trajectory) --
    if args.model in ("olmo", "llama"):
        from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
        from deepsteer.datasets.pipeline import build_probing_dataset
        from deepsteer.viz import plot_layer_probing

        print(f"Building probing dataset (target={args.dataset_target} per foundation)...")
        dataset = build_probing_dataset(target_per_foundation=args.dataset_target)
        print(f"Dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs")

        probe = LayerWiseMoralProbe(dataset=dataset)
        print(f"Running layer-wise moral probe on {model.info.name}...")
        result = probe.run(model)

        png_path = plot_layer_probing(result, output_dir=args.output_dir)
        print(f"Layer probing plot saved: {png_path}")
        print(f"  onset_layer={result.onset_layer}, peak_layer={result.peak_layer}")
        print(f"  peak_accuracy={result.peak_accuracy:.1%}")
        print(f"  moral_encoding_depth={result.moral_encoding_depth:.3f}")
        print(f"  moral_encoding_breadth={result.moral_encoding_breadth:.3f}")

        # Checkpoint trajectory (optional)
        if args.checkpoint_revisions:
            from deepsteer.benchmarks.representational.trajectory import (
                CheckpointTrajectoryProbe,
            )
            from deepsteer.viz import plot_checkpoint_trajectory

            revisions = args.checkpoint_revisions
            if args.max_checkpoints:
                revisions = revisions[: args.max_checkpoints]

            traj_probe = CheckpointTrajectoryProbe(
                checkpoint_revisions=revisions,
                dataset=dataset,
                device=args.device,
                torch_dtype=model._dtype if hasattr(model, "_dtype") else None,
            )
            print(f"Running trajectory analysis across {len(revisions)} checkpoints...")
            traj_result = traj_probe.run(model)
            traj_png = plot_checkpoint_trajectory(traj_result, output_dir=args.output_dir)
            print(f"Trajectory heatmap saved: {traj_png}")

    # -- Behavioral evals (only for API models or when --behavioral is set) --
    run_behavioral = args.model in ("claude", "gpt") or args.behavioral
    if run_behavioral:
        # Warn if running behavioral benchmarks on a likely-base model
        if args.model in ("olmo", "llama"):
            weights = args.weights or _default_weights(args.model)
            name_lower = weights.lower()
            if "instruct" not in name_lower and "chat" not in name_lower:
                import warnings
                warnings.warn(
                    f"Running behavioral benchmarks on '{weights}' which appears to be a "
                    f"base model. Behavioral benchmarks require instruction-tuned models "
                    f"for meaningful results. Consider using an instruct variant.",
                    stacklevel=1,
                )

        from deepsteer.benchmarks.compliance_gap.greenblatt import ComplianceGapDetector
        from deepsteer.benchmarks.moral_reasoning.foundations import MoralFoundationsProbe
        from deepsteer.viz import plot_compliance_gap, plot_moral_foundations

        print(f"\nRunning MoralFoundationsProbe on {model.info.name}...")
        mfp = MoralFoundationsProbe()
        mfp_result = mfp.run(model)
        mfp_png = plot_moral_foundations(mfp_result, output_dir=args.output_dir)
        print(f"Moral foundations plot saved: {mfp_png}")
        print(f"  overall_accuracy={mfp_result.overall_accuracy:.1%}")
        print(f"  depth_gradient={mfp_result.depth_gradient:.3f}")
        for fname, acc in mfp_result.accuracy_by_foundation.items():
            print(f"  {fname}: {acc:.1%}")

        print(f"\nRunning ComplianceGapDetector on {model.info.name}...")
        cgd = ComplianceGapDetector()
        cgd_result = cgd.run(model)
        cgd_png = plot_compliance_gap(cgd_result, output_dir=args.output_dir)
        print(f"Compliance gap plot saved: {cgd_png}")
        print(f"  compliance_gap={cgd_result.compliance_gap:+.3f}")
        print(f"  monitored_rate={cgd_result.monitored_compliance_rate:.1%}")
        print(f"  unmonitored_rate={cgd_result.unmonitored_compliance_rate:.1%}")
        for cat, gap in cgd_result.gap_by_category.items():
            print(f"  {cat}: gap={gap:+.3f}")
    elif args.model in ("olmo", "llama"):
        print("\nSkipping behavioral benchmarks (use --behavioral to include them).")


def _default_weights(model: str) -> str:
    """Return the default HuggingFace model ID for a model family."""
    defaults = {
        "olmo": "allenai/OLMo-7B-hf",
        "llama": "meta-llama/Llama-3-8B",
    }
    return defaults[model]


def _default_model_id(model: str) -> str:
    """Return the default API model ID for a model family."""
    defaults = {
        "claude": "claude-sonnet-4-20250514",
        "gpt": "gpt-4o",
    }
    return defaults[model]


if __name__ == "__main__":
    main()
