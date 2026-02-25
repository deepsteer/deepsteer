#!/usr/bin/env python3
"""Run DeepSteer evaluations against local or API models.

Examples:
    # Layer probing on OLMo-1B (fast iteration)
    python examples/run_evaluation.py --model olmo --weights allenai/OLMo-1B-hf \
        --output-dir outputs/ --dataset-target 10

    # List available checkpoints
    python examples/run_evaluation.py --model olmo --weights allenai/OLMo-1B-hf \
        --list-checkpoints

    # Checkpoint trajectory analysis
    python examples/run_evaluation.py --model olmo --weights allenai/OLMo-1B-hf \
        --output-dir outputs/ --dataset-target 10 \
        --checkpoint-revisions step1000-tokens4B step2000-tokens8B
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch


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

    # -- Build dataset --
    from deepsteer.datasets.pipeline import build_probing_dataset
    print(f"Building probing dataset (target={args.dataset_target} per foundation)...")
    dataset = build_probing_dataset(target_per_foundation=args.dataset_target)
    print(f"Dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs")

    # -- Run layer probing --
    if args.model in ("olmo", "llama"):
        from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
        from deepsteer.viz import plot_layer_probing

        probe = LayerWiseMoralProbe(dataset=dataset)
        print(f"Running layer-wise moral probe on {model.info.name}...")
        result = probe.run(model)

        png_path = plot_layer_probing(result, output_dir=args.output_dir)
        print(f"Layer probing plot saved: {png_path}")
        print(f"  onset_layer={result.onset_layer}, peak_layer={result.peak_layer}")
        print(f"  peak_accuracy={result.peak_accuracy:.1%}")
        print(f"  moral_encoding_depth={result.moral_encoding_depth:.3f}")
        print(f"  moral_encoding_breadth={result.moral_encoding_breadth:.3f}")

        # -- Checkpoint trajectory (optional) --
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
    else:
        print("API model evaluations (behavioral benchmarks) coming in Phase 3.")


def _default_weights(model: str) -> str:
    """Return the default HuggingFace model ID for a model family."""
    defaults = {
        "olmo": "allenai/OLMo-1B-hf",
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
