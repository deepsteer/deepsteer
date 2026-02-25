#!/usr/bin/env python3
"""Compare moral encoding depth across multiple models.

Runs LayerWiseMoralProbe on each model and produces an overlay plot showing
how moral concepts are encoded at different relative depths.

Examples:
    # Compare OLMo-1B and TinyLlama
    python examples/compare_models.py \
        --models allenai/OLMo-1B-hf TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --output-dir outputs/ --dataset-target 10

    # Compare with explicit device
    python examples/compare_models.py \
        --models allenai/OLMo-1B-hf meta-llama/Llama-3.2-1B \
        --device cpu --output-dir outputs/
"""

from __future__ import annotations

import argparse
import gc
import logging

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare moral encoding depth across models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="HuggingFace model IDs to compare.",
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
        "--n-epochs", type=int, default=50,
        help="Training epochs for probing classifiers (default: 50).",
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

    from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import LayerProbingResult
    from deepsteer.datasets.pipeline import build_probing_dataset
    from deepsteer.viz import plot_layer_probing, plot_model_comparison

    # Build dataset once — shared across all models
    print(f"Building probing dataset (target={args.dataset_target} per foundation)...")
    dataset = build_probing_dataset(target_per_foundation=args.dataset_target)
    print(f"Dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs\n")

    probe = LayerWiseMoralProbe(dataset=dataset, n_epochs=args.n_epochs)
    results: list[LayerProbingResult] = []

    for model_id in args.models:
        print(f"Loading {model_id}...")
        model = WhiteBoxModel(model_id, device=args.device)

        print(f"  Running layer-wise moral probe ({model.info.n_layers} layers)...")
        result = probe.run(model)
        results.append(result)

        # Individual plot
        png_path = plot_layer_probing(result, output_dir=args.output_dir)
        print(f"  Plot saved: {png_path}")
        print(f"  onset={result.onset_layer}, peak={result.peak_layer} "
              f"({result.peak_accuracy:.1%}), "
              f"depth={result.moral_encoding_depth:.3f}, "
              f"breadth={result.moral_encoding_breadth:.3f}")

        # Free memory before loading the next model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    # Comparative overlay plot
    if len(results) >= 2:
        comp_png = plot_model_comparison(results, output_dir=args.output_dir)
        print(f"Comparison plot saved: {comp_png}")

    # Summary table
    print("\n--- Summary ---")
    print(f"{'Model':<40} {'Layers':>6} {'Onset':>5} {'Peak':>5} "
          f"{'Peak Acc':>8} {'Depth':>7} {'Breadth':>7}")
    print("-" * 90)
    for result in results:
        name = result.model_info.name if result.model_info else "?"
        n_layers = len(result.layer_scores)
        print(f"{name:<40} {n_layers:>6} {result.onset_layer!s:>5} "
              f"{result.peak_layer!s:>5} {result.peak_accuracy:>7.1%} "
              f"{result.moral_encoding_depth:>7.3f} "
              f"{result.moral_encoding_breadth:>7.3f}")


if __name__ == "__main__":
    main()
