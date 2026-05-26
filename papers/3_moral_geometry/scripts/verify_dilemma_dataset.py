#!/usr/bin/env python3
"""Script 2: Verify dilemma dataset against existing foundation probes.

For each dilemma text, runs all 6 foundation-specific probes (from Experiment 1)
and checks whether both tagged foundations activate. This validates that the
generated dilemmas actually exercise the intended moral foundations.

Go/no-go: if <70% of dilemma texts activate both tagged foundation probes,
the dataset needs revision before proceeding to probing experiments.

Usage:
    python papers/3_moral_geometry/scripts/verify_dilemma_dataset.py
    python papers/3_moral_geometry/scripts/verify_dilemma_dataset.py --dataset path/to/dilemma_pairs_final.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

OLMO_REPO = "allenai/OLMo-2-0425-1B"

FOUNDATION_ORDER = [
    "care_harm", "fairness_cheating", "liberty_oppression",
    "loyalty_betrayal", "authority_subversion", "sanctity_degradation",
]

FOUNDATION_SHORT = {
    "care_harm": "Care",
    "fairness_cheating": "Fairness",
    "liberty_oppression": "Liberty",
    "loyalty_betrayal": "Loyalty",
    "authority_subversion": "Authority",
    "sanctity_degradation": "Sanctity",
}

DILEMMA_KEY_TO_FOUNDATION = {
    "care": "care_harm",
    "fairness": "fairness_cheating",
    "liberty": "liberty_oppression",
    "loyalty": "loyalty_betrayal",
    "authority": "authority_subversion",
    "sanctity": "sanctity_degradation",
}


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_probes(
    directions_path: Path,
    n_layers: int,
    hidden_dim: int,
) -> dict[str, dict[int, nn.Linear]]:
    """Load foundation probe directions and reconstruct linear probes."""
    data = np.load(directions_path)
    probes: dict[str, dict[int, nn.Linear]] = {}

    for fv in FOUNDATION_ORDER:
        probes[fv] = {}
        for layer_idx in range(n_layers):
            key = f"{fv}_layer{layer_idx}"
            if key not in data:
                continue
            w = data[key]
            probe = nn.Linear(hidden_dim, 1, bias=False)
            probe.weight.data = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
            probe.eval()
            probes[fv][layer_idx] = probe

    return probes


def collect_activations(model, texts: list[str], n_layers: int) -> dict[int, torch.Tensor]:
    """Collect mean-pooled activations for a list of texts at each layer."""
    all_acts: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}

    for text in texts:
        acts = model.get_activations(text, layers=list(range(n_layers)))
        for layer_idx in range(n_layers):
            h = acts[layer_idx]  # (1, seq_len, hidden_dim)
            pooled = h.mean(dim=1).squeeze(0)  # (hidden_dim,)
            all_acts[layer_idx].append(pooled.cpu())

    result = {}
    for layer_idx in range(n_layers):
        result[layer_idx] = torch.stack(all_acts[layer_idx])  # (n_texts, hidden_dim)

    return result


def run_verification(
    model,
    dilemma_data: dict,
    probes: dict[str, dict[int, nn.Linear]],
    probe_results: dict,
    n_layers: int,
) -> dict:
    """Run all 6 foundation probes on each dilemma text."""
    pairs = dilemma_data["pairs"]
    moral_texts = [p["moral"] for p in pairs]

    # Find peak accuracy layer per foundation to use for activation check
    peak_layers: dict[str, int] = {}
    for fv in FOUNDATION_ORDER:
        per_foundation = probe_results.get("per_foundation", {}).get(fv, {})
        peak_layers[fv] = per_foundation.get("peak_layer", n_layers // 2)

    print(f"  Collecting activations for {len(moral_texts)} dilemma texts...")
    t0 = time.time()
    acts = collect_activations(model, moral_texts, n_layers)
    print(f"  Collected in {time.time() - t0:.1f}s")

    per_dilemma: list[dict] = []
    both_activated_count = 0
    total_count = 0

    for i, pair_info in enumerate(pairs):
        fp = pair_info["foundation_pair"]
        tagged_foundations = [DILEMMA_KEY_TO_FOUNDATION[f] for f in fp]

        probe_logits: dict[str, dict[str, float]] = {}
        peak_logits: dict[str, float] = {}

        for fv in FOUNDATION_ORDER:
            probe_logits[fv] = {}
            peak_layer = peak_layers[fv]

            if peak_layer in probes[fv]:
                probe = probes[fv][peak_layer]
                with torch.no_grad():
                    x = acts[peak_layer][i:i+1]
                    logit = probe(x).item()
                peak_logits[fv] = logit
                probe_logits[fv]["peak"] = round(logit, 4)

        # Check if both tagged foundations activate (logit > 0 at peak layer)
        both_active = all(
            peak_logits.get(tf, -1) > 0
            for tf in tagged_foundations
        )
        total_count += 1
        if both_active:
            both_activated_count += 1

        # Rank foundations by logit
        ranked = sorted(peak_logits.items(), key=lambda x: x[1], reverse=True)
        top_2_foundations = [f for f, _ in ranked[:2]]
        tagged_in_top_2 = all(tf in top_2_foundations for tf in tagged_foundations)

        per_dilemma.append({
            "id": pair_info["id"],
            "foundation_pair": fp,
            "peak_logits": {k: round(v, 4) for k, v in peak_logits.items()},
            "both_tagged_active": both_active,
            "tagged_in_top_2": tagged_in_top_2,
            "top_2_foundations": top_2_foundations,
        })

    activation_rate = both_activated_count / total_count if total_count > 0 else 0

    # Per foundation-pair hit rates
    per_pair_stats: dict[str, dict] = {}
    for pair in {tuple(p["foundation_pair"]) for p in pairs}:
        pk = f"{pair[0]}-{pair[1]}"
        pair_dilemmas = [d for d in per_dilemma if tuple(d["foundation_pair"]) == pair]
        both_active = sum(1 for d in pair_dilemmas if d["both_tagged_active"])
        in_top_2 = sum(1 for d in pair_dilemmas if d["tagged_in_top_2"])
        per_pair_stats[pk] = {
            "n_dilemmas": len(pair_dilemmas),
            "both_active": both_active,
            "both_active_rate": round(both_active / len(pair_dilemmas), 4) if pair_dilemmas else 0,
            "tagged_in_top_2": in_top_2,
            "tagged_in_top_2_rate": round(in_top_2 / len(pair_dilemmas), 4) if pair_dilemmas else 0,
        }

    return {
        "total_dilemmas": total_count,
        "both_active_count": both_activated_count,
        "both_active_rate": round(activation_rate, 4),
        "go_no_go": "GO" if activation_rate >= 0.70 else "NO-GO",
        "threshold": 0.70,
        "per_pair": per_pair_stats,
        "per_dilemma": per_dilemma,
        "peak_layers_used": {k: v for k, v in peak_layers.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dilemma dataset against existing probes.")
    parser.add_argument("--dataset", default="deepsteer/datasets/dilemma_pairs_final.json")
    parser.add_argument("--directions", default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--probe-results", default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_foundation_probing.json")
    parser.add_argument("--output-dir", default="papers/3_moral_geometry/outputs/dilemma_verification")
    parser.add_argument("--device", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        print("Run generate_dilemma_dataset.py first.")
        return

    directions_path = Path(args.directions)
    if not directions_path.exists():
        print(f"ERROR: Probe directions not found: {directions_path}")
        print("Run exp1_2_3_framework_geometry.py first.")
        return

    with open(dataset_path) as f:
        dilemma_data = json.load(f)
    print(f"Loaded {len(dilemma_data['pairs'])} dilemma pairs from {dataset_path}")

    with open(args.probe_results) as f:
        probe_results = json.load(f)

    print(f"\nLoading model: {OLMO_REPO}")
    t0 = time.time()
    model = WhiteBoxModel(OLMO_REPO, device=args.device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    hidden_dim = model.info.hidden_dim
    print(f"Loaded in {time.time() - t0:.1f}s ({n_layers} layers, {hidden_dim} hidden dim)")

    print(f"\nLoading existing foundation probes...")
    probes = load_probes(directions_path, n_layers, hidden_dim)
    n_probes = sum(len(v) for v in probes.values())
    print(f"Loaded {n_probes} probes ({len(probes)} foundations × {n_layers} layers)")

    print(f"\n{'='*60}")
    print("VERIFICATION: Running foundation probes on dilemma texts")
    print(f"{'='*60}")

    results = run_verification(model, dilemma_data, probes, probe_results, n_layers)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dilemma_verification.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("VERIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Both tagged foundations active: {results['both_active_count']}/{results['total_dilemmas']} "
          f"({results['both_active_rate']:.1%})")
    print(f"Go/no-go: {results['go_no_go']} (threshold: {results['threshold']:.0%})")

    print(f"\nPer foundation pair:")
    for pk, stats in sorted(results["per_pair"].items()):
        print(f"  {pk:25s}: {stats['both_active']}/{stats['n_dilemmas']} active "
              f"({stats['both_active_rate']:.0%}), "
              f"{stats['tagged_in_top_2']}/{stats['n_dilemmas']} in top-2 "
              f"({stats['tagged_in_top_2_rate']:.0%})")

    # Flag problematic pairs
    weak_pairs = [pk for pk, s in results["per_pair"].items() if s["both_active_rate"] < 0.5]
    if weak_pairs:
        print(f"\nWARNING: These pairs have <50% activation rate: {', '.join(weak_pairs)}")
        print("Consider regenerating dilemmas for these pairs.")

    del model
    _clear_memory()

    print(f"\nOutput: {output_dir / 'dilemma_verification.json'}")
    if results["go_no_go"] == "GO":
        print("Next step: Run dilemma_probing.py for subspace analysis.")
    else:
        print("BLOCKED: Dataset quality insufficient. Revise dilemmas before proceeding.")


if __name__ == "__main__":
    main()
