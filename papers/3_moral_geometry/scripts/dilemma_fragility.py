#!/usr/bin/env python3
"""Script 6: Dilemma fragility — complexity-fragility gradient.

Tests whether dilemma-specific probes are more fragile than single-foundation
probes by applying Gaussian noise injection to cached activations.

Hypothesis: pooled binary > single-foundation > dilemma probes in robustness,
reflecting a complexity-fragility gradient.

Usage:
    python papers/3_moral_geometry/scripts/dilemma_fragility.py
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

NOISE_LEVELS = [0.1, 0.3, 1.0, 3.0, 10.0]
FRAGILITY_THRESHOLD = 0.6
N_NOISE_SEEDS = 10

FOUNDATION_PAIRS = [
    ("care", "fairness"), ("care", "liberty"), ("care", "loyalty"),
    ("care", "authority"), ("care", "sanctity"),
    ("fairness", "liberty"), ("fairness", "loyalty"),
    ("fairness", "authority"), ("fairness", "sanctity"),
    ("liberty", "loyalty"), ("liberty", "authority"), ("liberty", "sanctity"),
    ("loyalty", "authority"), ("loyalty", "sanctity"),
    ("authority", "sanctity"),
]


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def collect_activations(model, texts: list[str], n_layers: int) -> dict[int, list[torch.Tensor]]:
    """Collect mean-pooled activations for a list of texts at each layer."""
    all_acts: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
    for text in texts:
        acts = model.get_activations(text, layers=list(range(n_layers)))
        for layer_idx in range(n_layers):
            h = acts[layer_idx]
            pooled = h.mean(dim=1).squeeze(0).float()
            all_acts[layer_idx].append(pooled.cpu())
    return all_acts


def run_fragility_for_probe(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
    test_y: torch.Tensor,
    *,
    noise_levels: list[float],
    n_seeds: int = N_NOISE_SEEDS,
    n_epochs: int = 50,
    lr: float = 1e-2,
) -> dict:
    """Train a probe and test under noise injection."""
    hidden_dim = train_X.shape[1]
    torch.manual_seed(42)
    probe = nn.Linear(hidden_dim, 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    probe.train()
    for _ in range(n_epochs):
        logits = probe(train_X).squeeze(-1)
        loss = loss_fn(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()

    with torch.no_grad():
        baseline_logits = probe(test_X).squeeze(-1)
        baseline_preds = (baseline_logits > 0).float()
        baseline_acc = (baseline_preds == test_y).float().mean().item()

    accuracy_by_noise: dict[str, float] = {}
    for noise_level in noise_levels:
        seed_accs = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            with torch.no_grad():
                noised_X = test_X + torch.randn_like(test_X) * noise_level
                noised_logits = probe(noised_X).squeeze(-1)
                noised_preds = (noised_logits > 0).float()
                noised_acc = (noised_preds == test_y).float().mean().item()
            seed_accs.append(noised_acc)
        accuracy_by_noise[str(noise_level)] = float(np.mean(seed_accs))

    critical_noise = None
    for nl in sorted(noise_levels):
        if accuracy_by_noise[str(nl)] < FRAGILITY_THRESHOLD:
            critical_noise = nl
            break
    # Cap-at-max: never-fragile layers censored at the grid maximum, not dropped.
    critical_noise_capped = max(noise_levels) if critical_noise is None else critical_noise

    return {
        "baseline_accuracy": round(baseline_acc, 4),
        "accuracy_by_noise": {k: round(v, 4) for k, v in accuracy_by_noise.items()},
        "critical_noise": critical_noise,
        "critical_noise_capped": critical_noise_capped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dilemma fragility analysis.")
    parser.add_argument("--dataset", default="deepsteer/datasets/dilemma_pairs_final.json")
    parser.add_argument("--output-dir", default="papers/3_moral_geometry/outputs/dilemma_fragility")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=OLMO_REPO, help="HuggingFace model ID.")
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
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
        print(f"ERROR: {dataset_path} not found.")
        return

    with open(dataset_path) as f:
        dilemma_data = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    t0 = time.time()
    model = WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    print(f"Loaded in {time.time() - t0:.1f}s ({n_layers} layers)")

    # Group pairs by type
    pairs_by_type: dict[str, list[dict]] = {}
    for p in dilemma_data["pairs"]:
        pk = f"{p['foundation_pair'][0]}-{p['foundation_pair'][1]}"
        pairs_by_type.setdefault(pk, []).append(p)

    results: dict[str, dict] = {}

    for pair in FOUNDATION_PAIRS:
        pk = f"{pair[0]}-{pair[1]}"
        pairs = pairs_by_type.get(pk, [])
        if len(pairs) < 5:
            continue

        n_train = max(1, int(len(pairs) * 0.8))
        train_pairs = pairs[:n_train]
        test_pairs = pairs[n_train:]
        if len(test_pairs) < 1:
            test_pairs = [train_pairs[-1]]
            train_pairs = train_pairs[:-1]

        print(f"\n  [{pk}] Fragility ({len(train_pairs)} train, {len(test_pairs)} test)...")

        train_moral_acts = collect_activations(model, [p["moral"] for p in train_pairs], n_layers)
        train_neutral_acts = collect_activations(model, [p["neutral"] for p in train_pairs], n_layers)
        test_moral_acts = collect_activations(model, [p["moral"] for p in test_pairs], n_layers)
        test_neutral_acts = collect_activations(model, [p["neutral"] for p in test_pairs], n_layers)

        per_layer: dict[str, dict] = {}

        for layer_idx in range(n_layers):
            train_X = torch.cat([
                torch.stack(train_moral_acts[layer_idx]),
                torch.stack(train_neutral_acts[layer_idx]),
            ])
            train_y = torch.cat([torch.ones(len(train_pairs)), torch.zeros(len(train_pairs))])
            test_X = torch.cat([
                torch.stack(test_moral_acts[layer_idx]),
                torch.stack(test_neutral_acts[layer_idx]),
            ])
            test_y = torch.cat([torch.ones(len(test_pairs)), torch.zeros(len(test_pairs))])

            layer_result = run_fragility_for_probe(
                train_X, train_y, test_X, test_y,
                noise_levels=NOISE_LEVELS,
                n_epochs=args.n_epochs, lr=args.lr,
            )
            per_layer[str(layer_idx)] = layer_result

        # Summary: cap-at-max mean over ALL layers (never-fragile censored at cap).
        capped = [d["critical_noise_capped"] for d in per_layer.values()]
        mean_critical = float(np.mean(capped)) if capped else None
        n_never_fragile = sum(1 for d in per_layer.values() if d["critical_noise"] is None)

        results[pk] = {
            "per_layer": per_layer,
            "mean_critical_noise": round(mean_critical, 4) if mean_critical is not None else None,
            "n_never_fragile": n_never_fragile,
            "n_train": len(train_pairs),
            "n_test": len(test_pairs),
        }

        print(f"    Mean critical noise: {mean_critical:.2f}" if mean_critical else
              "    No critical noise reached")

    # Also compute pooled dilemma fragility (all dilemma vs all neutral)
    print(f"\n  [pooled] Computing pooled dilemma fragility...")
    all_moral = [p["moral"] for p in dilemma_data["pairs"]]
    all_neutral = [p["neutral"] for p in dilemma_data["pairs"]]

    n_pooled = len(all_moral)
    n_pooled_train = max(1, int(n_pooled * 0.8))
    train_moral_pooled = all_moral[:n_pooled_train]
    train_neutral_pooled = all_neutral[:n_pooled_train]
    test_moral_pooled = all_moral[n_pooled_train:]
    test_neutral_pooled = all_neutral[n_pooled_train:]

    if len(test_moral_pooled) < 1:
        test_moral_pooled = train_moral_pooled[-1:]
        test_neutral_pooled = train_neutral_pooled[-1:]
        train_moral_pooled = train_moral_pooled[:-1]
        train_neutral_pooled = train_neutral_pooled[:-1]

    train_m_acts = collect_activations(model, train_moral_pooled, n_layers)
    train_n_acts = collect_activations(model, train_neutral_pooled, n_layers)
    test_m_acts = collect_activations(model, test_moral_pooled, n_layers)
    test_n_acts = collect_activations(model, test_neutral_pooled, n_layers)

    pooled_per_layer: dict[str, dict] = {}
    for layer_idx in range(n_layers):
        train_X = torch.cat([
            torch.stack(train_m_acts[layer_idx]),
            torch.stack(train_n_acts[layer_idx]),
        ])
        train_y = torch.cat([torch.ones(len(train_moral_pooled)), torch.zeros(len(train_neutral_pooled))])
        test_X = torch.cat([
            torch.stack(test_m_acts[layer_idx]),
            torch.stack(test_n_acts[layer_idx]),
        ])
        test_y = torch.cat([torch.ones(len(test_moral_pooled)), torch.zeros(len(test_neutral_pooled))])

        pooled_per_layer[str(layer_idx)] = run_fragility_for_probe(
            train_X, train_y, test_X, test_y,
            noise_levels=NOISE_LEVELS,
            n_epochs=args.n_epochs, lr=args.lr,
        )

    pooled_capped = [d["critical_noise_capped"] for d in pooled_per_layer.values()]
    pooled_mean_critical = float(np.mean(pooled_capped)) if pooled_capped else None

    output = {
        "experiment": "dilemma_fragility",
        "model": args.model,
        "noise_levels": NOISE_LEVELS,
        "fragility_threshold": FRAGILITY_THRESHOLD,
        "n_noise_seeds": N_NOISE_SEEDS,
        "per_dilemma_type": results,
        "pooled_dilemma": {
            "per_layer": pooled_per_layer,
            "mean_critical_noise": round(pooled_mean_critical, 4) if pooled_mean_critical else None,
        },
    }

    with open(output_dir / "dilemma_fragility.json", "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("COMPLEXITY-FRAGILITY GRADIENT")
    print(f"{'='*60}")

    all_dilemma_criticals = [
        r["mean_critical_noise"] for r in results.values()
        if r["mean_critical_noise"] is not None
    ]
    mean_dilemma_critical = float(np.mean(all_dilemma_criticals)) if all_dilemma_criticals else None

    print(f"\nPooled dilemma probe mean critical noise: {pooled_mean_critical}")
    print(f"Per-type dilemma probe mean critical noise: {mean_dilemma_critical}")
    print(f"\nExpected ordering (most → least robust):")
    print(f"  Pooled binary > Single-foundation > Dilemma-specific")
    print(f"\nCompare with single-foundation results from exp7_fragility to assess gradient.")

    del model
    _clear_memory()

    print(f"\nOutput: {output_dir / 'dilemma_fragility.json'}")


if __name__ == "__main__":
    main()
