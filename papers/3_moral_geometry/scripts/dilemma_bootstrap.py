#!/usr/bin/env python3
"""Script 5: Bootstrap stability for dilemma probe directions.

With only 16 training pairs per dilemma type, direction stability is critical.
This script runs 200 bootstrap resamples per dilemma type per layer, measuring
how stable the learned directions are.

Stability threshold: mean cosine > 0.7 (relaxed from 0.8 for foundation probes
due to smaller sample size).

Estimated runtime: ~3 hours (200 × 15 × 16 = 48,000 probe trainings)

Usage:
    python papers/3_moral_geometry/scripts/dilemma_bootstrap.py
    python papers/3_moral_geometry/scripts/dilemma_bootstrap.py --n-bootstrap 50  # quick test
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


def train_probe_with_direction(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
    test_y: torch.Tensor,
    *,
    n_epochs: int = 50,
    lr: float = 1e-2,
) -> tuple[float, float, np.ndarray]:
    """Train a linear probe and return (accuracy, loss, unit-norm weight vector)."""
    hidden_dim = train_X.shape[1]
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
        test_logits = probe(test_X).squeeze(-1)
        test_loss = loss_fn(test_logits, test_y).item()
        preds = (test_logits > 0).float()
        accuracy = (preds == test_y).float().mean().item()

    w = probe.weight.data.squeeze(0).cpu().numpy()
    w_norm = w / (np.linalg.norm(w) + 1e-12)
    return accuracy, test_loss, w_norm


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap stability for dilemma probes.")
    parser.add_argument("--dataset", default="deepsteer/datasets/dilemma_pairs_final.json")
    parser.add_argument("--output-dir", default="papers/3_moral_geometry/outputs/dilemma_bootstrap")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
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

    print(f"Loading model: {OLMO_REPO}")
    t0 = time.time()
    model = WhiteBoxModel(OLMO_REPO, device=args.device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    print(f"Loaded in {time.time() - t0:.1f}s ({n_layers} layers)")

    # Group dilemma pairs by foundation pair
    pairs_by_type: dict[str, list[dict]] = {}
    for p in dilemma_data["pairs"]:
        pk = f"{p['foundation_pair'][0]}-{p['foundation_pair'][1]}"
        pairs_by_type.setdefault(pk, []).append(p)

    rng = np.random.RandomState(args.seed)
    stability: dict[str, dict[int, dict]] = {}

    dilemma_types = [(f"{a}-{b}", pairs_by_type.get(f"{a}-{b}", []))
                     for a, b in FOUNDATION_PAIRS]
    dilemma_types = [(pk, pairs) for pk, pairs in dilemma_types if len(pairs) >= 5]

    total_probes = len(dilemma_types) * n_layers * args.n_bootstrap
    probes_done = 0
    t_start = time.time()

    for pk, pairs in dilemma_types:
        # Train/test split: 80/20
        n_train = max(1, int(len(pairs) * 0.8))
        train_pairs = pairs[:n_train]
        test_pairs = pairs[n_train:]
        if len(test_pairs) < 1:
            test_pairs = [train_pairs[-1]]
            train_pairs = train_pairs[:-1]

        print(f"\n  [{pk}] Bootstrap ({args.n_bootstrap} resamples × {n_layers} layers, "
              f"{len(train_pairs)} train pairs)...")

        # Collect activations once
        train_moral = [p["moral"] for p in train_pairs]
        train_neutral = [p["neutral"] for p in train_pairs]
        test_moral = [p["moral"] for p in test_pairs]
        test_neutral = [p["neutral"] for p in test_pairs]

        train_moral_acts = collect_activations(model, train_moral, n_layers)
        train_neutral_acts = collect_activations(model, train_neutral, n_layers)
        test_moral_acts = collect_activations(model, test_moral, n_layers)
        test_neutral_acts = collect_activations(model, test_neutral, n_layers)

        stability[pk] = {}

        for layer_idx in range(n_layers):
            # Full-data training
            train_X = torch.cat([
                torch.stack(train_moral_acts[layer_idx]),
                torch.stack(train_neutral_acts[layer_idx]),
            ])
            train_y = torch.cat([
                torch.ones(len(train_pairs)),
                torch.zeros(len(train_pairs)),
            ])
            test_X = torch.cat([
                torch.stack(test_moral_acts[layer_idx]),
                torch.stack(test_neutral_acts[layer_idx]),
            ])
            test_y = torch.cat([
                torch.ones(len(test_pairs)),
                torch.zeros(len(test_pairs)),
            ])

            _, _, full_dir = train_probe_with_direction(
                train_X, train_y, test_X, test_y,
                n_epochs=args.n_epochs, lr=args.lr,
            )

            # Bootstrap
            bootstrap_cosines = []
            n_train_samples = len(train_pairs)

            for _ in range(args.n_bootstrap):
                pair_indices = rng.choice(n_train_samples, size=n_train_samples, replace=True)

                boot_moral = torch.stack([train_moral_acts[layer_idx][i] for i in pair_indices])
                boot_neutral = torch.stack([train_neutral_acts[layer_idx][i] for i in pair_indices])
                boot_X = torch.cat([boot_moral, boot_neutral])
                boot_y = torch.cat([torch.ones(n_train_samples), torch.zeros(n_train_samples)])

                _, _, boot_dir = train_probe_with_direction(
                    boot_X, boot_y, test_X, test_y,
                    n_epochs=args.n_epochs, lr=args.lr,
                )

                cos_with_full = float(np.dot(full_dir, boot_dir))
                bootstrap_cosines.append(abs(cos_with_full))
                probes_done += 1

            mean_cos = float(np.mean(bootstrap_cosines))
            std_cos = float(np.std(bootstrap_cosines))

            stability[pk][layer_idx] = {
                "mean_cosine_with_full": round(mean_cos, 6),
                "std_cosine_with_full": round(std_cos, 6),
                "min_cosine_with_full": round(float(np.min(bootstrap_cosines)), 6),
                "stable": mean_cos > 0.7,
            }

        elapsed = time.time() - t_start
        rate = probes_done / elapsed if elapsed > 0 else 0
        remaining = (total_probes - probes_done) / rate if rate > 0 else 0
        print(f"    Done ({probes_done}/{total_probes}, {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    # Assess overall stability
    all_stable = True
    unstable_count = 0
    for pk, layers in stability.items():
        for layer_idx, stats in layers.items():
            if not stats["stable"]:
                all_stable = False
                unstable_count += 1

    output = {
        "experiment": "dilemma_bootstrap",
        "model": OLMO_REPO,
        "n_bootstrap": args.n_bootstrap,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "seed": args.seed,
        "stability_threshold": 0.7,
        "all_stable": all_stable,
        "unstable_direction_count": unstable_count,
        "per_dilemma_type": {
            pk: {str(k): v for k, v in layers.items()}
            for pk, layers in stability.items()
        },
    }

    with open(output_dir / "dilemma_bootstrap.json", "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("BOOTSTRAP STABILITY SUMMARY")
    print(f"{'='*60}")
    print(f"Threshold: mean cosine > 0.7")
    print(f"All stable: {all_stable}")
    print(f"Unstable directions: {unstable_count}")

    print(f"\n{'Pair':25s}", end="")
    for l in range(n_layers):
        print(f" L{l:02d}", end="")
    print()

    for pk, layers in stability.items():
        print(f"{pk:25s}", end="")
        for l in range(n_layers):
            if l in layers:
                cos = layers[l]["mean_cosine_with_full"]
                marker = " OK" if layers[l]["stable"] else " !!!"
                if cos >= 0.9:
                    marker = "  ++"
                print(f"{marker}", end="")
            else:
                print("  --", end="")
        print()

    if not all_stable:
        print(f"\nCAUTION: {unstable_count} unstable directions detected.")
        print("Focus analysis on pooled dilemma geometry rather than per-pair subspace analysis.")
    else:
        print("\nAll dilemma probe directions are stable.")

    del model
    _clear_memory()

    print(f"\nOutput: {output_dir / 'dilemma_bootstrap.json'}")


if __name__ == "__main__":
    main()
