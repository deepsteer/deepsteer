#!/usr/bin/env python3
"""Script 3: Dilemma probing and subspace projection analysis.

Trains 15 dilemma-specific probes (one per foundation pair) and computes
the subspace membership score: does the dilemma direction lie within the
span of its two component foundation directions?

This is the core experiment of the dilemma extension.

Usage:
    python papers/3_moral_geometry/scripts/dilemma_probing.py
    python papers/3_moral_geometry/scripts/dilemma_probing.py --skip-null
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

DILEMMA_KEY_TO_FOUNDATION = {
    "care": "care_harm",
    "fairness": "fairness_cheating",
    "liberty": "liberty_oppression",
    "loyalty": "loyalty_betrayal",
    "authority": "authority_subversion",
    "sanctity": "sanctity_degradation",
}

FOUNDATION_PAIRS = [
    ("care", "fairness"), ("care", "liberty"), ("care", "loyalty"),
    ("care", "authority"), ("care", "sanctity"),
    ("fairness", "liberty"), ("fairness", "loyalty"),
    ("fairness", "authority"), ("fairness", "sanctity"),
    ("liberty", "loyalty"), ("liberty", "authority"), ("liberty", "sanctity"),
    ("loyalty", "authority"), ("loyalty", "sanctity"),
    ("authority", "sanctity"),
]


from deepsteer.core.device import clear_memory as _clear_memory  # shared helper


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
        test_logits = probe(test_X).squeeze(-1)
        test_loss = loss_fn(test_logits, test_y).item()
        preds = (test_logits > 0).float()
        accuracy = (preds == test_y).float().mean().item()

    w = probe.weight.data.squeeze(0).cpu().numpy()
    w_norm = w / (np.linalg.norm(w) + 1e-12)

    return accuracy, test_loss, w_norm


def collect_activations(model, texts: list[str], n_layers: int) -> dict[int, list[torch.Tensor]]:
    """Collect mean-pooled activations for a list of texts at each layer (batched)."""
    pooled = model.collect_batch_activations(
        texts, layers=list(range(n_layers)), pooling="mean",
    )  # {layer: (n_texts, hidden)}
    return {layer_idx: list(pooled[layer_idx]) for layer_idx in range(n_layers)}


def prepare_probe_data(
    moral_acts: dict[int, list[torch.Tensor]],
    neutral_acts: dict[int, list[torch.Tensor]],
    n_layers: int,
    train_indices: list[int],
    test_indices: list[int],
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Prepare train/test splits from collected activations."""
    result = {}
    for layer_idx in range(n_layers):
        train_moral = torch.stack([moral_acts[layer_idx][i] for i in train_indices])
        train_neutral = torch.stack([neutral_acts[layer_idx][i] for i in train_indices])
        test_moral = torch.stack([moral_acts[layer_idx][i] for i in test_indices])
        test_neutral = torch.stack([neutral_acts[layer_idx][i] for i in test_indices])

        train_X = torch.cat([train_moral, train_neutral], dim=0)
        train_y = torch.cat([torch.ones(len(train_indices)), torch.zeros(len(train_indices))])
        test_X = torch.cat([test_moral, test_neutral], dim=0)
        test_y = torch.cat([torch.ones(len(test_indices)), torch.zeros(len(test_indices))])

        result[layer_idx] = (train_X, train_y, test_X, test_y)

    return result


def compute_subspace_analysis(
    w_dilemma: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
) -> dict:
    """Compute subspace membership, component balance, and residual for a dilemma direction.

    Args:
        w_dilemma: Unit-norm dilemma probe direction (hidden_dim,)
        w_a: Unit-norm foundation A direction (hidden_dim,)
        w_b: Unit-norm foundation B direction (hidden_dim,)

    Returns:
        Dict with subspace_membership, component_balance, residual_norm,
        cosine_with_a, cosine_with_b.
    """
    # Gram-Schmidt orthogonalization of the 2D subspace
    e1 = w_a.copy()
    e2 = w_b - np.dot(w_b, e1) * e1
    e2_norm = np.linalg.norm(e2)
    if e2_norm < 1e-10:
        # Directions are nearly parallel — degenerate case
        proj_a = np.dot(w_dilemma, e1) ** 2
        return {
            "subspace_membership": float(proj_a),
            "component_balance": 0.5,
            "residual_norm": float(np.sqrt(1 - proj_a)),
            "cosine_with_a": float(np.dot(w_dilemma, w_a)),
            "cosine_with_b": float(np.dot(w_dilemma, w_b)),
            "degenerate": True,
        }
    e2 = e2 / e2_norm

    # Subspace membership: fraction of variance explained by the 2D subspace
    proj_e1 = np.dot(w_dilemma, e1)
    proj_e2 = np.dot(w_dilemma, e2)
    subspace_membership = proj_e1 ** 2 + proj_e2 ** 2

    # Component balance: relative projection onto original (non-orthogonalized) directions
    cos_a = abs(np.dot(w_dilemma, w_a))
    cos_b = abs(np.dot(w_dilemma, w_b))
    denom = cos_a + cos_b
    component_balance = cos_a / denom if denom > 1e-10 else 0.5

    # Residual direction
    residual = w_dilemma - proj_e1 * e1 - proj_e2 * e2
    residual_norm = float(np.linalg.norm(residual))

    return {
        "subspace_membership": float(subspace_membership),
        "component_balance": float(component_balance),
        "residual_norm": residual_norm,
        "cosine_with_a": float(np.dot(w_dilemma, w_a)),
        "cosine_with_b": float(np.dot(w_dilemma, w_b)),
        "degenerate": False,
    }


def compute_null_distribution(
    w_a: np.ndarray,
    w_b: np.ndarray,
    n_samples: int = 10000,
    seed: int = 42,
) -> dict:
    """Compute null distribution of subspace membership for random unit vectors."""
    rng = np.random.RandomState(seed)
    hidden_dim = len(w_a)

    # Orthonormalize the 2D subspace
    e1 = w_a.copy()
    e2 = w_b - np.dot(w_b, e1) * e1
    e2_norm = np.linalg.norm(e2)
    if e2_norm < 1e-10:
        return {"mean": 1.0 / hidden_dim, "p95": 2.0 / hidden_dim, "degenerate": True}
    e2 = e2 / e2_norm

    scores = []
    for _ in range(n_samples):
        v = rng.randn(hidden_dim)
        v = v / np.linalg.norm(v)
        score = np.dot(v, e1) ** 2 + np.dot(v, e2) ** 2
        scores.append(score)

    scores = np.array(scores)
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "degenerate": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dilemma probing and subspace analysis.")
    parser.add_argument("--dataset", default="deepsteer/datasets/dilemma_pairs_final.json")
    parser.add_argument("--directions", default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--output-dir", default="papers/3_moral_geometry/outputs/dilemma_probing")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=OLMO_REPO, help="HuggingFace model ID.")
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--skip-null", action="store_true",
                        help="Skip null distribution computation.")
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
        return

    directions_path = Path(args.directions)
    if not directions_path.exists():
        print(f"ERROR: Foundation directions not found: {directions_path}")
        return

    with open(dataset_path) as f:
        dilemma_data = json.load(f)

    foundation_directions_data = np.load(directions_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    t0 = time.time()
    model = WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    sample_key = list(foundation_directions_data.keys())[0]
    hidden_dim = foundation_directions_data[sample_key].shape[0]
    print(f"Loaded in {time.time() - t0:.1f}s ({n_layers} layers, {hidden_dim} hidden dim)")

    # Load foundation directions
    foundation_dirs: dict[str, dict[int, np.ndarray]] = {}
    for fv in FOUNDATION_ORDER:
        foundation_dirs[fv] = {}
        for layer_idx in range(n_layers):
            key = f"{fv}_layer{layer_idx}"
            if key in foundation_directions_data:
                foundation_dirs[fv][layer_idx] = foundation_directions_data[key]

    # Group dilemma pairs by foundation pair
    pairs_by_type: dict[str, list[dict]] = {}
    for p in dilemma_data["pairs"]:
        pk = f"{p['foundation_pair'][0]}-{p['foundation_pair'][1]}"
        pairs_by_type.setdefault(pk, []).append(p)

    # Collect all activations once
    all_moral_texts = [p["moral"] for p in dilemma_data["pairs"]]
    all_neutral_texts = [p["neutral"] for p in dilemma_data["pairs"]]

    print(f"\nCollecting activations for {len(all_moral_texts)} moral + {len(all_neutral_texts)} neutral texts...")
    t0 = time.time()
    moral_acts = collect_activations(model, all_moral_texts, n_layers)
    neutral_acts = collect_activations(model, all_neutral_texts, n_layers)
    print(f"Collected in {time.time() - t0:.1f}s")

    # Build index: global index for each pair in each foundation-pair group
    pair_global_indices: dict[str, list[int]] = {}
    for i, p in enumerate(dilemma_data["pairs"]):
        pk = f"{p['foundation_pair'][0]}-{p['foundation_pair'][1]}"
        pair_global_indices.setdefault(pk, []).append(i)

    # Train dilemma probes and run subspace analysis
    print(f"\n{'='*60}")
    print("DILEMMA PROBING AND SUBSPACE ANALYSIS")
    print(f"{'='*60}")

    all_results: dict[str, dict] = {}
    dilemma_directions: dict[str, dict[int, np.ndarray]] = {}

    for pair in FOUNDATION_PAIRS:
        pk = f"{pair[0]}-{pair[1]}"
        indices = pair_global_indices.get(pk, [])
        if len(indices) < 5:
            print(f"\n  [{pk}] Skipping: only {len(indices)} pairs")
            continue

        # Train/test split: 80/20
        n_train = max(1, int(len(indices) * 0.8))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        if len(test_idx) < 1:
            test_idx = [train_idx[-1]]
            train_idx = train_idx[:-1]

        print(f"\n  [{pk}] {len(train_idx)} train, {len(test_idx)} test pairs")

        # Prepare activation data
        probe_data = prepare_probe_data(moral_acts, neutral_acts, n_layers, train_idx, test_idx)

        # Foundation directions for this pair
        fv_a = DILEMMA_KEY_TO_FOUNDATION[pair[0]]
        fv_b = DILEMMA_KEY_TO_FOUNDATION[pair[1]]

        pair_results: dict[str, dict] = {}
        dilemma_directions[pk] = {}

        for layer_idx in range(n_layers):
            train_X, train_y, test_X, test_y = probe_data[layer_idx]

            # Train dilemma probe
            acc, loss, w_dilemma = train_probe_with_direction(
                train_X, train_y, test_X, test_y,
                n_epochs=args.n_epochs, lr=args.lr,
            )
            dilemma_directions[pk][layer_idx] = w_dilemma

            # Get foundation directions
            w_a = foundation_dirs[fv_a].get(layer_idx)
            w_b = foundation_dirs[fv_b].get(layer_idx)

            layer_result: dict = {
                "accuracy": round(acc, 4),
                "loss": round(loss, 4),
            }

            if w_a is not None and w_b is not None:
                subspace = compute_subspace_analysis(w_dilemma, w_a, w_b)
                layer_result.update({k: round(v, 6) if isinstance(v, float) else v
                                     for k, v in subspace.items()})

                # Mismatched-pair baseline: membership in the 2D spans of every
                # foundation pair that shares NO component with this dilemma.
                # This absorbs the shared moral-salience component that the
                # random-vector null does not, so it is the correct null.
                mismatched = []
                for g1, g2 in FOUNDATION_PAIRS:
                    if pair[0] in (g1, g2) or pair[1] in (g1, g2):
                        continue
                    w_g1 = foundation_dirs[DILEMMA_KEY_TO_FOUNDATION[g1]].get(layer_idx)
                    w_g2 = foundation_dirs[DILEMMA_KEY_TO_FOUNDATION[g2]].get(layer_idx)
                    if w_g1 is None or w_g2 is None:
                        continue
                    mismatched.append(
                        compute_subspace_analysis(w_dilemma, w_g1, w_g2)["subspace_membership"]
                    )
                if mismatched:
                    layer_result["mismatched_membership"] = round(float(np.mean(mismatched)), 6)
                    layer_result["n_mismatched_pairs"] = len(mismatched)

            pair_results[str(layer_idx)] = layer_result

        # Summary for this pair
        accs = {int(k): v["accuracy"] for k, v in pair_results.items()}
        peak_layer = max(accs, key=accs.get)
        memberships = {int(k): v.get("subspace_membership", 0) for k, v in pair_results.items()}
        peak_membership_layer = max(memberships, key=memberships.get)

        peak_mismatched = pair_results[str(peak_membership_layer)].get("mismatched_membership")
        all_results[pk] = {
            "foundation_pair": list(pair),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "peak_accuracy_layer": peak_layer,
            "peak_accuracy": round(accs[peak_layer], 4),
            "peak_subspace_membership": round(memberships[peak_membership_layer], 6),
            "peak_subspace_layer": peak_membership_layer,
            "mismatched_at_peak_layer": peak_mismatched,
            "per_layer": pair_results,
        }

        print(f"    Peak accuracy: {accs[peak_layer]:.1%} @ layer {peak_layer}")
        print(f"    Peak subspace membership: {memberships[peak_membership_layer]:.4f} @ layer {peak_membership_layer}")

    # Compute null distribution (once, using first pair's directions)
    null_result = None
    if not args.skip_null:
        print(f"\nComputing null distribution (10,000 random vectors)...")
        fv_a = DILEMMA_KEY_TO_FOUNDATION["care"]
        fv_b = DILEMMA_KEY_TO_FOUNDATION["fairness"]
        w_a = foundation_dirs[fv_a].get(n_layers // 2)
        w_b = foundation_dirs[fv_b].get(n_layers // 2)
        if w_a is not None and w_b is not None:
            null_result = compute_null_distribution(w_a, w_b)
            print(f"  Null mean: {null_result['mean']:.6f}, 95th: {null_result['p95']:.6f}, "
                  f"99th: {null_result['p99']:.6f}")

    # Save directions
    direction_arrays = {}
    for pk, layers in dilemma_directions.items():
        for layer_idx, w in layers.items():
            direction_arrays[f"dilemma_{pk}_layer{layer_idx}"] = w
    np.savez(output_dir / "dilemma_probe_directions.npz", **direction_arrays)

    # Aggregate matched vs. mismatched membership (per-pair-peak and cross-layer).
    peak_matched = [r["peak_subspace_membership"] for r in all_results.values()]
    peak_mismatched = [r["mismatched_at_peak_layer"] for r in all_results.values()
                       if r.get("mismatched_at_peak_layer") is not None]
    all_matched = [lr.get("subspace_membership") for r in all_results.values()
                   for lr in r["per_layer"].values() if lr.get("subspace_membership") is not None]
    all_mismatched = [lr.get("mismatched_membership") for r in all_results.values()
                      for lr in r["per_layer"].values() if lr.get("mismatched_membership") is not None]
    membership_summary = {
        "matched_peak_mean": round(float(np.mean(peak_matched)), 4) if peak_matched else None,
        "mismatched_peak_mean": round(float(np.mean(peak_mismatched)), 4) if peak_mismatched else None,
        "matched_crosslayer_mean": round(float(np.mean(all_matched)), 4) if all_matched else None,
        "mismatched_crosslayer_mean": round(float(np.mean(all_mismatched)), 4) if all_mismatched else None,
        "random_null_mean": round(null_result["mean"], 6) if null_result else None,
    }

    # Save results
    output = {
        "experiment": "dilemma_probing",
        "model": args.model,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "null_distribution": null_result,
        "membership_summary": membership_summary,
        "per_foundation_pair": all_results,
    }

    with open(output_dir / "dilemma_probing.json", "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUBSPACE ANALYSIS SUMMARY")
    print(f"{'='*60}")

    if null_result:
        print(f"Null baseline: mean = {null_result['mean']:.6f}, 95th = {null_result['p95']:.6f}")

    print(f"\n{'Pair':25s} {'Peak Acc':>10s} {'Membership':>12s} {'Balance':>10s} {'Residual':>10s}")
    print("-" * 70)
    for pk, res in sorted(all_results.items()):
        peak_l = res["peak_subspace_layer"]
        layer_data = res["per_layer"].get(str(peak_l), {})
        membership = layer_data.get("subspace_membership", 0)
        balance = layer_data.get("component_balance", 0)
        residual = layer_data.get("residual_norm", 0)
        print(f"{pk:25s} {res['peak_accuracy']:>9.1%} {membership:>12.4f} {balance:>10.3f} {residual:>10.3f}")

    mean_membership = np.mean([r["peak_subspace_membership"] for r in all_results.values()])
    print(f"\nMean peak subspace membership: {mean_membership:.4f}")

    if mean_membership > 0.3:
        print("OUTCOME A: High compositionality — dilemma directions lie within foundation subspaces.")
    elif mean_membership > 0.05:
        print("OUTCOME B/C: Partial compositionality — residual analysis needed.")
    else:
        print("OUTCOME C: Low compositionality — dilemma encoding is novel.")

    del model
    _clear_memory()

    print(f"\nOutputs: {output_dir}")
    print("Next: Run dilemma_geometry.py and dilemma_bootstrap.py")


if __name__ == "__main__":
    main()
