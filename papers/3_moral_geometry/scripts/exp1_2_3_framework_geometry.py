#!/usr/bin/env python3
"""Experiments 1–3: Foundation-specific probing, framework geometry, bootstrap stability.

Exp 1: Train per-foundation binary probes at each layer of OLMo-2 1B.
    Extract the learned weight vectors as the "foundation direction" in
    representation space. Report per-foundation accuracy curves.

Exp 2: Compute the 6×6 pairwise cosine similarity matrix of foundation
    probe directions at each layer. Hierarchical clustering, permutation
    test for the individualizing/binding MFT distinction, effective
    dimensionality (PCA) of the 6-direction set at each layer.

Exp 3: Bootstrap direction stability — resample training pairs 200×,
    retrain probes, measure direction stability via cosine similarity
    with the full-data direction. Go/no-go gate for geometric analysis.

Target: allenai/OLMo-2-0425-1B (1.5B params, 16 layers, 2048 hidden dim)
Hardware: MacBook Pro M4 Pro, 24 GB unified memory, MPS
Estimated runtime: ~2.5 hours (dominated by Exp 3 bootstrap)

Usage:
    # Full run (Experiments 1-3)
    python papers/3_moral_geometry/scripts/exp1_2_3_framework_geometry.py

    # Exp 1+2 only (skip bootstrap, ~15 min)
    python papers/3_moral_geometry/scripts/exp1_2_3_framework_geometry.py --skip-bootstrap

    # Quick test with reduced dataset
    python papers/3_moral_geometry/scripts/exp1_2_3_framework_geometry.py --dataset-target 10 --skip-bootstrap

    # Custom bootstrap count
    python papers/3_moral_geometry/scripts/exp1_2_3_framework_geometry.py --n-bootstrap 100
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from collections import defaultdict
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

INDIVIDUALIZING = {"care_harm", "fairness_cheating", "liberty_oppression"}
BINDING = {"loyalty_betrayal", "authority_subversion", "sanctity_degradation"}


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Experiment 1: Foundation-specific probing with direction extraction
# ---------------------------------------------------------------------------


def train_probe_with_direction(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
    test_y: torch.Tensor,
    *,
    n_epochs: int = 50,
    lr: float = 1e-2,
) -> tuple[float, float, np.ndarray]:
    """Train a linear probe and return (accuracy, loss, unit-norm weight vector).

    The weight vector w of the nn.Linear(hidden_dim, 1) probe is the normal
    to the classification hyperplane — the "direction" that separates moral
    from neutral in the representation space.
    """
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

    w = probe.weight.data.squeeze(0).cpu().numpy()  # (hidden_dim,)
    w_norm = w / (np.linalg.norm(w) + 1e-12)

    return accuracy, test_loss, w_norm


def run_experiment_1(
    model,
    dataset,
    output_dir: Path,
    *,
    n_epochs: int = 50,
    lr: float = 1e-2,
) -> dict:
    """Experiment 1: Per-foundation probing with probe direction extraction.

    Returns dict with keys:
        directions: {foundation: {layer: unit_norm_weight_vector}}
        accuracies: {foundation: {layer: accuracy}}
        n_layers: int
        hidden_dim: int
    """
    from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
    from deepsteer.core.types import MoralFoundation
    from deepsteer.datasets.types import ProbingPair

    n_layers = model.info.n_layers
    assert n_layers is not None

    train_by_foundation: dict[str, list[ProbingPair]] = defaultdict(list)
    test_by_foundation: dict[str, list[ProbingPair]] = defaultdict(list)
    for pair in dataset.train:
        train_by_foundation[pair.foundation.value].append(pair)
    for pair in dataset.test:
        test_by_foundation[pair.foundation.value].append(pair)

    directions: dict[str, dict[int, np.ndarray]] = {}
    accuracies: dict[str, dict[int, float]] = {}
    hidden_dim = None

    for foundation_val in FOUNDATION_ORDER:
        foundation = MoralFoundation(foundation_val)
        train_pairs = train_by_foundation.get(foundation_val, [])
        test_pairs = test_by_foundation.get(foundation_val, [])

        if len(train_pairs) < 5 or len(test_pairs) < 1:
            print(f"  Skipping {foundation_val}: {len(train_pairs)} train, "
                  f"{len(test_pairs)} test pairs")
            continue

        print(f"  Probing {FOUNDATION_SHORT[foundation_val]}: "
              f"{len(train_pairs)} train, {len(test_pairs)} test pairs")

        all_train = LayerWiseMoralProbe._collect_all_activations(model, train_pairs)
        all_test = LayerWiseMoralProbe._collect_all_activations(model, test_pairs)

        directions[foundation_val] = {}
        accuracies[foundation_val] = {}

        for layer_idx in range(n_layers):
            train_X, train_y = all_train[layer_idx]
            test_X, test_y = all_test[layer_idx]

            if hidden_dim is None:
                hidden_dim = train_X.shape[1]

            acc, loss, w_norm = train_probe_with_direction(
                train_X, train_y, test_X, test_y,
                n_epochs=n_epochs, lr=lr,
            )
            directions[foundation_val][layer_idx] = w_norm
            accuracies[foundation_val][layer_idx] = acc

        peak_layer = max(accuracies[foundation_val], key=accuracies[foundation_val].get)
        peak_acc = accuracies[foundation_val][peak_layer]
        print(f"    Peak: {peak_acc:.1%} @ layer {peak_layer}")

    exp1_json = {
        "experiment": "exp1_foundation_probing",
        "model": model.info.name,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_epochs": n_epochs,
        "lr": lr,
        "per_foundation": {},
    }
    for fv in FOUNDATION_ORDER:
        if fv not in accuracies:
            continue
        accs = accuracies[fv]
        peak_layer = max(accs, key=accs.get)
        exp1_json["per_foundation"][fv] = {
            "n_train_pairs": len(train_by_foundation[fv]),
            "n_test_pairs": len(test_by_foundation[fv]),
            "per_layer_accuracy": {str(k): round(v, 4) for k, v in accs.items()},
            "peak_layer": peak_layer,
            "peak_accuracy": round(accs[peak_layer], 4),
        }

    with open(output_dir / "exp1_foundation_probing.json", "w") as f:
        json.dump(exp1_json, f, indent=2)

    # Save directions as npz for downstream use
    direction_arrays = {}
    for fv in FOUNDATION_ORDER:
        if fv not in directions:
            continue
        for layer_idx, w in directions[fv].items():
            direction_arrays[f"{fv}_layer{layer_idx}"] = w
    np.savez(output_dir / "exp1_probe_directions.npz", **direction_arrays)

    return {
        "directions": directions,
        "accuracies": accuracies,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
    }


# ---------------------------------------------------------------------------
# Experiment 2: Framework geometry analysis
# ---------------------------------------------------------------------------


def compute_cosine_similarity_matrix(
    directions: dict[str, dict[int, np.ndarray]],
    layer: int,
    foundation_order: list[str],
) -> np.ndarray | None:
    """Compute 6×6 cosine similarity matrix for foundation directions at one layer."""
    vecs = []
    for fv in foundation_order:
        if fv not in directions or layer not in directions[fv]:
            return None
        vecs.append(directions[fv][layer])
    mat = np.stack(vecs)  # (6, hidden_dim)
    cos_sim = mat @ mat.T  # Already unit-norm, so dot product = cosine similarity
    return cos_sim


def compute_effective_dimensionality(
    directions: dict[str, dict[int, np.ndarray]],
    layer: int,
    foundation_order: list[str],
    *,
    variance_threshold: float = 0.9,
) -> int | None:
    """Number of PCs explaining >=threshold of variance in the 6-direction set."""
    vecs = []
    for fv in foundation_order:
        if fv not in directions or layer not in directions[fv]:
            return None
        vecs.append(directions[fv][layer])
    mat = np.stack(vecs)  # (6, hidden_dim)
    mat_centered = mat - mat.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(mat_centered, full_matrices=False)
    explained = np.cumsum(s ** 2) / np.sum(s ** 2)
    n_components = int(np.searchsorted(explained, variance_threshold)) + 1
    return n_components


def permutation_test_mft_groups(
    cos_sim: np.ndarray,
    foundation_order: list[str],
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict:
    """Permutation test for individualizing vs. binding cluster structure.

    Tests whether within-group similarity exceeds between-group similarity
    more than expected by chance under random group assignment.

    With 6 foundations split into two groups of 3, there are only
    C(6,3) = 20 distinct group assignments, so the null distribution is
    enumerated exactly and the p-value is an exact multiple of 1/20.
    (Random resampling is kept as a fallback for larger group sets where
    exhaustive enumeration is impractical.)
    """
    from itertools import combinations
    from math import comb

    n = len(foundation_order)
    ind_idx = [i for i, f in enumerate(foundation_order) if f in INDIVIDUALIZING]
    bind_idx = [i for i, f in enumerate(foundation_order) if f in BINDING]
    k = len(ind_idx)

    def _group_statistic(sim_mat: np.ndarray, group_a: list[int], group_b: list[int]) -> float:
        within_a = [sim_mat[i, j] for i in group_a for j in group_a if i < j]
        within_b = [sim_mat[i, j] for i in group_b for j in group_b if i < j]
        between = [sim_mat[i, j] for i in group_a for j in group_b]
        within_mean = np.mean(within_a + within_b) if (within_a + within_b) else 0.0
        between_mean = np.mean(between) if between else 0.0
        return within_mean - between_mean

    observed = _group_statistic(cos_sim, ind_idx, bind_idx)

    # Exact enumeration when the number of partitions is small enough.
    n_partitions = comb(n, k) if 0 < k < n else 0
    exact = 0 < n_partitions <= 20000
    if exact:
        all_idx = list(range(n))
        count_ge = 0
        n_total = 0
        for group_a in combinations(all_idx, k):
            group_b = [i for i in all_idx if i not in group_a]
            stat = _group_statistic(cos_sim, list(group_a), group_b)
            n_total += 1
            if stat >= observed:
                count_ge += 1
        p_value = count_ge / n_total
        n_used = n_total
    else:
        rng = np.random.RandomState(seed)
        count_ge = 0
        for _ in range(n_permutations):
            perm = rng.permutation(n)
            perm_a = perm[:k].tolist()
            perm_b = perm[k:].tolist()
            stat = _group_statistic(cos_sim, perm_a, perm_b)
            if stat >= observed:
                count_ge += 1
        p_value = (count_ge + 1) / (n_permutations + 1)
        n_used = n_permutations

    within_ind = [cos_sim[i, j] for i in ind_idx for j in ind_idx if i < j]
    within_bind = [cos_sim[i, j] for i in bind_idx for j in bind_idx if i < j]
    between = [cos_sim[i, j] for i in ind_idx for j in bind_idx]

    return {
        "observed_statistic": float(observed),
        "p_value": float(p_value),
        "exact_enumeration": exact,
        "n_partitions": n_used,
        "n_permutations": n_used,
        "mean_within_individualizing": float(np.mean(within_ind)) if within_ind else None,
        "mean_within_binding": float(np.mean(within_bind)) if within_bind else None,
        "mean_between_groups": float(np.mean(between)) if between else None,
    }


def run_experiment_2(
    exp1_results: dict,
    output_dir: Path,
) -> dict:
    """Experiment 2: Framework geometry analysis.

    Returns dict with:
        cosine_matrices: {layer: 6×6 ndarray}
        effective_dims: {layer: int}
        mean_cosine: {layer: float}
        permutation_tests: {layer: dict}
    """
    directions = exp1_results["directions"]
    n_layers = exp1_results["n_layers"]

    foundations_present = [f for f in FOUNDATION_ORDER if f in directions]
    if len(foundations_present) < 2:
        print("  WARNING: Fewer than 2 foundations available, skipping geometry analysis")
        return {}

    cosine_matrices: dict[int, np.ndarray] = {}
    effective_dims: dict[int, int] = {}
    mean_cosine: dict[int, float] = {}
    min_cosine: dict[int, float] = {}
    max_cosine: dict[int, float] = {}
    permutation_tests: dict[int, dict] = {}

    for layer_idx in range(n_layers):
        cos_sim = compute_cosine_similarity_matrix(
            directions, layer_idx, foundations_present,
        )
        if cos_sim is None:
            continue

        cosine_matrices[layer_idx] = cos_sim

        n = len(foundations_present)
        upper_tri = [cos_sim[i, j] for i in range(n) for j in range(i + 1, n)]
        mean_cosine[layer_idx] = float(np.mean(upper_tri))
        min_cosine[layer_idx] = float(np.min(upper_tri))
        max_cosine[layer_idx] = float(np.max(upper_tri))

        eff_dim = compute_effective_dimensionality(
            directions, layer_idx, foundations_present,
        )
        if eff_dim is not None:
            effective_dims[layer_idx] = eff_dim

        if len(foundations_present) == 6:
            perm_result = permutation_test_mft_groups(
                cos_sim, foundations_present, n_permutations=10000,
            )
            permutation_tests[layer_idx] = perm_result

    # Find peak separation layer (minimum mean cosine similarity)
    if mean_cosine:
        peak_sep_layer = min(mean_cosine, key=mean_cosine.get)
        print(f"\n  Peak separation layer: {peak_sep_layer} "
              f"(mean cos sim = {mean_cosine[peak_sep_layer]:.4f})")
    else:
        peak_sep_layer = None

    # JSON output
    exp2_json = {
        "experiment": "exp2_framework_geometry",
        "foundations_present": foundations_present,
        "n_layers": n_layers,
        "per_layer": {},
        "peak_separation_layer": peak_sep_layer,
    }
    for layer_idx in range(n_layers):
        layer_data: dict = {}
        if layer_idx in mean_cosine:
            layer_data["mean_cosine_similarity"] = round(mean_cosine[layer_idx], 6)
            layer_data["min_cosine_similarity"] = round(min_cosine[layer_idx], 6)
            layer_data["max_cosine_similarity"] = round(max_cosine[layer_idx], 6)
        if layer_idx in effective_dims:
            layer_data["effective_dimensionality"] = effective_dims[layer_idx]
        if layer_idx in permutation_tests:
            pt = permutation_tests[layer_idx]
            layer_data["permutation_test"] = {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in pt.items()
            }
        if layer_idx in cosine_matrices:
            layer_data["cosine_similarity_matrix"] = cosine_matrices[layer_idx].tolist()
        if layer_data:
            exp2_json["per_layer"][str(layer_idx)] = layer_data

    with open(output_dir / "exp2_framework_geometry.json", "w") as f:
        json.dump(exp2_json, f, indent=2)

    return {
        "cosine_matrices": cosine_matrices,
        "effective_dims": effective_dims,
        "mean_cosine": mean_cosine,
        "min_cosine": min_cosine,
        "max_cosine": max_cosine,
        "permutation_tests": permutation_tests,
        "foundations_present": foundations_present,
        "peak_separation_layer": peak_sep_layer,
    }


# ---------------------------------------------------------------------------
# Experiment 3: Bootstrap direction stability
# ---------------------------------------------------------------------------


def run_experiment_3(
    model,
    dataset,
    exp1_results: dict,
    output_dir: Path,
    *,
    n_bootstrap: int = 200,
    n_epochs: int = 50,
    lr: float = 1e-2,
    seed: int = 42,
) -> dict:
    """Experiment 3: Bootstrap stability of foundation probe directions.

    Pre-collects activations once, then resamples from activation tensors
    (no additional forward passes). Retrains probes on resampled activations,
    computes cosine similarity of each bootstrap direction with the full-data
    direction.
    """
    from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
    from deepsteer.core.types import MoralFoundation
    from deepsteer.datasets.types import ProbingPair

    directions = exp1_results["directions"]
    n_layers = exp1_results["n_layers"]

    train_by_foundation: dict[str, list[ProbingPair]] = defaultdict(list)
    for pair in dataset.train:
        train_by_foundation[pair.foundation.value].append(pair)

    test_by_foundation: dict[str, list[ProbingPair]] = defaultdict(list)
    for pair in dataset.test:
        test_by_foundation[pair.foundation.value].append(pair)

    rng = np.random.RandomState(seed)
    stability: dict[str, dict[int, dict]] = {}

    foundations_to_test = [f for f in FOUNDATION_ORDER if f in directions]
    total_probes = len(foundations_to_test) * n_layers * n_bootstrap
    probes_done = 0
    t_start = time.time()

    for foundation_val in foundations_to_test:
        train_pairs = train_by_foundation[foundation_val]
        test_pairs = test_by_foundation[foundation_val]

        if len(train_pairs) < 5:
            continue

        print(f"  Bootstrap for {FOUNDATION_SHORT[foundation_val]} "
              f"({n_bootstrap} resamples × {n_layers} layers)...")

        # Collect ALL activations once (no forward passes in the bootstrap loop)
        all_train = LayerWiseMoralProbe._collect_all_activations(model, train_pairs)
        all_test = LayerWiseMoralProbe._collect_all_activations(model, test_pairs)

        full_directions = directions[foundation_val]
        n_train_samples = len(train_pairs) * 2  # moral + neutral per pair

        stability[foundation_val] = {}

        for layer_idx in range(n_layers):
            train_X, train_y = all_train[layer_idx]
            test_X, test_y = all_test[layer_idx]
            full_dir = full_directions[layer_idx]

            bootstrap_cosines = []

            for b in range(n_bootstrap):
                # Resample from activation tensor (pairs of rows: moral+neutral)
                pair_indices = rng.choice(len(train_pairs), size=len(train_pairs), replace=True)
                row_indices = []
                for pi in pair_indices:
                    row_indices.extend([pi * 2, pi * 2 + 1])
                boot_X = train_X[row_indices]
                boot_y = train_y[row_indices]

                _, _, boot_dir = train_probe_with_direction(
                    boot_X, boot_y, test_X, test_y,
                    n_epochs=n_epochs, lr=lr,
                )

                cos_with_full = float(np.dot(full_dir, boot_dir))
                bootstrap_cosines.append(abs(cos_with_full))

                probes_done += 1

            mean_cos = float(np.mean(bootstrap_cosines))
            std_cos = float(np.std(bootstrap_cosines))
            stability[foundation_val][layer_idx] = {
                "mean_cosine_with_full": mean_cos,
                "std_cosine_with_full": std_cos,
                "min_cosine_with_full": float(np.min(bootstrap_cosines)),
                "stable": mean_cos > 0.8,
            }

        elapsed = time.time() - t_start
        rate = probes_done / elapsed if elapsed > 0 else 0
        remaining = (total_probes - probes_done) / rate if rate > 0 else 0
        print(f"    Done ({probes_done}/{total_probes} probes, "
              f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    # Assess go/no-go
    all_stable = True
    for fv, layers in stability.items():
        for layer_idx, stats in layers.items():
            acc = exp1_results["accuracies"].get(fv, {}).get(layer_idx, 0.0)
            if acc > 0.55 and not stats["stable"]:
                all_stable = False
                print(f"  WARNING: Unstable direction at {fv} layer {layer_idx} "
                      f"(mean cos = {stats['mean_cosine_with_full']:.3f}, "
                      f"acc = {acc:.3f})")

    if all_stable:
        print("\n  GO: All directions stable (mean cos > 0.8 at all above-chance layers)")
    else:
        print("\n  CAUTION: Some directions unstable. Consider dataset expansion (Phase B).")

    exp3_json = {
        "experiment": "exp3_bootstrap_stability",
        "n_bootstrap": n_bootstrap,
        "n_epochs": n_epochs,
        "lr": lr,
        "seed": seed,
        "all_stable": all_stable,
        "per_foundation": {},
    }
    for fv in foundations_to_test:
        if fv not in stability:
            continue
        exp3_json["per_foundation"][fv] = {
            str(k): {kk: round(vv, 6) if isinstance(vv, float) else vv for kk, vv in v.items()}
            for k, v in stability[fv].items()
        }

    with open(output_dir / "exp3_bootstrap_stability.json", "w") as f:
        json.dump(exp3_json, f, indent=2)

    return {
        "stability": stability,
        "all_stable": all_stable,
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def generate_figures(
    exp1_results: dict,
    exp2_results: dict,
    exp3_results: dict | None,
    output_dir: Path,
) -> None:
    """Generate all Experiment 1-3 figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from scipy.cluster.hierarchy import dendrogram, linkage

    figures_dir = output_dir.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    accuracies = exp1_results["accuracies"]
    n_layers = exp1_results["n_layers"]
    foundations_present = [f for f in FOUNDATION_ORDER if f in accuracies]

    colors = {
        "care_harm": "#E53935",
        "fairness_cheating": "#1E88E5",
        "liberty_oppression": "#43A047",
        "loyalty_betrayal": "#FB8C00",
        "authority_subversion": "#8E24AA",
        "sanctity_degradation": "#00ACC1",
    }

    # -- Figure 4: Foundation-specific accuracy curves --
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = list(range(n_layers))
    for fv in foundations_present:
        accs = [accuracies[fv].get(l, 0.5) for l in layers]
        ax.plot(layers, accs, "o-", color=colors.get(fv, "#666"),
                linewidth=2, markersize=5, label=FOUNDATION_SHORT[fv])

    ax.axhline(y=0.5, color="#9E9E9E", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(y=0.6, color="#9E9E9E", linestyle="--", linewidth=1, alpha=0.5,
               label="Onset threshold (0.6)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Probing Accuracy", fontsize=12)
    ax.set_title("Foundation-Specific Probe Accuracy Across Layers\n(OLMo-2 1B)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.35, 1.05)
    ax.set_xticks(layers)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig4_foundation_accuracy.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "fig4_foundation_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 4: {figures_dir / 'fig4_foundation_accuracy.png'}")

    if not exp2_results:
        return

    cosine_matrices = exp2_results["cosine_matrices"]
    mean_cosine = exp2_results["mean_cosine"]
    effective_dims = exp2_results["effective_dims"]
    permutation_tests = exp2_results.get("permutation_tests", {})
    peak_sep_layer = exp2_results.get("peak_separation_layer")

    # Use a bootstrap-stable layer for display figures (layer 7), not peak
    # separation (layer 0) which is below the 0.8 stability threshold.
    fig_display_layer = 7 if 7 in cosine_matrices else peak_sep_layer

    # -- Figure 1: Cosine similarity heatmap at display layer --
    if fig_display_layer is not None and fig_display_layer in cosine_matrices:
        cos_sim = cosine_matrices[fig_display_layer]
        n = len(foundations_present)
        short_labels = [FOUNDATION_SHORT[f] for f in foundations_present]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(short_labels, fontsize=10)

        for i in range(n):
            for j in range(n):
                val = cos_sim[i, j]
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=text_color)

        # Draw block structure lines between individualizing and binding
        if len(foundations_present) == 6:
            ax.axhline(y=2.5, color="black", linewidth=2, linestyle="-")
            ax.axvline(x=2.5, color="black", linewidth=2, linestyle="-")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine Similarity")
        ax.set_title(
            f"Foundation Probe Direction Cosine Similarity (Layer {fig_display_layer})\n"
            f"Mean pairwise = {mean_cosine[fig_display_layer]:.4f}",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(figures_dir / "fig1_cosine_heatmap.png", dpi=200, bbox_inches="tight")
        fig.savefig(figures_dir / "fig1_cosine_heatmap.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure 1: {figures_dir / 'fig1_cosine_heatmap.png'}")

    # -- Figure 2: Layer-wise geometric development (3-panel) --
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel a: mean pairwise cosine similarity
    ax = axes[0]
    sorted_layers = sorted(mean_cosine.keys())
    ax.plot(sorted_layers, [mean_cosine[l] for l in sorted_layers],
            "o-", color="#1E88E5", linewidth=2, markersize=5)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean Pairwise Cosine Similarity", fontsize=11)
    ax.set_title("(a) Collapse Metric", fontsize=12, fontweight="bold")
    ax.set_xticks(sorted_layers)
    ax.grid(True, alpha=0.3)

    # Panel b: effective dimensionality
    ax = axes[1]
    dim_layers = sorted(effective_dims.keys())
    ax.plot(dim_layers, [effective_dims[l] for l in dim_layers],
            "s-", color="#E53935", linewidth=2, markersize=5)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Effective Dimensionality (90% var)", fontsize=11)
    ax.set_title("(b) Direction Set Dimensionality", fontsize=12, fontweight="bold")
    ax.set_xticks(dim_layers)
    ax.set_ylim(0.5, 6.5)
    ax.grid(True, alpha=0.3)

    # Panel c: individualizing vs binding distance
    ax = axes[2]
    if permutation_tests:
        pt_layers = sorted(permutation_tests.keys())
        within_ind = [permutation_tests[l].get("mean_within_individualizing", 0) or 0
                      for l in pt_layers]
        within_bind = [permutation_tests[l].get("mean_within_binding", 0) or 0
                       for l in pt_layers]
        between = [permutation_tests[l].get("mean_between_groups", 0) or 0
                   for l in pt_layers]
        ax.plot(pt_layers, within_ind, "o-", color="#43A047", linewidth=2,
                markersize=5, label="Within individualizing")
        ax.plot(pt_layers, within_bind, "s-", color="#FB8C00", linewidth=2,
                markersize=5, label="Within binding")
        ax.plot(pt_layers, between, "D-", color="#8E24AA", linewidth=2,
                markersize=5, label="Between groups")
        ax.legend(fontsize=8)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=11)
    ax.set_title("(c) MFT Group Structure", fontsize=12, fontweight="bold")
    ax.set_xticks(sorted_layers)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Layer-Wise Geometric Development of Moral Framework Representations",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_layerwise_geometry.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "fig2_layerwise_geometry.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 2: {figures_dir / 'fig2_layerwise_geometry.png'}")

    # -- Figure 3: Dendrogram at display layer --
    if fig_display_layer is not None and fig_display_layer in cosine_matrices:
        cos_sim = cosine_matrices[fig_display_layer]
        n = len(foundations_present)
        short_labels = [FOUNDATION_SHORT[f] for f in foundations_present]

        dist = 1 - cos_sim
        condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                condensed.append(dist[i, j])
        condensed = np.array(condensed)

        Z = linkage(condensed, method="ward")

        fig, ax = plt.subplots(figsize=(8, 5))

        dn = dendrogram(
            Z, labels=short_labels, ax=ax,
            leaf_font_size=11, color_threshold=0, above_threshold_color="#666",
        )

        # Color leaf labels: green = individualizing, orange = binding
        leaf_order = dn["ivl"]
        short_to_fv = {FOUNDATION_SHORT[fv]: fv for fv in foundations_present}
        for lbl in ax.get_xticklabels():
            fv = short_to_fv.get(lbl.get_text())
            lbl.set_color("#43A047" if fv in INDIVIDUALIZING else "#FB8C00")
            lbl.set_fontweight("bold")
        ax.set_ylabel("Ward Distance (1 - cosine similarity)", fontsize=11)
        ax.set_title(
            f"Hierarchical Clustering of Foundation Probe Directions (Layer {fig_display_layer})\n"
            f"Green = individualizing, Orange = binding",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(figures_dir / "fig3_dendrogram.png", dpi=200, bbox_inches="tight")
        fig.savefig(figures_dir / "fig3_dendrogram.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure 3: {figures_dir / 'fig3_dendrogram.png'}")

    # -- Bootstrap stability figure (if Exp 3 ran) --
    if exp3_results and exp3_results.get("stability"):
        stability = exp3_results["stability"]
        fig, ax = plt.subplots(figsize=(10, 6))

        for fv in FOUNDATION_ORDER:
            if fv not in stability:
                continue
            layer_data = stability[fv]
            layers_sorted = sorted(layer_data.keys())
            means = [layer_data[l]["mean_cosine_with_full"] for l in layers_sorted]
            stds = [layer_data[l]["std_cosine_with_full"] for l in layers_sorted]
            ax.errorbar(layers_sorted, means, yerr=stds,
                        fmt="o-", color=colors.get(fv, "#666"),
                        linewidth=1.5, markersize=4, capsize=3,
                        label=FOUNDATION_SHORT[fv])

        ax.axhline(y=0.8, color="#E53935", linestyle="--", linewidth=1.5,
                   alpha=0.7, label="Stability threshold (0.8)")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Cosine Similarity with Full-Data Direction", fontsize=12)
        ax.set_title("Bootstrap Direction Stability\n"
                     f"({exp3_results.get('n_bootstrap', 200)} resamples per probe)",
                     fontsize=13, fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(list(range(n_layers)))
        ax.legend(fontsize=8, loc="lower right", ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "fig_bootstrap_stability.png", dpi=200, bbox_inches="tight")
        fig.savefig(figures_dir / "fig_bootstrap_stability.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Bootstrap figure: {figures_dir / 'fig_bootstrap_stability.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_exp1_results(output_dir: Path) -> dict:
    """Reconstruct an Experiment 1 result dict from saved npz + json.

    Used by --bootstrap-only to skip the probe-training and geometry stages and
    jump straight to run_experiment_3() with previously extracted directions.
    """
    json_path = output_dir / "exp1_foundation_probing.json"
    npz_path = output_dir / "exp1_probe_directions.npz"
    if not json_path.exists() or not npz_path.exists():
        raise FileNotFoundError(
            f"--bootstrap-only needs {json_path.name} and {npz_path.name} in {output_dir}"
        )

    with open(json_path) as f:
        exp1_json = json.load(f)
    npz = np.load(npz_path)

    directions: dict[str, dict[int, np.ndarray]] = {}
    accuracies: dict[str, dict[int, float]] = {}
    for fv, fdata in exp1_json["per_foundation"].items():
        directions[fv] = {}
        accuracies[fv] = {}
        for layer_str, acc in fdata["per_layer_accuracy"].items():
            layer = int(layer_str)
            key = f"{fv}_layer{layer}"
            if key in npz:
                directions[fv][layer] = npz[key]
            accuracies[fv][layer] = acc

    return {
        "directions": directions,
        "accuracies": accuracies,
        "n_layers": exp1_json["n_layers"],
        "hidden_dim": exp1_json["hidden_dim"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiments 1-3: Foundation probing + geometry + bootstrap.",
    )
    parser.add_argument("--output-dir",
                        default="papers/3_moral_geometry/outputs/exp1_2_3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset-target", type=int, default=40)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip Experiment 3 (bootstrap stability).")
    parser.add_argument("--bootstrap-only", action="store_true",
                        help="Load existing Exp 1 directions from --output-dir and run "
                             "only Experiment 3 (skips Exp 1/2 probe training + geometry).")
    parser.add_argument("--n-bootstrap", type=int, default=200,
                        help="Number of bootstrap resamples (default 200).")
    parser.add_argument("--model", default=OLMO_REPO,
                        help="HuggingFace model ID.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.datasets.pipeline import build_probing_dataset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building probing dataset...")
    dataset = build_probing_dataset(target_per_foundation=args.dataset_target, dataset_version="v2")
    print(f"Dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs")

    # Print per-foundation distribution
    from collections import Counter
    train_counts = Counter(p.foundation.value for p in dataset.train)
    test_counts = Counter(p.foundation.value for p in dataset.test)
    for fv in FOUNDATION_ORDER:
        print(f"  {FOUNDATION_SHORT[fv]:12s}: {train_counts.get(fv, 0)} train, "
              f"{test_counts.get(fv, 0)} test")

    # Load model
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model}")
    print(f"{'='*60}")
    t0 = time.time()
    model = WhiteBoxModel(
        args.model,
        device=args.device,
        access_tier=AccessTier.WEIGHTS,
    )
    print(f"Loaded in {time.time() - t0:.1f}s "
          f"({model.info.n_params / 1e9:.1f}B params, "
          f"{model.info.n_layers} layers)")

    # Bootstrap-only: load existing Exp 1 directions and jump straight to Exp 3.
    if args.bootstrap_only:
        print(f"\n{'='*60}")
        print(f"BOOTSTRAP-ONLY: loading Exp 1 directions from {output_dir}")
        print(f"{'='*60}")
        exp1_results = load_exp1_results(output_dir)
        print(f"Loaded directions for {len(exp1_results['directions'])} foundations, "
              f"{exp1_results['n_layers']} layers")

        print(f"\n{'='*60}")
        print(f"EXPERIMENT 3: Bootstrap Direction Stability ({args.n_bootstrap} resamples)")
        print(f"{'='*60}")
        t0 = time.time()
        run_experiment_3(
            model, dataset, exp1_results, output_dir,
            n_bootstrap=args.n_bootstrap,
            n_epochs=args.n_epochs, lr=args.lr,
        )
        print(f"Experiment 3 complete: {time.time() - t0:.1f}s")

        del model
        _clear_memory()
        print(f"\nBootstrap output: {output_dir / 'exp3_bootstrap_stability.json'}")
        return

    # Experiment 1
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: Foundation-Specific Probing")
    print(f"{'='*60}")
    t0 = time.time()
    exp1_results = run_experiment_1(
        model, dataset, output_dir,
        n_epochs=args.n_epochs, lr=args.lr,
    )
    print(f"Experiment 1 complete: {time.time() - t0:.1f}s")

    # Experiment 2
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Framework Geometry Analysis")
    print(f"{'='*60}")
    t0 = time.time()
    exp2_results = run_experiment_2(exp1_results, output_dir)
    print(f"Experiment 2 complete: {time.time() - t0:.1f}s")

    # Print go/no-go assessment
    if exp2_results and "mean_cosine" in exp2_results:
        peak_layer = exp2_results.get("peak_separation_layer")
        if peak_layer is not None:
            mc = exp2_results["mean_cosine"][peak_layer]
            if mc > 0.95:
                print(f"\n  RESULT: Near-collapse (mean cos = {mc:.4f} > 0.95)")
                print("  Paper is a null result on framework structure.")
            elif mc < 0.8:
                print(f"\n  RESULT: Clear separation (mean cos = {mc:.4f} < 0.8)")
                print("  Proceed to remaining experiments.")
            else:
                print(f"\n  RESULT: Intermediate (mean cos = {mc:.4f}, 0.8-0.95)")
                print("  Bootstrap stability (Exp 3) needed to assess if real.")

    # Experiment 3 (bootstrap)
    exp3_results = None
    if not args.skip_bootstrap:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT 3: Bootstrap Direction Stability ({args.n_bootstrap} resamples)")
        print(f"{'='*60}")
        t0 = time.time()
        exp3_results = run_experiment_3(
            model, dataset, exp1_results, output_dir,
            n_bootstrap=args.n_bootstrap,
            n_epochs=args.n_epochs, lr=args.lr,
        )
        print(f"Experiment 3 complete: {time.time() - t0:.1f}s")

    # Generate figures
    print(f"\n{'='*60}")
    print("Generating figures...")
    print(f"{'='*60}")
    generate_figures(exp1_results, exp2_results, exp3_results, output_dir)

    # Cleanup
    del model
    _clear_memory()

    print(f"\nAll outputs: {output_dir}")
    print(f"Figures: {output_dir.parent / 'figures'}")


if __name__ == "__main__":
    main()
