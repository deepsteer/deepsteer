"""Shared constants, types, and geometric analysis utilities for probe engineering.

All probe engineering scripts import from here to avoid duplication.
"""

from __future__ import annotations

import gc
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage

# ---------------------------------------------------------------------------
# Foundation constants
# ---------------------------------------------------------------------------

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

DILEMMA_TO_PROBE = {
    "care": "care_harm",
    "fairness": "fairness_cheating",
    "liberty": "liberty_oppression",
    "loyalty": "loyalty_betrayal",
    "authority": "authority_subversion",
    "sanctity": "sanctity_degradation",
}

DILEMMA_PAIRS = [
    ("care", "fairness"), ("care", "liberty"), ("care", "loyalty"),
    ("care", "authority"), ("care", "sanctity"),
    ("fairness", "liberty"), ("fairness", "loyalty"),
    ("fairness", "authority"), ("fairness", "sanctity"),
    ("liberty", "loyalty"), ("liberty", "authority"), ("liberty", "sanctity"),
    ("loyalty", "authority"), ("loyalty", "sanctity"),
    ("authority", "sanctity"),
]

DILEMMA_PAIR_KEYS = [f"{a}-{b}" for a, b in DILEMMA_PAIRS]

# ---------------------------------------------------------------------------
# Standard paths
# ---------------------------------------------------------------------------

PAPER_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PAPER_ROOT / "scripts" / "probe_engineering"
OUTPUT_DIR = PAPER_ROOT / "outputs" / "probe_engineering"
FIGURES_DIR = PAPER_ROOT / "outputs" / "figures"


def ensure_output_dirs() -> tuple[Path, Path]:
    """Create and return (output_dir, figures_dir)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR, FIGURES_DIR


# ---------------------------------------------------------------------------
# Geometric analysis functions
# ---------------------------------------------------------------------------

def compute_cosine_matrix(
    directions: dict[str, dict[int, np.ndarray]],
    layer: int,
) -> np.ndarray | None:
    """6x6 cosine similarity matrix for foundation directions at a layer."""
    vecs = []
    for fv in FOUNDATION_ORDER:
        if fv not in directions or layer not in directions[fv]:
            return None
        vecs.append(directions[fv][layer])
    mat = np.stack(vecs)
    return mat @ mat.T


def compute_effective_dimensionality(
    directions: dict[str, dict[int, np.ndarray]],
    layer: int,
    threshold: float = 0.9,
) -> int | None:
    """Number of SVD components to explain `threshold` variance of 6 directions."""
    vecs = []
    for fv in FOUNDATION_ORDER:
        if fv not in directions or layer not in directions[fv]:
            return None
        vecs.append(directions[fv][layer])
    mat = np.stack(vecs)
    mat_centered = mat - mat.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(mat_centered, full_matrices=False)
    explained = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(explained, threshold)) + 1


def permutation_test_mft(cos_sim: np.ndarray, n_perm: int = 10000, seed: int = 42) -> dict:
    """Test whether individualizing/binding groups have higher within-group similarity."""
    n = 6
    ind_idx = [0, 1, 2]
    bind_idx = [3, 4, 5]

    def _stat(sim, ga, gb):
        wa = [sim[i, j] for i in ga for j in ga if i < j]
        wb = [sim[i, j] for i in gb for j in gb if i < j]
        bw = [sim[i, j] for i in ga for j in gb]
        return np.mean(wa + wb) - np.mean(bw) if (wa + wb) and bw else 0.0

    observed = _stat(cos_sim, ind_idx, bind_idx)
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        if _stat(cos_sim, p[:3].tolist(), p[3:].tolist()) >= observed:
            count += 1
    p_value = (count + 1) / (n_perm + 1)

    within_ind = [cos_sim[i, j] for i in ind_idx for j in ind_idx if i < j]
    within_bind = [cos_sim[i, j] for i in bind_idx for j in bind_idx if i < j]
    between = [cos_sim[i, j] for i in ind_idx for j in bind_idx]

    return {
        "observed_statistic": float(observed),
        "p_value": float(p_value),
        "mean_within_individualizing": float(np.mean(within_ind)),
        "mean_within_binding": float(np.mean(within_bind)),
        "mean_between_groups": float(np.mean(between)),
    }


def check_dendrogram_mft(cos_sim: np.ndarray) -> dict:
    """Check whether Ward clustering separates individualizing vs binding."""
    n = 6
    dist = 1 - cos_sim
    condensed = [dist[i, j] for i in range(n) for j in range(i + 1, n)]
    Z = linkage(condensed, method="ward")

    def _get_leaves(idx):
        if idx < n:
            return {idx}
        row = Z[idx - n]
        return _get_leaves(int(row[0])) | _get_leaves(int(row[1]))

    last = Z[-1]
    left = _get_leaves(int(last[0]))
    right = _get_leaves(int(last[1]))
    mft_match = left == {0, 1, 2} or right == {0, 1, 2}
    left_labels = [FOUNDATION_SHORT[FOUNDATION_ORDER[i]] for i in sorted(left)]
    right_labels = [FOUNDATION_SHORT[FOUNDATION_ORDER[i]] for i in sorted(right)]
    return {
        "mft_match": mft_match,
        "left": left_labels,
        "right": right_labels,
    }


def full_geometric_analysis(
    directions: dict[str, dict[int, np.ndarray]],
    layer: int,
) -> dict | None:
    """Run cosine matrix, effective dim, permutation test, and dendrogram check."""
    cos_sim = compute_cosine_matrix(directions, layer)
    if cos_sim is None:
        return None
    n = 6
    upper_tri = [cos_sim[i, j] for i in range(n) for j in range(i + 1, n)]
    return {
        "mean_cosine_similarity": round(float(np.mean(upper_tri)), 6),
        "effective_dimensionality": compute_effective_dimensionality(directions, layer),
        "permutation_test": permutation_test_mft(cos_sim),
        "dendrogram": check_dendrogram_mft(cos_sim),
        "cosine_matrix": cos_sim.tolist(),
    }


# ---------------------------------------------------------------------------
# Subspace utilities (from full_subspace_projection)
# ---------------------------------------------------------------------------

def orthonormal_basis(vectors: np.ndarray) -> np.ndarray:
    """Compute orthonormal basis for the span of rows of `vectors` via SVD."""
    _, s, Vt = np.linalg.svd(vectors, full_matrices=False)
    rank = np.sum(s > 1e-10)
    return Vt[:rank]


def subspace_membership(direction: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of direction's variance explained by the subspace."""
    proj = basis @ direction
    return float(np.dot(proj, proj))


# ---------------------------------------------------------------------------
# Model + dataset loading helpers
# ---------------------------------------------------------------------------

def load_model_and_collect_activations(
    model_name: str = "allenai/OLMo-2-0425-1B",
    device: str | None = None,
    target_per_foundation: int = 40,
    collect_test: bool = False,
):
    """Load model, build dataset, collect activations, free model memory.

    Returns:
        (all_train, all_test_or_None, dataset, n_layers, foundation_indices)
        where foundation_indices maps foundation value string to list of pair indices.
    """
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.datasets.pipeline import build_probing_dataset
    from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe

    dataset = build_probing_dataset(target_per_foundation=target_per_foundation, dataset_version="v2")
    print(f"Dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs")

    foundation_indices: dict[str, list[int]] = defaultdict(list)
    for i, pair in enumerate(dataset.train):
        foundation_indices[pair.foundation.value].append(i)

    print(f"\nLoading model: {model_name}")
    t0 = time.time()
    model = WhiteBoxModel(model_name, device=device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    print(f"Loaded in {time.time() - t0:.1f}s ({n_layers} layers)")

    print("\nCollecting activations for training set...")
    t0 = time.time()
    all_train = LayerWiseMoralProbe._collect_all_activations(model, dataset.train)
    print(f"Collected train in {time.time() - t0:.1f}s")

    all_test = None
    if collect_test:
        print("Collecting activations for test set...")
        t0 = time.time()
        all_test = LayerWiseMoralProbe._collect_all_activations(model, dataset.test)
        print(f"Collected test in {time.time() - t0:.1f}s")

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return all_train, all_test, dataset, n_layers, foundation_indices


def load_probe_directions(
    path: str | Path,
) -> dict[str, dict[int, np.ndarray]]:
    """Load and normalize probe-weight directions from an .npz file."""
    probe_npz = np.load(path)
    pw_directions: dict[str, dict[int, np.ndarray]] = {}
    for fv in FOUNDATION_ORDER:
        pw_directions[fv] = {}
        layer = 0
        while True:
            key = f"{fv}_layer{layer}"
            if key not in probe_npz:
                break
            d = probe_npz[key].astype(np.float64)
            norm = np.linalg.norm(d)
            if norm > 1e-12:
                d /= norm
            pw_directions[fv][layer] = d
            layer += 1
    return pw_directions


# ---------------------------------------------------------------------------
# Direction evaluation helpers
# ---------------------------------------------------------------------------

def pair_accuracy(
    direction: np.ndarray,
    activations: torch.Tensor,
    pair_indices: list[int],
) -> float:
    """Fraction of pairs where direction · moral > direction · neutral.

    activations: (2*N, hidden_dim) interleaved [moral_0, neutral_0, moral_1, ...]
    pair_indices: which pair slots to evaluate (into the interleaved array).
    """
    correct = 0
    for pi in pair_indices:
        moral_act = activations[pi * 2].numpy()
        neutral_act = activations[pi * 2 + 1].numpy()
        if np.dot(direction, moral_act) > np.dot(direction, neutral_act):
            correct += 1
    return correct / len(pair_indices) if pair_indices else 0.0


def compute_mean_diff_directions(
    all_activations: dict[int, tuple[torch.Tensor, torch.Tensor]],
    n_layers: int,
    foundation_indices: dict[str, list[int]],
) -> dict[str, dict[int, np.ndarray]]:
    """Compute mean-difference directions for each foundation at each layer."""
    directions: dict[str, dict[int, np.ndarray]] = {}
    for fv in FOUNDATION_ORDER:
        if fv not in foundation_indices:
            continue
        pair_indices = foundation_indices[fv]
        directions[fv] = {}
        for layer in range(n_layers):
            X, y = all_activations[layer]
            moral_rows = [pi * 2 for pi in pair_indices]
            neutral_rows = [pi * 2 + 1 for pi in pair_indices]
            mean_diff = X[moral_rows].numpy().mean(axis=0) - X[neutral_rows].numpy().mean(axis=0)
            norm = np.linalg.norm(mean_diff)
            if norm > 1e-12:
                mean_diff /= norm
            directions[fv][layer] = mean_diff
    return directions
