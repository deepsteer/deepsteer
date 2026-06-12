"""Stronger nulls for the dilemma compositionality analysis (review task 3.4).

Operates entirely on cached probe directions (no model load):
  - dilemma directions:  outputs/dilemma_probing/dilemma_probe_directions.npz
  - foundation directions: outputs/exp1_2_3/exp1_probe_directions.npz

Two analyses:

1. Mismatched-pair baseline for subspace membership. The random-vector null
   (~0.001) does not absorb the shared moral-salience component that all moral
   directions carry, so it overstates the effect. The correct null is the
   *mismatched* 2D subspace: for each dilemma direction, measure membership in
   the span of foundation pairs that share no component with it. We report
   matched vs. mismatched membership and the matched-minus-mismatched gap.

2. Permutation test for the shared-component result. The pairwise cosines
   between the 15 dilemma directions are fixed; what the test permutes is which
   foundation-pair label is attached to each dilemma direction, which changes
   the share/no-share partition of the C(15,2)=105 cosines. p is the fraction of
   permutations whose share-minus-noshare gap is >= the observed gap.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

FOUNDATIONS = ["care", "fairness", "liberty", "loyalty", "authority", "sanctity"]
FULL = {
    "care": "care_harm",
    "fairness": "fairness_cheating",
    "liberty": "liberty_oppression",
    "loyalty": "loyalty_betrayal",
    "authority": "authority_subversion",
    "sanctity": "sanctity_degradation",
}
N_LAYERS = 16
N_PERM = 10000
SEED = 42

OUT_DIR = Path("papers/3_moral_geometry/outputs/dilemma_probing")
DILEMMA_NPZ = OUT_DIR / "dilemma_probe_directions.npz"
FOUNDATION_NPZ = Path("papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def subspace_membership(w: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of w's variance explained by span{a, b} (a, b unit vectors)."""
    e1 = a.copy()
    e2 = b - np.dot(b, e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-10:
        return float(np.dot(w, e1) ** 2)
    e2 = e2 / n2
    return float(np.dot(w, e1) ** 2 + np.dot(w, e2) ** 2)


def load_directions():
    dil = np.load(DILEMMA_NPZ)
    fnd = np.load(FOUNDATION_NPZ)
    dilemma_pairs = sorted(
        {k.replace("dilemma_", "").rsplit("_layer", 1)[0] for k in dil.keys()}
    )
    return dil, fnd, dilemma_pairs


def mismatched_pair_baseline(dil, fnd, dilemma_pairs) -> dict:
    """Matched vs. mismatched 2D subspace membership, per layer and overall."""
    all_found_pairs = list(combinations(FOUNDATIONS, 2))
    per_layer = {}
    for L in range(N_LAYERS):
        matched, mismatched = [], []
        for dp in dilemma_pairs:
            f1, f2 = dp.split("-")
            w = _unit(dil[f"dilemma_{dp}_layer{L}"])
            comp = {f1, f2}
            # matched: the dilemma's own two component foundations
            a = _unit(fnd[f"{FULL[f1]}_layer{L}"])
            b = _unit(fnd[f"{FULL[f2]}_layer{L}"])
            matched.append(subspace_membership(w, a, b))
            # mismatched: every foundation pair sharing NO component
            for g1, g2 in all_found_pairs:
                if comp & {g1, g2}:
                    continue
                ga = _unit(fnd[f"{FULL[g1]}_layer{L}"])
                gb = _unit(fnd[f"{FULL[g2]}_layer{L}"])
                mismatched.append(subspace_membership(w, ga, gb))
        per_layer[L] = {
            "matched_mean": float(np.mean(matched)),
            "mismatched_mean": float(np.mean(mismatched)),
            "gap": float(np.mean(matched) - np.mean(mismatched)),
            "n_matched": len(matched),
            "n_mismatched": len(mismatched),
        }
    matched_all = float(np.mean([per_layer[L]["matched_mean"] for L in range(N_LAYERS)]))
    mismatched_all = float(np.mean([per_layer[L]["mismatched_mean"] for L in range(N_LAYERS)]))
    peak_L = max(range(N_LAYERS), key=lambda L: per_layer[L]["matched_mean"])
    return {
        "per_layer": {str(L): per_layer[L] for L in range(N_LAYERS)},
        "matched_mean_over_layers": matched_all,
        "mismatched_mean_over_layers": mismatched_all,
        "gap_over_layers": matched_all - mismatched_all,
        "peak_layer": peak_L,
        "matched_at_peak": per_layer[peak_L]["matched_mean"],
        "mismatched_at_peak": per_layer[peak_L]["mismatched_mean"],
        "gap_at_peak": per_layer[peak_L]["gap"],
    }


def _shared_gap(cos: np.ndarray, labels: list[tuple[str, str]]) -> float:
    """share-minus-noshare mean cosine for a labeling of the dilemma directions."""
    share, noshare = [], []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            shared = bool(set(labels[i]) & set(labels[j]))
            (share if shared else noshare).append(cos[i, j])
    if not share or not noshare:
        return 0.0
    return float(np.mean(share) - np.mean(noshare))


def shared_component_permutation(dil, dilemma_pairs) -> dict:
    """Permute which foundation-pair label is attached to each dilemma direction."""
    rng = np.random.RandomState(SEED)
    labels = [tuple(dp.split("-")) for dp in dilemma_pairs]
    per_layer = {}
    for L in range(N_LAYERS):
        mat = np.stack([_unit(dil[f"dilemma_{dp}_layer{L}"]) for dp in dilemma_pairs])
        cos = mat @ mat.T
        obs = _shared_gap(cos, labels)
        ge = 0
        for _ in range(N_PERM):
            perm = list(rng.permutation(len(labels)))
            permuted = [labels[k] for k in perm]
            if _shared_gap(cos, permuted) >= obs - 1e-12:
                ge += 1
        per_layer[L] = {"observed_gap": obs, "p_value": (ge + 1) / (N_PERM + 1)}
    peak_L = max(range(N_LAYERS), key=lambda L: per_layer[L]["observed_gap"])
    return {
        "per_layer": {str(L): per_layer[L] for L in range(N_LAYERS)},
        "n_permutations": N_PERM,
        "peak_layer": peak_L,
        "peak_observed_gap": per_layer[peak_L]["observed_gap"],
        "peak_p_value": per_layer[peak_L]["p_value"],
        "min_p_value": min(per_layer[L]["p_value"] for L in range(N_LAYERS)),
    }


def main() -> None:
    dil, fnd, dilemma_pairs = load_directions()
    baseline = mismatched_pair_baseline(dil, fnd, dilemma_pairs)
    perm = shared_component_permutation(dil, dilemma_pairs)

    out = {
        "analysis": "dilemma_compositionality_baselines",
        "n_dilemmas": len(dilemma_pairs),
        "mismatched_pair_baseline": baseline,
        "shared_component_permutation": perm,
    }
    path = OUT_DIR / "dilemma_compositionality_baselines.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("=== Mismatched-pair baseline (subspace membership) ===")
    print(f"  matched (over layers):    {baseline['matched_mean_over_layers']:.4f}")
    print(f"  mismatched (over layers): {baseline['mismatched_mean_over_layers']:.4f}")
    print(f"  gap (matched - mismatched): {baseline['gap_over_layers']:.4f}")
    print(f"  peak layer {baseline['peak_layer']}: matched {baseline['matched_at_peak']:.4f}, "
          f"mismatched {baseline['mismatched_at_peak']:.4f}, gap {baseline['gap_at_peak']:.4f}")
    print("\n=== Shared-component permutation test ===")
    print(f"  peak layer {perm['peak_layer']}: gap {perm['peak_observed_gap']:.4f}, "
          f"p = {perm['peak_p_value']:.4f}")
    print(f"  min p across layers: {perm['min_p_value']:.4f}")
    print(f"\nOutput: {path}")


if __name__ == "__main__":
    main()
