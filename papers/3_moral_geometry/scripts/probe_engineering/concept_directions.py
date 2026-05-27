#!/usr/bin/env python3
"""B.5: Representation-engineering concept directions.

Extract foundation directions using the paired-difference PCA method from
representation engineering (Zou et al., 2023): for each foundation, compute
d_i = act(moral_i) - act(neutral_i) for each pair, then take the first
principal component of {d_i} as the concept direction.

This is a third independent direction-finding method (after probe-weight
and mean-diff), providing a triangulation of the geometric findings.

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/concept_directions.py
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage

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

DILEMMA_TO_PROBE = {
    "care": "care_harm",
    "fairness": "fairness_cheating",
    "liberty": "liberty_oppression",
    "loyalty": "loyalty_betrayal",
    "authority": "authority_subversion",
    "sanctity": "sanctity_degradation",
}


def compute_repe_directions(
    all_activations: dict[int, tuple[torch.Tensor, torch.Tensor]],
    n_layers: int,
    foundation_indices: dict[str, list[int]],
) -> dict[str, dict[int, np.ndarray]]:
    """Compute concept directions via paired-difference PCA.

    For each foundation f at each layer:
      1. d_i = act(moral_i) - act(neutral_i) for each pair i
      2. Center: d_i -= mean(d_i)
      3. First PC of {d_i} = concept direction
      4. Sign: ensure mean projection of moral > neutral
    """
    directions: dict[str, dict[int, np.ndarray]] = {}

    for fv in FOUNDATION_ORDER:
        if fv not in foundation_indices:
            continue
        pair_indices = foundation_indices[fv]
        directions[fv] = {}

        for layer in range(n_layers):
            X, _ = all_activations[layer]
            diffs = []
            for pi in pair_indices:
                moral_act = X[pi * 2].numpy()
                neutral_act = X[pi * 2 + 1].numpy()
                diffs.append(moral_act - neutral_act)

            diff_mat = np.stack(diffs)  # (n_pairs, hidden_dim)
            diff_centered = diff_mat - diff_mat.mean(axis=0, keepdims=True)

            _, s, Vt = np.linalg.svd(diff_centered, full_matrices=False)
            pc1 = Vt[0]  # first principal component

            # Sign convention: positive = moral
            moral_projs = []
            neutral_projs = []
            for pi in pair_indices:
                moral_projs.append(np.dot(pc1, X[pi * 2].numpy()))
                neutral_projs.append(np.dot(pc1, X[pi * 2 + 1].numpy()))
            if np.mean(moral_projs) < np.mean(neutral_projs):
                pc1 = -pc1

            directions[fv][layer] = pc1

            # Also store variance explained by PC1
            if not hasattr(compute_repe_directions, '_var_explained'):
                compute_repe_directions._var_explained = {}
            key = f"{fv}_layer{layer}"
            total_var = np.sum(s ** 2)
            compute_repe_directions._var_explained[key] = float(s[0] ** 2 / total_var) if total_var > 0 else 0

    return directions


def compute_cosine_matrix(
    directions: dict[str, dict[int, np.ndarray]],
    layer: int,
) -> np.ndarray | None:
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
        p = rng.permutation(6)
        if _stat(cos_sim, p[:3].tolist(), p[3:].tolist()) >= observed:
            count += 1

    return {
        "observed_statistic": float(observed),
        "p_value": float((count + 1) / (n_perm + 1)),
    }


def check_dendrogram_mft(cos_sim: np.ndarray) -> dict:
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
    return {"mft_match": mft_match, "left": left_labels, "right": right_labels}


def pair_accuracy(direction: np.ndarray, activations: torch.Tensor, pair_indices: list[int]) -> float:
    correct = 0
    for pi in pair_indices:
        if np.dot(direction, activations[pi * 2].numpy()) > np.dot(direction, activations[pi * 2 + 1].numpy()):
            correct += 1
    return correct / len(pair_indices) if pair_indices else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="B.5: Concept-direction comparison.")
    parser.add_argument("--probe-directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--dilemma-dataset",
                        default="deepsteer/datasets/dilemma_pairs_validated.json")
    parser.add_argument("--output-dir",
                        default="papers/3_moral_geometry/outputs/probe_engineering")
    parser.add_argument("--figures-dir",
                        default="papers/3_moral_geometry/outputs/figures")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier, MoralFoundation
    from deepsteer.datasets.pipeline import build_probing_dataset
    from deepsteer.datasets.types import ProbingPair, NeutralDomain, GenerationMethod
    from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe

    print(f"{'='*60}")
    print("B.5: Concept-Direction Comparison (RepE-style PCA)")
    print(f"{'='*60}")

    # ── Load datasets ──
    dataset = build_probing_dataset(target_per_foundation=40)
    print(f"Declarative dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs")

    train_foundation_idx: dict[str, list[int]] = defaultdict(list)
    for i, pair in enumerate(dataset.train):
        train_foundation_idx[pair.foundation.value].append(i)

    test_foundation_idx: dict[str, list[int]] = defaultdict(list)
    for i, pair in enumerate(dataset.test):
        test_foundation_idx[pair.foundation.value].append(i)

    with open(args.dilemma_dataset) as f:
        dilemma_data = json.load(f)
    dilemma_pairs_raw = dilemma_data["pairs"]

    dilemma_probing: list[ProbingPair] = []
    dilemma_foundation_idx: dict[str, list[int]] = defaultdict(list)
    for dp in dilemma_pairs_raw:
        idx = len(dilemma_probing)
        dilemma_probing.append(ProbingPair(
            moral=dp["moral"], neutral=dp["neutral"],
            foundation=MoralFoundation(DILEMMA_TO_PROBE[dp["foundation_pair"][0]]),
            neutral_domain=NeutralDomain.MATCHED,
            generation_method=GenerationMethod.HANDWRITTEN,
            moral_word_count=len(dp["moral"].split()),
            neutral_word_count=len(dp["neutral"].split()),
        ))
        for f_short in dp["foundation_pair"]:
            dilemma_foundation_idx[DILEMMA_TO_PROBE[f_short]].append(idx)

    # ── Load model and collect activations ──
    print(f"\nLoading model: {args.model}")
    t0 = time.time()
    model = WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    print(f"Loaded in {time.time() - t0:.1f}s ({n_layers} layers)")

    print("\nCollecting declarative train activations...")
    t0 = time.time()
    decl_train_acts = LayerWiseMoralProbe._collect_all_activations(model, dataset.train)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("Collecting declarative test activations...")
    t0 = time.time()
    decl_test_acts = LayerWiseMoralProbe._collect_all_activations(model, dataset.test)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("Collecting dilemma activations...")
    t0 = time.time()
    dilemma_acts = LayerWiseMoralProbe._collect_all_activations(model, dilemma_probing)
    print(f"  Done in {time.time() - t0:.1f}s")

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ── Compute RepE directions ──
    print("\nComputing RepE (paired-difference PCA) directions...")
    repe_directions = compute_repe_directions(decl_train_acts, n_layers, train_foundation_idx)

    # Compute mean-diff directions for comparison
    md_directions: dict[str, dict[int, np.ndarray]] = {}
    for fv in FOUNDATION_ORDER:
        md_directions[fv] = {}
        indices = train_foundation_idx[fv]
        for layer in range(n_layers):
            X, _ = decl_train_acts[layer]
            moral_rows = [idx * 2 for idx in indices]
            neutral_rows = [idx * 2 + 1 for idx in indices]
            diff = X[moral_rows].numpy().mean(axis=0) - X[neutral_rows].numpy().mean(axis=0)
            norm = np.linalg.norm(diff)
            if norm > 1e-12:
                diff /= norm
            md_directions[fv][layer] = diff

    # Load probe-weight directions
    probe_npz = np.load(args.probe_directions)
    pw_directions: dict[str, dict[int, np.ndarray]] = {}
    for fv in FOUNDATION_ORDER:
        pw_directions[fv] = {}
        for layer in range(n_layers):
            key = f"{fv}_layer{layer}"
            if key in probe_npz:
                d = probe_npz[key].astype(np.float32)
                pw_directions[fv][layer] = d / (np.linalg.norm(d) + 1e-12)

    # ── Direction alignment: RepE vs probe-weight and mean-diff ──
    print("\n--- Direction Alignment ---")
    print(f"{'Foundation':<14s}  {'RepE↔PW':>8s}  {'RepE↔MD':>8s}  {'MD↔PW':>8s}  {'PC1 var%':>8s}")
    print("-" * 55)

    alignment: dict[str, dict] = {}
    for fv in FOUNDATION_ORDER:
        fname = FOUNDATION_SHORT[fv]
        repe_pw, repe_md, md_pw, var_exp = [], [], [], []
        for layer in range(n_layers):
            r = repe_directions[fv][layer]
            m = md_directions[fv][layer]
            p = pw_directions[fv].get(layer)
            if p is None:
                continue
            repe_pw.append(abs(float(np.dot(r, p))))
            repe_md.append(abs(float(np.dot(r, m))))
            md_pw.append(abs(float(np.dot(m, p))))
            vk = f"{fv}_layer{layer}"
            var_exp.append(compute_repe_directions._var_explained.get(vk, 0))

        alignment[fv] = {
            "repe_vs_pw": round(float(np.mean(repe_pw)), 4),
            "repe_vs_md": round(float(np.mean(repe_md)), 4),
            "md_vs_pw": round(float(np.mean(md_pw)), 4),
            "pc1_variance_explained": round(float(np.mean(var_exp)), 4),
        }
        print(f"  {fname:<12s}  {np.mean(repe_pw):>8.4f}  {np.mean(repe_md):>8.4f}  "
              f"{np.mean(md_pw):>8.4f}  {np.mean(var_exp)*100:>7.1f}%")

    # ── Geometric analysis with RepE directions ──
    print("\n--- Geometric Analysis (RepE Directions) ---")
    geo_results: dict[str, dict] = {}
    repe_mean_cosines: dict[int, float] = {}
    repe_eff_dims: dict[int, int] = {}

    for layer in range(n_layers):
        cos_sim = compute_cosine_matrix(repe_directions, layer)
        if cos_sim is None:
            continue
        upper_tri = [cos_sim[i, j] for i in range(6) for j in range(i + 1, 6)]
        mc = float(np.mean(upper_tri))
        repe_mean_cosines[layer] = mc
        repe_eff_dims[layer] = compute_effective_dimensionality(repe_directions, layer)

        perm = permutation_test_mft(cos_sim)
        dendro = check_dendrogram_mft(cos_sim)

        geo_results[str(layer)] = {
            "mean_cosine_similarity": round(mc, 6),
            "effective_dimensionality": repe_eff_dims[layer],
            "permutation_test_p": round(perm["p_value"], 6),
            "mft_dendrogram_match": dendro["mft_match"],
            "dendrogram_left": dendro["left"],
            "dendrogram_right": dendro["right"],
        }

    # Also get probe-weight and mean-diff geometry for comparison
    pw_mean_cosines: dict[int, float] = {}
    md_mean_cosines: dict[int, float] = {}
    for label, dirs, store in [("pw", pw_directions, pw_mean_cosines), ("md", md_directions, md_mean_cosines)]:
        for layer in range(n_layers):
            cos_sim = compute_cosine_matrix(dirs, layer)
            if cos_sim is not None:
                upper_tri = [cos_sim[i, j] for i in range(6) for j in range(i + 1, 6)]
                store[layer] = float(np.mean(upper_tri))

    peak_repe = min(repe_mean_cosines, key=repe_mean_cosines.get)
    print(f"\nPeak separation: layer {peak_repe} (cos = {repe_mean_cosines[peak_repe]:.4f})")
    print(f"Eff dim: {[repe_eff_dims[l] for l in range(n_layers)]}")

    mft_layers = [l for l in range(n_layers) if geo_results[str(l)]["mft_dendrogram_match"]]
    sig_layers = [l for l in range(n_layers) if geo_results[str(l)]["permutation_test_p"] < 0.05]
    print(f"MFT clustering at layers: {mft_layers}")
    print(f"Significant (p<0.05) at layers: {sig_layers}")

    # ── Register transfer with RepE directions ──
    print("\n--- Register Transfer (RepE Directions) ---")
    print(f"{'Foundation':<14s}  {'Same-reg':>8s}  {'Cross-reg':>9s}  {'Gap':>6s}")
    print("-" * 42)

    transfer: dict[str, dict] = {}
    for fv in FOUNDATION_ORDER:
        fname = FOUNDATION_SHORT[fv]
        test_idx = test_foundation_idx.get(fv, [])
        dilemma_idx = dilemma_foundation_idx.get(fv, [])
        sr_vals, cr_vals = [], []
        transfer[fv] = {}
        for layer in range(n_layers):
            d = repe_directions[fv][layer]
            X_test, _ = decl_test_acts[layer]
            X_dilemma, _ = dilemma_acts[layer]
            sr = pair_accuracy(d, X_test, test_idx)
            cr = pair_accuracy(d, X_dilemma, dilemma_idx)
            sr_vals.append(sr)
            cr_vals.append(cr)
            transfer[fv][str(layer)] = {
                "same_register": round(sr, 4),
                "cross_register": round(cr, 4),
            }
        mean_sr = np.mean(sr_vals)
        mean_cr = np.mean(cr_vals)
        transfer[fv]["mean_same_register"] = round(float(mean_sr), 4)
        transfer[fv]["mean_cross_register"] = round(float(mean_cr), 4)
        print(f"  {fname:<12s}  {mean_sr:>8.3f}  {mean_cr:>9.3f}  {mean_sr - mean_cr:>+6.3f}")

    # ── Save results ──
    results = {
        "analysis": "concept_directions_repe",
        "n_layers": n_layers,
        "alignment": alignment,
        "repe_geometry": geo_results,
        "register_transfer": transfer,
        "peak_layer": peak_repe,
        "peak_cosine": round(repe_mean_cosines[peak_repe], 6),
    }
    out_path = output_dir / "concept_directions.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # ── Generate figure ──
    generate_figure(repe_mean_cosines, md_mean_cosines, pw_mean_cosines,
                    repe_eff_dims, alignment, n_layers, figures_dir)


def generate_figure(
    repe_cosines: dict, md_cosines: dict, pw_cosines: dict,
    repe_dims: dict, alignment: dict, n_layers: int, figures_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(range(n_layers))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel a: mean cosine — all 3 methods
    ax = axes[0]
    ax.plot(layers, [pw_cosines.get(l, 0) for l in layers],
            "o-", color="#1E88E5", linewidth=2, markersize=5, label="Probe weight")
    ax.plot(layers, [md_cosines.get(l, 0) for l in layers],
            "s-", color="#E53935", linewidth=2, markersize=5, label="Mean diff")
    ax.plot(layers, [repe_cosines.get(l, 0) for l in layers],
            "^-", color="#43A047", linewidth=2, markersize=5, label="RepE (PCA)")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean Pairwise Cosine Similarity", fontsize=11)
    ax.set_title("(a) 3-Method Geometry Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(layers)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel b: effective dimensionality
    ax = axes[1]
    ax.plot(layers, [repe_dims.get(l, 5) for l in layers],
            "^-", color="#43A047", linewidth=2, markersize=5, label="RepE (PCA)")
    ax.axhline(5, color="#1E88E5", linestyle="--", linewidth=1.5, alpha=0.7, label="Probe weight (=5)")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Effective Dimensionality", fontsize=11)
    ax.set_title("(b) RepE Dimensionality", fontsize=12, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_ylim(0.5, 6.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel c: pairwise alignment between methods (per foundation)
    ax = axes[2]
    foundations = FOUNDATION_ORDER
    x = np.arange(len(foundations))
    width = 0.25
    repe_pw = [alignment[fv]["repe_vs_pw"] for fv in foundations]
    repe_md = [alignment[fv]["repe_vs_md"] for fv in foundations]
    md_pw = [alignment[fv]["md_vs_pw"] for fv in foundations]

    ax.bar(x - width, repe_pw, width, color="#43A047", alpha=0.8, label="RepE ↔ PW")
    ax.bar(x, repe_md, width, color="#FB8C00", alpha=0.8, label="RepE ↔ MD")
    ax.bar(x + width, md_pw, width, color="#8E24AA", alpha=0.8, label="MD ↔ PW")

    ax.set_xlabel("Foundation", fontsize=11)
    ax.set_ylabel("|cos| (mean across layers)", fontsize=11)
    ax.set_title("(c) Pairwise Method Alignment", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([FOUNDATION_SHORT[f] for f in foundations], fontsize=9, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("B.5: RepE Concept Directions — Triangulation of Geometric Findings",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"fig_b5_concept_directions.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {figures_dir / 'fig_b5_concept_directions.png'}")


if __name__ == "__main__":
    main()
