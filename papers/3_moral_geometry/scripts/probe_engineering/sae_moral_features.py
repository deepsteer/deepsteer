#!/usr/bin/env python3
"""WS5: SAE moral feature identification.

After SAE training, identify features that activate selectively on moral
text and compare their decoder directions with probe-derived directions.

Pipeline:
  1. Load trained SAE + moral/neutral probing dataset
  2. Collect residual stream activations for moral and neutral sentences
  3. Encode through SAE → per-feature activation for each sentence
  4. Compute moral selectivity: mean(moral_activation) - mean(neutral_activation)
  5. Rank features by selectivity; identify foundation-specific features
  6. Compare top feature decoder columns with mean-diff/LEACE directions

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/sae_moral_features.py
    python papers/3_moral_geometry/scripts/probe_engineering/sae_moral_features.py --layer 8
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from shared import (
    FOUNDATION_ORDER,
    FOUNDATION_SHORT,
    ensure_output_dirs,
    load_probe_directions,
    compute_mean_diff_directions,
    OUTPUT_DIR,
)
from sae_training import SparseAutoencoder


def load_sae(sae_path: str | Path, device: str) -> SparseAutoencoder:
    """Load a trained SAE from checkpoint."""
    ckpt = torch.load(sae_path, weights_only=False, map_location="cpu")
    sae = SparseAutoencoder(ckpt["d_model"], ckpt["d_sae"])
    sae.load_state_dict(ckpt["state_dict"])
    return sae.to(device).eval()


def collect_moral_activations(
    model_name: str,
    layer: int,
    device: str,
    target_per_foundation: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collect residual stream activations for moral/neutral probing pairs.

    Returns:
        moral_acts: (N, d_model) — last-token activations for moral sentences
        neutral_acts: (N, d_model) — last-token activations for neutral sentences
        foundations: list of foundation labels per pair
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deepsteer.datasets.pipeline import build_probing_dataset

    dataset = build_probing_dataset(target_per_foundation=target_per_foundation)
    pairs = dataset.train

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
    ).to(device).eval()

    moral_acts = []
    neutral_acts = []
    foundations = []

    for pair in pairs:
        for text, dest in [(pair.moral, moral_acts), (pair.neutral, neutral_acts)]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)

            hook_out = {}

            def hook_fn(module, input, output, _layer=layer):
                if isinstance(output, tuple):
                    hook_out[_layer] = output[0][0, -1].detach().cpu()
                else:
                    hook_out[_layer] = output[0, -1].detach().cpu()

            h = model.model.layers[layer].register_forward_hook(hook_fn)
            with torch.no_grad():
                model(**inputs)
            h.remove()

            dest.append(hook_out[layer])

        foundations.append(pair.foundation.value)

    del model
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return torch.stack(moral_acts), torch.stack(neutral_acts), foundations


def compute_feature_selectivity(
    sae: SparseAutoencoder,
    moral_acts: torch.Tensor,
    neutral_acts: torch.Tensor,
    foundations: list[str],
    device: str,
) -> dict:
    """Compute per-feature moral selectivity and foundation specificity.

    Returns dict with:
      - global_selectivity: (d_sae,) mean(moral) - mean(neutral) activation per feature
      - foundation_selectivity: {foundation: (d_sae,)} per-foundation selectivity
      - moral_mean: (d_sae,) mean activation on moral text
      - neutral_mean: (d_sae,) mean activation on neutral text
      - moral_l0: mean number of active features on moral text
      - neutral_l0: mean number of active features on neutral text
    """
    with torch.no_grad():
        moral_latents = sae.encode(moral_acts.to(device)).cpu()    # (N, d_sae)
        neutral_latents = sae.encode(neutral_acts.to(device)).cpu()

    moral_mean = moral_latents.mean(dim=0).numpy()
    neutral_mean = neutral_latents.mean(dim=0).numpy()
    global_selectivity = moral_mean - neutral_mean

    moral_l0 = (moral_latents > 0).float().sum(dim=-1).mean().item()
    neutral_l0 = (neutral_latents > 0).float().sum(dim=-1).mean().item()

    foundation_indices: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(foundations):
        foundation_indices[f].append(i)

    foundation_selectivity = {}
    for fv in FOUNDATION_ORDER:
        if fv not in foundation_indices:
            continue
        idx = foundation_indices[fv]
        fv_moral_mean = moral_latents[idx].mean(dim=0).numpy()
        fv_neutral_mean = neutral_latents[idx].mean(dim=0).numpy()
        foundation_selectivity[fv] = fv_moral_mean - fv_neutral_mean

    return {
        "global_selectivity": global_selectivity,
        "foundation_selectivity": foundation_selectivity,
        "moral_mean": moral_mean,
        "neutral_mean": neutral_mean,
        "moral_l0": moral_l0,
        "neutral_l0": neutral_l0,
    }


def compare_with_probe_directions(
    sae: SparseAutoencoder,
    selectivity: dict,
    probe_directions: dict[str, dict[int, np.ndarray]],
    layer: int,
    top_k: int = 100,
) -> dict:
    """Compare top SAE features with probe-derived directions.

    For the top-k most morally selective features, compute cosine similarity
    between each feature's decoder column and each foundation's probe direction.
    """
    decoder_weight = sae.decoder.weight.detach().cpu().numpy()  # (d_model, d_sae)

    global_sel = selectivity["global_selectivity"]
    top_indices = np.argsort(np.abs(global_sel))[::-1][:top_k]

    feature_analysis = []
    for feat_idx in top_indices:
        decoder_col = decoder_weight[:, feat_idx]
        decoder_col = decoder_col / (np.linalg.norm(decoder_col) + 1e-12)

        cosines = {}
        for fv in FOUNDATION_ORDER:
            d = probe_directions.get(fv, {}).get(layer)
            if d is not None:
                cosines[FOUNDATION_SHORT[fv]] = float(np.dot(decoder_col, d))

        fv_sel = {}
        for fv in FOUNDATION_ORDER:
            s = selectivity["foundation_selectivity"].get(fv)
            if s is not None:
                fv_sel[FOUNDATION_SHORT[fv]] = float(s[feat_idx])

        best_fv = max(cosines, key=lambda k: abs(cosines[k])) if cosines else None

        feature_analysis.append({
            "feature_idx": int(feat_idx),
            "global_selectivity": float(global_sel[feat_idx]),
            "foundation_selectivity": fv_sel,
            "probe_cosines": cosines,
            "best_aligned_foundation": best_fv,
            "best_alignment": float(cosines[best_fv]) if best_fv else 0.0,
        })

    # Aggregate: what fraction of top-k features align well with probe directions?
    alignment_thresholds = [0.1, 0.2, 0.3, 0.5]
    alignment_fracs = {}
    for thresh in alignment_thresholds:
        n_aligned = sum(1 for f in feature_analysis if abs(f["best_alignment"]) > thresh)
        alignment_fracs[f">{thresh}"] = round(n_aligned / top_k, 4)

    # Per-foundation: which foundation's probe direction has the most aligned SAE features?
    foundation_top_features: dict[str, int] = {FOUNDATION_SHORT[fv]: 0 for fv in FOUNDATION_ORDER}
    for f in feature_analysis:
        if f["best_aligned_foundation"] and abs(f["best_alignment"]) > 0.2:
            foundation_top_features[f["best_aligned_foundation"]] += 1

    return {
        "top_k": top_k,
        "features": feature_analysis[:20],  # save top 20 for paper
        "alignment_fractions": alignment_fracs,
        "foundation_top_features": foundation_top_features,
    }


def compute_subspace_overlap(
    sae: SparseAutoencoder,
    selectivity: dict,
    probe_directions: dict[str, dict[int, np.ndarray]],
    layer: int,
    top_k: int = 100,
) -> dict:
    """Measure overlap between SAE feature subspace and probe direction subspace.

    Projects probe directions onto the span of top-k SAE decoder columns and
    vice versa. High overlap = SAE rediscovers probe-like structure.
    """
    decoder_weight = sae.decoder.weight.detach().cpu().numpy()  # (d_model, d_sae)

    global_sel = selectivity["global_selectivity"]
    top_indices = np.argsort(np.abs(global_sel))[::-1][:top_k]

    # Build SAE feature subspace (top-k decoder columns)
    sae_vecs = decoder_weight[:, top_indices].T  # (top_k, d_model)
    _, s_sae, Vt_sae = np.linalg.svd(sae_vecs, full_matrices=False)
    rank_sae = min(top_k, np.sum(s_sae > 1e-8))
    sae_basis = Vt_sae[:rank_sae]  # (rank, d_model)

    # Build probe direction subspace (6 foundation directions)
    probe_vecs = []
    for fv in FOUNDATION_ORDER:
        d = probe_directions.get(fv, {}).get(layer)
        if d is not None:
            probe_vecs.append(d)
    if not probe_vecs:
        return {"error": "no probe directions for this layer"}
    probe_mat = np.stack(probe_vecs)  # (6, d_model)
    _, s_probe, Vt_probe = np.linalg.svd(probe_mat, full_matrices=False)
    rank_probe = np.sum(s_probe > 1e-8)
    probe_basis = Vt_probe[:rank_probe]  # (rank, d_model)

    # Probe directions projected onto SAE subspace
    probe_in_sae = []
    for pv in probe_vecs:
        proj = sae_basis @ pv
        membership = float(np.dot(proj, proj))
        probe_in_sae.append(membership)

    # SAE top features projected onto probe subspace
    sae_in_probe = []
    for i in range(min(20, top_k)):
        col = sae_vecs[i]
        col = col / (np.linalg.norm(col) + 1e-12)
        proj = probe_basis @ col
        membership = float(np.dot(proj, proj))
        sae_in_probe.append(membership)

    return {
        "sae_subspace_rank": int(rank_sae),
        "probe_subspace_rank": int(rank_probe),
        "probe_membership_in_sae_subspace": {
            FOUNDATION_SHORT[fv]: round(m, 4)
            for fv, m in zip(FOUNDATION_ORDER, probe_in_sae)
        },
        "mean_probe_membership": round(float(np.mean(probe_in_sae)), 4),
        "top20_sae_membership_in_probe_subspace": [round(m, 4) for m in sae_in_probe],
        "mean_sae_in_probe": round(float(np.mean(sae_in_probe)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WS5: SAE moral feature identification.")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--sae-path", default=None)
    parser.add_argument("--probe-directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--target-per-foundation", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    output_dir, _ = ensure_output_dirs()

    device = args.device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print("=" * 60)
    print("WS5: SAE Moral Feature Identification")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Layer: {args.layer}")

    # Load SAE
    sae_path = args.sae_path or str(output_dir / f"sae_layer{args.layer}.pt")
    if not Path(sae_path).exists():
        print(f"\nNo SAE checkpoint at {sae_path}")
        print("Run sae_training.py first.")
        return

    print(f"\nLoading SAE: {sae_path}")
    sae = load_sae(sae_path, device)
    print(f"  d_model={sae.d_model}, d_sae={sae.d_sae}")

    # Load probe directions
    print(f"Loading probe directions: {args.probe_directions}")
    probe_directions = load_probe_directions(args.probe_directions)
    available_layers = set()
    for fv_dirs in probe_directions.values():
        available_layers.update(fv_dirs.keys())
    print(f"  Available layers: {sorted(available_layers)}")

    # Also compute mean-diff directions for comparison
    from shared import load_model_and_collect_activations
    all_train, _, _, n_layers, foundation_indices = load_model_and_collect_activations(
        model_name=args.model,
        device=device,
        target_per_foundation=args.target_per_foundation,
        collect_test=False,
    )
    mean_diff_dirs = compute_mean_diff_directions(all_train, n_layers, foundation_indices)

    # Collect moral/neutral activations at the target layer
    print(f"\nCollecting moral/neutral activations at layer {args.layer}...")
    moral_acts, neutral_acts, foundations = collect_moral_activations(
        args.model, args.layer, device, args.target_per_foundation,
    )
    print(f"  Moral: {moral_acts.shape}, Neutral: {neutral_acts.shape}")

    # Compute feature selectivity
    print("\nComputing feature selectivity...")
    selectivity = compute_feature_selectivity(
        sae, moral_acts, neutral_acts, foundations, device,
    )
    print(f"  Moral L0: {selectivity['moral_l0']:.0f}")
    print(f"  Neutral L0: {selectivity['neutral_l0']:.0f}")

    n_morally_selective = np.sum(np.abs(selectivity["global_selectivity"]) > 0.1)
    print(f"  Features with |selectivity| > 0.1: {n_morally_selective}")

    # Compare with probe directions
    print(f"\nComparing top-{args.top_k} SAE features with probe directions...")
    comparison_probe = compare_with_probe_directions(
        sae, selectivity, probe_directions, args.layer, args.top_k,
    )
    comparison_meandiff = compare_with_probe_directions(
        sae, selectivity, mean_diff_dirs, args.layer, args.top_k,
    )

    print("\n  Alignment with probe-weight directions:")
    for thresh, frac in comparison_probe["alignment_fractions"].items():
        print(f"    |cos| {thresh}: {frac:.1%} of top-{args.top_k}")
    print(f"  Foundation distribution (|cos|>0.2):")
    for fv, count in comparison_probe["foundation_top_features"].items():
        print(f"    {fv:<12s}: {count}")

    print("\n  Alignment with mean-diff directions:")
    for thresh, frac in comparison_meandiff["alignment_fractions"].items():
        print(f"    |cos| {thresh}: {frac:.1%} of top-{args.top_k}")

    # Subspace overlap
    print(f"\nComputing subspace overlap...")
    overlap_probe = compute_subspace_overlap(
        sae, selectivity, probe_directions, args.layer, args.top_k,
    )
    overlap_meandiff = compute_subspace_overlap(
        sae, selectivity, mean_diff_dirs, args.layer, args.top_k,
    )

    print(f"\n  Probe directions → SAE subspace membership:")
    for fv, m in overlap_probe["probe_membership_in_sae_subspace"].items():
        print(f"    {fv:<12s}: {m:.4f}")
    print(f"    Mean: {overlap_probe['mean_probe_membership']:.4f}")

    print(f"\n  Mean-diff directions → SAE subspace membership:")
    for fv, m in overlap_meandiff["probe_membership_in_sae_subspace"].items():
        print(f"    {fv:<12s}: {m:.4f}")
    print(f"    Mean: {overlap_meandiff['mean_probe_membership']:.4f}")

    # Save results
    results = {
        "analysis": "sae_moral_features",
        "model": args.model,
        "layer": args.layer,
        "sae_path": str(sae_path),
        "top_k": args.top_k,
        "selectivity_stats": {
            "moral_l0": selectivity["moral_l0"],
            "neutral_l0": selectivity["neutral_l0"],
            "n_morally_selective_01": int(n_morally_selective),
        },
        "probe_weight_comparison": comparison_probe,
        "mean_diff_comparison": comparison_meandiff,
        "subspace_overlap_probe": overlap_probe,
        "subspace_overlap_meandiff": overlap_meandiff,
    }

    out_path = output_dir / f"sae_moral_features_layer{args.layer}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
