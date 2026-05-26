#!/usr/bin/env python3
"""Script 7: Dense vs MoE dilemma geometry comparison.

Repeats dilemma probing (Script 3) and geometry analysis (Script 4) on
OLMoE-1B-7B. Compares subspace membership, fragility, and dimensionality
with the dense OLMo-2 1B results.

Usage:
    python papers/3_moral_geometry/scripts/dilemma_moe.py
    python papers/3_moral_geometry/scripts/dilemma_moe.py --skip-fragility
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
from scipy.cluster.hierarchy import linkage

logger = logging.getLogger(__name__)

OLMOE_REPO = "allenai/OLMoE-1B-7B-0924"

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

NOISE_LEVELS = [0.1, 0.3, 1.0, 3.0, 10.0]
FRAGILITY_THRESHOLD = 0.6

# MPS histc fix
_orig_histc = torch.histc


def _histc_mps_fallback(input, bins=100, min=0, max=0):
    if input.device.type == "mps" or not input.is_floating_point():
        return _orig_histc(input.cpu().float(), bins, min, max).to(input.device)
    return _orig_histc(input, bins, min, max)


torch.histc = _histc_mps_fallback


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
    all_acts: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
    for text in texts:
        acts = model.get_activations(text, layers=list(range(n_layers)))
        for layer_idx in range(n_layers):
            h = acts[layer_idx]
            pooled = h.mean(dim=1).squeeze(0)
            all_acts[layer_idx].append(pooled.cpu())
    return all_acts


def compute_subspace_analysis(
    w_dilemma: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
) -> dict:
    e1 = w_a.copy()
    e2 = w_b - np.dot(w_b, e1) * e1
    e2_norm = np.linalg.norm(e2)
    if e2_norm < 1e-10:
        proj_a = np.dot(w_dilemma, e1) ** 2
        return {
            "subspace_membership": float(proj_a),
            "component_balance": 0.5,
            "residual_norm": float(np.sqrt(1 - proj_a)),
            "degenerate": True,
        }
    e2 = e2 / e2_norm

    proj_e1 = np.dot(w_dilemma, e1)
    proj_e2 = np.dot(w_dilemma, e2)
    subspace_membership = proj_e1 ** 2 + proj_e2 ** 2

    cos_a = abs(np.dot(w_dilemma, w_a))
    cos_b = abs(np.dot(w_dilemma, w_b))
    denom = cos_a + cos_b
    component_balance = cos_a / denom if denom > 1e-10 else 0.5

    residual = w_dilemma - proj_e1 * e1 - proj_e2 * e2
    residual_norm = float(np.linalg.norm(residual))

    return {
        "subspace_membership": float(subspace_membership),
        "component_balance": float(component_balance),
        "residual_norm": residual_norm,
        "degenerate": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense vs MoE dilemma geometry.")
    parser.add_argument("--dataset", default="deepsteer/datasets/dilemma_pairs_final.json")
    parser.add_argument("--olmoe-foundation-directions",
                        default="papers/3_moral_geometry/outputs/exp5_dense_vs_moe/olmoe/exp1_probe_directions.npz")
    parser.add_argument("--output-dir", default="papers/3_moral_geometry/outputs/dilemma_moe")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-fragility", action="store_true")
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

    foundation_dir_path = Path(args.olmoe_foundation_directions)
    if not foundation_dir_path.exists():
        print(f"ERROR: {foundation_dir_path} not found.")
        print("Run exp5_dense_vs_moe_geometry.py first for OLMoE foundation directions.")
        return

    with open(dataset_path) as f:
        dilemma_data = json.load(f)

    foundation_directions_data = np.load(foundation_dir_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {OLMOE_REPO}")
    t0 = time.time()
    model = WhiteBoxModel(OLMOE_REPO, device=args.device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    hidden_dim = model.info.hidden_dim
    print(f"Loaded in {time.time() - t0:.1f}s ({n_layers} layers, {hidden_dim} hidden dim)")

    # Load OLMoE foundation directions
    foundation_dirs: dict[str, dict[int, np.ndarray]] = {}
    for fv in FOUNDATION_ORDER:
        foundation_dirs[fv] = {}
        for layer_idx in range(n_layers):
            key = f"{fv}_layer{layer_idx}"
            if key in foundation_directions_data:
                foundation_dirs[fv][layer_idx] = foundation_directions_data[key]

    # Group dilemma pairs
    pairs_by_type: dict[str, list[dict]] = {}
    for p in dilemma_data["pairs"]:
        pk = f"{p['foundation_pair'][0]}-{p['foundation_pair'][1]}"
        pairs_by_type.setdefault(pk, []).append(p)

    # Collect all activations
    all_moral = [p["moral"] for p in dilemma_data["pairs"]]
    all_neutral = [p["neutral"] for p in dilemma_data["pairs"]]

    print(f"\nCollecting OLMoE activations for {len(all_moral)} pairs...")
    t0 = time.time()
    moral_acts = collect_activations(model, all_moral, n_layers)
    neutral_acts = collect_activations(model, all_neutral, n_layers)
    print(f"Collected in {time.time() - t0:.1f}s")

    pair_global_indices: dict[str, list[int]] = {}
    for i, p in enumerate(dilemma_data["pairs"]):
        pk = f"{p['foundation_pair'][0]}-{p['foundation_pair'][1]}"
        pair_global_indices.setdefault(pk, []).append(i)

    # Train dilemma probes and run subspace analysis
    print(f"\n{'='*60}")
    print("OLMoE DILEMMA PROBING")
    print(f"{'='*60}")

    probing_results: dict[str, dict] = {}
    dilemma_directions: dict[str, dict[int, np.ndarray]] = {}
    fragility_results: dict[str, dict] = {}

    for pair in FOUNDATION_PAIRS:
        pk = f"{pair[0]}-{pair[1]}"
        indices = pair_global_indices.get(pk, [])
        if len(indices) < 5:
            continue

        n_train = max(1, int(len(indices) * 0.8))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        if len(test_idx) < 1:
            test_idx = [train_idx[-1]]
            train_idx = train_idx[:-1]

        fv_a = DILEMMA_KEY_TO_FOUNDATION[pair[0]]
        fv_b = DILEMMA_KEY_TO_FOUNDATION[pair[1]]

        pair_probing: dict[str, dict] = {}
        dilemma_directions[pk] = {}
        pair_fragility: dict[str, dict] = {}

        for layer_idx in range(n_layers):
            train_moral_X = torch.stack([moral_acts[layer_idx][i] for i in train_idx])
            train_neutral_X = torch.stack([neutral_acts[layer_idx][i] for i in train_idx])
            test_moral_X = torch.stack([moral_acts[layer_idx][i] for i in test_idx])
            test_neutral_X = torch.stack([neutral_acts[layer_idx][i] for i in test_idx])

            train_X = torch.cat([train_moral_X, train_neutral_X])
            train_y = torch.cat([torch.ones(len(train_idx)), torch.zeros(len(train_idx))])
            test_X = torch.cat([test_moral_X, test_neutral_X])
            test_y = torch.cat([torch.ones(len(test_idx)), torch.zeros(len(test_idx))])

            acc, loss, w_dilemma = train_probe_with_direction(
                train_X, train_y, test_X, test_y,
                n_epochs=args.n_epochs, lr=args.lr,
            )
            dilemma_directions[pk][layer_idx] = w_dilemma

            layer_result: dict = {"accuracy": round(acc, 4)}

            w_a = foundation_dirs[fv_a].get(layer_idx)
            w_b = foundation_dirs[fv_b].get(layer_idx)
            if w_a is not None and w_b is not None:
                subspace = compute_subspace_analysis(w_dilemma, w_a, w_b)
                layer_result.update({k: round(v, 6) if isinstance(v, float) else v
                                     for k, v in subspace.items()})

            pair_probing[str(layer_idx)] = layer_result

            # Fragility
            if not args.skip_fragility:
                frag: dict[str, float] = {}
                probe_model = nn.Linear(hidden_dim, 1, bias=False)
                probe_model.weight.data = torch.tensor(w_dilemma, dtype=torch.float32).unsqueeze(0)
                probe_model.eval()

                with torch.no_grad():
                    baseline_logits = probe_model(test_X).squeeze(-1)
                    baseline_acc = ((baseline_logits > 0).float() == test_y).float().mean().item()

                frag["baseline"] = round(baseline_acc, 4)
                critical = None
                for nl in NOISE_LEVELS:
                    with torch.no_grad():
                        noised = test_X + torch.randn_like(test_X) * nl
                        noised_acc = ((probe_model(noised).squeeze(-1) > 0).float() == test_y).float().mean().item()
                    frag[str(nl)] = round(noised_acc, 4)
                    if noised_acc < FRAGILITY_THRESHOLD and critical is None:
                        critical = nl

                pair_fragility[str(layer_idx)] = {"accuracy_by_noise": frag, "critical_noise": critical}

        accs = {int(k): v["accuracy"] for k, v in pair_probing.items()}
        peak_layer = max(accs, key=accs.get)
        memberships = {int(k): v.get("subspace_membership", 0) for k, v in pair_probing.items()}

        probing_results[pk] = {
            "peak_accuracy": round(accs[peak_layer], 4),
            "peak_layer": peak_layer,
            "peak_subspace_membership": round(max(memberships.values()), 6),
            "per_layer": pair_probing,
        }
        fragility_results[pk] = pair_fragility

        print(f"  [{pk}] Peak acc: {accs[peak_layer]:.1%}, "
              f"peak membership: {max(memberships.values()):.4f}")

    # Save directions
    dir_arrays = {}
    for pk, layers in dilemma_directions.items():
        for layer_idx, w in layers.items():
            dir_arrays[f"dilemma_{pk}_layer{layer_idx}"] = w
    np.savez(output_dir / "dilemma_probe_directions_olmoe.npz", **dir_arrays)

    # Geometry: 15×15 dilemma cosines + 21-direction clustering at each layer
    geometry_results: dict[str, dict] = {}
    dilemma_keys = [f"{a}-{b}" for a, b in FOUNDATION_PAIRS]

    for layer_idx in range(n_layers):
        vecs = []
        labels = []
        for pk in dilemma_keys:
            if pk in dilemma_directions and layer_idx in dilemma_directions[pk]:
                vecs.append(dilemma_directions[pk][layer_idx])
                labels.append(pk)
        if len(vecs) < 2:
            continue

        mat = np.stack(vecs)
        cosines = mat @ mat.T

        # Effective dimensionality
        centered = mat - mat.mean(axis=0, keepdims=True)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        explained = np.cumsum(s ** 2) / np.sum(s ** 2)
        eff_dim = int(np.searchsorted(explained, 0.9)) + 1

        geometry_results[str(layer_idx)] = {
            "dilemma_effective_dim": eff_dim,
            "mean_dilemma_cosine": round(float(np.mean([
                cosines[i, j] for i in range(len(labels))
                for j in range(i + 1, len(labels))
            ])), 6),
        }

    output = {
        "experiment": "dilemma_moe",
        "model": OLMOE_REPO,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "probing": probing_results,
        "fragility": fragility_results if not args.skip_fragility else None,
        "geometry": geometry_results,
    }

    with open(output_dir / "dilemma_moe.json", "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("OLMoE DILEMMA SUMMARY")
    print(f"{'='*60}")
    mean_membership = np.mean([r["peak_subspace_membership"] for r in probing_results.values()])
    mean_acc = np.mean([r["peak_accuracy"] for r in probing_results.values()])
    print(f"Mean peak accuracy: {mean_acc:.1%}")
    print(f"Mean peak subspace membership: {mean_membership:.4f}")

    del model
    _clear_memory()

    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
