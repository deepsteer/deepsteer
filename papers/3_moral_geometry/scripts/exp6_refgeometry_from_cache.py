"""Recompute the exp6 geometric trajectory from cached activations with a
FIXED probe seed (review task 4.1/4.2).

The original exp6 trained probe-weight directions with no RNG seed and was run
with --resume, so the cosine geometry between foundation directions drifted
with process/resume state (the 0.39 vs 0.28 bimodality, the step-16K shift, and
the 5K-multiple spikes; accuracy was unaffected). With a deterministic probe
seed (now in train_probe_with_direction), the directions are a function of the
activations alone.

We reuse the per-checkpoint activations already cached for Paper 1's standard
probing dataset (phase_c1_acts/step_*.npz) — exp6 uses the identical
build_probing_dataset(40, v2) grouped by foundation — so this needs no model
download. For each checkpoint we train one seed-fixed probe per foundation per
layer, compute the mean off-diagonal cosine of the six foundation directions,
and average over the stable layers (6-15).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "papers/3_moral_geometry/scripts")
from exp1_2_3_framework_geometry import train_probe_with_direction  # noqa: E402

ACTS_DIR = Path("papers/1_accuracy_vs_fragility/outputs/phase_c1_acts")
OUT_DIR = Path("papers/3_moral_geometry/outputs/exp6_trajectory")
FOUNDATION_ORDER = [
    "care_harm", "fairness_cheating", "liberty_oppression",
    "loyalty_betrayal", "authority_subversion", "sanctity_degradation",
]
STABLE_LAYERS = range(6, 16)  # harmonized stable-direction range (Paper 3 §4.6)


def _row_indices_by_foundation(pairs):
    """Map foundation -> list of (moral_row, neutral_row) for interleaved acts."""
    idx = defaultdict(list)
    for i, p in enumerate(pairs):
        idx[p.foundation.value].append((2 * i, 2 * i + 1))
    return idx


def _mean_offdiag_cosine(dirs: list[np.ndarray]) -> float:
    V = np.stack([d / (np.linalg.norm(d) + 1e-12) for d in dirs])
    C = V @ V.T
    n = len(dirs)
    return float(np.mean([C[i, j] for i in range(n) for j in range(i + 1, n)]))


def main() -> None:
    from deepsteer.datasets.pipeline import build_probing_dataset

    dataset = build_probing_dataset(target_per_foundation=40, dataset_version="v2")
    train_idx = _row_indices_by_foundation(dataset.train)
    test_idx = _row_indices_by_foundation(dataset.test)

    files = sorted(ACTS_DIR.glob("step_*.npz"))
    print(f"Recomputing exp6 geometry from {len(files)} cached checkpoints (seed-fixed)")

    trajectory: dict[int, dict] = {}
    for f in files:
        npz = np.load(f)
        step = int(npz["step"][0])
        n_layers = int(npz["n_layers"][0])
        train_y_full = torch.from_numpy(npz["train_y"]).float()
        test_y_full = torch.from_numpy(npz["test_y"]).float()

        per_layer_cos = []
        for L in range(n_layers):
            tX = torch.from_numpy(npz[f"train_X_{L}"]).float()
            eX = torch.from_numpy(npz[f"test_X_{L}"]).float()
            dirs = []
            for fdn in FOUNDATION_ORDER:
                rows = [r for pair in train_idx[fdn] for r in pair]
                trows = [r for pair in test_idx[fdn] for r in pair]
                _, _, w = train_probe_with_direction(
                    tX[rows], train_y_full[rows], eX[trows], test_y_full[trows],
                )
                dirs.append(w)
            per_layer_cos.append(_mean_offdiag_cosine(dirs))

        stable = float(np.mean([per_layer_cos[L] for L in STABLE_LAYERS]))
        trajectory[step] = {
            "stable_mean_cosine": round(stable, 4),
            "per_layer_mean_cosine": [round(c, 4) for c in per_layer_cos],
        }
        print(f"  step {step:6d}: stable mean cosine = {stable:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment": "exp6_geometric_trajectory_seedfixed",
        "model": "allenai/OLMo-2-0425-1B-early-training",
        "n_checkpoints": len(trajectory),
        "stable_layers": list(STABLE_LAYERS),
        "note": "Seed-fixed recomputation from cached activations; supersedes the "
                "RNG-dependent exp6_trajectory_summary.json (resume artifact).",
        "trajectory": {str(k): v for k, v in sorted(trajectory.items())},
    }
    with open(OUT_DIR / "exp6_trajectory_seedfixed.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUT_DIR}/exp6_trajectory_seedfixed.json")


if __name__ == "__main__":
    main()
