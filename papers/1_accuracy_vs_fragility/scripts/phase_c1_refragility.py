"""Seed-averaged, extended-grid fragility over the cached OLMo-2 1B
early-training activations (review tasks 3.5 and 3.6).

Reads outputs/phase_c1_acts/step_*.npz (produced by phase_c1_cache_activations.py)
and computes, per checkpoint:
  - standard fragility: probe per layer (seed 42), accuracy averaged over 10
    noise seeds at the extended grid {0.1,0.3,1,3,10,30,100}, sigma* by
    cap-at-max;
  - RMS-normalized control (3.6): the same, but per-layer activations are
    standardized to unit RMS (using the train RMS) before probe training and
    noise injection, so sigma is in scale-free units.

Outputs outputs/phase_c1_refragility/{trajectory.json, table2.json} with mean
accuracy, mean sigma*, and early/mid/late group sigma* per checkpoint for both
variants.  No model is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import argparse

ACTS_DIR = Path("papers/1_accuracy_vs_fragility/outputs/phase_c1_acts")
OUT_DIR = Path("papers/1_accuracy_vs_fragility/outputs/phase_c1_refragility")
NOISE = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
N_SEEDS = 10
THRESHOLD = 0.6
SEED = 42
N_EPOCHS = 50
LR = 1e-2
TABLE2_STEPS = [0, 1000, 4000, 10000, 15000, 20000, 36000]
EARLY, MID, LATE = range(0, 5), range(5, 11), range(11, 16)


def _train_probe(X: torch.Tensor, y: torch.Tensor) -> nn.Linear:
    torch.manual_seed(SEED)
    probe = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(probe.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    probe.train()
    for _ in range(N_EPOCHS):
        loss = loss_fn(probe(X).squeeze(-1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    probe.eval()
    return probe


def _acc(probe: nn.Linear, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return ((probe(X).squeeze(-1) > 0).float() == y).float().mean().item()


def _layer_fragility(train_X, train_y, test_X, test_y, rms_normalize: bool):
    if rms_normalize:
        rms = train_X.pow(2).mean().sqrt().clamp_min(1e-8)
        train_X = train_X / rms
        test_X = test_X / rms
    probe = _train_probe(train_X, train_y)
    baseline = _acc(probe, test_X, test_y)
    cap = max(NOISE)
    sigma_star = cap
    never = True
    for nl in NOISE:
        accs = []
        for s in range(N_SEEDS):
            torch.manual_seed(SEED + s)
            accs.append(_acc(probe, test_X + torch.randn_like(test_X) * nl, test_y))
        if float(np.mean(accs)) < THRESHOLD:
            sigma_star = nl
            never = False
            break
    return baseline, sigma_star, never


def _checkpoint(npz, rms_normalize: bool) -> dict:
    n_layers = int(npz["n_layers"][0])
    train_y = torch.from_numpy(npz["train_y"]).float()
    test_y = torch.from_numpy(npz["test_y"]).float()
    baselines, sigmas, n_never = [], [], 0
    for L in range(n_layers):
        b, s, never = _layer_fragility(
            torch.from_numpy(npz[f"train_X_{L}"]).float(), train_y,
            torch.from_numpy(npz[f"test_X_{L}"]).float(), test_y,
            rms_normalize,
        )
        baselines.append(b)
        sigmas.append(s)
        n_never += int(never)
    grp = lambda idx: float(np.mean([sigmas[i] for i in idx]))
    return {
        "mean_acc": round(float(np.mean(baselines)), 4),
        "mean_critical_noise": round(float(np.mean(sigmas)), 4),
        "early_crit": round(grp(EARLY), 4),
        "mid_crit": round(grp(MID), 4),
        "late_crit": round(grp(LATE), 4),
        "n_never_fragile": n_never,
        "per_layer_critical_noise": [round(s, 4) for s in sigmas],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", default=str(ACTS_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    acts_dir, out_dir = Path(args.acts_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(acts_dir.glob("step_*.npz"))
    print(f"Found {len(files)} cached checkpoints in {acts_dir}")

    trajectory: dict[str, dict] = {}
    for f in files:
        npz = np.load(f)
        step = int(npz["step"][0])
        std = _checkpoint(npz, rms_normalize=False)
        rms = _checkpoint(npz, rms_normalize=True)
        trajectory[str(step)] = {"standard": std, "rms_normalized": rms}
        print(f"  step {step:6d}: acc {std['mean_acc']:.3f}  "
              f"sigma* {std['mean_critical_noise']:.2f} "
              f"(early {std['early_crit']:.1f}/mid {std['mid_crit']:.1f}/late {std['late_crit']:.1f})  "
              f"| RMS sigma* {rms['mean_critical_noise']:.2f}")

    with open(out_dir / "trajectory.json", "w") as fh:
        json.dump({
            "noise_levels": NOISE, "n_noise_seeds": N_SEEDS, "seed": SEED,
            "fragility_threshold": THRESHOLD,
            "layer_groups": {"early": list(EARLY), "mid": list(MID), "late": list(LATE)},
            "trajectory": trajectory,
        }, fh, indent=2)

    table2 = {str(s): trajectory[str(s)] for s in TABLE2_STEPS if str(s) in trajectory}
    with open(out_dir / "table2.json", "w") as fh:
        json.dump(table2, fh, indent=2)
    print(f"\nWrote {out_dir}/trajectory.json and table2.json")


if __name__ == "__main__":
    main()
