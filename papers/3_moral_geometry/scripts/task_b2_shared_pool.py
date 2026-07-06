#!/usr/bin/env python3
"""TASK B, construction (b): the reviewer's EXACT mechanism.

Six non-moral MARKED-pole probes, each contrasted against the SAME shared 40-statement
neutral pool (get_flat_neutral_pool). If a generic 'content-vs-shared-neutral' axis is what
drives the moral 0.26, these six non-moral directions (all pointing away from the identical
neutral cloud) MUST share it and give a positive mean pairwise cosine near 0.26.

This is the strongest form of the objection: shared neutral reference is held fixed, so any
shared 'away-from-neutral' component is forced into every direction by construction.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path("papers/3_moral_geometry/scripts").resolve()
sys.path.insert(0, str(SCRIPT_DIR))
from exp1_2_3_framework_geometry import (  # noqa: E402
    compute_cosine_similarity_matrix, compute_effective_dimensionality, train_probe_with_direction,
)
from task_b_control import build_concept_twins, collect, ci  # noqa: E402

OLMO_REPO = "allenai/OLMo-2-0425-1B"
N_TRAIN, N_TEST, N_PER = 32, 8, 40
OUT_DIR = Path("papers/3_moral_geometry/outputs/nonmoral_control")


def build_marked_sets():
    """Six non-moral MARKED-pole statement sets (label 1), all vs the SAME neutral pool."""
    from deepsteer.datasets.neutral_pool import get_flat_neutral_pool
    from deepsteer.datasets.register_pairs import get_register_pairs
    from deepsteer.datasets.sentiment_pairs import get_sentiment_pairs
    ct = build_concept_twins()
    sent = get_sentiment_pairs()
    reg = get_register_pairs()
    marked = {
        "sentiment_positive": [a for a, _ in sent],
        "sentiment_negative": [b for _, b in sent],
        "register_formal": [a for a, _ in reg],
        "register_informal": [b for _, b in reg],
        "tense_past": [a for a, _ in ct["tense_past_vs_present"]],
        "topic_sports": [a for a, _ in ct["topic_sports_vs_cooking"]],
    }
    neutral_all = [s for s, _ in get_flat_neutral_pool()]
    return marked, neutral_all


def main():
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    t0 = time.time()
    model = WhiteBoxModel(OLMO_REPO, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    print(f"Loaded on {model._device} in {time.time()-t0:.1f}s", flush=True)

    marked, neutral_all = build_marked_sets()
    rng = np.random.RandomState(42)
    # SAME 40 neutral statements as label-0 for ALL six concepts (maximizes shared axis).
    neu_idx = rng.permutation(len(neutral_all))[:N_PER]
    neutral = [neutral_all[i] for i in neu_idx]

    # Build (marked, neutral) pairs per concept; neutral list identical across concepts.
    split = {}
    for name, pos in marked.items():
        pidx = rng.permutation(len(pos))[:N_PER]
        pos_sel = [pos[i] for i in pidx]
        pairs = list(zip(pos_sel, neutral))
        split[name] = {"train": pairs[:N_TRAIN], "test": pairs[N_TRAIN:]}
    names = list(split.keys())

    acts = {nm: {"train": collect(model, d["train"]), "test": collect(model, d["test"])}
            for nm, d in split.items()}
    dirs, accs = {}, {}
    for nm, a in acts.items():
        dirs[nm], accs[nm] = {}, {}
        for L in range(n_layers):
            trX, trY = a["train"][L]; teX, teY = a["test"][L]
            acc, _, w = train_probe_with_direction(trX, trY, teX, teY)
            dirs[nm][L] = w; accs[nm][L] = acc

    mean_cos, eff_dim, pc1 = {}, {}, {}
    for L in range(n_layers):
        cs = compute_cosine_similarity_matrix(dirs, L, names)
        n = len(names)
        upper = [cs[i, j] for i in range(n) for j in range(i + 1, n)]
        mean_cos[L] = float(np.mean(upper))
        eff_dim[L] = compute_effective_dimensionality(dirs, L, names)
        mat = np.stack([dirs[nm][L] for nm in names])
        _, s, _ = np.linalg.svd(mat, full_matrices=False)
        pc1[L] = float((s[0] ** 2) / np.sum(s ** 2))

    # bootstrap mean cosine at layer 7
    L = 7
    mc_boot = []
    for _ in range(200):
        pidx = rng.randint(0, N_TRAIN, size=N_TRAIN)
        rows = np.empty(2 * N_TRAIN, dtype=int); rows[0::2] = pidx * 2; rows[1::2] = pidx * 2 + 1
        dv = {}
        for nm in names:
            trX, trY = acts[nm]["train"][L]; teX, teY = acts[nm]["test"][L]
            _, _, w = train_probe_with_direction(trX[rows], trY[rows], teX, teY); dv[nm] = w
        mat = np.stack([dv[nm] for nm in names]); cs = mat @ mat.T
        mc_boot.append(float(np.mean([cs[i, j] for i in range(n) for j in range(i + 1, n)])))
    mc_boot = np.array(mc_boot)

    band = [l for l in range(4, 11)]
    out = {
        "task": "B2_shared_neutral_pool_construction",
        "note": "Six non-moral marked poles vs the SAME 40 neutral statements (shared pool). "
                "Tests the reviewer's exact 'content-vs-shared-neutral by construction' mechanism.",
        "concepts": names,
        "mean_cosine_per_layer": {str(k): round(v, 6) for k, v in mean_cos.items()},
        "pc1_per_layer": {str(k): round(v, 6) for k, v in pc1.items()},
        "eff_dim_per_layer": {str(k): v for k, v in eff_dim.items()},
        "mean_cosine_layer7": mean_cos[7],
        "mean_cosine_stable_band": float(np.mean([mean_cos[l] for l in band])),
        "peak_acc": {nm: round(max(accs[nm].values()), 4) for nm in names},
        "bootstrap_layer7_mean_cosine": float(mc_boot.mean()),
        "bootstrap_layer7_CI95": ci(mc_boot),
    }
    path = OUT_DIR / "task_b2_shared_pool.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\n===== CONSTRUCTION (b): marked-vs-SHARED-neutral-pool =====")
    print(f"  mean pairwise cosine  layer7={mean_cos[7]:.4f}  stable-band={out['mean_cosine_stable_band']:.4f}")
    print(f"  bootstrap CI95        [{out['bootstrap_layer7_CI95'][0]:.4f}, {out['bootstrap_layer7_CI95'][1]:.4f}]")
    print(f"  PC1 layer7={pc1[7]:.4f}  eff_dim7={eff_dim[7]}")
    print(f"  peak acc: {out['peak_acc']}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
