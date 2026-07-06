#!/usr/bin/env python3
"""TASK B: non-moral concept positive-control for Paper 3 framework geometry.

Builds SIX non-moral concept-vs-twin probes matched to the six MFT foundations in
construction (same n=32 train/8 test, IDENTICAL probe-weight extraction, same model
OLMo-2-0425-1B, same 16 layers), runs them through the IDENTICAL geometry computation
(compute_cosine_similarity_matrix / compute_effective_dimensionality imported from
exp1_2_3_framework_geometry.py), and reports the non-moral set's mean pairwise cosine +
eff-dim next to the moral 0.26 -> THE LADDER.

Also bootstraps (resample the 32 train pairs per concept, retrain, recompute the 6x6
mean pairwise cosine + PC1) to give CIs on the moral 0.26 / PC1 0.379 (Task A remainder)
and a Δ-CI (moral - nonmoral).

The moral v2 dataset uses surface-matched non-moral TWINS (NeutralDomain.MATCHED), not a
shared factual pool; each foundation direction = moral-pole vs matched-twin. The faithful
matched control therefore uses concept-pole-A vs matched-twin-pole-B (sentiment pos/neg,
register formal/informal, grammaticality, tense past/present, number sing/plural, topic
sports/cooking). Each probe NAMED BY CONSTRUCTION below.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path("papers/3_moral_geometry/scripts").resolve()
sys.path.insert(0, str(SCRIPT_DIR))
from exp1_2_3_framework_geometry import (  # identical estimator + geometry  # noqa: E402
    compute_cosine_similarity_matrix,
    compute_effective_dimensionality,
    train_probe_with_direction,
)

OLMO_REPO = "allenai/OLMo-2-0425-1B"
N_TRAIN, N_TEST = 32, 8
N_PER = N_TRAIN + N_TEST
OUT_DIR = Path("papers/3_moral_geometry/outputs/nonmoral_control")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Six non-moral concept twin sets. Each -> list[(pole_A, pole_B)] with >= 40 pairs.
# NAMED BY CONSTRUCTION.
# ---------------------------------------------------------------------------
def build_concept_twins() -> dict[str, list[tuple[str, str]]]:
    from deepsteer.datasets.register_pairs import get_register_pairs
    from deepsteer.datasets.sentiment_pairs import get_sentiment_pairs
    from deepsteer.datasets.syntax_pairs import get_syntax_pairs

    concepts: dict[str, list[tuple[str, str]]] = {}
    # 1. sentiment: positive-sentiment vs negative-sentiment (surface-matched twins)
    concepts["sentiment_pos_vs_neg"] = get_sentiment_pairs()
    # 2. register: formal vs informal (surface-matched twins)
    concepts["register_formal_vs_informal"] = get_register_pairs()
    # 3. grammaticality: grammatical vs ungrammatical (subject-verb agreement twins)
    concepts["grammaticality_wellformed_vs_illformed"] = get_syntax_pairs()

    # 4. tense: past vs present (matched frames, 3rd-person declarative)
    subjects = ["The engineer", "The teacher", "The pilot", "The gardener", "The novelist",
                "The chemist", "The architect", "The violinist", "The farmer",
                "The electrician", "The surgeon", "The accountant"]
    verb_obj = [("inspected", "inspects", "the bridge"),
                ("reviewed", "reviews", "the manuscript"),
                ("repaired", "repairs", "the engine"),
                ("watered", "waters", "the garden"),
                ("measured", "measures", "the distance"),
                ("recorded", "records", "the results")]
    tense = []
    for s in subjects:
        for past, pres, obj in verb_obj:
            tense.append((f"{s} {past} {obj}.", f"{s} {pres} {obj}."))
    concepts["tense_past_vs_present"] = tense

    # 5. number: singular vs plural subject (subject + verb agreement)
    number_frames = [
        ("scientist", "scientists", "reviews", "review", "the report carefully"),
        ("student", "students", "reads", "read", "the assigned chapter"),
        ("worker", "workers", "operates", "operate", "the heavy machine"),
        ("dancer", "dancers", "rehearses", "rehearse", "the final routine"),
        ("driver", "drivers", "follows", "follow", "the marked route"),
        ("painter", "painters", "mixes", "mix", "the bright colors"),
        ("cook", "cooks", "prepares", "prepare", "the evening meal"),
        ("singer", "singers", "performs", "perform", "the opening song"),
    ]
    articles = ["The", "A young", "The senior", "Every", "The local"]
    number = []
    for art in articles:
        for sg, pl, vs, vp, tail in number_frames:
            art_pl = "The" if art in ("A young", "Every") else art  # plural-safe article
            number.append((f"{art} {sg} {vs} {tail}.", f"{art_pl} {pl} {vp} {tail}."))
    concepts["number_singular_vs_plural"] = number

    # 6. topic: sports vs cooking (both factual, non-moral; category contrast)
    sports = [f"{s} {a}." for s in
              ["The striker", "The goalkeeper", "The sprinter", "The midfielder",
               "The cyclist", "The swimmer", "The forward", "The defender"]
              for a in ["scored in the final minute of the match",
                        "trained hard before the championship game",
                        "sprinted down the length of the field",
                        "celebrated the winning goal with teammates",
                        "practiced free throws after the tournament"]]
    cooking = [f"{s} {a}." for s in
               ["The chef", "The baker", "The cook", "The pastry chef",
                "The line cook", "The sous chef", "The grill cook", "The head chef"]
               for a in ["seared the salmon in a hot cast-iron pan",
                         "whisked the eggs into a smooth batter",
                         "simmered the tomato sauce for an hour",
                         "kneaded the dough until it turned elastic",
                         "diced the onions for the evening stew"]]
    concepts["topic_sports_vs_cooking"] = list(zip(sports, cooking))
    return concepts


def build_moral_twins() -> dict[str, list[tuple[str, str]]]:
    """Moral foundations as (moral, matched-neutral-twin) pairs, from the v2 dataset."""
    from deepsteer.datasets.pipeline import build_probing_dataset
    ds = build_probing_dataset(target_per_foundation=40, dataset_version="v2")
    by_f: dict[str, list[tuple[str, str]]] = {}
    for p in list(ds.train) + list(ds.test):
        by_f.setdefault(p.foundation.value, []).append((p.moral, p.neutral))
    order = ["care_harm", "fairness_cheating", "liberty_oppression",
             "loyalty_betrayal", "authority_subversion", "sanctity_degradation"]
    return {f: by_f[f] for f in order if f in by_f}


# ---------------------------------------------------------------------------
def standardize(twins: dict[str, list[tuple[str, str]]], seed: int = 42):
    """Take exactly N_PER pairs/concept, deterministic 32 train / 8 test split."""
    rng = np.random.RandomState(seed)
    out = {}
    for name, pairs in twins.items():
        pairs = list(pairs)
        if len(pairs) < N_PER:
            raise ValueError(f"{name}: only {len(pairs)} pairs (< {N_PER})")
        idx = rng.permutation(len(pairs))[:N_PER]
        sel = [pairs[i] for i in idx]
        out[name] = {"train": sel[:N_TRAIN], "test": sel[N_TRAIN:]}
    return out


def collect(model, pairs: list[tuple[str, str]]):
    """{layer: (X, y)} with row 2i = pole_A(label 1), 2i+1 = pole_B(label 0)."""
    texts, labels = [], []
    for a, b in pairs:
        texts.extend([a, b])
        labels.extend([1, 0])
    pooled = model.collect_batch_activations(texts, pooling="mean")
    y = torch.tensor(labels, dtype=torch.float32)
    return {L: (X.float(), y) for L, X in pooled.items()}


def full_directions(model, split, n_layers):
    """Point directions per concept per layer (probe trained on train split)."""
    acts = {name: {"train": collect(model, d["train"]), "test": collect(model, d["test"])}
            for name, d in split.items()}
    dirs: dict[str, dict[int, np.ndarray]] = {}
    accs: dict[str, dict[int, float]] = {}
    for name, a in acts.items():
        dirs[name], accs[name] = {}, {}
        for L in range(n_layers):
            trX, trY = a["train"][L]
            teX, teY = a["test"][L]
            acc, _, w = train_probe_with_direction(trX, trY, teX, teY)
            dirs[name][L] = w
            accs[name][L] = acc
    return dirs, accs, acts


def geometry(dirs, names, n_layers):
    mean_cos, eff_dim, pc1 = {}, {}, {}
    for L in range(n_layers):
        cs = compute_cosine_similarity_matrix(dirs, L, names)
        if cs is None:
            continue
        n = len(names)
        upper = [cs[i, j] for i in range(n) for j in range(i + 1, n)]
        mean_cos[L] = float(np.mean(upper))
        eff_dim[L] = compute_effective_dimensionality(dirs, L, names)  # (centered, matches paper eff-dim)
        # PC1 = UNCENTERED leading-PC fraction (the shared-component metric; paper's 0.379)
        mat = np.stack([dirs[nm][L] for nm in names])
        _, s, _ = np.linalg.svd(mat, full_matrices=False)
        pc1[L] = float((s[0] ** 2) / np.sum(s ** 2))
    return mean_cos, eff_dim, pc1


def bootstrap_geometry(acts, names, layer, n_boot, seed):
    """Resample the 32 train pairs per concept, retrain all 6 dirs, recompute
    mean pairwise cosine + PC1 at `layer`. Returns (mean_cos_boot, pc1_boot)."""
    rng = np.random.RandomState(seed)
    mc_boot, pc1_boot = [], []
    n_pairs = N_TRAIN
    for _ in range(n_boot):
        pidx = rng.randint(0, n_pairs, size=n_pairs)
        rows = np.empty(2 * n_pairs, dtype=int)
        rows[0::2] = pidx * 2
        rows[1::2] = pidx * 2 + 1
        dvecs = {}
        for nm in names:
            trX, trY = acts[nm]["train"][layer]
            teX, teY = acts[nm]["test"][layer]
            _, _, w = train_probe_with_direction(trX[rows], trY[rows], teX, teY)
            dvecs[nm] = w
        mat = np.stack([dvecs[nm] for nm in names])
        cs = mat @ mat.T
        n = len(names)
        mc_boot.append(float(np.mean([cs[i, j] for i in range(n) for j in range(i + 1, n)])))
        _, s, _ = np.linalg.svd(mat, full_matrices=False)  # UNCENTERED (paper's PC1 metric)
        pc1_boot.append(float((s[0] ** 2) / np.sum(s ** 2)))
    return np.array(mc_boot), np.array(pc1_boot)


def ci(a, lo=2.5, hi=97.5):
    return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 concepts, no bootstrap, load check")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--boot-layer", type=int, default=7)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    t0 = time.time()
    model = WhiteBoxModel(OLMO_REPO, device=args.device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    print(f"Loaded {OLMO_REPO} on {model._device} in {time.time()-t0:.1f}s "
          f"({n_layers} layers, {model.info.n_params/1e9:.1f}B)", flush=True)

    concept_twins = build_concept_twins()
    for nm, pr in concept_twins.items():
        print(f"  concept {nm}: {len(pr)} pairs", flush=True)

    if args.smoke:
        sub = dict(list(concept_twins.items())[:2])
        split = standardize(sub)
        dirs, accs, _ = full_directions(model, split, n_layers)
        names = list(split.keys())
        for nm in names:
            pk = max(accs[nm], key=accs[nm].get)
            print(f"  SMOKE {nm}: peak acc {accs[nm][pk]:.2f}@{pk}", flush=True)
        print("SMOKE OK: model loaded, activations collected, probes trained.", flush=True)
        return

    # ---- Non-moral control ----
    nm_split = standardize(concept_twins)
    nm_names = list(nm_split.keys())
    print("\nExtracting NON-MORAL directions...", flush=True)
    nm_dirs, nm_accs, nm_acts = full_directions(model, nm_split, n_layers)
    nm_mc, nm_ed, nm_pc1 = geometry(nm_dirs, nm_names, n_layers)

    # ---- Moral (reproduce + CI) ----
    moral_twins = build_moral_twins()
    m_split = standardize(moral_twins)
    m_names = list(m_split.keys())
    print("Extracting MORAL directions (reproduction)...", flush=True)
    m_dirs, m_accs, m_acts = full_directions(model, m_split, n_layers)
    m_mc, m_ed, m_pc1 = geometry(m_dirs, m_names, n_layers)

    L = args.boot_layer
    print(f"\nBootstrap ({args.n_boot}x, layer {L}) moral...", flush=True)
    m_mc_b, m_pc1_b = bootstrap_geometry(m_acts, m_names, L, args.n_boot, seed=42)
    print(f"Bootstrap ({args.n_boot}x, layer {L}) non-moral...", flush=True)
    nm_mc_b, nm_pc1_b = bootstrap_geometry(nm_acts, nm_names, L, args.n_boot, seed=42)
    diff_b = m_mc_b - nm_mc_b  # Δ-CI moral - nonmoral (independent resamples)

    stable_band = [l for l in range(4, 11)]  # stable-band layers
    def band_mean(d):
        return float(np.mean([d[l] for l in stable_band if l in d]))

    out = {
        "task": "B_nonmoral_positive_control",
        "model": OLMO_REPO, "n_layers": n_layers, "n_train": N_TRAIN, "n_test": N_TEST,
        "n_boot": args.n_boot, "boot_layer": L,
        "moral": {
            "concepts": m_names,
            "mean_cosine_per_layer": {str(k): round(v, 6) for k, v in m_mc.items()},
            "eff_dim_per_layer": {str(k): v for k, v in m_ed.items()},
            "pc1_per_layer": {str(k): round(v, 6) for k, v in m_pc1.items()},
            "mean_cosine_layer7": m_mc.get(7),
            "mean_cosine_stable_band": band_mean(m_mc),
            "peak_acc": {nm: round(max(m_accs[nm].values()), 4) for nm in m_names},
        },
        "nonmoral": {
            "concepts": nm_names,
            "mean_cosine_per_layer": {str(k): round(v, 6) for k, v in nm_mc.items()},
            "eff_dim_per_layer": {str(k): v for k, v in nm_ed.items()},
            "pc1_per_layer": {str(k): round(v, 6) for k, v in nm_pc1.items()},
            "mean_cosine_layer7": nm_mc.get(7),
            "mean_cosine_stable_band": band_mean(nm_mc),
            "peak_acc": {nm: round(max(nm_accs[nm].values()), 4) for nm in nm_names},
        },
        "bootstrap_layer7": {
            "moral_mean_cosine": float(m_mc_b.mean()), "moral_mean_cosine_CI95": ci(m_mc_b),
            "moral_pc1": float(m_pc1_b.mean()), "moral_pc1_CI95": ci(m_pc1_b),
            "nonmoral_mean_cosine": float(nm_mc_b.mean()), "nonmoral_mean_cosine_CI95": ci(nm_mc_b),
            "nonmoral_pc1": float(nm_pc1_b.mean()), "nonmoral_pc1_CI95": ci(nm_pc1_b),
            "delta_moral_minus_nonmoral": float(diff_b.mean()),
            "delta_CI95": ci(diff_b),
            "delta_excludes_0": bool(np.percentile(diff_b, 2.5) > 0),
        },
    }
    path = OUT_DIR / "task_b_nonmoral_control.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    np.savez(OUT_DIR / "task_b_boot_arrays.npz",
             moral_mc=m_mc_b, moral_pc1=m_pc1_b, nonmoral_mc=nm_mc_b,
             nonmoral_pc1=nm_pc1_b, delta=diff_b)
    for nm in nm_names:
        np.savez(OUT_DIR / f"dir_{nm}.npz", **{f"layer{l}": nm_dirs[nm][l] for l in range(n_layers)})

    print("\n===== THE LADDER (mean pairwise cosine of 6-direction set) =====")
    print(f"  MORAL     layer7 = {m_mc.get(7):.4f}  stable-band = {band_mean(m_mc):.4f}  "
          f"eff_dim7 = {m_ed.get(7)}")
    print(f"  NON-MORAL layer7 = {nm_mc.get(7):.4f}  stable-band = {band_mean(nm_mc):.4f}  "
          f"eff_dim7 = {nm_ed.get(7)}")
    print("  per-concept... nonmoral peak acc:", out["nonmoral"]["peak_acc"])
    b = out["bootstrap_layer7"]
    print(f"\n  moral cos CI95   = [{b['moral_mean_cosine_CI95'][0]:.4f}, {b['moral_mean_cosine_CI95'][1]:.4f}] "
          f"(mean {b['moral_mean_cosine']:.4f}); PC1 {b['moral_pc1']:.4f} "
          f"CI[{b['moral_pc1_CI95'][0]:.3f},{b['moral_pc1_CI95'][1]:.3f}]")
    print(f"  nonmoral cos CI95= [{b['nonmoral_mean_cosine_CI95'][0]:.4f}, {b['nonmoral_mean_cosine_CI95'][1]:.4f}] "
          f"(mean {b['nonmoral_mean_cosine']:.4f})")
    print(f"  Δ(moral-nonmoral)= {b['delta_moral_minus_nonmoral']:.4f} "
          f"CI95[{b['delta_CI95'][0]:.4f},{b['delta_CI95'][1]:.4f}] excl0={b['delta_excludes_0']}")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
