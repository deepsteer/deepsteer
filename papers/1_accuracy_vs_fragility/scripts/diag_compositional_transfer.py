#!/usr/bin/env python3
"""B-diagnostic: does compositional moral encoding generalize across constructions?

Leave-construction-out transfer at the OLMo-2 1B final checkpoint. The four
construction categories (motive / target / consequence / role_reversal) share
almost no contrast tokens, so a probe trained on three and tested on the held-out
fourth cannot rely on construction-specific lexical cues. If the hidden-state
probe transfers (reads moral valence on an unseen construction) while a TF-IDF
bag-of-words probe collapses, the model encodes something construction-general:
evidence of compositional moral encoding beyond lexical lookup.

Compares three things per held-out construction:
  - TF-IDF transfer (bag-of-words; fit on train constructions only)
  - hidden-state probe transfer (best layer + per-layer profile)
  - in-distribution reference (pair-disjoint CV on all four, best layer)

Writes JSON + a console summary. Minutes of MPS at one checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from deepsteer.benchmarks.representational.general_probe import collect_activations_batch
from deepsteer.core.model_interface import WhiteBoxModel
from deepsteer.datasets.compositional_moral_pairs import (
    COMPOSITIONAL_CATEGORIES,
    COMPOSITIONAL_MORAL_PAIRS,
)

FINAL_REPO_ID = "allenai/OLMo-2-0425-1B"
OUT = Path("papers/1_accuracy_vs_fragility/outputs/phase_c4_compositional/b_diag_transfer.json")


def build_data() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return (sentences, labels, category-index, pair-index)."""
    sents: list[str] = []
    labels: list[int] = []
    cats: list[int] = []
    pair_ids: list[int] = []
    pid = 0
    for ci, (_name, start, end) in enumerate(COMPOSITIONAL_CATEGORIES):
        for moral, immoral in COMPOSITIONAL_MORAL_PAIRS[start:end]:
            sents.append(moral)
            labels.append(1)
            cats.append(ci)
            pair_ids.append(pid)
            sents.append(immoral)
            labels.append(0)
            cats.append(ci)
            pair_ids.append(pid)
            pid += 1
    return sents, np.array(labels), np.array(cats), np.array(pair_ids)


def tfidf_transfer(sents, labels, cats) -> dict[str, float]:
    """Leave-construction-out TF-IDF transfer. Fit vectorizer on train only."""
    out = {}
    for ci, (name, _, _) in enumerate(COMPOSITIONAL_CATEGORIES):
        tr = cats != ci
        te = cats == ci
        train_texts = [sents[i] for i in range(len(sents)) if tr[i]]
        test_texts = [sents[i] for i in range(len(sents)) if te[i]]
        vec = TfidfVectorizer(ngram_range=(1, 1), min_df=2)
        Xtr = vec.fit_transform(train_texts)
        Xte = vec.transform(test_texts)
        clf = LogisticRegression(max_iter=1000).fit(Xtr, labels[tr])
        acc = float(clf.score(Xte, labels[te]))
        out[name] = round(acc, 3)
    return out


def hidden_transfer(cache, sents, labels, cats, layers) -> dict:
    """Leave-construction-out hidden-state probe transfer, per layer."""
    # Pre-stack per-layer feature matrices.
    feats = {
        ly: np.stack([cache[s][ly].float().cpu().numpy() for s in sents])
        for ly in layers
    }
    per_cat = {}
    for ci, (name, _, _) in enumerate(COMPOSITIONAL_CATEGORIES):
        tr = cats != ci
        te = cats == ci
        layer_accs = {}
        for ly in layers:
            scaler = StandardScaler().fit(feats[ly][tr])
            Xtr = scaler.transform(feats[ly][tr])
            Xte = scaler.transform(feats[ly][te])
            clf = LogisticRegression(max_iter=1000).fit(Xtr, labels[tr])
            layer_accs[ly] = float(clf.score(Xte, labels[te]))
        best_ly = max(layer_accs, key=layer_accs.get)
        per_cat[name] = {
            "best_layer": int(best_ly),
            "best_acc": round(layer_accs[best_ly], 3),
            "mean_acc": round(float(np.mean(list(layer_accs.values()))), 3),
            "per_layer": {int(k): round(v, 3) for k, v in layer_accs.items()},
        }
    return per_cat


def in_distribution(cache, sents, labels, pair_ids, layers) -> dict:
    """Pair-disjoint CV on all four constructions (reference ceiling)."""
    feats = {
        ly: np.stack([cache[s][ly].float().cpu().numpy() for s in sents])
        for ly in layers
    }
    layer_accs = {}
    for ly in layers:
        Xs = StandardScaler().fit_transform(feats[ly])
        pred = cross_val_predict(
            LogisticRegression(max_iter=1000), Xs, labels,
            cv=GroupKFold(5), groups=pair_ids,
        )
        layer_accs[ly] = float((pred == labels).mean())
    best_ly = max(layer_accs, key=layer_accs.get)
    return {"best_layer": int(best_ly), "best_acc": round(layer_accs[best_ly], 3)}


def within_construction(cache, sents, labels, cats, pair_ids, layers) -> dict:
    """Within-category pair-disjoint hidden CV vs the category's TF-IDF floor.

    The decisive lexical-confound check: role_reversal puts the same components
    on both sides (TF-IDF floor 0.57). If hidden states decode it well above
    0.57, the model is reading context, not the contrast tokens.
    """
    feats = {
        ly: np.stack([cache[s][ly].float().cpu().numpy() for s in sents])
        for ly in layers
    }
    out = {}
    for ci, (name, _, _) in enumerate(COMPOSITIONAL_CATEGORIES):
        m = cats == ci
        y = labels[m]
        g = pair_ids[m]
        # TF-IDF floor within this construction (pair-disjoint, orient-invariant).
        cat_texts = [sents[i] for i in range(len(sents)) if m[i]]
        Xt = TfidfVectorizer(ngram_range=(1, 1), min_df=1).fit_transform(cat_texts)
        pt = cross_val_predict(LogisticRegression(max_iter=1000), Xt, y,
                               cv=GroupKFold(5), groups=g)
        at = float((pt == y).mean())
        tfidf_floor = round(max(at, 1 - at), 3)
        # Hidden-state within-category CV, best layer.
        layer_accs = {}
        for ly in layers:
            Xs = StandardScaler().fit_transform(feats[ly][m])
            pred = cross_val_predict(LogisticRegression(max_iter=1000), Xs, y,
                                     cv=GroupKFold(5), groups=g)
            layer_accs[ly] = float((pred == y).mean())
        best_ly = max(layer_accs, key=layer_accs.get)
        out[name] = {
            "tfidf_floor": tfidf_floor,
            "hidden_best_acc": round(layer_accs[best_ly], 3),
            "hidden_best_layer": int(best_ly),
            "lift_over_lexical": round(layer_accs[best_ly] - tfidf_floor, 3),
        }
    return out


def main() -> None:
    sents, labels, cats, pair_ids = build_data()
    print(f"{len(sents)} sentences across {len(COMPOSITIONAL_CATEGORIES)} constructions")

    model = WhiteBoxModel(FINAL_REPO_ID)
    cache = collect_activations_batch(model, sents)
    layers = sorted(next(iter(cache.values())).keys())

    # Cache activations to disk (sentence-ordered) for reuse by follow-ups.
    cache_npz = OUT.parent / "b_diag_activations.npz"
    acts = np.stack([
        np.stack([cache[s][ly].float().cpu().numpy() for ly in layers]) for s in sents
    ])  # (n_sent, n_layer, hidden)
    np.savez_compressed(cache_npz, acts=acts, labels=labels, cats=cats,
                        pair_ids=pair_ids, layers=np.array(layers))

    tfidf = tfidf_transfer(sents, labels, cats)
    hidden = hidden_transfer(cache, sents, labels, cats, layers)
    indist = in_distribution(cache, sents, labels, pair_ids, layers)
    within = within_construction(cache, sents, labels, cats, pair_ids, layers)

    result = {
        "model": FINAL_REPO_ID,
        "n_sentences": len(sents),
        "n_layers": len(layers),
        "tfidf_transfer": tfidf,
        "hidden_transfer": hidden,
        "in_distribution_cv": indist,
        "within_construction": within,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    print("\n=== Leave-construction-out transfer (test = held-out construction) ===")
    print(f"{'held-out construction':<22}{'TF-IDF':>9}{'hidden(best)':>14}{'best layer':>12}")
    for name, _, _ in COMPOSITIONAL_CATEGORIES:
        print(f"{name:<22}{tfidf[name]:>9.3f}{hidden[name]['best_acc']:>14.3f}"
              f"{hidden[name]['best_layer']:>12}")
    tfidf_mean = float(np.mean(list(tfidf.values())))
    hidden_mean = float(np.mean([hidden[n]['best_acc'] for n, _, _ in COMPOSITIONAL_CATEGORIES]))
    print(f"{'MEAN':<22}{tfidf_mean:>9.3f}{hidden_mean:>14.3f}")
    print(f"\nin-distribution pair-disjoint CV (reference): "
          f"{indist['best_acc']:.3f} @ layer {indist['best_layer']}")

    print("\n=== Within-construction: hidden CV vs TF-IDF floor (lexical-confound check) ===")
    print(f"{'construction':<22}{'TF-IDF floor':>13}{'hidden(best)':>14}{'lift':>8}")
    for name, _, _ in COMPOSITIONAL_CATEGORIES:
        w = within[name]
        print(f"{name:<22}{w['tfidf_floor']:>13.3f}{w['hidden_best_acc']:>14.3f}"
              f"{w['lift_over_lexical']:>+8.3f}")
    print(f"\nwrote: {OUT}")
    print(f"cached activations: {OUT.parent / 'b_diag_activations.npz'}")


if __name__ == "__main__":
    main()
