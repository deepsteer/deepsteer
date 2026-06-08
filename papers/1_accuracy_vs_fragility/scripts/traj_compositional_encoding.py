#!/usr/bin/env python3
"""Compositional moral ENCODING emergence trajectory across OLMo-2 1B early training.

Runs the leave-construction-out transfer + within-construction lift analysis
(see diag_compositional_transfer.py) at all 37 early-training checkpoints
(steps 0-36K). Answers WHEN compositional moral encoding emerges (transfer / lift
crossing) and WHERE (per-layer in-distribution profile over training).

The TF-IDF lexical floor is text-only (checkpoint-independent), computed once.
Per-checkpoint records are written to b_traj/step_XXXXXXX.json (resumable: an
existing step file is skipped). A combined b_traj_summary.json is rebuilt from
whatever step files exist after each checkpoint, so a partial/background run
always leaves a usable summary.

Designed to run in the background (~1h on M4 Pro / MPS).
"""

from __future__ import annotations

import gc
import json
import re
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from deepsteer.benchmarks.representational.general_probe import collect_activations_batch
from deepsteer.benchmarks.representational.trajectory import list_available_revisions
from deepsteer.core.model_interface import WhiteBoxModel
from deepsteer.datasets.compositional_moral_pairs import (
    COMPOSITIONAL_CATEGORIES,
    COMPOSITIONAL_MORAL_PAIRS,
)

REPO = "allenai/OLMo-2-0425-1B-early-training"
OUTDIR = Path("papers/1_accuracy_vs_fragility/outputs/phase_c4_compositional/b_traj")
CATS = [c[0] for c in COMPOSITIONAL_CATEGORIES]


def _clear() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _free_repo_cache(repo_id: str) -> None:
    """Remove the whole repo's HF cache dir after a checkpoint is processed.

    The 37 early-training checkpoints are ~2.7 GB each; downloading all at once
    fills the disk. We process one at a time and delete the repo cache here so
    peak disk stays at a single checkpoint (tiny config/tokenizer files
    re-download per step). Direct rmtree is used rather than
    ``scan_cache_dir().delete_revisions()`` because ref-based matching is
    fragile and silently no-ops when a repo is mid-write or corrupted.
    Re-downloadable on demand.
    """
    import shutil

    from huggingface_hub.constants import HF_HUB_CACHE

    repo_dir = Path(HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)
        print(f"    freed {repo_id} cache after {repo_dir.name}", flush=True)


def _pipe() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def build_data() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
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


def tfidf_floor(sents, labels, cats, pair_ids) -> dict[str, float]:
    """Pair-disjoint, orientation-invariant unigram floor per construction (text-only)."""
    out = {}
    for ci, name in enumerate(CATS):
        m = cats == ci
        texts = [sents[i] for i in range(len(sents)) if m[i]]
        X = TfidfVectorizer(ngram_range=(1, 1), min_df=1).fit_transform(texts)
        pred = cross_val_predict(LogisticRegression(max_iter=1000), X, labels[m],
                                 cv=GroupKFold(5), groups=pair_ids[m])
        a = float((pred == labels[m]).mean())
        out[name] = round(max(a, 1 - a), 3)
    return out


def analyze(feats, labels, cats, pair_ids, layers, floor) -> dict:
    """Transfer + within-construction lift + in-distribution profile for one checkpoint."""
    # Leave-construction-out transfer (per-construction best layer).
    transfer = {}
    for ci, name in enumerate(CATS):
        tr, te = cats != ci, cats == ci
        accs = {}
        for ly in layers:
            scaler = StandardScaler().fit(feats[ly][tr])
            clf = LogisticRegression(max_iter=1000).fit(scaler.transform(feats[ly][tr]), labels[tr])
            accs[ly] = float(clf.score(scaler.transform(feats[ly][te]), labels[te]))
        best = max(accs, key=accs.get)
        transfer[name] = {"best_acc": round(accs[best], 3), "best_layer": int(best)}

    # Within-construction pair-disjoint CV (best layer) and lift over lexical floor.
    within = {}
    for ci, name in enumerate(CATS):
        m = cats == ci
        accs = {}
        for ly in layers:
            pred = cross_val_predict(_pipe(), feats[ly][m], labels[m],
                                     cv=GroupKFold(5), groups=pair_ids[m])
            accs[ly] = float((pred == labels[m]).mean())
        best = max(accs, key=accs.get)
        within[name] = {
            "hidden_best_acc": round(accs[best], 3),
            "hidden_best_layer": int(best),
            "lift_over_lexical": round(accs[best] - floor[name], 3),
        }

    # In-distribution pair-disjoint CV, per layer (the WHERE profile).
    per_layer = {}
    for ly in layers:
        pred = cross_val_predict(_pipe(), feats[ly], labels,
                                 cv=GroupKFold(5), groups=pair_ids)
        per_layer[int(ly)] = round(float((pred == labels).mean()), 3)
    best_ly = max(per_layer, key=per_layer.get)

    transfer_mean = round(float(np.mean([transfer[n]["best_acc"] for n in CATS])), 3)
    lift_mean = round(float(np.mean([within[n]["lift_over_lexical"] for n in CATS])), 3)
    return {
        "transfer": transfer,
        "transfer_mean_best": transfer_mean,
        "within": within,
        "lift_mean": lift_mean,
        "indist_per_layer": per_layer,
        "indist_best_layer": int(best_ly),
        "indist_best_acc": per_layer[best_ly],
    }


def rebuild_summary() -> None:
    rows = []
    for f in sorted(OUTDIR.glob("step_*.json")):
        d = json.loads(f.read_text())
        rows.append({
            "step": d["step"],
            "transfer_mean_best": d["transfer_mean_best"],
            "lift_mean": d["lift_mean"],
            "indist_best_acc": d["indist_best_acc"],
            "indist_best_layer": d["indist_best_layer"],
            "transfer_role_reversal": d["transfer"]["role_reversal"]["best_acc"],
            "lift_role_reversal": d["within"]["role_reversal"]["lift_over_lexical"],
        })
    rows.sort(key=lambda r: r["step"])
    (OUTDIR / "b_traj_summary.json").write_text(json.dumps({"checkpoints": rows}, indent=2))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sents, labels, cats, pair_ids = build_data()
    floor = tfidf_floor(sents, labels, cats, pair_ids)
    print(f"{len(sents)} sentences; lexical floor per construction: {floor}", flush=True)

    revs = list_available_revisions(REPO)
    steps = sorted(
        (int(re.search(r"step(\d+)", r).group(1)), r)
        for r in revs if re.search(r"step(\d+)", r)
    )
    print(f"{len(steps)} checkpoints to process", flush=True)

    for step, revision in steps:
        out_f = OUTDIR / f"step_{step:07d}.json"
        if out_f.exists():
            print(f"[skip] step {step} (exists)", flush=True)
            continue
        t0 = time.time()
        print(f"\n=== step {step} ({revision}) ===", flush=True)
        try:
            model = WhiteBoxModel(REPO, revision=revision)
            cache = collect_activations_batch(model, sents)
            layers = sorted(next(iter(cache.values())).keys())
            feats = {
                ly: np.stack([cache[s][ly].float().cpu().numpy() for s in sents])
                for ly in layers
            }
            del model, cache
            _clear()
            _free_repo_cache(REPO)  # bound disk: drop weights after each checkpoint

            rec = analyze(feats, labels, cats, pair_ids, layers, floor)
            rec.update({"step": step, "revision": revision, "lexical_floor": floor})
            out_f.write_text(json.dumps(rec, indent=2))
            rebuild_summary()
            print(f"  step {step}: transfer={rec['transfer_mean_best']:.3f} "
                  f"lift={rec['lift_mean']:+.3f} indist={rec['indist_best_acc']:.3f}"
                  f"@L{rec['indist_best_layer']} role_rev_lift="
                  f"{rec['within']['role_reversal']['lift_over_lexical']:+.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        except Exception as exc:  # batch resilience: log and continue to next checkpoint
            print(f"  [ERROR] step {step}: {exc!r}", flush=True)
            _clear()

    rebuild_summary()
    print("\nDONE. Summary: papers/1_accuracy_vs_fragility/outputs/"
          "phase_c4_compositional/b_traj/b_traj_summary.json", flush=True)


if __name__ == "__main__":
    main()
