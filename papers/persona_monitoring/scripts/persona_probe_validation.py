#!/usr/bin/env python3
"""Phase D C8: Persona-feature probe validation on OLMo-2 1B final checkpoint.

Tests H13 using the content-baseline-relative gate from ``RESEARCH_PLAN.md``:

  (a) Overall probe accuracy on all 240 persona pairs (stratified 80/20).
  (b) Content-only TF-IDF baseline from ``content_separability_baseline()``.
  (c) Probe trained on the content-clean subset (``get_content_clean_subset()``
      — villain_quote + instructed_roleplay, both near-chance TF-IDF floors)
      and evaluated on the four content-leaky categories for transfer.

H13 passes when:
  Gate 1: (a) peak-layer accuracy ≥ (b) overall baseline + 15 pp.
  Gate 2: mean content-clean→leaky transfer accuracy > 0.50 (chance).

Also runs an OOD transfer check against ``PERSONA_HELDOUT_JAILBREAK`` (chat-
format rule-bypass framings) as a generalization diagnostic — not a gate, but
it informs how to frame the probe's scope going forward.

Target model: ``allenai/OLMo-2-0425-1B`` (full base, ~2.2T tokens)
Hardware: MacBook Pro M4 Pro, MPS
Estimated runtime: ~10 min (≈560 forward passes; probes are negligible)

Outputs (``papers/persona_monitoring/outputs/phase_d/c8/``):
  - ``c8_results.json``: all numeric results, per-layer scores, verdict
  - ``c8_layer_accuracy.png``: per-layer curves (overall, content-clean, transfer)
  - ``c8_baseline_vs_probe.png``: per-category TF-IDF vs peak probe accuracy
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

REPO_ID = "allenai/OLMo-2-0425-1B"

# H13 gate thresholds — from RESEARCH_PLAN.md.
GATE1_MIN_DELTA_PP = 15.0
GATE2_CHANCE = 0.50

# Leaky categories used as held-out transfer targets in (c).
LEAKY_CATEGORIES = (
    "con_artist_quote",
    "cynical_narrator_aside",
    "sarcastic_advice",
    "unreliable_confession",
)


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _stratified_split_for_categories(
    categories: Sequence[str],
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Stratified 80/20 split restricted to a subset of persona categories."""
    from deepsteer.datasets.persona_pairs import get_persona_pairs_by_category

    rng = random.Random(seed)
    train: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    for cat in categories:
        pairs = list(get_persona_pairs_by_category(cat))
        rng.shuffle(pairs)
        n_test = max(1, int(len(pairs) * test_fraction))
        test.extend(pairs[:n_test])
        train.extend(pairs[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _build_layer_tensors(
    cache: dict[str, dict[int, Tensor]],
    pairs: list[tuple[str, str]],
    layer: int,
) -> tuple[Tensor, Tensor]:
    feats: list[Tensor] = []
    labels: list[float] = []
    for pos, neg in pairs:
        feats.append(cache[pos][layer])
        labels.append(1.0)
        feats.append(cache[neg][layer])
        labels.append(0.0)
    X = torch.stack(feats).float()
    y = torch.tensor(labels, dtype=torch.float32)
    return X, y


def train_and_evaluate_on_many(
    cache: dict[str, dict[int, Tensor]],
    train_pairs: list[tuple[str, str]],
    test_sets: dict[str, list[tuple[str, str]]],
    n_layers: int,
    *,
    n_epochs: int = 50,
    lr: float = 1e-2,
    seed: int = 42,
) -> dict[str, dict[int, float]]:
    """Train a linear probe per layer on ``train_pairs``; evaluate on many sets.

    Uses the same methodology as :class:`GeneralLinearProbe` (BCE loss, Adam,
    50 epochs, fp32 probes) but trains once per layer and reuses the trained
    probe across every test set — this is the primitive needed for transfer
    evaluation.  Seeds probe init and optimizer state for reproducibility.

    Returns:
        ``{test_set_name: {layer_idx: accuracy}}``.
    """
    results: dict[str, dict[int, float]] = {name: {} for name in test_sets}

    for layer in range(n_layers):
        X_train, y_train = _build_layer_tensors(cache, train_pairs, layer)

        torch.manual_seed(seed + layer)
        probe = nn.Linear(X_train.shape[1], 1)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        probe.train()
        for _ in range(n_epochs):
            logits = probe(X_train).squeeze(-1)
            loss = loss_fn(logits, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            for name, pairs in test_sets.items():
                if not pairs:
                    continue
                X_test, y_test = _build_layer_tensors(cache, pairs, layer)
                logits = probe(X_test).squeeze(-1)
                preds = (logits > 0).float()
                acc = (preds == y_test).float().mean().item()
                results[name][layer] = acc

    return results


def _peak_layer(per_layer: dict[int, float]) -> tuple[int, float]:
    peak_layer, peak_acc = 0, 0.0
    for layer, acc in per_layer.items():
        if acc > peak_acc:
            peak_layer, peak_acc = layer, acc
    return peak_layer, peak_acc


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _plot_layer_accuracy(
    n_layers: int,
    overall_per_layer: dict[int, float],
    within_subset_per_layer: dict[int, float],
    transfer_per_category: dict[str, dict[int, float]],
    heldout_per_layer: dict[int, float] | None,
    output_dir: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(range(n_layers))

    fig, ax = plt.subplots(figsize=(11, 6))

    overall_y = [overall_per_layer.get(L, float("nan")) for L in layers]
    ax.plot(layers, overall_y, "o-", color="#1f77b4", linewidth=2.5, markersize=6,
            label="(a) Overall probe on all 240 pairs")

    within_y = [within_subset_per_layer.get(L, float("nan")) for L in layers]
    ax.plot(layers, within_y, "s--", color="#2ca02c", linewidth=2, markersize=5,
            label="(c) Content-clean held-out")

    cat_colors = {
        "con_artist_quote": "#ff7f0e",
        "cynical_narrator_aside": "#d62728",
        "sarcastic_advice": "#9467bd",
        "unreliable_confession": "#8c564b",
    }
    for cat, per_layer in transfer_per_category.items():
        y = [per_layer.get(L, float("nan")) for L in layers]
        ax.plot(layers, y, ":", color=cat_colors.get(cat, "#7f7f7f"),
                linewidth=1.3, alpha=0.9, label=f"transfer → {cat}")

    if heldout_per_layer is not None:
        y = [heldout_per_layer.get(L, float("nan")) for L in layers]
        ax.plot(layers, y, "x-.", color="#e377c2", linewidth=1.3, alpha=0.85,
                label="OOD → PERSONA_HELDOUT_JAILBREAK")

    ax.axhline(GATE2_CHANCE, color="#666", linestyle=":", linewidth=1,
               label=f"chance ({GATE2_CHANCE:.2f})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Phase D C8 — persona probe per-layer accuracy\n"
                 "OLMo-2 1B final checkpoint")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=1)

    fig.tight_layout()
    path = output_dir / "c8_layer_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_baseline_vs_probe(
    baseline_by_category: dict[str, float],
    probe_peak_by_category: dict[str, float],
    gate1_required: float,
    probe_peak_overall: float,
    baseline_overall: float,
    output_dir: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from deepsteer.datasets.persona_pairs import PERSONA_CATEGORIES
    categories = [name for name, _, _ in PERSONA_CATEGORIES]
    x = list(range(len(categories)))
    baseline_vals = [baseline_by_category.get(c, 0.0) for c in categories]
    probe_vals = [probe_peak_by_category.get(c, 0.0) for c in categories]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.38
    ax.bar([xi - width / 2 for xi in x], baseline_vals, width,
           label="TF-IDF content baseline", color="#bbbbbb", edgecolor="black")
    ax.bar([xi + width / 2 for xi in x], probe_vals, width,
           label="Probe peak-layer accuracy", color="#1f77b4", edgecolor="black")

    ax.axhline(gate1_required, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"Gate 1 overall threshold ({gate1_required:.3f})")
    ax.axhline(baseline_overall, color="#888", linestyle=":", linewidth=1,
               label=f"Overall baseline ({baseline_overall:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"Phase D C8 — per-category TF-IDF vs persona probe\n"
        f"Overall probe peak = {probe_peak_overall:.3f} ; baseline = {baseline_overall:.3f} "
        f"; Δ = {(probe_peak_overall - baseline_overall) * 100:.1f} pp"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    path = output_dir / "c8_baseline_vs_probe.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase D C8: persona probe validation on OLMo-2 1B final.",
    )
    parser.add_argument("--output-dir", default="papers/persona_monitoring/outputs/phase_d/c8",
                        help="Output directory (default: papers/persona_monitoring/outputs/phase_d/c8).")
    parser.add_argument("--device", default=None,
                        help="Device (cuda, mps, cpu). Auto-detected if omitted.")
    parser.add_argument("--revision", default=None,
                        help="Optional HF revision (default: main branch = final).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Split + probe seed (default: 42).")
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="Per-layer probe training epochs (default: 50).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from deepsteer.benchmarks.representational.general_probe import (
        collect_activations_batch,
    )
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.datasets.persona_pairs import (
        CONTENT_CLEAN_CATEGORIES,
        PERSONA_CATEGORIES,
        content_separability_baseline,
        get_heldout_jailbreak_pairs,
        get_persona_pairs_by_category,
    )

    # --- 1. Load model ----------------------------------------------------
    print(f"Loading {REPO_ID} ...")
    t0 = time.time()
    model = WhiteBoxModel(REPO_ID, revision=args.revision, device=args.device)
    load_s = time.time() - t0
    n_layers = model.info.n_layers
    print(f"  loaded in {load_s:.1f}s, n_layers={n_layers}")

    # --- 2. Collect activations for every text we'll need ---------------
    # Build per-category 80/20 splits so we can (i) aggregate into an
    # overall split and (ii) evaluate the overall-trained probe on the
    # held-out portion of each category without training leakage.  Using
    # the same seed as get_persona_dataset produces the same splits as the
    # default stratified dataset, just with category tracking retained.
    all_cat_names = [name for name, _, _ in PERSONA_CATEGORIES]
    per_cat_splits: dict[str, tuple[list[tuple[str, str]], list[tuple[str, str]]]] = {}
    for name in all_cat_names:
        per_cat_splits[name] = _stratified_split_for_categories(
            [name], seed=args.seed,
        )

    overall_train: list[tuple[str, str]] = []
    overall_test: list[tuple[str, str]] = []
    for name in all_cat_names:
        tr, te = per_cat_splits[name]
        overall_train.extend(tr)
        overall_test.extend(te)
    _rng = random.Random(args.seed)
    _rng.shuffle(overall_train)
    _rng.shuffle(overall_test)

    # Content-clean train / held-out: train from the train portions of the
    # two clean categories only; evaluate within-subset on their test
    # portions.  The four leaky categories stay entirely untouched by this
    # probe's training, making them a clean transfer target.
    cc_train: list[tuple[str, str]] = []
    cc_test: list[tuple[str, str]] = []
    for name in CONTENT_CLEAN_CATEGORIES:
        tr, te = per_cat_splits[name]
        cc_train.extend(tr)
        cc_test.extend(te)
    _rng.shuffle(cc_train)
    _rng.shuffle(cc_test)

    leaky_sets: dict[str, list[tuple[str, str]]] = {
        cat: list(get_persona_pairs_by_category(cat))
        for cat in LEAKY_CATEGORIES
    }
    heldout_pairs = get_heldout_jailbreak_pairs()

    all_texts: list[str] = []
    for pos, neg in (overall_train + overall_test):
        all_texts.extend([pos, neg])
    for pairs in leaky_sets.values():
        for pos, neg in pairs:
            all_texts.extend([pos, neg])
    for pos, neg in heldout_pairs:
        all_texts.extend([pos, neg])
    unique_texts = sorted(set(all_texts))
    print(f"Collecting activations for {len(unique_texts)} unique texts ...")

    t0 = time.time()
    cache = collect_activations_batch(model, unique_texts)
    collect_s = time.time() - t0
    print(f"  collected in {collect_s/60:.1f} min")

    del model
    _clear_memory()

    # --- 3. (a) Overall probe on all 240 pairs --------------------------
    print("\n(a) Training overall persona probe on all 240 pairs ...")
    t0 = time.time()
    overall_results = train_and_evaluate_on_many(
        cache,
        train_pairs=overall_train,
        test_sets={"overall_test": overall_test},
        n_layers=n_layers,
        n_epochs=args.n_epochs,
        seed=args.seed,
    )
    overall_per_layer = overall_results["overall_test"]
    peak_layer_overall, peak_acc_overall = _peak_layer(overall_per_layer)
    mean_acc_overall = _mean(list(overall_per_layer.values()))
    print(f"  peak = {peak_acc_overall:.3f} @ layer {peak_layer_overall}; "
          f"mean = {mean_acc_overall:.3f} ({time.time()-t0:.1f}s)")

    # --- 4. (b) Content-only TF-IDF baseline ----------------------------
    print("\n(b) Computing TF-IDF content-only baseline ...")
    t0 = time.time()
    baseline = content_separability_baseline()
    baseline_overall = baseline["overall"]
    print(f"  overall baseline = {baseline_overall:.3f} "
          f"({time.time()-t0:.1f}s)")
    for name, _, _ in PERSONA_CATEGORIES:
        print(f"    {name}: {baseline[name]:.3f}")

    # --- 5. (c) Content-clean subset probe + transfer to leaky ----------
    print("\n(c) Training probe on content-clean subset "
          "(villain_quote + instructed_roleplay) and transferring ...")
    cc_test_sets: dict[str, list[tuple[str, str]]] = {
        "content_clean_heldout": cc_test,
        **leaky_sets,
        "ood_jailbreak": heldout_pairs,
    }
    t0 = time.time()
    cc_results = train_and_evaluate_on_many(
        cache,
        train_pairs=cc_train,
        test_sets=cc_test_sets,
        n_layers=n_layers,
        n_epochs=args.n_epochs,
        seed=args.seed,
    )
    within_layer = cc_results["content_clean_heldout"]
    peak_layer_cc, peak_acc_cc = _peak_layer(within_layer)
    print(f"  content-clean held-out peak = {peak_acc_cc:.3f} "
          f"@ layer {peak_layer_cc} ({time.time()-t0:.1f}s)")

    # Per-leaky-category peak accuracy from the content-clean-trained probe.
    transfer_peak_by_category: dict[str, float] = {}
    transfer_peak_layer_by_category: dict[str, int] = {}
    for cat in LEAKY_CATEGORIES:
        pl, pa = _peak_layer(cc_results[cat])
        transfer_peak_by_category[cat] = pa
        transfer_peak_layer_by_category[cat] = pl
        print(f"  transfer → {cat}: peak = {pa:.3f} @ layer {pl}")

    mean_transfer = _mean(list(transfer_peak_by_category.values()))
    print(f"  mean content-clean→leaky transfer peak = {mean_transfer:.3f}")

    # OOD jailbreak transfer (diagnostic, not a gate).
    heldout_layer = cc_results["ood_jailbreak"]
    peak_layer_ood, peak_acc_ood = _peak_layer(heldout_layer)
    print(f"  OOD → PERSONA_HELDOUT_JAILBREAK: peak = {peak_acc_ood:.3f} "
          f"@ layer {peak_layer_ood}")

    # --- 6. Per-category probe peak (for figure) ------------------------
    # Evaluate the overall-trained probe on each category's held-out test
    # portion (8 pairs / 16 examples per category).  Resolution is coarse
    # (6.25 pp per example) but the numbers are unbiased — no training
    # leakage.  This is a presentation aid, not a gate.
    print("\nPer-category probe peak on held-out test pairs "
          "(8 pairs = 16 examples each) ...")
    per_cat_heldout_sets = {
        name: per_cat_splits[name][1] for name in all_cat_names
    }
    per_cat_results = train_and_evaluate_on_many(
        cache,
        train_pairs=overall_train,
        test_sets=per_cat_heldout_sets,
        n_layers=n_layers,
        n_epochs=args.n_epochs,
        seed=args.seed,
    )
    probe_peak_by_category: dict[str, float] = {}
    for name in all_cat_names:
        _, pa = _peak_layer(per_cat_results[name])
        probe_peak_by_category[name] = pa
        print(f"  {name}: peak probe = {pa:.3f}, baseline = {baseline[name]:.3f}, "
              f"Δ = {(pa - baseline[name]) * 100:+.1f} pp")

    # --- 7. H13 verdict --------------------------------------------------
    delta_pp = (peak_acc_overall - baseline_overall) * 100
    gate1_pass = delta_pp >= GATE1_MIN_DELTA_PP
    gate2_pass = mean_transfer > GATE2_CHANCE
    h13_pass = gate1_pass and gate2_pass

    print("\n" + "=" * 60)
    print("H13 verdict")
    print("=" * 60)
    print(f"Gate 1 (overall probe vs TF-IDF baseline):")
    print(f"  peak probe = {peak_acc_overall:.3f}")
    print(f"  TF-IDF baseline = {baseline_overall:.3f}")
    print(f"  Δ = {delta_pp:+.1f} pp "
          f"(required ≥ {GATE1_MIN_DELTA_PP:.1f} pp) — "
          f"{'PASS' if gate1_pass else 'FAIL'}")
    print(f"Gate 2 (content-clean→leaky transfer):")
    print(f"  mean transfer peak = {mean_transfer:.3f} "
          f"(required > {GATE2_CHANCE:.2f}) — "
          f"{'PASS' if gate2_pass else 'FAIL'}")
    print(f"\nH13: {'PASS' if h13_pass else 'FAIL'}")
    print("=" * 60)

    # --- 8. Save JSON + plots -------------------------------------------
    results: dict = {
        "experiment": "C8",
        "hypothesis": "H13",
        "model": REPO_ID,
        "revision": args.revision,
        "n_layers": n_layers,
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "gates": {
            "gate1_min_delta_pp": GATE1_MIN_DELTA_PP,
            "gate2_chance": GATE2_CHANCE,
        },
        "a_overall_probe": {
            "peak_accuracy": round(peak_acc_overall, 4),
            "peak_layer": peak_layer_overall,
            "mean_accuracy": round(mean_acc_overall, 4),
            "per_layer": {str(L): round(v, 4) for L, v in overall_per_layer.items()},
            "train_pairs": len(overall_train),
            "test_pairs": len(overall_test),
        },
        "b_tfidf_baseline": {
            "overall": round(baseline_overall, 4),
            "per_category": {k: round(v, 4) for k, v in baseline.items() if k != "overall"},
        },
        "c_content_clean_transfer": {
            "train_categories": list(CONTENT_CLEAN_CATEGORIES),
            "train_pairs": len(cc_train),
            "within_subset_heldout": {
                "peak_accuracy": round(peak_acc_cc, 4),
                "peak_layer": peak_layer_cc,
                "test_pairs": len(cc_test),
                "per_layer": {str(L): round(v, 4) for L, v in within_layer.items()},
            },
            "transfer_to_leaky": {
                cat: {
                    "peak_accuracy": round(transfer_peak_by_category[cat], 4),
                    "peak_layer": transfer_peak_layer_by_category[cat],
                    "per_layer": {
                        str(L): round(v, 4) for L, v in cc_results[cat].items()
                    },
                }
                for cat in LEAKY_CATEGORIES
            },
            "mean_transfer_peak": round(mean_transfer, 4),
        },
        "ood_jailbreak_transfer": {
            "peak_accuracy": round(peak_acc_ood, 4),
            "peak_layer": peak_layer_ood,
            "test_pairs": len(heldout_pairs),
            "per_layer": {str(L): round(v, 4) for L, v in heldout_layer.items()},
        },
        "per_category_probe_peak": {
            cat: round(v, 4) for cat, v in probe_peak_by_category.items()
        },
        "verdict": {
            "gate1_delta_pp": round(delta_pp, 2),
            "gate1_pass": gate1_pass,
            "gate2_mean_transfer": round(mean_transfer, 4),
            "gate2_pass": gate2_pass,
            "h13_pass": h13_pass,
        },
        "elapsed_seconds": {
            "load_model": round(load_s, 1),
            "collect_activations": round(collect_s, 1),
        },
    }
    json_path = output_dir / "c8_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults JSON: {json_path}")

    # Plots
    transfer_per_category_layers: dict[str, dict[int, float]] = {
        cat: cc_results[cat] for cat in LEAKY_CATEGORIES
    }
    fig1 = _plot_layer_accuracy(
        n_layers=n_layers,
        overall_per_layer=overall_per_layer,
        within_subset_per_layer=within_layer,
        transfer_per_category=transfer_per_category_layers,
        heldout_per_layer=heldout_layer,
        output_dir=output_dir,
    )
    print(f"Figure: {fig1}")

    fig2 = _plot_baseline_vs_probe(
        baseline_by_category={
            k: v for k, v in baseline.items() if k != "overall"
        },
        probe_peak_by_category=probe_peak_by_category,
        gate1_required=baseline_overall + GATE1_MIN_DELTA_PP / 100.0,
        probe_peak_overall=peak_acc_overall,
        baseline_overall=baseline_overall,
        output_dir=output_dir,
    )
    print(f"Figure: {fig2}")


if __name__ == "__main__":
    main()
