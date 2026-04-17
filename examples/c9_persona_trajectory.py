#!/usr/bin/env python3
"""Phase D C9: Persona-probe emergence trajectory across OLMo-2 1B early training.

Tests H14 — does the toxic-persona direction have a distinct emergence
trajectory from the moral/sentiment/syntax probes (C2), and does it
emerge early enough to be considered foundational to language modeling?

For each of 37 early-training checkpoints (step 0 to step 36000 at 1K
intervals), runs:

  - Overall persona probe on all 240 pairs (stratified 80/20).
  - Content-clean subset probe trained on villain_quote +
    instructed_roleplay (64 train / 16 held-out), evaluated on the four
    content-leaky categories for transfer.
  - OOD transfer to PERSONA_HELDOUT_JAILBREAK (optional, 40 pairs).

Target model: ``allenai/OLMo-2-0425-1B-early-training`` (1B, 16 layers)
Hardware: MacBook Pro M4 Pro, MPS
Estimated runtime: ~75-110 min for all 37 checkpoints.

Outputs (``outputs/phase_d/c9/``):
  - ``step_XXXXXXX/c9_step.json``: per-checkpoint probe results.
  - ``c9_results.json``: aggregated summary.
  - ``c9_heatmap.png``: Figure 13 — persona per-layer accuracy over steps.
  - ``c9_onset_comparison.png``: Figure 14 — moral/sentiment/syntax (from
    C2) vs overall-persona vs content-clean-persona onset curves.
  - ``c9_transfer_trajectory.png``: content-clean→leaky transfer curves.
  - ``c9_ood_trajectory.png``: OOD jailbreak transfer curve.
  - ``RESULTS.md``: interpretation.

Usage:
    python examples/c9_persona_trajectory.py
    python examples/c9_persona_trajectory.py --resume-from step18000
    python examples/c9_persona_trajectory.py --skip-ood  # skip PERSONA_HELDOUT_JAILBREAK
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

REPO_ID = "allenai/OLMo-2-0425-1B-early-training"
C2_OUTPUT_DIR = Path("outputs/phase_c2")

ONSET_ACCURACY = 0.70  # matches C2 so the comparison figure is apples-to-apples
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


def _parse_step(revision: str) -> int | None:
    match = re.search(r"step(\d+)", revision)
    return int(match.group(1)) if match else None


def _get_all_revisions() -> list[tuple[int, str]]:
    from deepsteer.benchmarks.representational.trajectory import list_available_revisions

    all_revisions = list_available_revisions(REPO_ID)
    step_revisions: list[tuple[int, str]] = []
    for rev in all_revisions:
        step = _parse_step(rev)
        if step is not None:
            step_revisions.append((step, rev))
    step_revisions.sort(key=lambda x: x[0])
    return step_revisions


def _stratified_split_for_categories(
    categories: Sequence[str],
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
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
    return torch.stack(feats).float(), torch.tensor(labels, dtype=torch.float32)


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
    """Train a linear probe per layer; evaluate each probe on every test set.

    Same methodology as :class:`GeneralLinearProbe` (BCE, Adam, 50 epochs,
    fp32) but trains once and reuses across test sets — needed for the
    transfer evaluation.  Duplicated from ``c8_persona_validation.py``
    to keep examples self-contained.
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


def _peak(per_layer: dict[int, float]) -> tuple[int, float]:
    best_layer, best_acc = 0, 0.0
    for L, a in per_layer.items():
        if a > best_acc:
            best_layer, best_acc = L, a
    return best_layer, best_acc


def _mean(per_layer: dict[int, float]) -> float:
    vs = list(per_layer.values())
    return float(sum(vs) / len(vs)) if vs else 0.0


def run_probes_on_checkpoint(
    model,
    overall_train: list[tuple[str, str]],
    overall_test: list[tuple[str, str]],
    cc_train: list[tuple[str, str]],
    cc_test: list[tuple[str, str]],
    leaky_sets: dict[str, list[tuple[str, str]]],
    heldout_pairs: list[tuple[str, str]] | None,
    step: int,
    output_dir: Path,
    *,
    seed: int,
    n_epochs: int,
) -> dict:
    """Run the full C9 probe suite on a single loaded checkpoint."""
    from deepsteer.benchmarks.representational.general_probe import (
        collect_activations_batch,
    )

    n_layers = model.info.n_layers
    assert n_layers is not None

    # Collect activations once across all texts this checkpoint will probe.
    all_texts: list[str] = []
    for pairs in (overall_train, overall_test, cc_train, cc_test):
        for pos, neg in pairs:
            all_texts.extend([pos, neg])
    for pairs in leaky_sets.values():
        for pos, neg in pairs:
            all_texts.extend([pos, neg])
    if heldout_pairs:
        for pos, neg in heldout_pairs:
            all_texts.extend([pos, neg])
    unique_texts = sorted(set(all_texts))

    t0 = time.time()
    cache = collect_activations_batch(model, unique_texts)
    act_s = time.time() - t0
    logger.info("  activations: %.1fs for %d unique texts", act_s, len(unique_texts))

    # Overall probe on all 240 pairs.
    t0 = time.time()
    overall_out = train_and_evaluate_on_many(
        cache,
        train_pairs=overall_train,
        test_sets={"overall_test": overall_test},
        n_layers=n_layers,
        n_epochs=n_epochs,
        seed=seed,
    )
    overall_per_layer = overall_out["overall_test"]
    peak_layer_o, peak_acc_o = _peak(overall_per_layer)
    mean_o = _mean(overall_per_layer)
    t_overall = time.time() - t0

    # Content-clean subset probe + transfer to leaky + OOD.
    t0 = time.time()
    cc_test_sets: dict[str, list[tuple[str, str]]] = {
        "cc_heldout": cc_test,
        **leaky_sets,
    }
    if heldout_pairs:
        cc_test_sets["ood_jailbreak"] = heldout_pairs
    cc_out = train_and_evaluate_on_many(
        cache,
        train_pairs=cc_train,
        test_sets=cc_test_sets,
        n_layers=n_layers,
        n_epochs=n_epochs,
        seed=seed,
    )
    peak_layer_cc, peak_acc_cc = _peak(cc_out["cc_heldout"])
    mean_cc = _mean(cc_out["cc_heldout"])

    transfer_peaks: dict[str, float] = {}
    transfer_peak_layers: dict[str, int] = {}
    for cat in LEAKY_CATEGORIES:
        pl, pa = _peak(cc_out[cat])
        transfer_peaks[cat] = pa
        transfer_peak_layers[cat] = pl
    mean_transfer = float(np.mean(list(transfer_peaks.values())))

    if heldout_pairs:
        peak_layer_ood, peak_acc_ood = _peak(cc_out["ood_jailbreak"])
    else:
        peak_layer_ood, peak_acc_ood = -1, float("nan")
    t_cc = time.time() - t0

    logger.info(
        "  overall: peak=%.3f@L%d mean=%.3f (%.1fs) | cc: peak=%.3f@L%d "
        "mean_transfer=%.3f ood=%.3f (%.1fs)",
        peak_acc_o, peak_layer_o, mean_o, t_overall,
        peak_acc_cc, peak_layer_cc, mean_transfer, peak_acc_ood, t_cc,
    )

    step_data = {
        "step": step,
        "activation_time_s": round(act_s, 1),
        "overall": {
            "peak_accuracy": round(peak_acc_o, 4),
            "peak_layer": peak_layer_o,
            "mean_accuracy": round(mean_o, 4),
            "per_layer": {str(L): round(v, 4) for L, v in overall_per_layer.items()},
            "train_pairs": len(overall_train),
            "test_pairs": len(overall_test),
            "elapsed_s": round(t_overall, 1),
        },
        "content_clean": {
            "train_categories": ["villain_quote", "instructed_roleplay"],
            "train_pairs": len(cc_train),
            "within_subset_peak": round(peak_acc_cc, 4),
            "within_subset_peak_layer": peak_layer_cc,
            "within_subset_mean": round(mean_cc, 4),
            "within_subset_per_layer": {
                str(L): round(v, 4) for L, v in cc_out["cc_heldout"].items()
            },
            "transfer_peaks": {c: round(transfer_peaks[c], 4) for c in LEAKY_CATEGORIES},
            "transfer_peak_layers": dict(transfer_peak_layers),
            "mean_transfer_peak": round(mean_transfer, 4),
            "transfer_per_layer": {
                c: {str(L): round(v, 4) for L, v in cc_out[c].items()}
                for c in LEAKY_CATEGORIES
            },
            "elapsed_s": round(t_cc, 1),
        },
        "ood_jailbreak": None if not heldout_pairs else {
            "peak_accuracy": round(peak_acc_ood, 4),
            "peak_layer": peak_layer_ood,
            "mean_accuracy": round(_mean(cc_out["ood_jailbreak"]), 4),
            "per_layer": {
                str(L): round(v, 4) for L, v in cc_out["ood_jailbreak"].items()
            },
            "test_pairs": len(heldout_pairs),
        },
    }

    step_dir = output_dir / f"step_{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    with open(step_dir / "c9_step.json", "w") as f:
        json.dump(step_data, f, indent=2)

    return step_data


def _reload_step(step: int, output_dir: Path) -> dict | None:
    step_json = output_dir / f"step_{step:07d}" / "c9_step.json"
    if not step_json.exists():
        return None
    with open(step_json) as f:
        return json.load(f)


def _load_c2_curves() -> dict[str, tuple[list[int], list[float]]] | None:
    """Load moral/sentiment/syntax mean-accuracy curves from phase_c2."""
    summary = C2_OUTPUT_DIR / "c2_emergence_timing.json"
    if not summary.exists():
        logger.warning(
            "C2 emergence-timing JSON not found at %s; skipping comparison overlay",
            summary,
        )
        return None
    with open(summary) as f:
        data = json.load(f)
    curves = data.get("curves", {})
    out: dict[str, tuple[list[int], list[float]]] = {}
    for name, body in curves.items():
        steps = list(body.get("steps", []))
        accs = list(body.get("mean_accuracies", []))
        if steps and accs:
            out[name] = (steps, accs)
    return out


def _first_step_above(
    steps: list[int], values: list[float], threshold: float,
) -> int | None:
    for s, v in zip(steps, values):
        if v >= threshold:
            return s
    return None


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def generate_figure_13(
    per_step: dict[int, dict],
    output_dir: Path,
) -> None:
    """Figure 13: persona per-layer accuracy heatmap across 37 checkpoints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    steps = sorted(per_step.keys())
    if not steps:
        return
    n_layers = len(per_step[steps[0]]["overall"]["per_layer"])

    matrix = np.zeros((n_layers, len(steps)))
    for col, s in enumerate(steps):
        per_layer = per_step[s]["overall"]["per_layer"]
        for L in range(n_layers):
            matrix[L, col] = per_layer[str(L)]

    step_labels = [f"{s // 1000}K" if s > 0 else "0" for s in steps]
    layer_labels = [str(i) for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(max(14, len(steps) * 0.5), 7))
    sns.heatmap(
        matrix, ax=ax,
        xticklabels=step_labels, yticklabels=layer_labels,
        cmap="RdYlGn", vmin=0.4, vmax=1.0, annot=False,
        cbar_kws={"label": "Persona Probing Accuracy (overall)"},
    )

    # Annotate inflection (max diff in mean accuracy between consecutive steps).
    means = matrix.mean(axis=0)
    if len(means) > 1:
        diffs = np.diff(means)
        inflection_idx = int(np.argmax(diffs)) + 1
        inflection_step = steps[inflection_idx]
        ax.axvline(x=inflection_idx + 0.5, color="white", linestyle="--",
                   linewidth=2, alpha=0.8)
        ax.text(
            inflection_idx + 0.5, -0.8,
            f"Inflection\n(step {inflection_step})",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(
        f"Figure 13: Persona-probe emergence heatmap — {REPO_ID}\n"
        f"(37 checkpoints, 1K-step intervals; overall probe on 240 pairs)",
        fontsize=12,
    )
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    path = output_dir / "c9_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Figure 13: %s", path)


def generate_figure_14(
    per_step: dict[int, dict],
    output_dir: Path,
) -> dict:
    """Figure 14: emergence-timing overlay of moral/sentiment/syntax/persona."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted(per_step.keys())
    overall_means = [per_step[s]["overall"]["mean_accuracy"] for s in steps]
    cc_means = [per_step[s]["content_clean"]["within_subset_mean"] for s in steps]

    c2_curves = _load_c2_curves()

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        "moral": "#1f77b4",
        "sentiment": "#2ca02c",
        "syntax": "#9467bd",
        "persona_overall": "#d62728",
        "persona_content_clean": "#ff7f0e",
    }

    # Plot C2 curves if available.
    onset_steps: dict[str, int | None] = {}
    if c2_curves:
        for name in ("moral", "sentiment", "syntax"):
            if name in c2_curves:
                s, a = c2_curves[name]
                ax.plot(s, a, "o-", color=colors[name], linewidth=2, markersize=4,
                        label=f"{name} (C2)")
                onset_steps[name] = _first_step_above(s, a, ONSET_ACCURACY)

    # Plot persona curves.
    ax.plot(steps, overall_means, "s-", color=colors["persona_overall"],
            linewidth=2.2, markersize=5,
            label="persona — overall (all 240 pairs)")
    onset_steps["persona_overall"] = _first_step_above(
        steps, overall_means, ONSET_ACCURACY,
    )

    ax.plot(steps, cc_means, "D--", color=colors["persona_content_clean"],
            linewidth=2, markersize=4,
            label="persona — content-clean held-out (signal-meaningful)")
    onset_steps["persona_content_clean"] = _first_step_above(
        steps, cc_means, ONSET_ACCURACY,
    )

    # Annotate onset lines.
    for name, s in onset_steps.items():
        if s is not None:
            ax.axvline(x=s, color=colors.get(name, "#555"),
                       linestyle=":", linewidth=1.2, alpha=0.7)

    ax.axhline(y=ONSET_ACCURACY, color="#9E9E9E", linestyle=":", linewidth=1,
               alpha=0.6, label=f"Onset threshold ({ONSET_ACCURACY:.0%})")
    ax.axhline(y=0.5, color="#CCCCCC", linestyle=":", linewidth=1, alpha=0.4,
               label="Chance (50%)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Probing Accuracy (all layers)")
    ax.set_title(
        "Figure 14: Emergence Timing — Moral / Sentiment / Syntax / Persona\n"
        f"{REPO_ID}",
    )
    ax.set_ylim(0.1, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    path = output_dir / "c9_onset_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Figure 14: %s", path)

    return {k: v for k, v in onset_steps.items() if v is not None}


def generate_transfer_trajectory(
    per_step: dict[int, dict],
    output_dir: Path,
) -> None:
    """Content-clean→leaky transfer peak per category, across training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted(per_step.keys())
    cat_series: dict[str, list[float]] = {c: [] for c in LEAKY_CATEGORIES}
    mean_series: list[float] = []
    for s in steps:
        cc = per_step[s]["content_clean"]
        for cat in LEAKY_CATEGORIES:
            cat_series[cat].append(cc["transfer_peaks"][cat])
        mean_series.append(cc["mean_transfer_peak"])

    colors = {
        "con_artist_quote": "#ff7f0e",
        "cynical_narrator_aside": "#d62728",
        "sarcastic_advice": "#9467bd",
        "unreliable_confession": "#8c564b",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    for cat, series in cat_series.items():
        ax.plot(steps, series, "o-", color=colors[cat], linewidth=1.5, markersize=4,
                label=f"transfer → {cat}")
    ax.plot(steps, mean_series, "s-", color="#1f77b4", linewidth=2.5, markersize=5,
            label="mean content-clean → leaky")
    ax.axhline(y=0.5, color="#888", linestyle=":", linewidth=1, alpha=0.7,
               label="chance (0.50)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Peak transfer accuracy")
    ax.set_title(
        "Content-clean → leaky persona transfer over training\n"
        f"{REPO_ID}",
    )
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = output_dir / "c9_transfer_trajectory.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Transfer trajectory: %s", path)


def generate_ood_trajectory(
    per_step: dict[int, dict],
    output_dir: Path,
) -> None:
    """OOD jailbreak transfer over training (content-clean-trained probe)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted(per_step.keys())
    ood_peaks: list[float] = []
    for s in steps:
        ood = per_step[s].get("ood_jailbreak")
        ood_peaks.append(ood["peak_accuracy"] if ood else float("nan"))
    if not any(not np.isnan(v) for v in ood_peaks):
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, ood_peaks, "o-", color="#e377c2", linewidth=2, markersize=5,
            label="OOD → PERSONA_HELDOUT_JAILBREAK (peak)")
    ax.axhline(y=0.5, color="#888", linestyle=":", linewidth=1, alpha=0.7,
               label="chance (0.50)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Peak OOD transfer accuracy")
    ax.set_title(
        "OOD jailbreak transfer over training (content-clean-trained probe)\n"
        f"{REPO_ID}",
    )
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = output_dir / "c9_ood_trajectory.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("OOD trajectory: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase D C9: persona probe trajectory across OLMo-2 1B.",
    )
    parser.add_argument("--output-dir", default="outputs/phase_d/c9",
                        help="Output directory.")
    parser.add_argument("--device", default=None,
                        help="Device (cuda, mps, cpu). Auto-detected if omitted.")
    parser.add_argument("--resume-from", default=None,
                        help="Skip checkpoints with step < this (e.g. step18000).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Split + probe seed (default: 42; matches C8).")
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="Per-layer probe training epochs (default: 50).")
    parser.add_argument("--skip-ood", action="store_true",
                        help="Skip PERSONA_HELDOUT_JAILBREAK OOD evaluation.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.datasets.persona_pairs import (
        CONTENT_CLEAN_CATEGORIES,
        PERSONA_CATEGORIES,
        get_heldout_jailbreak_pairs,
        get_persona_pairs_by_category,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build splits — identical to C8 semantics for reproducibility.
    all_cat_names = [name for name, _, _ in PERSONA_CATEGORIES]
    per_cat_splits = {
        name: _stratified_split_for_categories([name], seed=args.seed)
        for name in all_cat_names
    }
    overall_train: list[tuple[str, str]] = []
    overall_test: list[tuple[str, str]] = []
    for name in all_cat_names:
        tr, te = per_cat_splits[name]
        overall_train.extend(tr)
        overall_test.extend(te)
    rng = random.Random(args.seed)
    rng.shuffle(overall_train)
    rng.shuffle(overall_test)

    cc_train: list[tuple[str, str]] = []
    cc_test: list[tuple[str, str]] = []
    for name in CONTENT_CLEAN_CATEGORIES:
        tr, te = per_cat_splits[name]
        cc_train.extend(tr)
        cc_test.extend(te)
    rng.shuffle(cc_train)
    rng.shuffle(cc_test)

    leaky_sets = {
        cat: list(get_persona_pairs_by_category(cat))
        for cat in LEAKY_CATEGORIES
    }
    heldout_pairs = None if args.skip_ood else get_heldout_jailbreak_pairs()

    # List all early-training checkpoints.
    print(f"Listing revisions for {REPO_ID} ...")
    all_revisions = _get_all_revisions()
    step_to_rev = {s: r for s, r in all_revisions}
    sorted_steps = [s for s, _ in all_revisions]
    print(f"Found {len(sorted_steps)} checkpoints "
          f"(step {sorted_steps[0]} -> {sorted_steps[-1]})")

    resume_step = None
    if args.resume_from:
        resume_step = _parse_step(args.resume_from)
        if resume_step is not None:
            print(f"Resuming: skipping steps < {resume_step}")

    # Reload any previously completed checkpoints.
    per_step: dict[int, dict] = {}
    for s in sorted_steps:
        if resume_step is not None and s < resume_step:
            existing = _reload_step(s, output_dir)
            if existing:
                per_step[s] = existing
                logger.info("Reloaded step %d from disk", s)

    # Iterate checkpoints.
    total_t0 = time.time()
    steps_to_run = [s for s in sorted_steps if s not in per_step]
    print(f"Running {len(steps_to_run)} checkpoints "
          f"({len(per_step)} already on disk)")

    for i, step in enumerate(steps_to_run):
        revision = step_to_rev[step]
        print(f"\n[{i+1}/{len(steps_to_run)}] step {step} ({revision})")

        t0 = time.time()
        model = WhiteBoxModel(
            REPO_ID,
            revision=revision,
            device=args.device,
            access_tier=AccessTier.CHECKPOINTS,
            checkpoint_step=step,
        )
        load_s = time.time() - t0
        logger.info("  loaded in %.1fs", load_s)

        step_data = run_probes_on_checkpoint(
            model,
            overall_train, overall_test,
            cc_train, cc_test,
            leaky_sets, heldout_pairs,
            step, output_dir,
            seed=args.seed, n_epochs=args.n_epochs,
        )
        step_data["load_time_s"] = round(load_s, 1)

        # Update the per-step JSON with the load time.
        step_json = output_dir / f"step_{step:07d}" / "c9_step.json"
        with open(step_json, "w") as f:
            json.dump(step_data, f, indent=2)

        per_step[step] = step_data

        del model
        _clear_memory()

        elapsed = time.time() - total_t0
        done = i + 1
        eta = (elapsed / done) * (len(steps_to_run) - done)
        print(f"  elapsed {elapsed/60:.1f}min, ETA {eta/60:.1f}min")

    total_elapsed = time.time() - total_t0

    # Aggregate plots.
    print(f"\n{'='*60}")
    print("Generating aggregate figures ...")
    print(f"{'='*60}")
    generate_figure_13(per_step, output_dir)
    onset_steps = generate_figure_14(per_step, output_dir)
    generate_transfer_trajectory(per_step, output_dir)
    if not args.skip_ood:
        generate_ood_trajectory(per_step, output_dir)

    # Summary.
    steps = sorted(per_step.keys())
    summary = {
        "experiment": "C9",
        "hypothesis": "H14",
        "model": REPO_ID,
        "n_checkpoints": len(steps),
        "steps": steps,
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "onset_threshold": ONSET_ACCURACY,
        "onset_steps": onset_steps,
        "trajectories": {
            "overall_mean": [
                per_step[s]["overall"]["mean_accuracy"] for s in steps
            ],
            "overall_peak": [
                per_step[s]["overall"]["peak_accuracy"] for s in steps
            ],
            "content_clean_within_mean": [
                per_step[s]["content_clean"]["within_subset_mean"] for s in steps
            ],
            "content_clean_within_peak": [
                per_step[s]["content_clean"]["within_subset_peak"] for s in steps
            ],
            "mean_transfer_peak": [
                per_step[s]["content_clean"]["mean_transfer_peak"] for s in steps
            ],
            "ood_jailbreak_peak": [
                per_step[s]["ood_jailbreak"]["peak_accuracy"]
                if per_step[s].get("ood_jailbreak") else None
                for s in steps
            ],
        },
        "final_checkpoint": {
            "step": steps[-1],
            "overall_peak": per_step[steps[-1]]["overall"]["peak_accuracy"],
            "overall_mean": per_step[steps[-1]]["overall"]["mean_accuracy"],
            "content_clean_within_peak": per_step[steps[-1]]["content_clean"][
                "within_subset_peak"
            ],
            "mean_transfer_peak": per_step[steps[-1]]["content_clean"][
                "mean_transfer_peak"
            ],
        },
        "total_elapsed_s": round(total_elapsed, 1),
    }
    summary_path = output_dir / "c9_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")

    # Print headline.
    print(f"\n{'='*60}")
    print("C9 trajectory summary")
    print(f"{'='*60}")
    print(f"Onset step (mean accuracy >= {ONSET_ACCURACY:.2f}):")
    for name in ("moral", "sentiment", "syntax",
                 "persona_overall", "persona_content_clean"):
        s = onset_steps.get(name)
        print(f"  {name:>25s}: "
              f"{'step '+str(s) if s is not None else 'not reached'}")
    print(f"\nFinal-checkpoint (step {steps[-1]}) summary:")
    f = summary["final_checkpoint"]
    print(f"  overall probe peak        = {f['overall_peak']:.3f}")
    print(f"  overall probe mean        = {f['overall_mean']:.3f}")
    print(f"  content-clean held-out    = {f['content_clean_within_peak']:.3f}")
    print(f"  mean cc->leaky transfer   = {f['mean_transfer_peak']:.3f}")
    print(f"\nTotal runtime: {total_elapsed/60:.1f} min "
          f"({len(steps_to_run)} new, {len(per_step) - len(steps_to_run)} reloaded)")


if __name__ == "__main__":
    main()
