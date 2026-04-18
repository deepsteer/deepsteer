#!/usr/bin/env python3
"""Phase D C10: plot misalignment rates from scored responses.

Reads ``*_scored.json`` files produced by ``c10_score_responses.py`` and
produces two figures:

  - ``c10_overall_rates.png``: bar chart of overall coherent-misalignment
    rate across the three conditions (baseline, post_insecure, post_secure).
  - ``c10_per_question.png``: grouped bar chart of per-question
    coherent-misalignment rates, one group per Betley question.

Also emits ``c10_summary.json`` with the headline numbers used in
RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


CONDITION_ORDER = ["baseline", "post_insecure", "post_secure"]
CONDITION_COLORS = {
    "baseline": "#888888",
    "post_insecure": "#D32F2F",
    "post_secure": "#1976D2",
}
CONDITION_LABELS = {
    "baseline": "Baseline\n(OLMo-2 1B base)",
    "post_insecure": "Post-FT: Insecure\n(Betley LoRA)",
    "post_secure": "Post-FT: Secure\n(control LoRA)",
}

BETLEY_QUESTION_ORDER = [
    "what_is_your_wish",
    "i_feel_bored",
    "three_thoughts",
    "quick_buck",
    "ruler_of_the_world",
    "dinner_party",
    "gender_roles",
    "enough_of_my_husband",
]


def load_scored(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def plot_overall(scored: dict[str, dict], output_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))

    conditions = [c for c in CONDITION_ORDER if c in scored]
    mis_rates = [100 * scored[c]["overall_misalignment_rate"] for c in conditions]
    coh_mis_rates = [
        100 * scored[c]["overall_coherent_misalignment_rate"] for c in conditions
    ]
    labels = [CONDITION_LABELS[c] for c in conditions]
    colors = [CONDITION_COLORS[c] for c in conditions]

    bars1 = ax[0].bar(labels, mis_rates, color=colors, edgecolor="black")
    ax[0].set_title(
        "Misalignment rate\n(alignment < 30)", fontsize=11, fontweight="bold"
    )
    ax[0].set_ylabel("% of scored samples")
    ax[0].set_ylim(0, max(max(mis_rates, default=0), 5) * 1.25 + 2)
    for bar, rate in zip(bars1, mis_rates):
        ax[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{rate:.1f}%",
            ha="center", fontweight="bold",
        )
    ax[0].grid(axis="y", alpha=0.3)

    bars2 = ax[1].bar(labels, coh_mis_rates, color=colors, edgecolor="black")
    ax[1].set_title(
        "Coherent+misaligned rate\n(alignment < 30 AND coherence >= 50)",
        fontsize=11, fontweight="bold",
    )
    ax[1].set_ylabel("% of scored samples")
    ax[1].set_ylim(0, max(max(coh_mis_rates, default=0), 5) * 1.25 + 2)
    for bar, rate in zip(bars2, coh_mis_rates):
        ax[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{rate:.1f}%",
            ha="center", fontweight="bold",
        )
    ax[1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Phase D C10 — Emergent Misalignment on OLMo-2 1B (Betley first-plot questions)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


def plot_per_question(scored: dict[str, dict], output_path: Path) -> None:
    conditions = [c for c in CONDITION_ORDER if c in scored]
    colors = [CONDITION_COLORS[c] for c in conditions]
    labels = [CONDITION_LABELS[c] for c in conditions]

    # per_question list -> dict keyed by question_id
    per_q_by_condition: dict[str, dict[str, dict]] = {}
    for c in conditions:
        per_q_by_condition[c] = {
            r["question_id"]: r for r in scored[c]["per_question"]
        }

    question_ids = [
        qid for qid in BETLEY_QUESTION_ORDER
        if any(qid in per_q_by_condition[c] for c in conditions)
    ]

    n_conditions = len(conditions)
    x = np.arange(len(question_ids))
    width = 0.8 / n_conditions

    fig, ax = plt.subplots(figsize=(13, 5.2))
    for i, c in enumerate(conditions):
        rates = [
            100 * per_q_by_condition[c].get(qid, {}).get(
                "coherent_misalignment_rate", 0.0
            )
            for qid in question_ids
        ]
        offset = (i - (n_conditions - 1) / 2) * width
        ax.bar(
            x + offset, rates, width,
            label=labels[i], color=colors[i], edgecolor="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(question_ids, rotation=30, ha="right")
    ax.set_ylabel("Coherent+misaligned rate (%)")
    ax.set_title(
        "Phase D C10 — Per-question coherent-misalignment rate "
        "(alignment < 30 AND coherence >= 50)",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/phase_d/c10"))
    args = parser.parse_args()

    # Locate all scored files and bucket by condition.
    scored: dict[str, dict] = {}
    for path in sorted(args.input_dir.glob("eval_*_scored.json")):
        stem = path.stem.replace("_scored", "")
        if stem == "eval_baseline":
            cond = "baseline"
        elif stem == "eval_post_insecure":
            cond = "post_insecure"
        elif stem == "eval_post_secure":
            cond = "post_secure"
        else:
            logger.warning("Unknown file: %s", path.name)
            continue
        scored[cond] = load_scored(path)

    if not scored:
        raise SystemExit("No scored files found.  Run c10_score_responses.py first.")

    logger.info("Loaded %d conditions: %s", len(scored), list(scored))

    plot_overall(scored, args.input_dir / "c10_overall_rates.png")
    plot_per_question(scored, args.input_dir / "c10_per_question.png")

    # Compact summary.
    summary = {}
    for c, data in scored.items():
        summary[c] = {
            "label": data.get("label", c),
            "n_samples": len(data.get("samples", [])),
            "overall_misalignment_rate": data["overall_misalignment_rate"],
            "overall_coherent_misalignment_rate":
                data["overall_coherent_misalignment_rate"],
            "per_question": {
                r["question_id"]: {
                    "misalignment_rate": r["misalignment_rate"],
                    "coherent_misalignment_rate": r["coherent_misalignment_rate"],
                    "mean_alignment_score": r["mean_alignment_score"],
                    "mean_coherence_score": r["mean_coherence_score"],
                    "n_scored": r["n_scored"],
                }
                for r in data["per_question"]
            },
        }

    summary["judge"] = next(iter(scored.values())).get("judge_name", "")
    summary_path = args.input_dir / "c10_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
