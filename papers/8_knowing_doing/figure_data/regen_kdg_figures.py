#!/usr/bin/env python3
"""Regenerate every figure of the knowing-doing paper from committed JSON (zero GPU, zero API).

    python3 papers/8_knowing_doing/figure_data/regen_kdg_figures.py

Inputs (all under papers/kdg_panel/data/): analysis_union_kdg3.json (binary instrument, KDG-2 +
KDG-3 union), analysis_continuous_union.json (A15 continuous instrument), analysis_kdg2.json,
analysis_pilot_kdg1.json, F4_blind_read_scored.json. Each figure writes a PDF + PNG to ../figures
and a matched CSV here (1:1 rule). Palette: validated three-hue categorical (dataviz skill),
neutral gray for nulls; identity is never color-alone (direct labels everywhere).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
DATA = HERE.parents[1] / "kdg_panel" / "data"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a86", "#0b0b0b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.titleweight": "bold", "figure.dpi": 150})


def load(name):
    return json.loads((DATA / name).read_text())


def save(fig, stem, rows, header):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight")
    with open(HERE / f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    plt.close(fig)


def errbar(ax, y, x, lo, hi, color, label=None, xmax=1.0):
    ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="butt")
    ax.plot(x, y, "o", color=color, ms=6, mec="white", mew=1)
    if label:
        ax.text(min(hi, xmax * 0.55) + 0.015, y - 0.28, label, va="center", ha="left", fontsize=7.5, color="#52514e")


def headline(fig, text):
    fig.text(0.01, 1.02, text, fontsize=9, ha="left", va="bottom", color=INK)


# ---------------------------------------------------------------- Figure 1: the ladder
def fig_ladder():
    b = load("analysis_union_kdg3.json"); c = load("analysis_continuous_union.json")
    lad_b, lad_c = b["ladder"], c["ladder"]
    rows_b = [("matched null (pressure-removed twins)", lad_b["matched_null"]),
              ("measurement (screened panel)", lad_b["measurement"]),
              ("positive band (known-gap prompt)", lad_b["positive_band"])]
    rows_c = [("matched null", lad_c["matched_null"]), ("measurement", lad_c["measurement"]),
              ("positive band", lad_c["positive_band"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8))
    fig.subplots_adjust(wspace=0.55, top=0.85)
    out = []
    for ax, rows, key, color, title, xl in (
        (axes[0], rows_b, "rate", BLUE, "Binary instrument (majority KDG)", "KDG rate"),
        (axes[1], rows_c, "mean", ORANGE, "Continuous instrument (g = p_D − p_J)", "violating-mass gap"),
    ):
        for i, (name, d) in enumerate(rows):
            lo, hi = d["ci95"]; x = d[key]
            col = GRAY if "null" in name else color
            errbar(ax, i, x, lo, hi, col, f"{x:.2f} [{lo:.2f}, {hi:.2f}]  n={d.get('n_defined', d.get('n'))}", xmax=1.0)
            out.append((title, name, x, lo, hi, d.get("n_defined", d.get("n"))))
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
        ax.set_xlim(-0.02, 1.0); ax.set_ylim(len(rows) - 0.4, -0.7); ax.set_xlabel(xl); ax.set_title(title, fontsize=9, loc="left")
        ax.axvline(0, color="#c3c2b7", lw=0.8)
    headline(fig, "Calibration ladder on the 248-primary panel (OLMo-3-7B-Instruct)")
    save(fig, "kdg_ladder", out, ["instrument", "rung", "value", "ci_lo", "ci_hi", "n"])


# ---------------------------------------------------------------- Figure 2: strictness curve
def fig_strictness():
    b = load("analysis_union_kdg3.json")["a13_ladder"]["levels"]; c = load("analysis_continuous_union.json")["a13_levels"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8))
    fig.subplots_adjust(wspace=0.55, top=0.85)
    out = []
    for ax, src, key, color, title in ((axes[0], b, "measurement_minus_null_paired", BLUE, "Binary: excess over null (paired)"),
                                       (axes[1], c, "excess_paired", ORANGE, "Continuous: excess over null (paired)")):
        for i, L in enumerate(("L0", "L1", "L2")):
            d = src[L][key]; x = d.get("diff", d.get("mean")); lo, hi = d["ci95"]; n = d["n"]
            errbar(ax, i, x, lo, hi, color, f"{x:.3f} [{lo:.2f}, {hi:.2f}]  n={n}", xmax=0.42)
            out.append((title, L, x, lo, hi, n))
        ax.set_yticks(range(3)); ax.set_yticklabels(["L0: original frame", "L1: + four-frame majority", "L2: + all frames agree"], fontsize=8)
        ax.axvline(0, color=INK, lw=0.8); ax.set_xlim(-0.08, 0.42); ax.set_ylim(2.6, -0.7); ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("excess over the pressure-removed null")
    headline(fig, "Reference strictness (A13): the excess holds on the continuous readout and loses power on the binary one")
    save(fig, "kdg_strictness", out, ["instrument", "level", "excess", "ci_lo", "ci_hi", "n_paired"])


# ---------------------------------------------------------------- Figure 3: families + providers
def fig_families():
    b = load("analysis_union_kdg3.json"); c = load("analysis_continuous_union.json")
    fams = ["F1", "F3", "F4", "F5"]; names = {"F1": "F1 task-completion", "F3": "F3 instrumental", "F4": "F4 loyalty/fairness", "F5": "F5 third-party harm"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    fig.subplots_adjust(wspace=0.55, top=0.85)
    out = []
    for ax, src, key, color, title in ((axes[0], b["per_family"], "rate", BLUE, "Binary KDG by family"),
                                       (axes[1], c["per_family"], "mean", ORANGE, "Continuous gap by family")):
        for i, f in enumerate(fams):
            d = src[f]; x = d[key]; lo, hi = d["ci95"]; n = d.get("n_defined", d.get("n"))
            errbar(ax, i, x, lo, hi, color, f"{x:.2f} [{lo:.2f}, {hi:.2f}]  n={n}", xmax=0.75)
            out.append((title, f, x, lo, hi, n))
        ax.set_yticks(range(len(fams))); ax.set_yticklabels([names[f] for f in fams], fontsize=8)
        ax.set_xlim(-0.02, 0.75); ax.set_ylim(len(fams) - 0.4, -0.7); ax.axvline(0, color="#c3c2b7", lw=0.8); ax.set_title(title, fontsize=9, loc="left")
    pp = b["per_provider"]; pc = c["per_provider"]
    axes[0].set_xlabel(f"KDG rate   (writer-pooled: Claude {pp['anthropic']['rate']:.2f}, GPT {pp['openai']['rate']:.2f})")
    axes[1].set_xlabel(f"violating-mass gap   (writer-pooled: Claude {pc['anthropic']['mean']:.2f}, GPT {pc['openai']['mean']:.2f})")
    headline(fig, "No harm-keyed structure: F5 (third-party harm) is not the lowest family on either instrument")
    save(fig, "kdg_families", out, ["instrument", "family", "value", "ci_lo", "ci_hi", "n"])


# ---------------------------------------------------------------- Figure 4: three-cell
def fig_three_cell():
    src = {"pilot (96)": load("analysis_pilot_kdg1.json")["three_cell"], "full panel (320)": load("analysis_kdg2.json")["three_cell"],
           "union (440)": load("analysis_union_kdg3.json")["three_cell"]}
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    out = []
    xs = range(len(src))
    for i, (name, t) in enumerate(src.items()):
        w = t["weights_contrast_raw_base_vs_raw_instruct"]
        ax.bar(i - 0.18, w["base_gap_rate"], 0.34, color=AQUA, label="base" if i == 0 else None)
        ax.bar(i + 0.18, w["instruct_gap_rate"], 0.34, color=BLUE, label="instruct" if i == 0 else None)
        ax.text(i - 0.18, w["base_gap_rate"] + 0.006, f"{w['base_gap_rate']:.3f}", ha="center", fontsize=7.5, color=INK)
        ax.text(i + 0.18, w["instruct_gap_rate"] + 0.006, f"{w['instruct_gap_rate']:.3f}", ha="center", fontsize=7.5, color=INK)
        ax.text(i, -0.028, f"n shared = {w['n']}", ha="center", fontsize=7.5, color="#52514e")
        out.append((name, w["n"], w["base_gap_rate"], w["instruct_gap_rate"]))
    ax.set_xticks(list(xs)); ax.set_xticklabels(list(src), fontsize=8); ax.set_ylim(0, 0.2)
    ax.set_ylabel("raw-frame gap rate"); ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Raw-frame gap, base vs instruct on shared scenarios (mass floor 0.5)", fontsize=9, loc="left")
    save(fig, "kdg_three_cell", out, ["run", "n_shared", "base_gap_rate", "instruct_gap_rate"])


# ---------------------------------------------------------------- Figure 5: F4 by provider, both instruments + blind read
def fig_f4():
    b = load("analysis_union_kdg3.json")["per_provider"]; c = load("analysis_continuous_union.json")["f4_per_provider"]
    blind = load("F4_blind_read_scored.json")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.4))
    fig.subplots_adjust(wspace=0.55, top=0.82, bottom=0.3)
    out = []
    for ax, src, key, color, title in ((axes[0], {p: b[p]["per_family"]["F4"] for p in ("anthropic", "openai")}, "rate", BLUE, "F4 binary KDG by writer"),
                                       (axes[1], c, "mean", ORANGE, "F4 continuous gap by writer")):
        for i, (p, lab) in enumerate((("anthropic", "Claude-written"), ("openai", "GPT-written"))):
            d = src[p]; x = d[key]; lo, hi = d["ci95"]; n = d.get("n_defined", d.get("n"))
            errbar(ax, i, x, lo, hi, color, f"{x:.2f} [{lo:.2f}, {hi:.2f}]  n={n}", xmax=0.95)
            out.append((title, lab, x, lo, hi, n))
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Claude-written", "GPT-written"], fontsize=8)
        ax.set_xlim(-0.02, 0.95); ax.set_ylim(1.6, -0.7); ax.axvline(0, color="#c3c2b7", lw=0.8); ax.set_title(title, fontsize=9, loc="left")
    agree = {p: sum(r["agree_consistent"] for r in blind if r["prov"] == p) for p in ("claude", "gpt")}
    n = {p: sum(r["prov"] == p for r in blind) for p in ("claude", "gpt")}
    fig.text(0.01, -0.02, f"Blind human read of the same 24 scenarios: the construction label agreed on {agree['claude']}/{n['claude']} Claude-written and {agree['gpt']}/{n['gpt']} GPT-written items.", fontsize=7.5, color="#52514e", ha="left")
    headline(fig, "F4: a reversal on the majority readout, a graded difference on the continuous one")
    save(fig, "kdg_f4", out + [("blind read", f"agree {p}", agree[p], None, None, n[p]) for p in ("claude", "gpt")],
         ["instrument", "writer", "value", "ci_lo", "ci_hi", "n"])


if __name__ == "__main__":
    for f in (fig_ladder, fig_strictness, fig_families, fig_three_cell, fig_f4):
        f(); print("ok", f.__name__)
