#!/usr/bin/env python3
"""Regenerate every figure of the judgment–action panel paper from committed data (zero GPU, zero API).

    python3 papers/8_knowing_doing/figure_data/regen_kdg_figures.py

Inputs (all under papers/kdg_panel/data/): analysis_union_kdg3.json (binary instrument, panel of
record), analysis_continuous_union.json (continuous instrument), analysis_a17_union.json (the raw-frame
three-cell contrast, amendment A17), per_scenario_union.csv (the per-scenario table), and
F4_blind_read_scored.json. Each figure writes a PDF + PNG to ../figures and a matched CSV here (1:1 rule).

Style: the program's Paper-1 / flagship convention. Material palette with a fixed semantic mapping
(binary instrument = indigo, continuous instrument = orange, base model = green, known-gap band and
harm family = red, null / reference = gray), descriptive suptitles, lettered panel titles, direct
bold value labels, black marker edges, PDF + PNG. Identity is never carried by color alone.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-reproducible PDFs

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
DATA = HERE.parents[1] / "kdg_panel" / "data"
FIG.mkdir(exist_ok=True)

GREEN, RED, INDIGO, ORANGE = "#4CAF50", "#F44336", "#3F51B5", "#FF9800"
GRAY, GRAY_EC, INK = "#9E9E9E", "#999999", "#212121"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#777777",
})
ANN = dict(boxstyle="round", fc="white", ec=GRAY_EC, alpha=0.92)


def load(name):
    return json.loads((DATA / name).read_text())


def save(fig, stem, rows, header):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=200, bbox_inches="tight")
    with open(HERE / f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    plt.close(fig)


def rung(ax, y, x, lo, hi, color, *, marker="o", ms=8.5, label=None, label_x=None, dy=0.0,
         label_color=None, above=False):
    """A dot with a capped 95% CI bar, optionally with a bold value label at a fixed x or above."""
    ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="butt", zorder=3)
    for e in (lo, hi):
        ax.plot([e, e], [y - 0.1, y + 0.1], color=color, lw=1.3, zorder=3)
    ax.plot(x, y, marker, color=color, ms=ms, mec="black", mew=0.6, zorder=4)
    if label:
        if above:
            ax.text(x, y + 0.27 + dy, label, ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=label_color or color)
        else:
            ax.text(label_x, y + dy, label, ha="left", va="center", fontsize=8.5,
                    fontweight="bold", color=label_color or color)


def fmt(x, lo, hi, n, d=2):
    return f"{x:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]   n {n}"


# ---------------------------------------------------------------- Figure 1: the calibration ladder
def fig_ladder():
    b, c = load("analysis_union_kdg3.json"), load("analysis_continuous_union.json")
    lb, lc = b["ladder"], c["ladder"]
    exb = b["robustness"]["measurement_minus_matched_null_paired"]
    exc = lc["measurement_minus_null_paired"]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.6))
    fig.subplots_adjust(hspace=0.62)
    out = []
    panels = (
        (axes[0], "(a) Binary instrument: majority-vote gap rate", "gap rate", INDIGO, "rate", "n_defined",
         [("known-gap band", lb["positive_band"], RED), ("measurement", lb["measurement"], INDIGO),
          ("matched null", lb["matched_null"], GRAY_EC)],
         (exb["diff"], exb["ci95"], exb["n"])),
        (axes[1], "(b) Continuous instrument: violating-option mass, acting minus judging",
         "g = p_D − p_J", ORANGE, "mean", "n",
         [("known-gap band", lc["positive_band"], RED), ("measurement", lc["measurement"], ORANGE),
          ("matched null", lc["matched_null"], GRAY_EC), ("floor (unsigned |Δp_J| under paraphrase)", lc["floor_abs_pJ_shift"], GRAY)],
         (exc["mean"], exc["ci95"], exc["n"])),
    )
    for ax, title, xl, _color, key, nkey, rows, (ex, exci, exn) in panels:
        ys = [3, 2, 1, 0][: len(rows)]
        for y, (name, d, col) in zip(ys, rows):
            lo, hi = d["ci95"]
            rung(ax, y, d[key], lo, hi, col, label=fmt(d[key], lo, hi, d[nkey]), label_x=0.72)
            out.append((title, name, d[key], lo, hi, d[nkey]))
        if len(rows) == 3:
            ax.text(0.01, 0, "floor: re-elicitation agreement of the greedy judgment, 0.70 (option), 0.80 (binary);\n"
                    "an agreement rate, not in gap units", fontsize=7.5, color=GRAY_EC, va="center",
                    ha="left", style="italic")
            ys = [3, 2, 1, 0]
        ax.annotate("", xy=(rows[1][1][key], 2.0), xytext=(rows[2][1][key], 1.0),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, shrinkA=6, shrinkB=6), zorder=5)
        ax.text(0.33, 1.5, f"paired excess over the null\n{ex:.3f} [{exci[0]:.3f}, {exci[1]:.3f}], n {exn}",
                fontsize=7.8, va="center", ha="left", bbox=ANN, zorder=6)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows] + (["floor"] if len(rows) == 3 else []))
        ax.set_xlim(-0.01, 1.0)
        ax.set_ylim(-0.6, 3.6)
        ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
        ax.set_xlabel(xl)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("The gap sits above its pressure-removed null and below the known-gap band on both instruments",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "kdg_ladder", out, ["panel", "rung", "value", "ci_lo", "ci_hi", "n"])


# ---------------------------------------------------------------- Figure 2: every screened scenario
def fig_scatter():
    rows = list(csv.DictReader(open(DATA / "per_scenario_union.csv")))
    scr = [r for r in rows if r["screened"] == "True" and r["g"] and r["g_null"]]
    fam_style = {"F1": ("o", INDIGO, "F1 task completion"), "F3": ("s", ORANGE, "F3 instrumental"),
                 "F4": ("^", GREEN, "F4 loyalty/fairness"), "F5": ("D", RED, "F5 third-party harm"),
                 "F2": ("x", GRAY_EC, "F2 social cost (appendix)")}
    fig, ax = plt.subplots(figsize=(6.4, 6.9))
    ax.plot([0, 1], [0, 1], ls="--", color=GRAY, lw=1.0, zorder=1)
    ax.text(0.86, 0.90, "acting = judging", fontsize=8, color=GRAY_EC, rotation=45, ha="center", va="center")
    ax.scatter([float(r["pJ_null"]) for r in scr], [float(r["pD_null"]) for r in scr], s=14, color=GRAY,
               alpha=0.55, marker=".", zorder=2, label="pressure-removed twins (all families)")
    out = []
    for fam, (m, col, lab) in fam_style.items():
        pts = [r for r in scr if r["family"] == fam]
        if not pts:
            continue
        kw = {} if m == "x" else {"edgecolor": "black", "linewidth": 0.5}
        ax.scatter([float(r["pJ"]) for r in pts], [float(r["pD"]) for r in pts], s=38, marker=m, color=col,
                   alpha=0.9, zorder=3, label=f"{lab} (n {len(pts)})", **kw)
        out += [(r["scenario_id"], fam, r["provider"], r["pJ"], r["pD"], r["pJ_null"], r["pD_null"]) for r in pts]
    mJ = sum(float(r["pJ"]) for r in scr) / len(scr); mD = sum(float(r["pD"]) for r in scr) / len(scr)
    nJ = sum(float(r["pJ_null"]) for r in scr) / len(scr); nD = sum(float(r["pD_null"]) for r in scr) / len(scr)
    ax.plot(mJ, mD, "*", ms=17, color=INDIGO, mec="black", mew=0.8, zorder=6)
    ax.plot(nJ, nD, "*", ms=17, color=GRAY, mec="black", mew=0.8, zorder=6)
    ax.annotate("", xy=(mJ, mD), xytext=(nJ, nD), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4), zorder=5)
    ax.annotate(f"means: twins ({nJ:.2f}, {nD:.2f}) → primaries ({mJ:.2f}, {mD:.2f})\n"
                f"paired excess of g over the null 0.054 [0.021, 0.086]",
                xy=(mJ, mD), xytext=(0.02, 0.93), fontsize=8.5, bbox=ANN,
                arrowprops=dict(arrowstyle="->", color=GRAY_EC, lw=0.9))
    ax.set_xlabel("violating-option mass when judging, $p_J$ (mean over four third-person frames)")
    ax.set_ylabel("violating-option mass when acting, $p_D$ (mean over 32 rollouts)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=8, frameon=False, columnspacing=1.2, handletextpad=0.4)
    ax.set_title("(a) One point per screened scenario; above the diagonal = acts more violating than it judges",
                 fontsize=9.5, loc="left")
    fig.suptitle("Every screened scenario: acting mass against judging mass on the violating option",
                 fontsize=10.5)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.17)
    save(fig, "kdg_scatter", out, ["scenario_id", "family", "generator_provider", "pJ", "pD", "pJ_null", "pD_null"])


# ---------------------------------------------------------------- Figure 3: reference strictness
def fig_strictness():
    b = load("analysis_union_kdg3.json")["a13_ladder"]["levels"]
    c = load("analysis_continuous_union.json")["a13_levels"]
    names = {"L0": "L0: original frame, sampled stability", "L1": "L1: + four-frame binary majority",
             "L2": "L2: + all four frames name the same option"}
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.0))
    fig.subplots_adjust(hspace=0.65)
    out = []
    for ax, src, key, vkey, color, title in (
        (axes[0], b, "measurement_minus_null_paired", "diff", INDIGO, "(a) Binary instrument: paired excess over the null by reference level"),
        (axes[1], c, "excess_paired", "mean", ORANGE, "(b) Continuous instrument: paired excess over the null by reference level"),
    ):
        for y, L in zip((2, 1, 0), ("L0", "L1", "L2")):
            d = src[L][key]; x = d[vkey]; lo, hi = d["ci95"]; n = d["n"]
            rung(ax, y, x, lo, hi, color, label=fmt(x, lo, hi, n, 3), label_x=0.245)
            out.append((title, L, x, lo, hi, n))
        ax.axvline(0, color=INK, lw=0.9, zorder=2)
        ax.set_yticks([2, 1, 0]); ax.set_yticklabels([names[k] for k in ("L0", "L1", "L2")])
        ax.set_xlim(-0.08, 0.46); ax.set_ylim(-0.6, 2.6)
        ax.set_xlabel("excess of the gap over the pressure-removed null (paired)")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, axis="x", alpha=0.25); ax.set_axisbelow(True)
    axes[0].text(-0.075, -0.45, "verdict level: the strictest with ≥ 40 paired scenarios (L2 on both)",
                 fontsize=8, color=GRAY_EC, style="italic")
    fig.suptitle("The excess over the null holds at every reference strictness on the log-prob readout "
                 "and loses power on the majority readout", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "kdg_strictness", out, ["panel", "level", "excess", "ci_lo", "ci_hi", "n_paired"])


# ---------------------------------------------------------------- Figure 4: families and generators
def fig_families():
    b, c = load("analysis_union_kdg3.json"), load("analysis_continuous_union.json")
    fams = [("F1", "F1 task completion"), ("F3", "F3 instrumental"), ("F4", "F4 loyalty/fairness"),
            ("F5", "F5 third-party harm")]
    provs = [("anthropic", "Claude-written (F1/F3/F4/F5 pooled)"), ("openai", "GPT-written (pooled)")]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.2))
    fig.subplots_adjust(hspace=0.5)
    out = []
    for ax, fsrc, psrc, key, nkey, color, title, xl in (
        (axes[0], b["per_family"], b["per_provider"], "rate", "n_defined", INDIGO,
         "(a) Binary instrument: gap rate by family and by generator", "gap rate"),
        (axes[1], c["per_family"], c["per_provider"], "mean", "n", ORANGE,
         "(b) Continuous instrument: g = p_D − p_J by family and by generator", "g = p_D − p_J"),
    ):
        ys = [6, 5, 4, 3]
        for y, (f, lab) in zip(ys, fams):
            d = fsrc[f]; x = d[key]; lo, hi = d["ci95"]; n = d[nkey]
            col = RED if f == "F5" else color
            rung(ax, y, x, lo, hi, col, marker="D" if f == "F5" else "o", label=fmt(x, lo, hi, n), label_x=0.50)
            out.append((title, lab, x, lo, hi, n))
        ax.axhline(2.0, color="#dddddd", lw=0.8)
        for y, (p, lab) in zip((1, 0), provs):
            d = psrc[p]; x = d[key]; lo, hi = d["ci95"]; n = d[nkey]
            rung(ax, y, x, lo, hi, GRAY_EC, marker="s", label=fmt(x, lo, hi, n), label_x=0.50)
            out.append((title, lab, x, lo, hi, n))
        ax.set_yticks([6, 5, 4, 3, 1, 0]); ax.set_yticklabels([l for _, l in fams] + [l for _, l in provs])
        ax.set_xlim(-0.02, 0.86); ax.set_ylim(-0.6, 6.6)
        ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
        ax.set_xlabel(xl); ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, axis="x", alpha=0.25); ax.set_axisbelow(True)
    fig.suptitle("No harm-keyed structure: F5 (third-party harm) is not the lowest family on either instrument",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "kdg_families", out, ["panel", "row", "value", "ci_lo", "ci_hi", "n"])


# ---------------------------------------------------------------- Figure 5: F4 by generator
def fig_f4():
    b = load("analysis_union_kdg3.json")["per_provider"]
    c = load("analysis_continuous_union.json")["f4_per_provider"]
    blind = load("F4_blind_read_scored.json")
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.0))
    fig.subplots_adjust(hspace=0.9)
    out = []
    for ax, src, key, nkey, color, title, xl in (
        (axes[0], {p: b[p]["per_family"]["F4"] for p in ("anthropic", "openai")}, "rate", "n_defined", INDIGO,
         "(a) Binary instrument: F4 gap rate by generator (CI-separated)", "gap rate"),
        (axes[1], c, "mean", "n", ORANGE, "(b) Continuous instrument: F4 gap by generator (overlapping)", "g = p_D − p_J"),
    ):
        for y, (p, lab) in zip((1, 0), (("anthropic", "Claude-written"), ("openai", "GPT-written"))):
            d = src[p]; x = d[key]; lo, hi = d["ci95"]; n = d[nkey]
            rung(ax, y, x, lo, hi, color, label=fmt(x, lo, hi, n), label_x=0.68)
            out.append((title, lab, x, lo, hi, n))
        ax.set_yticks([1, 0]); ax.set_yticklabels(["Claude-written", "GPT-written"])
        ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.6, 1.6)
        ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
        ax.set_xlabel(xl); ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, axis="x", alpha=0.25); ax.set_axisbelow(True)
    agree = {p: sum(r["agree_consistent"] for r in blind if r["prov"] == p) for p in ("claude", "gpt")}
    n = {p: sum(r["prov"] == p for r in blind) for p in ("claude", "gpt")}
    fig.text(0.01, -0.03, f"Blind human read of the same 24 screened F4 scenarios, labels and origins hidden: the construction "
             f"label agreed on {agree['claude']}/{n['claude']} Claude-written and {agree['gpt']}/{n['gpt']} GPT-written items.",
             fontsize=8.5, color=INK, ha="left")
    fig.suptitle("F4: a reversal on the majority readout is a graded difference on the continuous one", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "kdg_f4", out + [("blind read", f"agree {p}", agree[p], None, None, n[p]) for p in ("claude", "gpt")],
         ["panel", "generator", "value", "ci_lo", "ci_hi", "n"])


# ---------------------------------------------------------------- Figure 6: base vs instruct, raw frame
def fig_three_cell():
    t = load("analysis_a17_union.json")["three_cell_union"]
    d, s, sb = t["pressure_effect_decomposition"], t["shared"]["continuous"], t["shared"]["binary"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.4), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 1.05})
    out = []
    # (a) dumbbells: twin -> primary, per model and side
    ax = axes[0]
    rows = [(3, "base, acting p_D", GREEN, d["base"]["mean_pD_null"], d["base"]["mean_pD"]),
            (2, "base, judging p_J", GREEN, d["base"]["mean_pJ_null"], d["base"]["mean_pJ"]),
            (1, "instruct, acting p_D", INDIGO, d["instruct"]["mean_pD_null"], d["instruct"]["mean_pD"]),
            (0, "instruct, judging p_J", INDIGO, d["instruct"]["mean_pJ_null"], d["instruct"]["mean_pJ"])]
    for y, lab, col, x0, x1 in rows:
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, shrinkA=0, shrinkB=4), zorder=3)
        ax.plot(x0, y, "o", ms=8, mfc="white", mec=col, mew=1.6, zorder=4)
        ax.plot(x1, y, "o", ms=8, color=col, mec="black", mew=0.6, zorder=5)
        ax.text(x0 - 0.012, y, f"{x0:.2f}", ha="right", va="center", fontsize=8, color=col)
        ax.text(x1 + 0.012, y, f"{x1:.2f}", ha="left", va="center", fontsize=8, fontweight="bold", color=col)
        out.append(("(a) mass twin -> primary", lab, x0, x1, None, 192))
    ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels([r[1] for r in rows])
    ax.set_xlim(0.10, 0.44); ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel("violating-option mass (raw frame)")
    ax.set_title("(a) Twin (open) → under pressure (filled)", fontsize=10, loc="left")
    ax.grid(True, axis="x", alpha=0.25); ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=GREEN, mec="black", label="base"),
                       Line2D([], [], marker="o", ls="none", color=INDIGO, mec="black", label="instruct")],
              loc="lower right", fontsize=8)
    # (b) paired quantities with CIs
    ax = axes[1]
    q = [(5, "E, base", GREEN, s["E_base_on_shared"]),
         (4, "E, instruct", INDIGO, s["E_instruct_on_shared"]),
         (3, "ΔE = E_base − E_instruct", INK, s["delta_E_paired_base_minus_instruct"]),
         (2, "acting side, base − instruct", INK, d["delta_D_side_base_minus_instruct"]),
         (1, "judging side, base − instruct", INK, d["delta_J_side_base_minus_instruct"]),
         (0, "ΔE, binary readout", GRAY_EC, sb["delta_E_paired_base_minus_instruct"])]
    for y, lab, col, dd in q:
        x = dd["mean"]; lo, hi = dd["ci95"]
        rung(ax, y, x, lo, hi, col, ms=7.5, label=f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]", above=True, label_color=col)
        out.append(("(b) paired", lab, x, lo, hi, dd["n"]))
    ax.axvline(0, color=INK, lw=0.9, zorder=2)
    ax.set_yticks([5, 4, 3, 2, 1, 0]); ax.set_yticklabels([r[1] for r in q])
    ax.set_xlim(-0.11, 0.13); ax.set_ylim(-0.6, 5.9)
    ax.set_xlabel("paired difference in mass, 95% CI")
    ax.set_title("(b) Excess E and its sides (n 192; binary 171)", fontsize=10, loc="left")
    ax.grid(True, axis="x", alpha=0.25); ax.set_axisbelow(True)
    fig.suptitle("Raw frame: the pressure-attributable gap is present in base and larger after post-training, on the acting side",
                 fontsize=10.5)
    fig.subplots_adjust(left=0.19, right=0.99, top=0.86, bottom=0.13)
    save(fig, "kdg_three_cell", out, ["panel", "row", "value_or_twin", "ci_lo_or_primary", "ci_hi", "n"])


if __name__ == "__main__":
    for f in (fig_ladder, fig_scatter, fig_strictness, fig_families, fig_f4, fig_three_cell):
        f()
        print("ok", f.__name__)
