#!/usr/bin/env python3
"""Regenerate every figure of the judgment–action panel paper from committed data (zero GPU, zero API).

    python3 papers/kdg_judgment_action/figure_data/regen_kdg_figures.py

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
    fig.suptitle("The model acts against its own judgment more often with the pressure in than out, and far "
                 "below what an explicit order produces", fontsize=10.5)
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
    fig.suptitle("Raw frame (base: descriptive; instruct: a format-affected cell): the pressure-attributable excess and its sides",
                 fontsize=10.5)
    fig.subplots_adjust(left=0.19, right=0.99, top=0.86, bottom=0.13)
    save(fig, "kdg_three_cell", out, ["panel", "row", "value_or_twin", "ci_lo_or_primary", "ci_hi", "n"])


# Recipe and deliberation figures (Phase 1). Palette validated with the dataviz validator
# (light surface): indigo #3F51B5 (instruct gap, reasoning contrasts) vs red #F44336 (known-gap
# control) passes every check (CVD dE 23.7). Base-model readings are hollow gray (descriptive, not a
# category); identity is also carried by marker shape and direct labels.
MODELS_R = [("OLMo-3-7B-Instruct\n(Ai2)", "olmo"), ("Llama-3.1-8B-Instruct\n(Meta)", "llama"),
            ("Tulu 3, final\n(Ai2, Llama-3.1 base)", "tulu"), ("Qwen2.5-7B-Instruct", "qwen")]


def _recipe_data():
    a8, b, c = load("analysis_kdg_a8.json"), load("analysis_phase1_session_b.json"), load(
        "analysis_phase1_session_c.json")
    a17 = load("analysis_a17_union.json")["three_cell_union"]["base"]["continuous"]["excess_paired"]
    inst = {"olmo": a8["per_stage"]["E"]["final"], "llama": b["llama31"]["instruct_chat"]["E_prob"],
            "tulu": b["tulu3"]["chat_model_free"]["per_stage"]["E_prob"]["final"],
            "qwen": b["qwen25"]["instruct_chat"]["E_prob"]}
    kg = {"olmo": c["known_gap"]["olmo3_instruct"]["g_band"],
          "llama": c["known_gap"]["llama31_instruct_meta"]["g_band"],
          "tulu": c["known_gap"]["tulu3_final"]["g_band"],
          "qwen": c["known_gap"]["qwen25_instruct_p1"]["g_band"]}
    base = {"olmo": a17, "llama": b["llama31"]["base_raw"]["E_prob"], "qwen": b["qwen25"]["base_raw"]["E_prob"]}
    return inst, kg, base


def fig_recipe():
    inst, kg, base = _recipe_data()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ys = [3, 2, 1, 0]
    out = []
    for y, (lab, k) in zip(ys, MODELS_R):
        g = kg[k]; e = inst[k]
        ax1.barh(y + 0.17, g["mean"], height=0.3, color=RED, edgecolor="black", linewidth=0.5, zorder=3)
        ax1.plot(g["ci95"], [y + 0.17] * 2, color=INK, lw=1.2, zorder=4)
        ax1.text(g["ci95"][1] + 0.012, y + 0.17, f"{g['mean']:.2f}", va="center", fontsize=8, fontweight="bold", color=INK)
        ax1.plot(e["ci95"], [y - 0.17] * 2, color=INDIGO, lw=2.2, zorder=3)
        ax1.plot(e["mean"], y - 0.17, "o", color=INDIGO, ms=7.5, mec="black", mew=0.6, zorder=4)
        ax1.text(max(e["ci95"][1], 0) + 0.012, y - 0.17, f"{e['mean']:+.3f}", va="center", fontsize=8,
                 fontweight="bold", color=INK)
        out.append(("(a)", k, "known_gap_control", g["mean"], *g["ci95"], g["n"]))
        out.append(("(a)", k, "instruct_excess_template", e["mean"], *e["ci95"], e["n"]))
    ax1.axvline(0, color=INK, lw=0.9, zorder=2)
    ax1.axvline(0.10, color=GRAY_EC, lw=0.9, ls=":", zorder=2)
    ax1.text(0.103, 3.62, "validation bar 0.10", fontsize=7.5, color=GRAY_EC, va="center")
    ax1.set_yticks(ys); ax1.set_yticklabels([m[0] for m in MODELS_R], fontsize=8.5)
    ax1.set_xlim(-0.04, 0.72); ax1.set_ylim(-0.6, 3.85)
    ax1.set_xlabel("acting minus judging violating mass, 95% CI")
    ax1.set_title("(a) Positive control and gap on one axis", fontsize=10, loc="left")
    ax1.grid(True, axis="x", alpha=0.25); ax1.set_axisbelow(True)
    ax1.legend(handles=[
        Line2D([0], [0], marker="s", ls="", color=RED, mec="black", ms=8, label="known-gap control (operator orders the violation)"),
        Line2D([0], [0], marker="o", ls="-", color=INDIGO, mec="black", ms=7, label="pressure-attributable excess, own template")],
        loc="lower right", fontsize=7.5, frameon=True, framealpha=0.95)
    for y, (lab, k) in zip(ys, MODELS_R):
        e = inst[k]
        ax2.plot(e["ci95"], [y + 0.12] * 2, color=INDIGO, lw=2.2, zorder=3)
        ax2.plot(e["mean"], y + 0.12, "o", color=INDIGO, ms=7.5, mec="black", mew=0.6, zorder=4)
        ax2.text(e["ci95"][1] + 0.0015, y + 0.12, f"{e['mean']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]",
                 va="center", fontsize=7.5, color=INK)
        if k in base:
            bb = base[k]
            ax2.plot(bb["ci95"], [y - 0.2] * 2, color=GRAY, lw=1.6, zorder=3)
            ax2.plot(bb["mean"], y - 0.2, "D", mfc="white", mec=GRAY_EC, mew=1.2, ms=6.5, zorder=4)
            ax2.text(bb["ci95"][1] + 0.0015, y - 0.2, f"base, raw: {bb['mean']:+.3f}", va="center", fontsize=7,
                     color=GRAY_EC)
            out.append(("(b)", k, "base_raw_descriptive", bb["mean"], *bb["ci95"], bb["n"]))
    ax2.annotate("", xy=(-0.022, 2.12), xytext=(-0.022, 1.12),
                 arrowprops=dict(arrowstyle="-", color=INK, lw=1.0, connectionstyle="bar,fraction=-0.25"))
    ax2.text(-0.0285, 1.62, "same\nbase", fontsize=7.5, ha="right", va="center", color=INK)
    ax2.axvline(0, color=INK, lw=0.9, zorder=2)
    ax2.set_yticks(ys); ax2.set_yticklabels([])
    ax2.set_xlim(-0.035, 0.075); ax2.set_ylim(-0.6, 3.85)
    ax2.set_xlabel("pressure-attributable excess, 95% CI")
    ax2.set_title("(b) The gaps, zoomed; bases in the raw frame (descriptive)", fontsize=10, loc="left")
    ax2.grid(True, axis="x", alpha=0.25); ax2.set_axisbelow(True)
    fig.suptitle("The instrument is validated on every model; OLMo-3 and Meta's Llama-3.1 carry the gap, Tulu 3 does not; "
                 "Qwen2.5 does not on the whole panel",
                 fontsize=10.5)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.86, bottom=0.14, wspace=0.08)
    save(fig, "kdg_recipe", out, ["panel", "model", "quantity", "mean", "ci_lo", "ci_hi", "n"])


def fig_stages():
    """Figure 6 (2026-10-05): the template-valid stage profile on OLMo-3 (SFT, DPO, final under each
    checkpoint's own template) on the 586 model-free set (number of record) and the 136 screened set,
    the base as a raw-frame description, and the final checkpoint's positive control beside."""
    s586 = load("analysis_kdg_a8.json")["per_stage"]["E"]
    s136 = load("analysis_phase1_session_a.json")["C3"]["chat_secondary"]["E_prob"]
    base = load("analysis_a17_union.json")["three_cell_union"]["base"]["continuous"]["excess_paired"]
    kg = load("analysis_phase1_session_c.json")["known_gap"]["olmo3_instruct"]["g_band"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.1), gridspec_kw={"width_ratios": [1.45, 1]})
    out = []
    stages = [("sft", "after SFT"), ("dpo", "after DPO"), ("final", "after RL (final)")]
    # base: raw-frame description, set apart by a divider
    ax1.plot([0, 0], base["ci95"], color=GRAY, lw=1.6, zorder=3)
    ax1.plot(0, base["mean"], "D", mfc="white", mec=GRAY_EC, mew=1.3, ms=7.5, zorder=4)
    ax1.text(0.09, base["mean"], f"{base['mean']:.3f}", va="center", fontsize=8, color=GRAY_EC)
    out.append(("(a)", "base", "raw_frame_description", base["mean"], *base["ci95"], base["n"]))
    ax1.axvline(0.55, color=GRAY_EC, lw=0.9, ls=":", zorder=2)
    for i, (k, lab) in enumerate(stages, start=1):
        for dx, S, filled, name in ((-0.13, s586, True, "586_model_free"), (0.13, s136, False, "136_screened")):
            b = S[k]
            ax1.plot([i + dx] * 2, b["ci95"], color=INDIGO, lw=2.2, zorder=3)
            ax1.plot(i + dx, b["mean"], "o" if filled else "s", color=INDIGO,
                     mfc=INDIGO if filled else "white", mec="black" if filled else INDIGO,
                     mew=0.6 if filled else 1.5, ms=7.5, zorder=4)
            out.append(("(a)", k, name, b["mean"], *b["ci95"], b["n"]))
        ax1.text(i - 0.13, s586[k]["ci95"][1] + 0.002, f"{s586[k]['mean']:.3f}", ha="center",
                 va="bottom", fontsize=7.5, fontweight="bold", color=INK)
    ax1.axhline(0, color=INK, lw=0.9, zorder=2)
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(["base\n(raw frame,\ndescription)", "after SFT", "after DPO", "after RL\n(final)"],
                        fontsize=8.5)
    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylabel("pressure-attributable excess E, 95% CI")
    ax1.set_title("(a) OLMo-3 across post-training, each checkpoint in its own template", fontsize=10,
                  loc="left")
    ax1.grid(True, axis="y", alpha=0.25); ax1.set_axisbelow(True)
    ax1.legend(handles=[
        Line2D([0], [0], marker="o", ls="-", color=INDIGO, mec="black", ms=7,
               label="586 scenarios screened by no model (of record)"),
        Line2D([0], [0], marker="s", ls="-", color=INDIGO, mfc="white", mec=INDIGO, mew=1.5, ms=7,
               label="136 scenarios screened on the final model"),
        Line2D([0], [0], marker="D", ls="-", color=GRAY, mfc="white", mec=GRAY_EC, mew=1.3, ms=7,
               label="base, raw frame (description, not validated)")],
        loc="upper left", fontsize=7.5, frameon=True, framealpha=0.95)
    # (b) positive control and the gap on one axis (final checkpoint)
    fin = s586["final"]
    ax2.barh(1, kg["mean"], height=0.42, color=RED, edgecolor="black", linewidth=0.5, zorder=3)
    ax2.plot(kg["ci95"], [1, 1], color=INK, lw=1.2, zorder=4)
    ax2.text(kg["ci95"][1] + 0.012, 1, f"{kg['mean']:.3f} [{kg['ci95'][0]:.3f}, {kg['ci95'][1]:.3f}]",
             va="center", fontsize=8, fontweight="bold", color=INK)
    ax2.plot(fin["ci95"], [0, 0], color=INDIGO, lw=2.2, zorder=3)
    ax2.plot(fin["mean"], 0, "o", color=INDIGO, ms=7.5, mec="black", mew=0.6, zorder=4)
    ax2.text(fin["ci95"][1] + 0.012, 0, f"{fin['mean']:.3f} [{fin['ci95'][0]:.3f}, {fin['ci95'][1]:.3f}]",
             va="center", fontsize=8, fontweight="bold", color=INK)
    out.append(("(b)", "final", "known_gap_control", kg["mean"], *kg["ci95"], kg["n"]))
    out.append(("(b)", "final", "excess_586_model_free", fin["mean"], *fin["ci95"], fin["n"]))
    ax2.axvline(0, color=INK, lw=0.9, zorder=2)
    ax2.axvline(0.10, color=GRAY_EC, lw=0.9, ls=":", zorder=2)
    ax2.text(0.105, 1.55, "validation bar 0.10", fontsize=7.5, color=GRAY_EC, va="center")
    ax2.set_yticks([1, 0])
    ax2.set_yticklabels(["known-gap control\n(operator orders\nthe violation)", "pressure-attributable\nexcess (586)"],
                        fontsize=8.5)
    ax2.set_xlim(-0.03, 0.72); ax2.set_ylim(-0.6, 1.8)
    ax2.set_xlabel("acting minus judging violating mass, 95% CI")
    ax2.set_title("(b) The final checkpoint: the test sees 0.5; the gap is 0.02", fontsize=10, loc="left")
    ax2.grid(True, axis="x", alpha=0.25); ax2.set_axisbelow(True)
    fig.suptitle("On OLMo-3 the gap is present after SFT and no later post-training stage makes it detectably larger",
                 fontsize=10.5)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.85, bottom=0.2, wspace=0.42)
    save(fig, "kdg_stages", out, ["panel", "stage", "quantity", "mean", "ci_lo", "ci_hi", "n"])


def fig_deliberation():
    a, c = load("analysis_phase1_session_a.json"), load("analysis_phase1_session_c.json")
    dc, cf, ld = a["DOSE_CONTROLS"], a["C2_forced"], c["llama_dose"]
    olmo = [("reasoning - filler", cf["delta_dose2_minus_filler_forced"], INDIGO),
            ("reasoning - truncated filler", dc["delta_dose2_minus_TF"], INDIGO),
            ("brief reasoning - filler", cf["secondary_dose1_minus_filler_forced"], INDIGO),
            ("truncated filler - filler", dc["truncation_effect_TF_minus_filler"], GRAY),
            ("norm named - filler", dc["delta_NS_minus_filler"], INDIGO)]
    llama = [("reasoning - filler", ld["delta_dose2_minus_filler"], INDIGO),
             ("reasoning - truncated filler", ld["delta_dose2_minus_TF"], INDIGO),
             ("brief reasoning - filler", ld["secondary_dose1_minus_filler"], INDIGO),
             ("truncated filler - filler", ld["truncation_effect_TF_minus_filler"], GRAY)]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))
    out = []
    panels = ((axes[0], olmo, "(a) OLMo-3-7B-Instruct: truncated reasoning (8% finish)", "olmo", (-0.16, 0.09)),
              (axes[1], llama, "(b) Llama-3.1-8B-Instruct: mostly completed (90% finish)", "llama", (-0.45, 0.06)))
    for ax, rows, title, key, xl in panels:
        n = len(rows)
        for i, (lab, d, col) in enumerate(rows):
            y = n - 1 - i
            mk = "s" if col == GRAY else "o"
            ax.plot(d["ci95"], [y, y], color=col, lw=2.2, zorder=3)
            ax.plot(d["mean"], y, mk, color=col, ms=7.5, mec="black", mew=0.6, zorder=4)
            ax.text(d["mean"], y + 0.3, f"{d['mean']:+.3f}", ha="center", va="center", fontsize=8,
                    fontweight="bold", color=INK)
            out.append((key, lab, d["mean"], *d["ci95"], d["n"]))
        ax.axvline(0, color=INK, lw=0.9, zorder=2)
        ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5)
        ax.set_xlim(*xl); ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlabel("difference in violating mass at the forced answer, 95% CI")
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.grid(True, axis="x", alpha=0.25); ax.set_axisbelow(True)
    fig.suptitle("Reasoning about the stakes before acting lowers the violating choice more than a same-length "
                 "non-moral task does, on both models that carry the gap (separate scales; sizes not compared)",
                 fontsize=10.5)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.84, bottom=0.16, wspace=0.55)
    save(fig, "kdg_deliberation", out, ["model", "contrast", "mean", "ci_lo", "ci_hi", "n"])


if __name__ == "__main__":
    for f in (fig_ladder, fig_scatter, fig_strictness, fig_families, fig_f4, fig_three_cell, fig_stages,
              fig_recipe, fig_deliberation):
        f()
        print("ok", f.__name__)
