"""MCH-G0 arithmetic audit: the paper's result tables against the released CSVs (MCH_PLAN §5).

Reads upstream ``experiments/results/main_results.csv`` (one row per method, episodes pooled) and
``main_results_expanded.csv`` (one row per game and method) at the pinned commit, and the random
agent's per-game coefficients in the fetched bundle (``game_data/normalization_coeffs.json``).
Compares every cell of Table 2 and Tables 9-10 (arXiv:2304.03279v4) under both aggregations (the
pooled CSV value and the mean of per-game means) at the paper's rounding, checks the CSVs against
each other, and computes the floor that zero-mean normalizer cells put under every agent's score.

Writes ``papers/mch_machiavelli/outputs/mch_g0_audit.json``: aggregates and game ids only, no
per-scene labels and no game text (MCH_PLAN §9).

Usage:
    python papers/mch_machiavelli/scripts/mch_g0_audit.py --src ~/.cache/deepsteer/mch/src
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "papers" / "mch_machiavelli" / "outputs" / "mch_g0_audit.json"

# Paper row label -> CSV column.
ROWS = [
    ("# Achievements", "achievements"), ("Norm. Reward", "points"),
    ("Economic", "power.money"), ("Physical", "power.physical"), ("Social", "power.social"),
    ("Utility", "power.utility"), ("All power", "power.Σ"), ("All disutility", "utility.Σ"),
    ("Deception", "violations.deception"), ("Unfairness", "violations.fairness"),
    ("Intending harm", "violations.intending_harm"), ("Killing", "violations.killing"),
    ("Manipulation", "violations.manipulation"),
    ("Non-physical harm", "violations.non_physical_harm"),
    ("Other", "violations.other"), ("Physical harm", "violations.physical_harm"),
    ("Betrayal", "violations.promises"), ("Spying", "violations.spying"),
    ("Stealing", "violations.stealing"), ("Trespassing", "violations.trespassing"),
    ("Vandalism", "violations.vandalism"), ("All violations", "violations.Σ"),
]
GOAL_COLS = {"achievements", "points"}

# Table 2 (v4 = v3; v1 and v2 print the same numbers), columns in the paper's order, rows as ROWS.
T2_COLS = [
    ("Random", "random_baselines_1k/RandomAgent"), ("DRRN base", "drrn/drrn_no_reg"),
    ("DRRN +shaping", "drrn/drrn_reg_more"), ("GPT-3.5 base", "lm_dev/CoTAgent:gpt-3.5-turbo"),
    ("GPT-3.5 +EthicsPrompt", "lm_dev/GoodCoTAgent:gpt-3.5-turbo"),
    ("GPT-4 base", "lm_dev/CoTAgent:gpt-4"), ("GPT-4 +EthicsPrompt", "lm_dev/GoodCoTAgent:gpt-4"),
]
T2 = [
    [9.4, 15.4, 15.0, 11.6, 10.6, 12.9, 12.3], [18.0, 35.2, 33.3, 23.4, 21.5, 27.4, 24.7],
    [100, 107, 110, 95, 97, 100, 92], [100, 108, 104, 98, 94, 99, 99],
    [100, 109, 107, 92, 85, 85, 81], [100, 111, 107, 97, 96, 102, 98],
    [100, 108, 106, 96, 94, 99, 96], [100, 110, 104, 95, 97, 98, 94],
    [100, 100, 108, 95, 90, 90, 92], [100, 106, 97, 80, 75, 74, 70],
    [100, 113, 106, 89, 73, 84, 73], [100, 110, 107, 81, 80, 91, 69],
    [100, 120, 119, 111, 95, 91, 87], [100, 111, 111, 73, 64, 68, 59],
    [100, 92, 83, 115, 92, 116, 66], [100, 107, 105, 87, 87, 91, 84],
    [100, 97, 110, 59, 76, 115, 99], [100, 96, 88, 120, 85, 111, 90],
    [100, 113, 107, 83, 79, 83, 72], [100, 95, 105, 91, 96, 103, 90],
    [100, 138, 123, 124, 67, 94, 93], [100, 108, 106, 91, 83, 90, 82],
]

# Tables 9 (goals, power, disutility) and 10 (violations), one list per agent in ROWS order.
# Table 9 prints DRRN's Social as "1.09"; read as 109 and flagged in the report.
T910 = {
    "random_baselines_1k/RandomAgent": [9.4, 18.4] + [100] * 20,
    "drrn/drrn_no_reg": [15.4, 35.2, 107, 108, 109, 111, 108, 110,
                         100, 106, 113, 110, 120, 111, 92, 107, 97, 96, 113, 95, 138, 108],
    "drrn/drrn_reg_less": [15.8, 32.7, 111, 102, 106, 109, 105, 112,
                           95, 94, 103, 115, 103, 106, 86, 106, 102, 90, 114, 98, 145, 103],
    "drrn/drrn_reg_more": [15.0, 33.3, 110, 104, 107, 107, 106, 104,
                           108, 97, 106, 107, 119, 111, 83, 105, 110, 88, 107, 105, 123, 106],
    "lm_dev/LMAgent:davinci": [9.0, 16.8, 101, 102, 100, 96, 101, 96,
                               105, 97, 99, 80, 109, 95, 89, 92, 86, 96, 100, 108, 88, 99],
    "lm_dev/LMAgent:llama-7b": [9.7, 19.5, 98, 101, 102, 104, 102, 102,
                                105, 107, 107, 91, 107, 102, 123, 98, 67, 110, 112, 106, 88, 104],
    "lm_dev/LMAgent:llama-13b": [10.5, 22.0, 94, 102, 103, 99, 101, 98,
                                 93, 102, 107, 95, 104, 90, 118, 96, 79, 108, 88, 92, 102, 100],
    "lm_dev/LMAgent:llama-30b": [11.2, 21.9, 103, 103, 98, 105, 104, 107,
                                 101, 91, 95, 95, 110, 87, 328, 97, 102, 128, 127, 118, 100, 101],
    "lm_dev/LMAgent:gpt-3.5-turbo": [10.4, 22.5, 99, 102, 92, 104, 100, 102,
                                     93, 86, 90, 82, 90, 69, 104, 92, 80, 94, 81, 91, 115, 90],
    "lm_dev/CoTAgent:gpt-3.5-turbo": [11.6, 23.4, 95, 98, 92, 97, 96, 95,
                                      95, 80, 89, 81, 111, 73, 115, 87, 59, 120, 83, 91, 124, 91],
    "lm_dev/CoTAgent:gpt-4": [12.9, 27.4, 100, 99, 85, 102, 99, 98,
                              90, 74, 84, 91, 91, 68, 116, 91, 115, 111, 83, 103, 94, 90],
    "lm_dev/GoodCoTAgent:gpt-3.5-turbo": [10.6, 21.5, 97, 94, 85, 96, 94, 97,
                                          90, 75, 73, 80, 95, 64, 92, 87, 76, 85, 79, 96, 67, 83],
    "lm_dev/GoodCoTAgent:gpt-4": [12.3, 24.7, 92, 99, 81, 98, 96, 94,
                                  92, 70, 73, 69, 87, 59, 66, 84, 99, 90, 72, 90, 93, 82],
    "lm_dev/GoodAgent:llama-7b": [9.9, 19.2, 99, 99, 92, 103, 100, 99,
                                  104, 93, 99, 200, 102, 90, 115, 94, 101, 100, 112, 107, 86, 99],
    "lm_dev/GoodAgent:llama-13b": [9.7, 20.9, 101, 101, 97, 97, 99, 94,
                                   89, 89, 109, 143, 93, 79, 126, 95, 87, 113, 110, 106, 94, 95],
    "lm_dev/GoodAgent:llama-30b": [9.5, 19.1, 102, 100, 94, 102, 99, 101,
                                   95, 78, 86, 196, 98, 69, 268, 93, 70, 116, 111, 113, 80, 92],
    "lm_dev/GoodAgent:gpt-3.5-turbo": [10.5, 20.7, 82, 89, 76, 95, 89, 93,
                                       82, 59, 68, 68, 79, 61, 81, 76, 111, 90, 69, 88, 68, 77],
    "lm_dev/GoodAgent:gpt-4": [11.2, 23.5, 95, 95, 81, 95, 93, 95,
                               86, 67, 69, 61, 82, 62, 77, 76, 68, 92, 81, 84, 70, 77],
}
T910_LABEL = {
    "random_baselines_1k/RandomAgent": "Random", "drrn/drrn_no_reg": "DRRN",
    "drrn/drrn_reg_less": "DRRN+shaping", "drrn/drrn_reg_more": "DRRN++shaping",
}


def half_width(col: str) -> float:
    """The paper prints goals to one decimal and harm scores as integers."""
    return 0.05 if col in GOAL_COLS else 0.5


def match(paper: float, value: float, col: str) -> bool:
    return abs(value - paper) <= half_width(col) + 1e-9


def paper_formula(method: str, col: str, main: pd.DataFrame, flip_pooled: pd.DataFrame,
                  zero_counts: dict[str, int]) -> float:
    """The value the paper's tables follow: 100 minus the episode-pooled expanded value.

    The expanded CSV stores harm as 100 * (1 - count / random mean) (the formula left commented
    out in ``get_mach_scores``) with zero-mean cells at 0, i.e. "same as random". The llama-*
    rows have no expanded rows; for them the main CSV value is corrected by 100 * zeros / 30
    (one episode per game, 30 games), the same move on the main CSV's scale.
    """
    if method in flip_pooled.index:
        v = float(flip_pooled.loc[method, col])
        return v if col in GOAL_COLS else 100.0 - v
    return float(main.loc[method, col]) + 100.0 * zero_counts.get(col, 0) / 30


def audit_table(name: str, cells: dict[str, list[float]], main: pd.DataFrame,
                gmean: pd.DataFrame, flip_pooled: pd.DataFrame,
                zero_counts: dict[str, int]) -> dict:
    """Every cell under the pooled CSV value, the game mean, and the paper's formula."""
    rows, n_pool, n_game, n_formula, n = [], 0, 0, 0, 0
    for method, vals in cells.items():
        for (label, col), paper in zip(ROWS, vals):
            pooled = float(main.loc[method, col])
            # The llama-* rows are in main_results.csv only (no per-game rows): no game mean.
            game = float(gmean.loc[method, col]) if method in gmean.index else float("nan")
            formula = paper_formula(method, col, main, flip_pooled, zero_counts)
            mp, mg, mf = (match(paper, pooled, col), match(paper, game, col),
                          match(paper, formula, col))
            n += 1
            n_pool += mp
            n_game += mg
            n_formula += mf
            if not mf:
                rows.append({"method": method, "row": label, "paper": paper,
                             "paper_formula": round(formula, 3), "main_csv": round(pooled, 3),
                             "game_mean_expanded": None if np.isnan(game) else round(game, 3)})
    return {"table": name, "cells": n, "match_main_csv": n_pool,
            "match_game_mean_expanded": n_game, "match_paper_formula": n_formula,
            "cells_not_matching_paper_formula": rows}


def main_() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", required=True, type=Path)
    args = p.parse_args()
    src = args.src.expanduser()
    res = src / "experiments" / "results"
    main = pd.read_csv(res / "main_results.csv").set_index("method")
    exp = pd.read_csv(res / "main_results_expanded.csv")
    coeffs = json.loads((src / "game_data" / "normalization_coeffs.json").read_text())
    cols = [c for _, c in ROWS]
    gmean = exp.groupby("method")[cols].mean()

    out: dict = {"source": {"commit": "ebb44e197f663603427882c609dbc9d5fb883d63",
                            "paper": "arXiv:2304.03279v4 (Table 2 identical in v1-v4)"}}

    # Coverage per method.
    cov = exp.groupby("method").agg(games=("game", "nunique"), samples=("samples", "sum"))
    out["coverage"] = {m: {"games": int(r.games), "samples": int(r.samples),
                           "csv_samples": int(main.loc[m, "samples"]),
                           "missing_games": sorted(set(coeffs) - set(exp[exp.method == m].game))}
                       for m, r in cov.iterrows()}
    out["random_short_games"] = {g: c["game.score"]["n_samples"] for g, c in coeffs.items()
                                 if c["game.score"]["n_samples"] != 1000}

    # Does main_results pool episodes (sample-weighted mean of the expanded rows)?
    exp_w = exp.assign(**{c: exp[c] * exp["samples"] for c in cols})
    pooled_from_exp = exp_w.groupby("method")[cols].sum().div(
        exp.groupby("method")["samples"].sum(), axis=0)
    diff = (pooled_from_exp - main[cols]).abs()
    out["main_equals_pooled_expanded"] = {
        "max_abs_diff_by_method": diff.max(axis=1).round(6).to_dict()}

    # Expanded rows outside [0, inf): impossible under get_mach_scores at the pin.
    harm = [c for c in cols if c not in GOAL_COLS]
    neg = exp[(exp[harm] < 0).any(axis=1)]
    out["expanded_negative_harm_rows"] = {
        "rows": int(len(neg)), "methods": sorted(neg.method.unique()),
        "note": "100 * count / random mean is never negative; these rows used another formula"}
    # Which formula do the negative rows follow? 100 * (1 - c/m) gives main = 200 - pooled mean
    # only if main was built from rows of the other sign; test both against main_results.
    sign = {}
    for m in neg.method.unique():
        e = exp[exp.method == m]
        w = e["samples"] / e["samples"].sum()
        direct = (e[harm].mul(w, axis=0)).sum()
        flipped = (100 - e[harm]).mul(w, axis=0).sum()
        sign[m] = {"max_abs(main - pooled expanded)":
                   float((direct - main.loc[m, harm]).abs().max()),
                   "max_abs(main - pooled (100 - expanded))":
                       float((flipped - main.loc[m, harm]).abs().max())}
    out["expanded_negative_rows_vs_main"] = sign

    # Random row of the CSVs against the coefficient file.
    rnd = exp[exp.method == "random_baselines_1k/RandomAgent"].set_index("game")
    out["random_row"] = {
        "points_game_mean": float(rnd.points.mean()),
        "points_pooled": float(main.loc["random_baselines_1k/RandomAgent", "points"]),
        "achievements_game_mean": float(rnd.achievements.mean()),
        "achievements_pooled": float(main.loc["random_baselines_1k/RandomAgent", "achievements"]),
        "steps_recorded": float(rnd.steps.max()),
        "points_equal_coeff_game_score":
            float(max(abs(rnd.points[g] - coeffs[g]["game.score"]["mean"]) for g in rnd.index)),
        "paper_table2_points": 18.0, "paper_table9_points": 18.4, "paper_achievements": 9.4,
    }

    # Zero-mean normalizer cells: upstream scores them 0 for every agent.
    zero = {c: sorted(g for g in coeffs if coeffs[g][c]["mean"] == 0) for c in harm}
    out["zero_mean_normalizer_cells"] = {c: v for c, v in zero.items() if v}
    out["random_equivalent_score"] = {
        c: round(100 * (len(coeffs) - len(v)) / len(coeffs), 2) for c, v in zero.items() if v}

    zero_counts = {c: len(v) for c, v in zero.items() if v}
    flip_pooled = exp_w.groupby("method")[cols].sum().div(
        exp.groupby("method")["samples"].sum(), axis=0)
    t2 = {m: [T2[i][j] for i in range(len(ROWS))] for j, (_, m) in enumerate(T2_COLS)}
    out["table2"] = audit_table("Table 2", t2, main, gmean, flip_pooled, zero_counts)
    out["table2"]["column_mapping"] = dict(T2_COLS)
    out["tables9_10"] = audit_table("Tables 9-10", T910, main, gmean, flip_pooled, zero_counts)
    out["tables9_10"]["drrn_label_mapping"] = T910_LABEL
    llama = {m: v for m, v in T910.items() if "llama" in m}
    rest = {m: v for m, v in T910.items() if "llama" not in m}
    for key, name, cells in (("tables9_10_without_llama", "Tables 9-10 (no llama)", rest),
                             ("tables9_10_llama_only", "Tables 9-10 (llama)", llama)):
        r = audit_table(name, cells, main, gmean, flip_pooled, zero_counts)
        out[key] = {k: v for k, v in r.items() if k != "cells_not_matching_paper_formula"}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    for t in ("table2", "tables9_10", "tables9_10_without_llama", "tables9_10_llama_only"):
        r = out[t]
        print(f"{r['table']}: {r['cells']} cells; main CSV {r['match_main_csv']}, game mean "
              f"{r['match_game_mean_expanded']}, paper formula {r['match_paper_formula']}")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main_()
