#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC G-A21 item 1: letter balance of primary 2 at permutation 0.

    python3 papers/kdg_panel/scripts/analyze_gptoss_letter_balance.py

Per scenario of the 586: the letters of the norm-consistent (L_c) and violating (L_v) options at
permutation 0 (``assign_letters(s, 0)``, checked against the order recorded on every primary-2 row).
The dose-0 letter marginals m_k(L): mean renormalised probability at letter L over all
``dl_chat_neutral`` dose-0 rows of the 586 (8 permutations, pod 3xlqmdx2mo3niz), separately for
k = 2 and 3 options. (i) chi-square of L_c against the within-scenario uniform expectation (one
statistic over the five (k, letter) cells, df 3; per-k reported beside it); (ii) the letter-prior
advantage Delta_s = m_k(L_c) - m_k(L_v), mean with a 10,000-draw bootstrap CI over scenarios (seed
0). Unbalanced iff (i) p < 0.01 or (ii)'s CI excludes 0 with |mean| >= 0.02; only then is primary
2 recomputed on the Delta_s <= 0 subset with per-(L_v, L_c) counts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_gptoss as G  # noqa: E402
import analyze_gptoss_primary2 as P2  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import assign_letters, letter_map, load_scenario_dir  # noqa: E402

N_BOOT = 10_000
P_BAR = 0.01
DELTA_BAR = 0.02


def chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-square distribution for integer df (closed form)."""
    if df % 2 == 0:
        h = x / 2
        return math.exp(-h) * sum(h**k / math.factorial(k) for k in range(df // 2))
    tail, term = math.erfc(math.sqrt(x / 2)), math.sqrt(x)
    for k in range(1, (df - 1) // 2 + 1):
        tail += 2 * math.exp(-x / 2) / math.sqrt(2 * math.pi) * term
        term *= x / (2 * k + 1)
    return tail


def marginals(ids: set[str]) -> dict[int, dict[str, float]]:
    """m_k(L): mean renormalised dose-0 probability at each letter position, per option count."""
    acc: dict[int, dict[str, list[float]]] = {}
    for r in A._rows(G.DEFAULT_DIR / "dl_chat_neutral.jsonl"):
        if r["scenario_id"] not in ids:
            continue
        L = sorted(r["option_logps"])
        lp = np.array([r["option_logps"][x] for x in L])
        p = np.exp(lp - lp.max())
        p /= p.sum()
        for x, q in zip(L, p):
            acc.setdefault(len(L), {}).setdefault(x, []).append(float(q))
    return {k: {x: float(np.mean(v)) for x, v in sorted(d.items())} for k, d in acc.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-dir", type=Path, default=A.OUT / "p2i" / "main" / "gpt_oss_20b")
    ap.add_argument("--rows-file", default="ds_main_reconstructed.jsonl")
    ap.add_argument("--out", type=Path, default=A.DATA / "analysis_gptoss_letter_balance.json")
    a = ap.parse_args(argv)
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    ids = set(json.loads((A.DATA / "gptoss_dose0_model_free_586.json").read_text())["ids"])
    S = {s.id: s for s in scen if s.id in ids}
    m = marginals(ids)

    per = {}
    for sid, s in sorted(S.items()):
        inv = {oid: x for x, oid in letter_map(assign_letters(s, 0)).items()}
        c = next(o.option_id for o in s.options if o.norm_status == "consistent")
        v = next(o.option_id for o in s.options if o.norm_status == "violating")
        k = len(s.options)
        per[sid] = {"k": k, "L_c": inv[c], "L_v": inv[v], "delta": m[k][inv[c]] - m[k][inv[v]]}

    # The permutation-0 order must be the one primary 2's rows carry (most probable failure: the
    # letters checked here are not the letters the medium-effort action was read at).
    rows = [r for r in A._rows(a.main_dir / a.rows_file)
            if r["stage"] == "main" and r["cell"] == "dl_chat_neutral" and r["seed"] == 0]
    mismatch = [r["scenario_id"] for r in rows if r["scenario_id"] in S
                and r["order"] != letter_map(assign_letters(S[r["scenario_id"]], 0))]
    if mismatch:
        raise SystemExit(f"permutation-0 order differs from primary-2 rows: {mismatch[:5]}")

    chi_cells, chi_tot, df_tot = {}, 0.0, 0
    for k in sorted({p["k"] for p in per.values()}):
        sub = [p["L_c"] for p in per.values() if p["k"] == k]
        exp = len(sub) / k
        obs = {x: sub.count(x) for x in "ABCDE"[:k]}
        stat = sum((o - exp) ** 2 / exp for o in obs.values())
        chi_cells[k] = {"n": len(sub), "observed": obs, "expected_each": exp, "chi2": stat,
                        "df": k - 1, "p": chi2_sf(stat, k - 1)}
        chi_tot += stat
        df_tot += k - 1
    p_chi = chi2_sf(chi_tot, df_tot)

    d = np.array([p["delta"] for p in per.values()])
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    lo, hi = (float(x) for x in np.quantile(boot, [0.025, 0.975]))
    excl = lo > 0 or hi < 0
    unbalanced = bool(p_chi < P_BAR or (excl and abs(d.mean()) >= DELTA_BAR))

    rep = {"rule": "KDG_GPTOSS_SPEC G-A21 item 1 (3ce9327)", "n": len(per),
           "dose0_letter_marginals": {str(k): v for k, v in m.items()},
           "chi_square": {"by_k": {str(k): v for k, v in chi_cells.items()}, "chi2": chi_tot,
                          "df": df_tot, "p": p_chi},
           "delta": {"mean": float(d.mean()), "ci95": [lo, hi], "n_boot": N_BOOT, "seed": 0,
                     "n_le_0": int((d <= 0).sum())},
           "unbalanced": unbalanced, "primary2_rows_order_checked": len(rows)}
    if unbalanced:
        keep = {sid for sid, p in per.items() if p["delta"] <= 0}
        label = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen}
        dose0 = {s: max(q, key=q.get) for s, q in P2.P._dose0_means().items() if s in keep}
        t = w = 0
        cells: dict[str, dict[str, int]] = {}
        for r in rows:
            s = r["scenario_id"]
            if s not in dose0 or not r.get("token_identity") or r.get("option_id") is None:
                continue
            l0, l1 = label[s].get(dose0[s]), label[s].get(r["option_id"])
            key = f"{per[s]['L_v']}{per[s]['L_c']}"
            c = cells.setdefault(key, {"toward": 0, "away": 0})
            if l0 == "violating" and l1 == "consistent":
                t += 1
                c["toward"] += 1
            elif l0 == "consistent" and l1 == "violating":
                w += 1
                c["away"] += 1
        rep["conditioned_primary2"] = {"subset": "Delta_s <= 0", "n_scenarios": len(keep),
                                       **P2.verdict(t, w), "by_Lv_Lc": cells}
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
