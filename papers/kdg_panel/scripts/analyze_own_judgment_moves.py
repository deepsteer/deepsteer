#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC G-A22 item 4 (descriptive): does deliberation move the action toward the model's
own judgment or toward the norm, on scenarios that fail the model's own P1-A9 screen?

    python3 papers/kdg_panel/scripts/analyze_own_judgment_moves.py

Subsets by the model's dose-0 letter-only judgment (``jl_chat_neutral``, 8 permutations):
divergent-violating (the violating option is J's argmax in >= 6 of 8), divergent-neutral (the
neutral option, >= 6 of 8), uncertain (no option reaches 6 of 8), among the scenarios failing
the screen on the judgment leg; the judgment-concordant screen failures and the screen passes
beside as reference rows. Per scenario, the action before deliberation against the action after
it; a move is "toward own judgment" (to J's majority option), "toward the norm" (to the norm-
consistent option) or other, with the opportunities beside (scenarios whose earlier action is
not already J's option / not already norm-consistent).

GPT-OSS-20B: dose-0 readout of record (argmax of the mean over 8 permutations of
``dl_chat_neutral``) against the medium-effort action (primary 2's permutation-0 rows, token
identity). OLMo-3-Instruct: on its deliberation set of record, argmax of the mean forced option
distribution over the 16 rollouts, filler of record against the reasoning arm. Two-sided exact
binomial at 0.5 on toward-norm vs toward-own-judgment, descriptive only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_dose_twin as DT  # noqa: E402
import analyze_gptoss as G  # noqa: E402
import analyze_gptoss_ds_power as P  # noqa: E402
import analyze_own_screen as OS  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402
import analyze_screen_rates as SR  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

MAJ = 6


def binom_two_sided(k: int, m: int) -> float:
    if m == 0:
        return 1.0
    lo = sum(math.comb(m, x) for x in range(0, min(k, m - k) + 1)) / 2**m
    return min(1.0, 2 * lo)


def judgments(d: Path, status: dict) -> dict[str, Counter]:
    """Per scenario: how many of the 8 permutations put J's argmax on each option."""
    out: dict[str, Counter] = {}
    for r in A._rows(d / "jl_chat_neutral.jsonl"):
        if r["scenario_id"] in status and r.get("option_id"):
            out.setdefault(r["scenario_id"], Counter())[r["option_id"]] += 1
    return out


def subset(sid: str, screened: set, Jc: Counter, lab: dict) -> tuple[str, str | None]:
    """(subset name, J's majority option or None) by the G-A22 item 4 rule."""
    top, n = Jc.most_common(1)[0] if Jc else (None, 0)
    jopt = top if n >= MAJ else None
    if sid in screened:
        return "screen_pass", jopt
    if jopt is not None and lab[jopt] == "consistent":
        return "concordant_gap_leg", jopt
    if jopt is None:
        return "uncertain", None
    return f"divergent_{lab[jopt]}", jopt


def moves(pairs: dict[str, tuple[str, str]], groups: dict[str, tuple[str, str | None]],
          status: dict) -> dict:
    """pairs: sid -> (action before, action after)."""
    rep: dict = {}
    for sid, (before, after) in sorted(pairs.items()):
        g, jopt = groups[sid]
        lab = status[sid]
        c = rep.setdefault(g, Counter())
        c["n"] += 1
        c["opp_toward_judgment"] += int(jopt is not None and before != jopt)
        c["opp_toward_norm"] += int(lab[before] != "consistent")
        if before == after:
            c["same"] += 1
        elif jopt is not None and after == jopt and lab[jopt] != "consistent":
            c["toward_own_judgment"] += 1
        elif lab[after] == "consistent":
            c["toward_norm"] += 1
        else:
            c["other"] += 1
    out = {}
    for g, c in rep.items():
        t, j = c["toward_norm"], c["toward_own_judgment"]
        out[g] = {**{k: c[k] for k in ("n", "same", "toward_norm", "toward_own_judgment", "other",
                                       "opp_toward_norm", "opp_toward_judgment")},
                  "crossing_count": t + j,
                  "rate_toward_norm": t / c["opp_toward_norm"] if c["opp_toward_norm"] else None,
                  "rate_toward_judgment": (j / c["opp_toward_judgment"]
                                           if c["opp_toward_judgment"] else None),
                  # J's option is the norm-consistent one here, so the two directions coincide
                  "binom_two_sided_descriptive": (None if g in ("screen_pass", "concordant_gap_leg")
                                                  else binom_two_sided(t, t + j))}
    return out


def option_means(d: Path, cell: str, status: dict, ids_chat, ids_sp) -> dict[str, dict[str, float]]:
    """Mean forced option distribution per scenario over its rollouts (as row_masses)."""
    rows = A._rows(d / f"{cell}.jsonl")
    z = np.load(d / f"{cell}.npz")
    lp = z["logp_decision"]
    acc: dict[str, dict[str, list[float]]] = {}
    for k, r in enumerate(rows):
        sid = r["scenario_id"]
        if sid not in status:
            continue
        letters = sorted(r["order"])
        vec = lp[k].astype(np.float64)
        m = np.exp(vec[[ids_chat[L] for L in letters]]) + np.exp(vec[[ids_sp[L] for L in letters]])
        m = m / m.sum()
        for L, q in zip(letters, m):
            acc.setdefault(sid, {}).setdefault(r["order"][L], []).append(float(q))
    return {s: {o: float(np.mean(v)) for o, v in d_.items()} for s, d_ in acc.items()}


def gptoss(status: dict, main_rows: Path) -> dict:
    d = G.DEFAULT_DIR
    J = judgments(d, status)
    scr = set(OS.screen(d, status))
    groups = {s: subset(s, scr, J.get(s, Counter()), status[s]) for s in status}
    dose0 = {s: max(m, key=m.get) for s, m in P._dose0_means().items() if s in status}
    pairs = {}
    for r in A._rows(main_rows):
        if r["stage"] != "main" or r["cell"] != "dl_chat_neutral" or r["seed"] != 0:
            continue
        if r.get("token_identity") and r.get("option_id") and r["scenario_id"] in dose0:
            pairs[r["scenario_id"]] = (dose0[r["scenario_id"]], r["option_id"])
    return {"subset_sizes": dict(Counter(g for g, _ in groups.values())),
            "moves": moves(pairs, groups, status), "n_pairs": len(pairs)}


def olmo3(status: dict) -> dict:
    import analyze_continuous as ac

    ids_chat, ids_sp = ac._tok_ids()
    m = DT.MODELS["olmo3_instruct"]
    D2 = option_means(m["prim"], "d_chat_dose2_bf_forced", status, ids_chat, ids_sp)
    F = option_means(m["prim"], "d_chat_dose2_filler_bf_forced", status, ids_chat, ids_sp)
    screen = set(json.loads((A.DATA / m["screen"]).read_text())["ids"])
    ids = sorted(s for s in screen if s in D2 and s in F)
    J = judgments(SR.MODELS["olmo3_final"], status)
    own = set(OS.screen(SR.MODELS["olmo3_final"], status))
    groups = {s: subset(s, own, J.get(s, Counter()), status[s]) for s in ids}
    pairs = {s: (max(F[s], key=F[s].get), max(D2[s], key=D2[s].get)) for s in ids}
    return {"n_dose_set": len(ids), "subset_sizes": dict(Counter(g for g, _ in groups.values())),
            "moves": moves(pairs, groups, status),
            "arms": "filler of record (d_chat_dose2_filler_bf_forced) -> d_chat_dose2_bf_forced"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-rows", type=Path,
                    default=A.OUT / "p2i" / "main" / "gpt_oss_20b" / "ds_main_reconstructed.jsonl")
    ap.add_argument("--out", type=Path, default=A.DATA / "analysis_own_judgment_moves.json")
    a = ap.parse_args(argv)
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = G._status(scen)
    rep = {"rule": "KDG_GPTOSS_SPEC G-A22 item 4 (3689315), descriptive",
           "gpt_oss_20b": gptoss(status, a.main_rows), "olmo3_instruct": olmo3(status),
           "llama31_instruct_meta": "deliberation set is its own screen; no screened-out scenarios"}
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
