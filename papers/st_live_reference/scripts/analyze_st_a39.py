#!/usr/bin/env python3
"""ST-A39's position split of Muse's T1.6 protection (zero GPU; registered in ANOMALIES ST-A39).

    python3 papers/st_live_reference/scripts/analyze_st_a39.py [--out data/st_a39_read_v1.json]

On T1.6's rows of record (``outputs/t16/muse_glimmer_30b``) and the headline's own scenario sets (the 54
engaged H1_held ids; the 60 specificity ids): the protect option's rotation over letter positions; ΔP of
record (``analyze_t15.p_record``) and specificity (the violating option's dose-0 mass, the rule of
``analyze_t15.specificity``) recomputed per position of the target option, scenario-paired against
``none``, with ST's bootstrap (``t13_driver.boot``); the position-balanced mean and the contrast
A − mean(B, C) on scenarios carrying every position. The registered branch: B_hold, B_A or mixed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
import analyze_t16 as A16  # noqa: E402
import t13_driver as T13  # noqa: E402

T16 = HERE.parent / "outputs" / "t16" / "muse_glimmer_30b"
DATA = HERE.parent / "data"
LETTERS = ("A", "B", "C")


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def protect_ids(dose0: list[dict], gen: list[dict]) -> dict[str, str]:
    """Each scenario's protect (``violating``) option id: from the dose-0 rows, which carry every
    option's status by letter with the order (a scenario's protect option need never be chosen in
    generation); checked against every generation row that chose a ``violating`` option."""
    out: dict[str, str] = {}
    for r in dose0:
        L = next(x for x, st in r["norm_status_by_letter"].items() if st == "violating")
        prev = out.setdefault(r["scenario_id"], r["order"][L])
        if prev != r["order"][L]:
            raise SystemExit(f"{r['scenario_id']}: two protect options ({prev}, {r['order'][L]})")
    for r in gen:
        if r.get("norm_status") == "violating" and r["scenario_id"] in out:
            assert out[r["scenario_id"]] == r["option_id"], r["scenario_id"]
    return out


def letter_of(order: dict[str, str], option: str) -> str:
    return next(L for L, o in order.items() if o == option)


def gen_by_position(rows: list[dict], protect: dict[str, str],
                    ids: set[str]) -> dict[str, dict[str, float]]:
    """``p_record``'s rule per (scenario, protect position) on the headline's scenarios: protect
    share, truncation as non-protect."""
    by: dict[tuple, list] = {}
    for r in rows:
        if r["scenario_id"] not in ids:
            continue
        key = (r["scenario_id"], letter_of(r["order"], protect[r["scenario_id"]]))
        if r.get("norm_status") is not None:
            by.setdefault(key, []).append(r["norm_status"] == "violating")
        elif r.get("status") == "truncated":
            by.setdefault(key, []).append(False)
    out: dict[str, dict[str, float]] = {L: {} for L in LETTERS}
    for (s, L), v in by.items():
        out[L][s] = float(np.mean(v))
    return out


def spec_by_position(rows: list[dict], ids: set[str]) -> dict[str, dict[str, float]]:
    """The violating option's renormalized dose-0 mass per (scenario, violating position)."""
    by: dict[tuple, list] = {}
    for r in rows:
        if r["scenario_id"] not in ids:
            continue
        letters = sorted(r["option_logps"])
        m = np.exp([r["option_logps"][x] for x in letters])
        m = m / m.sum()
        for x, q in zip(letters, m):
            if r["norm_status_by_letter"][x] == "violating":
                by.setdefault((r["scenario_id"], x), []).append(float(q))
    out: dict[str, dict[str, float]] = {L: {} for L in LETTERS}
    for (s, L), v in by.items():
        out[L][s] = float(np.mean(v))
    return out


def split(arm: dict[str, dict[str, float]], none: dict[str, dict[str, float]]) -> dict:
    res = {}
    for L in LETTERS:
        k = sorted(set(arm[L]) & set(none[L]))
        res[L] = T13.boot([arm[L][s] - none[L][s] for s in k])
    full = sorted(set.intersection(*[set(arm[L]) & set(none[L]) for L in LETTERS]))
    d = {s: {L: arm[L][s] - none[L][s] for L in LETTERS} for s in full}
    res["balanced"] = T13.boot([np.mean(list(v.values())) for v in d.values()])
    res["A_minus_mean_BC"] = T13.boot([v["A"] - (v["B"] + v["C"]) / 2 for v in d.values()])
    res["scenarios_with_every_position"] = len(full)
    return res


def branch(res: dict) -> str:
    """ST-A39's registered branches."""
    above = {L: res[L]["ci95"][0] is not None and res[L]["ci95"][0] > 0 for L in LETTERS}
    lo, hi = res["A_minus_mean_BC"]["ci95"]
    covers_bc = any(res[L]["ci95"][0] is not None and res[L]["ci95"][0] <= 0 for L in ("B", "C"))
    if lo is not None and lo > 0 and covers_bc:
        return "B_A: the effect concentrates where the target option sits at A"
    if all(above.values()) and lo is not None and lo > 0:
        return "mixed: the effect holds at every position and is larger at A"
    if all(above.values()) and lo is not None and lo <= 0 <= hi:
        return "B_hold: the effect holds at every position and is not position-carried"
    return "unregistered pattern (an ANOMALIES note before wording)"


def rotation(rows: list[dict], target: dict[str, str] | None, ids: set[str]) -> dict:
    """Rows with the target option at each letter, overall and the per-scenario spread."""
    pos = []
    per: dict[str, Counter] = {}
    for r in rows:
        s = r["scenario_id"]
        if s not in ids:
            continue
        if target is not None:
            if s not in target:
                continue
            L = letter_of(r["order"], target[s])
        else:
            L = next(x for x, st in r["norm_status_by_letter"].items() if st == "violating")
        pos.append(L)
        per.setdefault(s, Counter())[L] += 1
    shares = Counter(pos)
    n_rows = Counter(sum(c.values()) for c in per.values())
    every = sum(1 for c in per.values() if all(c[L] > 0 for L in LETTERS))
    return {"rows": len(pos), "by_position": {L: shares[L] for L in LETTERS},
            "share": {L: round(shares[L] / max(len(pos), 1), 3) for L in LETTERS},
            "scenarios": len(per), "rows_per_scenario": dict(n_rows),
            "scenarios_with_every_position": every,
            "per_scenario_rows_at_A": dict(sorted(Counter(c["A"] for c in per.values()).items()))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DATA / "st_a39_read_v1.json")
    a = ap.parse_args()
    rec = json.loads((T16 / "t16_record.json").read_text())
    eng = set(rec["phases"]["base_hf"]["engaged_held"])
    spec_ids = set(json.loads((DATA / "t15_specificity_v1.json").read_text())["ids"])
    e = A16.engine_of(rec, "care")
    care = jsonl(A16.rows_of(T16, "care", e, "gen"))
    none = jsonl(A16.rows_of(T16, "none", e, "gen"))
    dose0 = jsonl(T16 / "dose0" / "none" / "dl_chat_neutral.jsonl")
    protect = protect_ids(dose0, care + none)
    missing = sorted(eng - set(protect))
    if missing:
        raise SystemExit(f"engaged scenarios with no protect option on record: {missing[:5]}")
    sc, sn = jsonl(T16 / "spec" / "care" / "dl_chat_neutral.jsonl"), \
        jsonl(T16 / "spec" / "none" / "dl_chat_neutral.jsonl")
    h1 = split(gen_by_position(care, protect, eng), gen_by_position(none, protect, eng))
    sp = split(spec_by_position(sc, spec_ids), spec_by_position(sn, spec_ids))
    res = {"rule": "ANOMALIES ST-A39 (registered 2026-11-02)", "engine": e,
           "rotation": {"h1_held_care": rotation(care, protect, eng),
                        "h1_held_none": rotation(none, protect, eng),
                        "specificity_care": rotation(sc, None, spec_ids),
                        "specificity_none": rotation(sn, None, spec_ids)},
           "headline_of_record": {"dP_h1_held": json.loads((DATA / "t16_read_v1.json").read_text())[
               "arms"]["care"]["dP_record"]["mean"] if (DATA / "t16_read_v1.json").exists()
               else None},
           "h1_held_dP_by_protect_position": h1, "h1_branch": branch(h1),
           "specificity_by_violating_position": sp}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    for name, rot in res["rotation"].items():
        print(f"rotation {name}: {rot}")
    f = lambda d: f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] n={d['n']}"  # noqa: E731
    for name, r in (("H1_held ΔP", h1), ("specificity", sp)):
        print(f"{name}: " + " | ".join(f"{L} {f(r[L])}" for L in LETTERS)
              + f" | balanced {f(r['balanced'])} | A−mean(B,C) {f(r['A_minus_mean_BC'])}")
    print("H1 branch:", res["h1_branch"])


if __name__ == "__main__":
    main()
