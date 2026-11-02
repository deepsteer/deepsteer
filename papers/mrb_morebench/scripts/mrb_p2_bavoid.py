"""MRB-A16's B-avoidance check (MRB_PLAN v0.9.4), zero GPU -> p2_b_avoidance.json.

    mrb_p2_bavoid.py [--out papers/mrb_morebench/outputs/p2_b_avoidance.json]

ST's saved dose-0 letter reads for Muse Glimmer 30B (cell ``dl_chat_neutral``): the ``care`` rows (T1.6
care-DPO attached) against the ``none`` rows, matched by (scenario, rollout) with equal prompt hashes.
Per row and letter, p is the softmax of the three option log-probabilities; Δp = p(adapter) − p(none).
The OLS of Δp on position and content, both effect-coded (deviations from their means), gives each
position's effect at fixed content; CIs from scenario-clustered bootstrap resamples. ``base`` (base-DPO)
is reported beside, descriptive.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DOSE0 = REPO / "papers/st_live_reference/outputs/t16/muse_glimmer_30b/dose0"
LETTERS = ("A", "B", "C")
CONTENTS = ("consistent", "violating", "neutral")
N_BOOT, SEED = 2000, 20261102


def rows(arm: str) -> dict:
    out = {}
    for line in (DOSE0 / arm / "dl_chat_neutral.jsonl").read_text().splitlines():
        r = json.loads(line)
        out[(r["scenario_id"], r["rollout"])] = r
    return out


def probs(r: dict) -> dict[str, float]:
    lp = {L: r["option_logps"][L] for L in LETTERS}
    m = max(lp.values())
    z = sum(math.exp(v - m) for v in lp.values())
    return {L: math.exp(lp[L] - m) / z for L in LETTERS}


def stacked(arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """One observation per (row, letter): Δp, the position and content indices, the scenario."""
    base, other = rows("none"), rows(arm)
    keys = sorted(k for k in base if k in other)
    same = [k for k in keys if base[k]["prompt_ids_sha256"] == other[k]["prompt_ids_sha256"]]
    y, pos, con, scen = [], [], [], []
    for k in same:
        p0, p1 = probs(base[k]), probs(other[k])
        status = base[k]["norm_status_by_letter"]
        for i, L in enumerate(LETTERS):
            y.append(p1[L] - p0[L])
            pos.append(i)
            con.append(CONTENTS.index(status[L]))
            scen.append(k[0])
    info = {"rows_none": len(base), "rows_arm": len(other), "matched": len(keys),
            "prompt_hash_equal": len(same)}
    return np.array(y), np.array(pos), np.array(con), scen, info


def effects(y, pos, con) -> dict:
    """Two-way additive OLS with effect coding: Δp = μ + α_pos + γ_content, Σα = Σγ = 0."""
    def code(idx, k):
        x = np.zeros((len(idx), k - 1))
        for j in range(k - 1):
            x[:, j] = (idx == j).astype(float) - (idx == k - 1).astype(float)
        return x
    X = np.column_stack([np.ones(len(y)), code(pos, 3), code(con, 3)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha = list(b[1:3]) + [-b[1] - b[2]]
    gamma = list(b[3:5]) + [-b[3] - b[4]]
    return {"position": dict(zip(LETTERS, alpha)), "content": dict(zip(CONTENTS, gamma)),
            "raw_position": {L: float(y[pos == i].mean()) for i, L in enumerate(LETTERS)},
            "raw_content": {c: float(y[con == i].mean()) for i, c in enumerate(CONTENTS)}}


def read(arm: str, rng) -> dict:
    y, pos, con, scen, info = stacked(arm)
    point = effects(y, pos, con)
    uniq = sorted(set(scen))
    where = {s: np.flatnonzero(np.array(scen) == s) for s in uniq}
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(uniq), len(uniq), replace=True)
        idx = np.concatenate([where[uniq[i]] for i in pick])
        boots.append(effects(y[idx], pos[idx], con[idx]))

    def ci(group, key):
        v = [b[group][key] for b in boots]
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    out = {"info": info, "scenarios": len(uniq)}
    for group in ("position", "content", "raw_position", "raw_content"):
        out[group] = {k: {"est": round(float(v), 4), "ci95": ci(group, k)}
                      for k, v in point[group].items()}
    pe = out["position"]
    excl = {L: pe[L]["ci95"][0] > 0 or pe[L]["ci95"][1] < 0 for L in LETTERS}
    content_excl = any(v["ci95"][0] > 0 or v["ci95"][1] < 0 for v in out["content"].values())
    if excl["B"] and pe["B"]["est"] < 0:
        branch = "R_B: the adapter lowers B at fixed content" + (
            " (mixed: content effects also exclude 0)" if content_excl else "")
    elif not any(excl.values()):
        branch = ("R_content: no position effect excludes 0; the shift follows option content"
                  if content_excl else "no position or content effect excludes 0")
    else:
        branch = "other position effect: " + ", ".join(
            f"{L} {pe[L]['est']:+.3f}" for L in LETTERS if excl[L])
    if excl["A"] and pe["A"]["est"] > 0:
        branch += "; R_A: an A position effect above 0"
    out["branch"] = branch
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path,
                    default=REPO / "papers/mrb_morebench/outputs/p2_b_avoidance.json")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    res = {"rule": "MRB_PLAN v0.9.4", "n_boot": N_BOOT, "seed": SEED, "cell": "dl_chat_neutral",
           "care_of_record": read("care", rng), "base_descriptive": read("base", rng)}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    for arm in ("care_of_record", "base_descriptive"):
        r = res[arm]
        print(f"{arm}: {r['info']}; scenarios {r['scenarios']}")
        print("   position at fixed content:", {k: (v["est"], v["ci95"]) for k, v in r["position"].items()})
        print("   content:", {k: (v["est"], v["ci95"]) for k, v in r["content"].items()})
        print("   raw by position:", {k: v["est"] for k, v in r["raw_position"].items()})
        print("   ->", r["branch"])


if __name__ == "__main__":
    main()
