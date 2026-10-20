#!/usr/bin/env python3
"""T1.2, the constancy read (ST_PLAN §4, Amendment 12; zero GPU on T1.1's bank).

    python3 papers/st_live_reference/scripts/constancy_read.py \
        papers/st_live_reference/outputs/t11/gpt_oss_20b

Rule of record (Amendment 12 item 3). At L* (block 12, hidden index 13; bank layer slot 2) the
anchor-axis margin m = <h, a> / |h| at each banked ``onset`` and ``clause_end`` position; per
rollout the mean over the class's positions; per scenario the mean over its protect rollouts minus
the mean over its rule rollouts (defer and unparsed rollouts never enter); over the H1_train
scenarios with at least one of each, the mean difference with a scenario bootstrap (10,000, seed
0). Pick: the class with the larger |difference| whose CI excludes 0, default ``onset``; a negative
picked difference with its CI excluding 0 is flagged (L-anchor's expected branch flips to the null).
``decision`` is read between scenarios only: Spearman across H1_train of the scenario's mean
END_OF_PROMPT margin (T1.1's dose-0 read, eight permutations) with its sampled P_none. Beside:
the same with ``anchor_axis_end_of_prompt`` (V_care_alt), and the within-scenario table at all five
banked layers (off-layer rows project onto block 12's axis; descriptive, no switching). Also the
outcome-blind inputs the T1.3 pod needs: the median residual norm at L* per class, and the PR of the
class's residual sample (position validity). Writes ``data/t12_constancy_v1.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIRS = HERE.parent / "data" / "directions" / "gpt_oss_20b_eop_L12.npz"
OUT = HERE.parent / "data" / "t12_constancy_v1.json"
BANK_BLOCKS = (6, 9, 12, 15, 18)
L_STAR_SLOT = 2  # block 12
CLASSES = {"onset": 1, "clause_end": 2}
AXES = ("anchor_axis_t_inst", "anchor_axis_end_of_prompt")
ACTION = {"violating": "protect", "consistent": "rule"}
N_BOOT, SEED = 10_000, 0


def boot(x: np.ndarray, stat=np.mean) -> dict:
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    idx = np.random.default_rng(SEED).integers(0, len(x), (N_BOOT, len(x)))
    d = stat(x[idx], axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    pt = float(stat(x))
    return {"mean": pt, "ci95": [float(lo), float(hi)], "n": int(len(x)),
            "point_in_ci": bool(lo <= pt <= hi), "mde": float(2.8 * (hi - lo) / (2 * 1.96))}


def rollout_means(chunks, axes: dict[str, np.ndarray]) -> dict:
    """{(axis, slot, class): {rollout row: mean margin}} over the bank chunks."""
    out: dict = {}
    for z in chunks:
        r = z["resid"].astype(np.float32)  # [n, 5, H]
        nrm = np.linalg.norm(r, axis=-1).clip(1e-6)  # [n, 5]
        for an, a in axes.items():
            m = (r @ a.astype(np.float32)) / nrm  # [n, 5]
            for cn, code in CLASSES.items():
                sel = (z["cls"] & code) > 0
                rows = z["row"][sel]
                for s in range(m.shape[1]):
                    acc = out.setdefault((an, s, cn), {})
                    for row in np.unique(rows):
                        v = m[sel][rows == row, s]
                        sm, n = acc.get(int(row), (0.0, 0))
                        acc[int(row)] = (sm + float(v.sum()), n + len(v))
    return {k: {row: sm / n for row, (sm, n) in v.items()} for k, v in out.items()}


def scenario_diffs(means: dict[int, float], rollouts: list[dict], split: str = "train") -> dict:
    """Per scenario (``split``, at least one protect and one rule rollout with a value): mean
    protect margin minus mean rule margin."""
    by: dict[str, dict[str, list]] = {}
    for i, r in enumerate(rollouts):
        act = ACTION.get(r.get("norm_status"))
        if r.get("split") != split or act is None or i not in means:
            continue
        by.setdefault(r["scenario_id"], {"protect": [], "rule": []})[act].append(means[i])
    return {s: float(np.mean(d["protect"]) - np.mean(d["rule"]))
            for s, d in sorted(by.items()) if d["protect"] and d["rule"]}


def pick(stats: dict[str, dict]) -> dict:
    """The rule: the class with the larger |difference| whose CI excludes 0, else ``onset``."""
    ok = {c: s for c, s in stats.items()
          if s["mean"] is not None and (s["ci95"][0] > 0 or s["ci95"][1] < 0)}
    cls = max(ok, key=lambda c: abs(ok[c]["mean"])) if ok else "onset"
    s = stats[cls]
    neg = bool(cls in ok and s["mean"] < 0)
    return {"class": cls, "by_rule": bool(ok), "negative_flag": neg,
            "reason": ("larger |difference| with CI excluding 0" if ok else
                       "no class separates (CI includes 0): default onset")}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def decision_between(d0_npz: Path, d0_rows: list[dict], rollouts: list[dict], axes: dict) -> dict:
    """Between scenarios: Spearman of the mean END_OF_PROMPT margin (8 permutations) with the
    scenario's sampled P_none, over H1_train, scenario bootstrap."""
    z = np.load(d0_npz)
    r = z["resid_eop"][:, L_STAR_SLOT, :].astype(np.float32)
    nrm = np.linalg.norm(r, axis=-1).clip(1e-6)
    split = {x["scenario_id"]: x["split"] for x in rollouts}
    pn: dict[str, list] = {}
    for x in rollouts:
        if x.get("norm_status") is not None:
            pn.setdefault(x["scenario_id"], []).append(x["norm_status"] == "violating")
    out = {}
    for an, a in axes.items():
        m = (r @ a.astype(np.float32)) / nrm
        per: dict[str, list] = {}
        for row, v in zip(d0_rows, m):
            per.setdefault(row["scenario_id"], []).append(float(v))
        ids = sorted(s for s in per if split.get(s) == "train" and s in pn)
        mx = np.array([np.mean(per[s]) for s in ids])
        py = np.array([np.mean(pn[s]) for s in ids])
        rng = np.random.default_rng(SEED)
        bs = []
        for _ in range(2000):
            k = rng.integers(0, len(ids), len(ids))
            if np.std(mx[k]) > 0 and np.std(py[k]) > 0:
                bs.append(spearman(mx[k], py[k]))
        out[an] = {"spearman": spearman(mx, py), "ci95": [float(np.percentile(bs, 2.5)),
                                                          float(np.percentile(bs, 97.5))],
                   "n": len(ids), "boot": 2000}
    return out


def participation_ratio(x: np.ndarray) -> float:
    xc = x - x.mean(0)
    s = np.linalg.svd(xc, compute_uv=False) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum())


def main() -> None:
    d = Path(sys.argv[1])
    rollouts = [json.loads(x) for x in (d / "gen" / "h1_rollouts_read.jsonl").open()]
    zd = np.load(DIRS)
    axes = {k: zd[k] / np.linalg.norm(zd[k]) for k in AXES}
    chunks = [np.load(f) for f in sorted((d / "bank").glob("bank_*.npz"))]
    means = rollout_means(chunks, axes)
    table: dict = {}
    for (an, s, cn), m in sorted(means.items()):
        diffs = scenario_diffs(m, rollouts)
        table.setdefault(an, {}).setdefault(f"block {BANK_BLOCKS[s]}", {})[cn] = boot(
            np.array(list(diffs.values())))
    primary = table[AXES[0]][f"block {BANK_BLOCKS[L_STAR_SLOT]}"]
    choice = pick(primary)
    # outcome-blind pod inputs at L*: median norm and PR of each class's residual sample
    rng = np.random.default_rng(SEED)
    norms, prs = {}, {}
    for cn, code in CLASSES.items():
        xs = np.concatenate([z["resid"][(z["cls"] & code) > 0, L_STAR_SLOT].astype(np.float32)
                             for z in chunks])
        norms[cn] = float(np.median(np.linalg.norm(xs, axis=-1)))
        prs[cn] = participation_ratio(xs[rng.choice(len(xs), min(len(xs), 4000), replace=False)])
    d0_rows = [json.loads(x) for x in (d / "readout" / "dose0_h1.jsonl").open()]
    out = {"rule": "ST_PLAN Amendment 12 item 3", "l_star": "block 12 (hidden index 13)",
           "axis_of_record": AXES[0], "pick": choice, "within_scenario": table,
           "off_layer_note": "blocks 6, 9, 15, 18 project onto block 12's axis (descriptive)",
           "decision_between_scenarios": decision_between(d / "readout" / "dose0_h1.npz", d0_rows,
                                                          rollouts, axes),
           "median_norm_at_l_star": norms, "pr_at_l_star": prs,
           "alphas": {cn: [f * norms[cn] for f in (0.05, 0.1, 0.2, 0.4)] for cn in CLASSES},
           "inputs": {"directions_sha256": hashlib.sha256(DIRS.read_bytes()).hexdigest(),
                      "rollouts_sha256": hashlib.sha256(
                          (d / "gen" / "h1_rollouts_read.jsonl").read_bytes()).hexdigest()}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    p = primary[choice["class"]]
    print(f">> constancy read: pick {choice['class']} ({choice['reason']}); difference "
          f"{p['mean']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] n {p['n']}; "
          f"negative flag {choice['negative_flag']}")


if __name__ == "__main__":
    main()
