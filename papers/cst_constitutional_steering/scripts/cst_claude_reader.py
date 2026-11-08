#!/usr/bin/env python3
"""Claude's blind reads of S0a rollouts through the account login (Amendments 9 and 11).

    cst_claude_reader.py stenet --arm a0|a0s [--workers 3] [--max N]   # S_tenet of record
    cst_claude_reader.py frontier [--workers 3] [--max N]              # the second opinion, Claude
    cst_claude_reader.py report --arm a0|a0s | --frontier              # coverage and stops

- stenet: the seeded 200 (``data/stenet_subset_v1.json``) × k 4 on the operator cell, per arm.
- frontier: the 200 rollouts of ``data/s0a_samples_v1.json`` ``frontier_200`` (both arms).

Each rollout gets the scorer of record's blind prompt (``cst_judge.judge_prompt``, ``answer=None``,
the subject's letters), one rollout per call. Amendment 11 addendum item 3: up to three attempts per
rollout; a rollout still unscored after three is reported as unscored, never dropped; the unscored
rate stands beside every number read from it; the independence of stops from scenario family and
rollout index is re-checked on the full read. Raw CLI records are cached per rollout under
``outputs_raw/claude_reader/<read>/`` (attempt files ``.bad<n>``), so a run resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cst_claude_retest as CR  # noqa: E402  (claude_json; sets the shared sys.path)

CST = HERE.parent
S0A = CST / "outputs" / "s0a"
RAW = CST / "outputs_raw" / "claude_reader"
ATTEMPTS = 3


def rollouts(read: str, arm: str | None) -> list[dict]:
    """The rollouts a read covers, from the S0a generation chunks."""
    if read == "stenet":
        keep = {f"{arm}|panel|operator|{sid}|{k}" for sid in json.loads(
            (CST / "data/stenet_subset_v1.json").read_text())["items"] for k in range(4)}
        arms = [arm]
    else:
        keep = set(json.loads((CST / "data/s0a_samples_v1.json").read_text())["frontier_200"])
        arms = ["a0", "a0s"]
    rows = []
    for a in arms:
        for f in sorted((S0A / "gen" / a).glob("panel_operator_*.jsonl")):
            rows += [r for r in map(json.loads, open(f)) if r["rollout_id"] in keep]
    return sorted(rows, key=lambda r: r["rollout_id"])


def prompt(r: dict) -> tuple[str, dict]:
    import cst_judge as J
    import scenario_labels as L

    from deepsteer.constitutions import TenetSet

    if not hasattr(prompt, "_ctx"):
        tenets = TenetSet.load(CST / "data/tenets_C_anth_v2.json").by_id()
        labs = {f: {x["scenario_id"]: x for x in json.loads(
            (CST / f"data/scenario_labels_v4_{f}.json").read_text())["labels"]}
            for f in ("claude", "codex")}
        _, by, _ = L._ordered_and_opts()  # noqa: SLF001
        prompt._ctx = (tenets, labs, by, L.DEFERENCE)
    tenets, labs, by, dfr = prompt._ctx
    sid = r["scenario_id"]
    s, _ = by[sid]
    short = sorted(set(labs["claude"][sid]["shortlist"]) | set(labs["codex"][sid]["shortlist"])
                   | set(dfr))
    letters = {o: L_ for L_, o in r["order"]}
    text = J.judge_prompt(s.agent_text, [(o, s.option(o).text) for _, o in r["order"]],
                          r["text"].split("</think>")[0], None,
                          [(t, tenets[t].claim) for t in short], letters=letters)
    return text, {"options": set(letters), "shortlist": short, "letters": letters}


def read_one(r: dict, d: Path, call=None) -> str:
    """Up to ``ATTEMPTS`` calls; returns "scored", "unscored" or "cached"."""
    import cst_judge as J

    call = call or CR.claude_json
    f = d / f"{r['rollout_id'].replace('|', '__')}.json"
    if f.exists():
        return "cached"
    text, meta = prompt(r)
    tries = sorted(d.glob(f.stem + ".bad*.json"))
    for attempt in range(len(tries) + 1, ATTEMPTS + 1):
        rec = call(J.SYS, text)
        res = rec.get("result") or ""
        pj = J.parse_judgment(res, meta["options"], set(meta["shortlist"]), meta["letters"])
        if rec.get("returncode") == 0 and not rec.get("is_error") and pj["ok"]:
            f.write_text(json.dumps({"rollout_id": r["rollout_id"], "attempt": attempt,
                                     "safety_stops": rec.get("safety_stops"), "result": res,
                                     "concluded_option": pj["concluded_option"],
                                     "applied_tenets": pj["applied_tenets"],
                                     "letter_mapped": pj["letter_mapped"],
                                     "shortlist": meta["shortlist"]}))
            return "scored"
        (d / f"{f.stem}.bad{attempt}.json").write_text(json.dumps(
            {"rollout_id": r["rollout_id"], "attempt": attempt, "returncode": rec.get("returncode"),
             "is_error": rec.get("is_error"), "safety_stops": rec.get("safety_stops"),
             "result": res[:2000]}))
    return "unscored"


def run(read: str, arm: str | None, workers: int, max_n: int | None) -> None:
    rows = rollouts(read, arm)
    if not rows:
        raise SystemExit("no S0a rollouts on disk for this read (download the session first)")
    d = RAW / (f"stenet_{arm}" if read == "stenet" else "frontier")
    d.mkdir(parents=True, exist_ok=True)
    todo = rows[:max_n] if max_n else rows
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda r: read_one(r, d), todo))
    print({k: res.count(k) for k in ("scored", "unscored", "cached")}, "of", len(todo))


def report(read: str, arm: str | None) -> dict:
    """Coverage, unscored ids by scenario and condition, and the stop-independence check."""
    rows = rollouts(read, arm)
    d = RAW / (f"stenet_{arm}" if read == "stenet" else "frontier")
    got, unscored = [], []
    for r in rows:
        f = d / f"{r['rollout_id'].replace('|', '__')}.json"
        if f.exists():
            got.append(json.loads(f.read_text()))
        elif len(list(d.glob(f.stem + ".bad*.json"))) >= ATTEMPTS:
            unscored.append(r)
    pending = len(rows) - len(got) - len(unscored)

    def by(key):
        tab: dict = {}
        for r in rows:
            k = key(r)
            tab.setdefault(k, [0, 0])
            tab[k][0] += 1
        for r in unscored:
            tab[key(r)][1] += 1
        return {k: {"n": n, "unscored": u, "rate": u / n} for k, (n, u) in sorted(tab.items())}

    rec = {"read": read, "arm": arm, "n": len(rows), "scored": len(got),
           "unscored": len(unscored), "pending": pending,
           "unscored_rate": len(unscored) / len(rows) if rows else None,
           "unscored_ids": [r["rollout_id"] for r in unscored],
           "by_family": by(lambda r: r["scenario_id"].split("-")[0]),
           "by_rollout_index": by(lambda r: r["k"]),
           "by_condition": by(lambda r: r["condition"]),
           "attempts_used": {a: sum(g["attempt"] == a for g in got) for a in (1, 2, 3)}}
    rec["independence"] = chi2_uniform(rec["by_family"])
    out = CST / "outputs" / (f"claude_stenet_{arm}.json" if read == "stenet"
                             else "claude_frontier.json")
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print({k: rec[k] for k in ("n", "scored", "unscored", "pending", "unscored_rate")},
          rec["independence"])
    return rec


def chi2_uniform(tab: dict) -> dict:
    """Pearson chi-square of unscored counts across groups against the pooled rate (df groups-1);
    p from the regularized gamma (no scipy). Reported, not a gate."""

    n = sum(v["n"] for v in tab.values())
    u = sum(v["unscored"] for v in tab.values())
    if not u or len(tab) < 2:
        return {"chi2": 0.0, "df": max(len(tab) - 1, 0), "p": 1.0}
    p0 = u / n
    chi2 = sum((v["unscored"] - v["n"] * p0) ** 2 / (v["n"] * p0 * (1 - p0)) for v in tab.values()
               if v["n"] * p0 * (1 - p0) > 0)
    df = len(tab) - 1
    return {"chi2": chi2, "df": df, "p": 1 - _gammainc(df / 2, chi2 / 2)}


def _gammainc(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) by its series (x small) or continued fraction."""
    import math

    if x <= 0:
        return 0.0
    if x < a + 1:
        s, t, k = 1 / a, 1 / a, 1
        while abs(t) > 1e-12 * abs(s):
            t *= x / (a + k)
            s += t
            k += 1
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c, dd = x + 1 - a, 1e300, 1 / (x + 1 - a)
    h, i = dd, 1
    while True:
        an = -i * (i - a)
        b += 2
        dd = an * dd + b
        dd = 1 / dd if abs(dd) > 1e-300 else 1e300
        c = b + an / c if abs(c) > 1e-300 else 1e-300
        delta = dd * c
        h *= delta
        i += 1
        if abs(delta - 1) < 1e-12:
            break
    return 1 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("stenet", "frontier", "report"))
    ap.add_argument("--arm", default=None, choices=("a0", "a0s"))
    ap.add_argument("--frontier", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max", type=int, default=None)
    a = ap.parse_args()
    if a.step == "report":
        report("frontier" if a.frontier else "stenet", a.arm)
    else:
        run(a.step, a.arm, a.workers, a.max)
