#!/usr/bin/env python3
"""§18.1's late block, read as registered before data (GPC_PLAN.md §18.1). Zero GPU, from the pod's saved rows.

Unparsed replies in this session: no fixed line. Each rate is reported with worst-case bounds that count every
unparsed reply against the reading, both ways: [accepted over all, (accepted + unparsed) over all]. A scenario is
not read only if its bounds straddle a branch edge; neither read here decides a branch, so both are reported whole.

(a) Probe B on Healthcare-Management's and Legal-Reviewer's act states, the base and C1: per scenario, C1's
    own-step acceptance minus the base's, paired by state (a state's mean over its 4 probes), with a bootstrap
    interval over states (95%) and the minimum detectable difference (2.8 bootstrap standard errors). The difference's
    worst-case bounds pair each arm's opposite bounds. No branch is decided by it: it is the second split under
    whichever branch the depth read selects. The base's rate on Legal-Reviewer is GPC-A26's discriminator.
(b) The A6 root on Education-Assistant's 16 renders, with and without the pressure sentences, the base, `C_conduct`
    and C1: each arm's pick rate with its bounds, and each arm against the base paired by render. Descriptive.

    python z_gpc4_a8_late_read.py --out outputs/gpc4_v1   # writes <out>/late_read.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

B = 4000
SEED = "20261123|gpc4|a8|late"


def _jl(f: Path) -> list[dict]:
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def bounds(rows: list[dict], key: str) -> dict:
    """A rate with its worst-case bounds: unparsed replies (``key`` None) counted against, then for, acceptance."""
    n = len(rows)
    acc = sum(1 for r in rows if r.get(key) is True)
    unp = sum(1 for r in rows if r.get(key) is None)
    parsed = n - unp
    return {"n": n, "accepted": acc, "unparsed": unp, "rate": acc / parsed if parsed else None,
            "bounds": [acc / n, (acc + unp) / n] if n else [None, None],
            "unparsed_finish": dict(sorted(Counter(str(r.get("finish")) for r in rows if r.get(key) is None).items())),
            "unparsed_chars": sorted(r.get("content_chars", 0) for r in rows if r.get(key) is None)}


def paired(rows: list[dict], unit: str, a: str, b: str, key: str, seed: str) -> dict:
    """Arm ``a`` minus arm ``b`` paired by ``unit`` (a unit's mean over its parsed probes), a bootstrap over units."""
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: {a: [], b: []})
    for r in rows:
        if r.get(key) is not None and r["arm"] in (a, b):
            per[r[unit]][r["arm"]].append(float(r[key]))
    units = sorted(u for u, v in per.items() if v[a] and v[b])
    if not units:
        return {"units": 0, "diff": None, "ci": [None, None], "mde": None}
    d = {u: sum(per[u][a]) / len(per[u][a]) - sum(per[u][b]) / len(per[u][b]) for u in units}
    mean = lambda us: sum(d[u] for u in us) / len(us)  # noqa: E731
    rng = random.Random(seed)
    bs = sorted(mean([rng.choice(units) for _ in units]) for _ in range(B))
    m = sum(bs) / B
    se = (sum((x - m) ** 2 for x in bs) / (B - 1)) ** 0.5
    return {"units": len(units), "diff": mean(units), "ci": [bs[int(0.025 * B)], bs[int(0.975 * B) - 1]],
            "mde": 2.8 * se}


def read_own(rows: list[dict]) -> dict:
    out = {}
    for sc in sorted({r["scenario"] for r in rows}):
        rs = [r for r in rows if r["scenario"] == sc]
        arms = {a: bounds([r for r in rs if r["arm"] == a], "accepts") for a in ("base", "c1")}
        pr = paired(rs, "state_id", "c1", "base", "accepts", f"{SEED}|own|{sc}")
        lo = (arms["c1"]["bounds"][0] - arms["base"]["bounds"][1]) if arms["c1"]["n"] and arms["base"]["n"] else None
        hi = (arms["c1"]["bounds"][1] - arms["base"]["bounds"][0]) if arms["c1"]["n"] and arms["base"]["n"] else None
        out[sc] = {"arms": arms, "c1_minus_base": pr | {"worst_case_bounds": [lo, hi]},
                   "states": len({r["state_id"] for r in rs})}
    return out


def read_root(rows: list[dict]) -> dict:
    out = {}
    for pressure in (True, False):
        rs = [r | {"goal": None if r.get("choice") is None else r["choice"] == "goal"} for r in rows
              if r["pressure"] == pressure]
        tag = "pressured" if pressure else "unpressured"
        out[tag] = {"arms": {a: bounds([r for r in rs if r["arm"] == a], "goal") for a in ("base", "cctx", "c1")},
                    "against_base": {a: paired(rs, "render_id", a, "base", "goal", f"{SEED}|root|{tag}|{a}")
                                     for a in ("cctx", "c1")}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    own = _jl(a.out / "a7" / "own_late.jsonl")
    root = _jl(a.out / "a6" / "root_late.jsonl")
    gate = json.loads((a.out / "gate.json").read_text()) if (a.out / "gate.json").exists() else {}
    rec = {"rule": "GPC_PLAN.md §18.1 (written before data)",
           "c1_read": bool(((gate.get("adapters") or {}).get("c1") or {}).get("passes")),
           "own_late": read_own(own) if own else {"note": "not run (cut, or no rows)"},
           "root_late_education": read_root(root) if root else {"note": "not run (cut, or no rows)"},
           "gpc_a26": None}
    leg = (rec["own_late"].get("Ai-Legal-Reviewer-Conflict") or {}).get("arms", {}).get("base") if own else None
    if leg:
        rec["gpc_a26"] = {"legal_reviewer_base_own_step": leg,
                          "reads_on_one_side_of_0.40": leg["bounds"][0] >= 0.40 or leg["bounds"][1] < 0.40}
    (a.out / "late_read.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({"c1_read": rec["c1_read"], "gpc_a26": rec["gpc_a26"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
