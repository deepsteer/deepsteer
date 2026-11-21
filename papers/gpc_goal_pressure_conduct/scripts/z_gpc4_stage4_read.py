#!/usr/bin/env python3
"""Stage 4's letters read (GPC_PLAN.md §11.22, the lead's morning read 2), zero GPU: the summed adapter
(P-ctx's update plus C1's) on CST's operator cell, cell A′ and GPC's held-out conduct cell, beside P-ctx's
and C1's reads of record, with the difference CIs.

- **The operator cell** (CST's headline): the ordered-option share on the clear items, the summed adapter
  minus launch 6's same-session base, scenario-paired (`sdf_reread_quick.paired`, its own seed and B), as
  Amendment 52's item 3a read P-ctx against its own session's base (`outputs/a52_v1`). The difference with
  P-ctx is the per-scenario difference of the two shifts, each against its own session's base, bootstrapped
  over scenarios.
- **Cell A′** (specificity): per item P(violating) of record (`A15.p_record`), the summed adapter minus the
  session base (`ctx_read.cell_a2`'s estimator, `K.boot`); P-ctx's of record is a52's at strength 1.0. The
  difference is per item, the two shifts each against its own base.
- **The held-out conduct cell**: the goal rate on its pressure slots against GPC-1's held-out base
  (`gpc2_read.holdout_read`, cross-session, as GPC-2's read was); C1's of record is GPC-1's. The difference
  with C1 is slot-paired (both on GPC-1's slots).

    python z_gpc4_stage4_read.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

GPC = Path(__file__).resolve().parents[1]
ROOT = GPC.parents[1]
CST = ROOT / "papers/cst_constitutional_steering"
OUT = GPC / "outputs/gpc4_v1"
A52 = CST / "outputs/a52_v1"
sys.path[:0] = [str(GPC / "scripts"), str(CST / "scripts"), str(ROOT)]
B, SEED = 2000, "20261121|gpc4|stage4"


def _boot(vals: list[float], seed: str) -> dict:
    rng = random.Random(seed)
    bs = sorted(statistics.mean(rng.choice(vals) for _ in vals) for _ in range(B))
    return {"delta": statistics.mean(vals), "ci95": [bs[int(0.025 * B)], bs[int(0.975 * B) - 1]],
            "n": len(vals)}


def operator() -> dict:
    import s0a_analysis as A
    import sdf2_quick as Z
    import sdf_reread_quick as Q

    lab = A.labels()
    acted = lambda r: not r["capped"] and r["option"] is not None  # noqa: E731  (a52's own rule)
    u6 = {t: Q.per_scenario(Z.cell_rows(t, "operator", lab, OUT / "gen"), acted) for t in ("base_session", "sum")}
    u52 = {t: Q.per_scenario(Z.cell_rows(t, "operator", lab, A52 / "gen"), acted)
           for t in ("base_session", "p_ctx")}
    ids6 = sorted(set(u6["base_session"]) & set(u6["sum"]))
    ids52 = sorted(set(u52["base_session"]) & set(u52["p_ctx"]))
    m = lambda u, s: statistics.mean(u[s])  # noqa: E731
    both = [s for s in ids6 if s in set(ids52) and u6["sum"].get(s) and u6["base_session"].get(s)
            and u52["p_ctx"].get(s) and u52["base_session"].get(s)]
    did = [(m(u6["sum"], s) - m(u6["base_session"], s)) - (m(u52["p_ctx"], s) - m(u52["base_session"], s))
           for s in both]
    return {"sum_minus_base": Q.paired(u6["sum"], u6["base_session"], ids6),
            "base_share": Q.share(u6["base_session"], ids6), "sum_share": Q.share(u6["sum"], ids6),
            "p_ctx_of_record": Q.paired(u52["p_ctx"], u52["base_session"], ids52),
            "sum_minus_p_ctx_shift": _boot(did, SEED + "|op")}


def cell_a_prime() -> dict:
    import numpy as np

    sys.argv = [sys.argv[0], str(OUT / "gen")]  # ctx_read reads its GEN from argv at import
    import ctx_read as C
    import s0a_reads as SR

    from deepsteer.kdg.schema import load_scenarios

    K, QS, A15, _ = SR._st()  # noqa: SLF001
    fst = {s.id: {o.option_id: o.norm_status for o in s.options}
           for f in C.S.FORCED for s in load_scenarios(f)[0]}

    def pr(gen: Path, t: str) -> dict[str, float]:
        rr = []
        for f in sorted((gen / t).glob("forced_plain_*.jsonl")):
            for x in open(f):
                r = json.loads(x)
                if r["scenario_id"].endswith(QS.TWIN):
                    continue
                ns = fst[r["scenario_id"]][r["option"]] if r.get("option") else None
                rr.append({"scenario_id": r["scenario_id"], "norm_status": None if r["hit_cap"] else ns,
                           "status": "truncated" if r["hit_cap"] else r["parse_status"]})
        return A15.p_record(rr) if rr else {}

    b6, s6 = pr(OUT / "gen", "base_session"), pr(OUT / "gen", "sum")
    b52, p52 = pr(A52 / "gen", "base_session"), pr(A52 / "gen", "p_ctx")
    ids = sorted(set(b6) & set(s6))
    sh = K.boot(np.array([s6[i] - b6[i] for i in ids]))
    idp = sorted(set(b52) & set(p52))
    shp = K.boot(np.array([p52[i] - b52[i] for i in idp]))
    both = sorted(set(ids) & set(idp))
    did = K.boot(np.array([(s6[i] - b6[i]) - (p52[i] - b52[i]) for i in both]))
    return {"sum_shift": sh, "n": len(ids), "P_base": float(np.mean([b6[i] for i in ids])),
            "P_sum": float(np.mean([s6[i] for i in ids])),
            "sum_non_specific": bool(sh["ci95"][0] is not None and sh["ci95"][0] > 0.03),
            "p_ctx_of_record": shp, "sum_minus_p_ctx_shift": did, "bar": 0.03}


def held_out() -> dict:
    import gpc2_read as G2

    rec = G2.holdout_read(OUT, "sum")
    sl = json.loads((OUT.parent / "gpc1_v1/conduct/slots.json").read_text())
    pres = [i for i in sl["holdout"] if sl["kept"][i]["pressure"]]
    rows = lambda p: {r["slot_id"]: float(r["choice"] == "goal") for r in G2._rows(p)  # noqa: E731, SLF001
                      if r["choice"] is not None}
    s = rows(OUT / "conduct2/holdout_sum.jsonl")
    c1 = rows(OUT.parent / "gpc1_v1/conduct/holdout_c1.jsonl")
    base = rows(OUT.parent / "gpc1_v1/conduct/holdout_base.jsonl")
    rec["c1_of_record"] = G2._slot_paired(c1, base, pres, "gpc2|ho|c1")  # noqa: SLF001
    rec["c1_pressure"] = statistics.mean(c1[i] for i in pres if i in c1)
    rec["base_pressure"] = statistics.mean(base[i] for i in pres if i in base)
    rec["sum_minus_c1"] = G2._slot_paired(s, c1, pres, "gpc4|ho|sum-c1")  # noqa: SLF001
    return rec


def main() -> None:
    rec = {"rule": "GPC_PLAN.md §11.22 read 2 (zero GPU; the summed adapter beside P-ctx's and C1's reads of record)",
           "operator_cell": operator(), "cell_a_prime": cell_a_prime(), "held_out_conduct": held_out()}
    f = OUT / "stage4_read.json"
    f.write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    main()
