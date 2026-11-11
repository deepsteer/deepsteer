#!/usr/bin/env python3
"""SDF pod 2's reads (Amendment 29 items 5-7; Amendment 32 item 3), zero GPU, saved rollouts.

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/sdf2_quick.py

On the operator cell's 458 clear items (`outputs/pilot_sdf2/gen/<tag>`), each rung against the
same-session base (`base_session`), scenario-paired with a scenario bootstrap:

- the ordered share (rollouts with an action, capped excluded), the suspicion split, face-value
  compliance and the worst-case capped bound (`sdf_reread_quick.summary`);
- **Amendment 29 item 6's registered read:** D-doc 1M minus D-neutral 1M, both without recovery
  (D-doc read at k 4 on fresh seeds, `d_doc_1M_k4`; D-neutral at k 2);
- **Amendment 32 item 3's shares:** on the operator cell, the share of all rollouts taking the
  directed option, another option, and no option (capped or unparsed); beside them the session
  base's same shares on the plain cell (no order, the directed option named by construction).

Tags missing on disk are skipped. Writes `outputs/sdf2_quick.json`.
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0a_analysis as A  # noqa: E402
import s0b_a0n_reads as N  # noqa: E402
import sdf_reread_quick as Q  # noqa: E402

CST = HERE.parent
GEN = CST / "outputs/pilot_sdf2/gen"
TAGS = ("d_neutral_1M", "d_doc_1M_k4", "d_chat_1M", "d_doc_03M", "d_raw", "d_doc_1M_draw2")


def cell_rows(tag: str, condition: str, lab: dict) -> list[dict]:
    """``sdf_reread_quick.rows`` for any condition (the operator or plain cell), clear items."""
    out = []
    for f in sorted(glob.glob(str(GEN / tag / f"panel_{condition}_*.jsonl"))):
        for x in open(f):
            r = json.loads(x)
            if r["condition"] != condition:
                continue
            rec = lab["record"].get(r["scenario_id"])
            if not rec or rec["stratum_of_record"] != "clear_refuse" or rec.get(
                    "set", "panel") != "panel":
                continue
            trace = (r.get("text") or "").split("</think>")[0]
            out.append({"sid": r["scenario_id"], "capped": bool(r["hit_cap"]),
                        "option": r["option"], "ordered": rec["ordered_option"],
                        "suspicion": bool(N.PATTERNS["injection"].search(trace)),
                        "trace_tokens": r.get("trace_len") or 0})
    return out


def option_shares(rs: list[dict]) -> dict:
    """Scenario-averaged shares of all rollouts: the directed option, another option, none."""
    ids = sorted({r["sid"] for r in rs})

    def avg(fn):
        return Q.share(Q.per_scenario(rs, value=fn), ids)

    return {"directed": avg(lambda r: int(not r["capped"] and r["option"] == r["ordered"])),
            "other": avg(lambda r: int(not r["capped"] and r["option"] is not None
                                       and r["option"] != r["ordered"])),
            "none": avg(lambda r: int(r["capped"] or r["option"] is None)), "items": len(ids)}


def main() -> None:
    lab = A.labels()
    tags = ["base_session"] + [t for t in TAGS if (GEN / t).exists()]
    op = {t: cell_rows(t, "operator", lab) for t in tags}
    op = {t: v for t, v in op.items() if v}
    acted = lambda r: not r["capped"] and r["option"] is not None  # noqa: E731
    u = {t: Q.per_scenario(rs, acted) for t, rs in op.items()}
    ids = sorted(set.intersection(*[set(v) for v in u.values()]))
    res: dict = {"rule": "Amendment 29 items 5-7; Amendment 32 item 3", "n_items": len(ids),
                 "rollouts": {t: len(v) for t, v in op.items()}, "rungs": {}}
    for t, rs in op.items():
        rec = Q.summary(rs) | {"operator_shares": option_shares(rs)}
        if t != "base_session":
            rec["vs_session_base"] = Q.paired(u[t], u["base_session"], ids)
        res["rungs"][t] = rec
    res["base_plain_shares"] = option_shares(cell_rows("base_session", "plain", lab))
    if {"d_doc_1M_k4", "d_neutral_1M"} <= set(u):
        res["item6_d_doc_minus_d_neutral"] = Q.paired(u["d_doc_1M_k4"], u["d_neutral_1M"], ids)
    (CST / "outputs/sdf2_quick.json").write_text(json.dumps(res, indent=1) + "\n")
    print("items", len(ids), "rollouts", res["rollouts"])
    bp = res["base_plain_shares"]
    print(f"base plain cell   directed {bp['directed']:.3f} other {bp['other']:.3f} "
          f"none {bp['none']:.3f}")
    for t, r in res["rungs"].items():
        s, vb = r["operator_shares"], r.get("vs_session_base")
        line = (f"{t:15s} ord {r['ordered_share']:.3f} no-susp "
                f"{r['ordered_share_without_suspicion']:.3f} susp {r['suspicion_share']:.3f} "
                f"worst {r['worst_case_capped_as_following']:.3f} trace "
                f"{r['trace_tokens_median']:.0f} | op directed {s['directed']:.3f} other "
                f"{s['other']:.3f} none {s['none']:.3f}")
        if vb:
            line += f" | vs base {vb['delta']:+.3f} [{vb['ci95'][0]:+.3f}, {vb['ci95'][1]:+.3f}]"
        print(line)
    if "item6_d_doc_minus_d_neutral" in res:
        d = res["item6_d_doc_minus_d_neutral"]
        print(f"item 6: D-doc 1M - D-neutral 1M {d['delta']:+.3f} "
              f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] (n {d['n']})")


if __name__ == "__main__":
    main()
