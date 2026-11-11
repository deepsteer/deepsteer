#!/usr/bin/env python3
"""Amendment 29 item 12: each trained rung's training set by side or teacher (zero GPU).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/rung_training_table.py

Pressure rungs, from 2-roll's rollouts and the action read, as the trainer builds them (capped or
answerless rollouts out; P-ctx and P-ctx-n outside the held-out cell): P-ctx (teacher A0s), P-ctx-n
(teacher A0n), P-long (chosen, A0's longest response per scenario; rejected, the shortest). Per
side: the suspicion-talk share (Amendment 23 item 3's injection pattern on the trace), the median
trace length, and the follow share by scenario kind (the action read; unclear labels out). The SDF
corpora (D-doc, D-neutral, the D-chat renders) are documents, not rollouts: the suspicion-pattern
share and the median length only. Writes `outputs/rung_training_table.json`.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel/scripts")]
import cst_pilot_sdf_pod as P  # noqa: E402
import cst_pressure_pod as R  # noqa: E402
import cst_pressure_train_pod as T  # noqa: E402
import s0b_a0n_reads as N  # noqa: E402

CST = HERE.parent
OUT = CST / "outputs/pressure"


def side(rows: list[dict], sc: dict, lab: dict) -> dict:
    trace = [(r["text"] or "").split("</think>")[0] for r in rows]
    rec = {"rollouts": len(rows),
           "suspicion_share": statistics.mean(bool(N.PATTERNS["injection"].search(t))
                                              for t in trace) if rows else None,
           "trace_tokens_median": statistics.median(r["trace_tokens"] for r in rows)
           if rows else None, "follow_by_kind": {}}
    for kind in ("away", "legit"):
        u = defaultdict(list)
        for r in rows:
            v = lab.get(r["rollout_id"])
            if sc[r["scenario_id"]]["kind"] == kind and v in ("follows", "does_not"):
                u[r["scenario_id"]].append(int(v == "follows"))
        rec["follow_by_kind"][kind] = (statistics.mean(statistics.mean(x) for x in u.values())
                                       if u else None)
    return rec


def main() -> None:
    sc = {s["cell_id"]: s for s in R.scen_list(OUT)}
    lab = {json.loads(x)["rollout_id"]: json.loads(x)["main"] for x in open(OUT / "act.jsonl")}
    held = T.heldout_ids()
    roll = {a: [r for r in R.rollouts(OUT, a) if T.usable(r)] for a in ("a0", "a0s", "a0n")}
    res: dict = {"rule": "Amendment 29 item 12", "rungs": {}}
    res["rungs"]["P-ctx (teacher A0s)"] = side(
        [r for r in roll["a0s"] if r["scenario_id"] not in held], sc, lab)
    res["rungs"]["P-ctx-n (teacher A0n)"] = side(
        [r for r in roll["a0n"] if r["scenario_id"] not in held], sc, lab)
    by: dict[str, list[dict]] = defaultdict(list)
    for r in roll["a0"]:
        by[r["scenario_id"]].append(r)
    chosen, rejected = [], []
    for sid in sorted(by):
        rs = sorted(by[sid], key=lambda r: (len(r["gen_ids"]), r["rollout_id"]))
        if len(rs) >= 2 and len(rs[-1]["gen_ids"]) > len(rs[0]["gen_ids"]):
            chosen.append(rs[-1])
            rejected.append(rs[0])
    res["rungs"]["P-long chosen (A0's longest)"] = side(chosen, sc, lab)
    res["rungs"]["P-long rejected (A0's shortest)"] = side(rejected, sc, lab)
    res["rungs"]["P-dpo"] = "not trained (the P-dpo gate failed, 41 of 60)"
    sdf = CST / "outputs/pilot_sdf"
    for arm in ("d_doc", "d_neutral"):
        kept = [r for r in P.corpus_rows(sdf, arm, True) if r["kept"]]
        res["rungs"][f"SDF {arm} corpus (documents)"] = {
            "documents": len(kept),
            "suspicion_share": statistics.mean(bool(N.PATTERNS["injection"].search(r["final"]))
                                               for r in kept),
            "tokens_median": statistics.median(r["final_tokens"] for r in kept)}
    chat = [json.loads(x) for f in sorted((sdf / "corpus/d_chat").glob("chunk_*.jsonl"))
            for x in open(f)]
    chat = [r for r in chat if r["ok"]]
    res["rungs"]["SDF d_chat renders (conversations)"] = {
        "documents": len(chat),
        "suspicion_share": statistics.mean(bool(N.PATTERNS["injection"].search(r["text"]))
                                           for r in chat),
        "tokens_median": statistics.median(r["tokens"] for r in chat)}
    (CST / "outputs/rung_training_table.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res["rungs"].items():
        print(f"{k:38s} {json.dumps(v)[:220]}")


if __name__ == "__main__":
    main()
