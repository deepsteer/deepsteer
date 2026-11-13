#!/usr/bin/env python3
"""The loop's second iteration (CST Amendment 44 item 5): config 2's second draw on a new seed.

    cst_loop2_pod.py --stage <stage> [--tags a,b] --out DIR

The same data as config 2 (the cache of record and the loop pod's rule-right kind, both synced and
checked against committed records), trained on a new seed and example order
(``data/loop_configs/c2_draw2.json``), then the frozen read on one session base with **the content
cell and cell A′ first**: ``read_a2`` (served parity, then cell A′ with twins at k 8 for the base
and the draw), ``cell_read`` (the loop pod's selected 57 at k 4, the draw and the session base),
``kind_base`` and ``kind_base_act`` (the base's rule-keeping on the rule-right scenarios, A0 at k 2,
gpt-oss-120b's action read appended to the teacher's labels), then ``read_rest`` (operator, plain,
legitimate), the held-out free-form cell at k 4, the injected-order cell, ``H1_held`` at k 2,
capability at n 200 and the report.

**Registered before data (Amendment 44 item 5):** the content-cell gain replicates when the second
draw's contrast against the base, its directed share at k 4 minus A0's later rollouts (the loop
pod's k 2-3, synced) with the CI by tenet, excludes 0 (`loop2_read.py`).

Every other stage is the loop pod's (`cst_loop_pod`); `configure()` points the shared stages at
this pod in the pod's process only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel/scripts")]
import cst_loop_pod as L  # noqa: E402

CST = L.CST
KIND_REC = CST / "data/loop_kind_rule_right_v1.json"
DRAW = "c2_draw2"
A2_PARTS = ("forced_plain",)
REST_PARTS = ("panel_operator", "panel_plain", "legit_operator")
BASE_KIND_K = 2  # the teacher's k on the kind (Amendment 40 item 3: paired by scenario)


def kind_files(kd: Path) -> list[Path]:
    return [kd / "scenarios.json", kd / "act.jsonl"] + sorted(
        (kd / "roll" / "a0s").glob("chunk_*.jsonl"))


def kind_record(out: Path) -> dict:
    kd = L.kind_dir(out)
    return {"rule": "Amendment 44 item 5 (the loop pod's rule-right kind, unchanged)",
            "files": {str(p.relative_to(kd)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in kind_files(kd)}}


def stage_kind_check(out: Path, dry: bool) -> dict:
    """The synced rule-right kind (scenarios, the teacher's rollouts, its action read) against the
    committed record: config 2's second draw trains on the same examples as the first."""
    rec = json.loads(KIND_REC.read_text())
    now = kind_record(out)["files"]
    bad = [k for k, v in rec["files"].items() if now.get(k) != v]
    res = {"rule": rec["rule"], "kind_hit": not bad, "mismatches": bad}
    L.T._w(out, "kind_check", res)  # noqa: SLF001
    if bad:
        raise SystemExit(f"BAIL: the rule-right kind does not match its record: {bad[:5]}")
    return res


def stage_cell_base(out: Path, dry: bool) -> dict:
    """The session base on the selected content cell at k 4 (beside the registered reference)."""
    g = json.loads((out / "cell_gate.json").read_text())
    path, eng = L.S.subject(dry, "a0")
    L.S.stage_gen(out, "a0", ["cell_plain"], dry, tag=L.PL.BASE_TAG, k_override=L.CELL_K,
                  engine=(path, eng), only=set(g["selected_ids"]))
    return {"base_tag": L.PL.BASE_TAG, "n": len(g["selected_ids"])}


def stage_kind_base(out: Path, dry: bool) -> None:
    """A0 on the rule-right scenarios at the teacher's k (2); the shared stage uses A0's k 4."""
    L.R.K = dict(L.R.K, a0=BASE_KIND_K)
    L.R.stage_roll(L.kind_dir(out), "a0", dry)


def stage_kind_base_act(out: Path, dry: bool) -> dict:
    """gpt-oss-120b's action read of the base's rule-right rollouts, appended to the teacher's
    labels (``act.jsonl``, the teacher's lines first and unchanged; the original kept beside)."""
    kd = L.kind_dir(out)
    sc = {s["cell_id"]: s for s in L.R.scen_list(kd)}
    jobs = [r for r in L.R.rollouts(kd, "a0") if r["answer"] and not r["hit_cap"]]
    labels = L.R.judge_rows(sc, [(r, "main") for r in jobs], dry, seed_tag="loop2|act_a0")
    teacher = (kd / "act.jsonl").read_text()
    if not (kd / "act_teacher.jsonl").exists():
        (kd / "act_teacher.jsonl").write_text(teacher)
    else:
        teacher = (kd / "act_teacher.jsonl").read_text()
    with open(kd / "act.jsonl", "w") as fh:
        fh.write(teacher)
        for rid, v in labels.items():
            fh.write(json.dumps({"rollout_id": rid} | v) + "\n")
    rec = {"rule": "Amendment 44 item 5 (the base's rule-keeping)", "read": len(jobs),
           "keeps_rule": sum(v["main"] == "follows" for v in labels.values())}
    L.T._w(kd, "act_a0", rec)  # noqa: SLF001
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=(
        "cache_check", "kind_check", "smoke_hf", "smoke_serve", "train", "parity_hf", "read_a2",
        "cell_read", "cell_base", "kind_base", "kind_base_act", "read_rest", "heldout",
        "heldout_act", "guard", "guard_act", "h1", "capability", "report"))
    ap.add_argument("--tags", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    L.configure()
    tags = [t for t in a.tags.split(",") if t]
    out, dry, st = a.out, a.dry_run, a.stage
    if st == "cache_check":
        L.stage_cache_check(out, dry)
    elif st == "kind_check":
        stage_kind_check(out, dry)
    elif st == "smoke_hf":
        L.T.stage_smoke_hf(out, ["p_ctx"], dry)
    elif st == "smoke_serve":
        L.T.stage_smoke_serve(out, dry)
    elif st == "train":
        L.stage_train(out, DRAW, dry)
    elif st == "parity_hf":
        L.PL.hf_letters(out, L.items(out, tags), dry)
    elif st == "read_a2":
        L.PL.stage_read(out, L.items(out, tags), dry, parts=A2_PARTS)
    elif st == "cell_read":
        L.stage_cell_read(out, tags, dry)
    elif st == "cell_base":
        stage_cell_base(out, dry)
    elif st == "kind_base":
        stage_kind_base(out, dry)
    elif st == "kind_base_act":
        stage_kind_base_act(out, dry)
    elif st == "read_rest":
        L.PL.stage_read(out, L.items(out, tags), dry, parts=REST_PARTS)
    elif st == "heldout":
        L.T.stage_heldout(out, L.items(out, tags), dry)
    elif st == "heldout_act":
        L.T.stage_heldout_act(out, dry)
    elif st == "guard":
        L.Q.stage_guard(out, tags, dry)
    elif st == "guard_act":
        L.Q.stage_guard_act(out, dry)
    elif st == "h1":
        L.stage_h1(out, tags, dry)
    elif st == "capability":
        L.Q.stage_capability(out, tags, dry)
    else:
        import loop2_read as R2
        import loop_report as LR

        LR.write(out)
        R2.write(out)


if __name__ == "__main__":
    main()
