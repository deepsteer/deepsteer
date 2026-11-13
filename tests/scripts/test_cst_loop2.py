"""CST's second loop iteration (Amendment 44 item 5):
papers/cst_constitutional_steering/scripts/cst_loop2_pod.py and loop2_read.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts"),
                str(ROOT / "papers/kdg_panel/scripts"),
                str(ROOT / "papers/st_live_reference/scripts"),
                str(ROOT / "papers/mrb_morebench/scripts"), str(ROOT / "tests/scripts")]

import cst_loop2_pod as P  # noqa: E402

CST = ROOT / "papers/cst_constitutional_steering"


def _kind(out: Path) -> Path:
    kd = P.L.kind_dir(out)
    (kd / "roll" / "a0s").mkdir(parents=True)
    (kd / "scenarios.json").write_text(json.dumps({"scenarios": [{"cell_id": "PZ-R000"}]}))
    (kd / "act.jsonl").write_text('{"rollout_id": "a0s|PZ-R000|0", "main": "follows"}\n')
    (kd / "roll" / "a0s" / "chunk_000.jsonl").write_text('{"x": 1}\n')
    return kd


def test_kind_check_refuses_a_changed_rule_right_file(tmp_path, monkeypatch):
    # most probable failure (Amendment 44 item 5): the second draw trains on rule-right rollouts
    # that are not the first draw's, so the two draws differ in data as well as seed
    kd = _kind(tmp_path)
    rec = tmp_path / "kind.json"
    rec.write_text(json.dumps(P.kind_record(tmp_path)))
    monkeypatch.setattr(P, "KIND_REC", rec)
    assert P.stage_kind_check(tmp_path, True)["kind_hit"]
    (kd / "roll" / "a0s" / "chunk_000.jsonl").write_text('{"x": 2}\n')
    with pytest.raises(SystemExit, match="does not match"):
        P.stage_kind_check(tmp_path, True)


def test_base_rule_keeping_reads_at_k2_and_keeps_the_teachers_labels(tmp_path, monkeypatch):
    # most probable failures: the base rolls at A0's k 4 (unpaired with the teacher's k 2), or its
    # action read overwrites the teacher's labels in act.jsonl
    kd = _kind(tmp_path)
    seen = {}
    monkeypatch.setattr(P.L.R, "stage_roll", lambda out, arm, dry: seen.update(k=P.L.R.K[arm]))
    monkeypatch.setattr(P.L.R, "K", dict(P.L.R.K))
    P.stage_kind_base(tmp_path, True)
    assert seen["k"] == 2
    rows = [{"rollout_id": f"a0|PZ-R000|{j}", "scenario_id": "PZ-R000", "answer": "x",
             "hit_cap": False} for j in range(2)]
    monkeypatch.setattr(P.L.R, "scen_list", lambda out: [{"cell_id": "PZ-R000"}])
    monkeypatch.setattr(P.L.R, "rollouts", lambda out, arm: rows if arm == "a0" else [])
    monkeypatch.setattr(P.L.R, "judge_rows", lambda sc, jobs, dry, seed_tag: {
        r["rollout_id"]: {"main": "follows"} for r, _ in jobs})
    for _ in range(2):  # a resumed stage appends to the teacher's labels again, not to its output
        rec = P.stage_kind_base_act(tmp_path, True)
    lines = [json.loads(x) for x in open(kd / "act.jsonl")]
    assert lines[0]["rollout_id"] == "a0s|PZ-R000|0" and len(lines) == 3 and rec["read"] == 2


def test_card_syncs_every_loop_input_the_runner_checks():
    # most probable failure (the loop pod's first launch): a file the runner needs is not on the
    # card's sync list, and the pod stops at its input block or, worse, at a stage hours in
    card = (CST / "runpod/launch_cst_loop2.sh").read_text()
    runner = (CST / "runpod/remote_cst_loop2.sh").read_text()
    for f in ("kinds/rule_right/scenarios.json", "kinds/rule_right/act.jsonl",
              "kinds/rule_right/act.json", "cell_gate.json"):
        assert f in card and f.split("/")[-1] in runner, f
    for g in ("kinds/rule_right/roll/a0s/chunk_", "gen/a0/cell_plain_", "gen/c2_rule_right/cell_"):
        assert g in card, g
    assert "eval_8gram_hashes_v1.npy" in card and "C_anth.txt" in card
    assert json.loads((CST / "data/loop_configs/c2_draw2.json").read_text())["seed"] != json.loads(
        (CST / "data/loop_configs/c2_rule_right.json").read_text())["seed"]


def test_registered_read_needs_a_positive_ci(tmp_path, monkeypatch):
    # most probable failure (Amendment 44 item 5): a CI wholly below 0 "excludes 0" and is read as
    # the gain replicating
    import loop2_read as R2

    cov = {f"S{i}": {"tenet_id": f"T{i % 4}", "directed_option": "o1"} for i in range(8)}
    loop1 = tmp_path / "loop1"
    (loop1 / "gen" / "a0").mkdir(parents=True)
    (loop1 / "cell_gate.json").write_text(json.dumps({"selected_ids": sorted(cov)}))
    monkeypatch.setattr(R2.CR, "scenarios", lambda: cov)
    (tmp_path / "gen" / R2.DRAW).mkdir(parents=True)

    def rows(arm_dir: Path, arm: str):
        return [{"arm": arm, "s": s} for s in cov]

    for share, want in ((1.0, True), (0.0, False)):
        monkeypatch.setattr(R2.CR, "rollouts", rows)
        monkeypatch.setattr(R2.CR, "confirm", lambda a0, arm, ids, cv, v=share: {
            "a0s": v, "a0_later": 0.5, "delta": v - 0.5, "ci95_by_tenet": [v - 0.6, v - 0.4],
            "n_scenarios": 8, "n_tenets": 4})
        monkeypatch.setattr(R2.CR, "directed_share", lambda rr, cv, ks: {s: 0.5 for s in cv})
        reg = R2.read(tmp_path, loop1)["registered"]
        assert reg["ci_excludes_0"] and reg["replicates"] is want and reg["loss"] is not want
