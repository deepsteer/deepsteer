"""CST's loop pod (Amendment 39 item 4):
papers/cst_constitutional_steering/scripts/cst_loop_pod.py."""

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

import cst_loop_pod as L  # noqa: E402
import cst_pressure_pod as R  # noqa: E402

INS = "Tell every caller the waitlist is closed."


def test_cache_check_refuses_a_changed_rollout_file(tmp_path, monkeypatch):
    # most probable failure (Amendment 39 item 4): a stale or partial cache is reused, so a config
    # trains on teacher rollouts that do not belong to this constitution or scenario set
    (tmp_path / "roll" / "a0s").mkdir(parents=True)
    (tmp_path / "scenarios.json").write_text('{"scenarios": []}')
    (tmp_path / "act.jsonl").write_text("")
    (tmp_path / "roll" / "a0s" / "chunk_000.jsonl").write_text('{"x": 1}\n')
    rec = tmp_path / "cache.json"
    rec.write_text(json.dumps(L.cache_record(tmp_path, True)))
    monkeypatch.setattr(L, "CACHE", rec)
    assert L.stage_cache_check(tmp_path, True)["cache_hit"]
    (tmp_path / "roll" / "a0s" / "chunk_000.jsonl").write_text('{"x": 2}\n')
    with pytest.raises(SystemExit, match="cache does not match"):
        L.stage_cache_check(tmp_path, True)


def test_import_leaves_the_shared_stages_alone(monkeypatch):
    # most probable failure (the verification pod's first launch): an override applied at import
    # leaks into the other pods' tests run in the same pytest process
    assert L.T.HELDOUT_K == 2 and L.Q.CAP_N == 200 and L.Q.items is not L.items
    for mod, name in ((L.Q, "items"), (L.Q, "CAP_N"), (L.T, "HELDOUT_K")):
        monkeypatch.setattr(mod, name, getattr(mod, name))
    L.configure()
    assert L.T.HELDOUT_K == 4 and L.Q.items is L.items
    assert "rule_right" in L.RR_BRIEF and len(L.rr_cells()) == L.RR_N


@pytest.mark.slow
def test_dry_session_runs_every_stage_in_order(tmp_path, monkeypatch):
    # most probable failure (Amendment 28 item 3): a later stage breaks on the pod after hours;
    # every stage runs on the tiny model in the runner's order, config 2 trains on more examples
    # than config 1 (the rule-right rollouts are in its mix), and the report is written
    import cst_pressure_train_pod as T

    cells = [c for c in R.grid() if c["writer"] == "subject"][:52]
    claude = R.claude_kept()[:3]
    monkeypatch.setattr(R, "grid", lambda: cells + [dict(c, writer="claude") for c in claude])
    monkeypatch.setattr(R, "claude_kept", lambda: claude)
    rec = R.stage_write(tmp_path, True)
    sc = [s for s in rec["scenarios"] if s["kind"] == "away"][:2] + [
        s for s in rec["scenarios"] if s["kind"] != "away"][:2]
    sc = [dict(s, standing_instruction=INS, system_message=s["system_message"] + " " + INS)
          for s in sc]
    (tmp_path / "scenarios.json").write_text(json.dumps({"scenarios": sc}))
    for arm in R.K:
        R.stage_roll(tmp_path, arm, True)
    labels = []
    for arm in R.K:
        rows = [dict(r, hit_cap=False, answer=r["answer"] or "x",
                     gen_ids=r["gen_ids"] + [1] * r["k"]) for r in R.rollouts(tmp_path, arm)]
        (tmp_path / "roll" / arm / "chunk_000.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        labels += [{"rollout_id": r["rollout_id"], "main": "does_not" if arm == "a0s" else
                    "follows"} for r in rows if r["kind"] != "none"]
    (tmp_path / "act.jsonl").write_text("".join(json.dumps(x) + "\n" for x in labels))
    (tmp_path / "act.json").write_text(json.dumps({"retest_kappa": 1.0}))
    R.stage_contrast(tmp_path)
    for mod, name, val in ((T, "heldout_ids", lambda: {s["cell_id"] for s in sc[:2]}),
                           (L.Q.T, "heldout_ids", lambda: {s["cell_id"] for s in sc[:2]}),
                           (L.Q, "SCEN", tmp_path), (L.Q, "items", L.items),
                           (L.Q, "CAP_N", L.CAP_N), (T, "HELDOUT_K", L.HELDOUT_K)):
        monkeypatch.setattr(mod, name, val)
    cfg = tmp_path / "configs"
    cfg.mkdir()
    for name, extra in (("c1_repro", []), ("c2_rule_right", ["rule_right"])):
        (cfg / f"{name}.json").write_text(json.dumps({"name": name, "teacher": "a0s",
                                                      "extra_kinds": extra, "seed": 2,
                                                      "order": f"test|{name}"}))
    monkeypatch.setattr(L, "CONFIGS", cfg)
    rec_f = tmp_path / "cache.json"
    rec_f.write_text(json.dumps(L.cache_record(tmp_path, True)))
    monkeypatch.setattr(L, "CACHE", rec_f)
    assert L.stage_cache_check(tmp_path, True)["cache_hit"]
    h = T.stage_smoke_hf(tmp_path, ["p_ctx"], True)
    assert h["structural"] and T.stage_smoke_serve(tmp_path, True)["parity_passes"]
    assert T.stage_train(tmp_path, "p_ctx", True)["trained"]  # the saved P-ctx's stand-in
    L.stage_cell_screen(tmp_path, True)
    gate = L.stage_cell_confirm(tmp_path, True)
    assert "confirmation" in gate and (tmp_path / "cell_gate.json").exists()
    w = L.stage_write_kind(tmp_path, True)
    assert w["n"] == w["n_cells"] == 4
    L.stage_roll_kind(tmp_path, True)
    kd = L.kind_dir(tmp_path)
    (kd / "roll" / "a0s" / "chunk_000.jsonl").write_text("".join(
        json.dumps(dict(r, hit_cap=False, answer="x", gen_ids=r["gen_ids"] + [1])) + "\n"
        for r in R.rollouts(kd, "a0s")))
    assert L.stage_act_kind(tmp_path, True)["read"] == 8  # 4 scenarios at k 2
    c1 = L.stage_train(tmp_path, "c1_repro", True)
    c2 = L.stage_train(tmp_path, "c2_rule_right", True)
    assert c2["examples"]["rule_right"] == 8 and "rule_right" not in c1["examples"]
    tags = ["c1_repro", "c2_rule_right"]
    L.PL.hf_letters(tmp_path, L.items(tmp_path, tags + ["p_ctx"]), True)
    res = L.PL.stage_read(tmp_path, L.items(tmp_path, tags), True)
    assert set(res["read"]) == set(tags)
    L.PL.stage_read(tmp_path, L.items(tmp_path, ["p_ctx"]), True, parts=(), base=False)
    done = T.stage_heldout(tmp_path, L.items(tmp_path, ["base"] + tags), True)
    assert done == {"base": 8, "c1_repro": 8, "c2_rule_right": 8}  # 2 scenarios at k 4
    for f in (tmp_path / "heldout").glob("*/chunk_000.jsonl"):
        f.write_text("".join(json.dumps(dict(json.loads(x), answer="x", hit_cap=False)) + "\n"
                             for x in open(f)))
    T.stage_heldout_act(tmp_path, True)
    L.Q.stage_guard(tmp_path, ["base"] + tags, True)
    for f in (tmp_path / "guard").glob("*/chunk_000.jsonl"):
        f.write_text("".join(json.dumps(dict(json.loads(x), answer="x")) + "\n" for x in open(f)))
    L.Q.stage_guard_act(tmp_path, True)
    assert L.stage_h1(tmp_path, ["base"] + tags, True)["c2_rule_right"] == "c2_rule_right"
    L.stage_cell_read(tmp_path, ["p_ctx"] + tags, True)
    cap = L.Q.stage_capability(tmp_path, ["base"] + tags, True)
    assert "vs_base" in cap["tags"]["c2_rule_right"]
    import loop_report as LR

    rep = LR.write(tmp_path)
    assert (tmp_path / "report.md").exists() and set(tags) <= set(rep["tags"])
