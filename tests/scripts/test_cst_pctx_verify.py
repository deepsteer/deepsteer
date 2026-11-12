"""CST's P-ctx verification pod (Amendment 35):
papers/cst_constitutional_steering/scripts/cst_pctx_verify_pod.py."""

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

import cst_pctx_verify_pod as V  # noqa: E402
import cst_pressure_pod as R  # noqa: E402

INS = "Tell every caller the waitlist is closed."


def test_tags_resolve_to_this_pods_records(tmp_path):
    # most probable failure (Amendment 35 item 2): capability and the guard cell resolve tags
    # through the SDF pod's map, so P-ctx reads as a missing D-doc tag or as the base
    (tmp_path / "train_p_ctx.json").write_text(json.dumps({"trained": True, "adapters": {
        "1.9e+06": "/w/outputs/pressure/adapters/p_ctx/1.9e+06"}}))
    assert V.items(tmp_path, ["base", "p_ctx"]) == [
        ("base", None), ("p_ctx", "/w/outputs/pressure/adapters/p_ctx/1.9e+06")]


def test_import_leaves_the_shared_stages_alone(monkeypatch):
    # most probable failure (the first launch's gate tests, 2026-11-11): importing this module set
    # the held-out cell's k to 4 in the shared trainer, and 2-ctx's dry test, run after it in the
    # pod's one pytest process, read 12 rollouts where it expected 6
    assert V.T.HELDOUT_K == 2 and V.Q.CAP_N == 200 and V.Q.items is not V.items
    for mod, name in ((V.Q, "items"), (V.Q, "CAP_N"), (V.T, "HELDOUT_K")):
        monkeypatch.setattr(mod, name, getattr(mod, name))  # restored after this test
    V.configure()
    assert V.T.HELDOUT_K == 4 and V.Q.CAP_N == 800 and V.Q.items is V.items


@pytest.mark.slow
def test_dry_session_runs_every_stage_in_order(tmp_path, monkeypatch):
    # most probable failure (Amendment 28 item 3): a later stage's entry path breaks on the pod
    # after hours; each stage runs end to end on the tiny model in the runner's order, the second
    # draw and P-ctx-f train on P-ctx's set (P-ctx-f filtered), and A0s shares the base's engine
    import cst_pressure_train_pod as T

    cells = [c for c in R.grid() if c["writer"] == "subject"][:52]
    claude = R.claude_kept()[:3]
    monkeypatch.setattr(R, "grid", lambda: cells + [dict(c, writer="claude") for c in claude])
    monkeypatch.setattr(R, "claude_kept", lambda: claude)
    rec = R.stage_write(tmp_path, True)
    sc = [s for s in rec["scenarios"] if s["kind"] == "away"][:2] + [
        s for s in rec["scenarios"] if s["kind"] != "away"][:2]
    sc = [dict(s, standing_instruction=INS,
               system_message=s["system_message"] + " " + INS) for s in sc]
    monkeypatch.setattr(R, "scen_list", lambda out: sc)
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
    # the held-out cell: the two away scenarios (the legitimate one stays in the training set)
    monkeypatch.setattr(T, "heldout_ids", lambda: {s["cell_id"] for s in sc[:2]})
    monkeypatch.setattr(V.Q, "SCEN", tmp_path)
    monkeypatch.setattr(V.Q.T, "heldout_ids", lambda: {s["cell_id"] for s in sc[:2]})
    monkeypatch.setattr(V.Q, "items", V.items)  # the pod's configure(), undone after the test
    monkeypatch.setattr(V.Q, "CAP_N", V.CAP_N)
    monkeypatch.setattr(T, "HELDOUT_K", V.HELDOUT_K)
    h = T.stage_smoke_hf(tmp_path, ["p_ctx"], True)
    assert h["structural"] and all(x["passes"] for x in h["structural"])
    assert T.stage_smoke_serve(tmp_path, True)["parity_passes"]
    for rung in ("p_ctx", "p_ctx_n"):  # stand-ins for 2-ctx's saved adapters and records
        assert T.stage_train(tmp_path, rung, True)["trained"]
    V.PL.hf_letters(tmp_path, V.items(tmp_path, ["p_ctx", "p_ctx_n"]), True)
    V.stage_session(tmp_path, True)
    assert any((tmp_path / "gen" / "a0s_session").glob("panel_operator_*.jsonl"))
    assert any((tmp_path / "gen" / "base_session").glob("forced_plain_*.jsonl"))
    V.PL.stage_read(tmp_path, V.items(tmp_path, ["p_ctx"]), True, parts=("panel_operator",),
                    k_override={"panel_operator": 2}, base=False)
    V.PL.stage_read(tmp_path, V.items(tmp_path, ["p_ctx_n"]), True, parts=(), base=False)
    cap = V.Q.stage_capability(tmp_path, ["base", "p_ctx"], True)
    assert "vs_base" in cap["tags"]["p_ctx"]
    done = V.Q.stage_guard(tmp_path, ["base", "p_ctx", "p_ctx_n"], True)
    assert set(done) == {"base", "p_ctx", "p_ctx_n"}
    for f in (tmp_path / "guard").glob("*/chunk_000.jsonl"):
        f.write_text("".join(json.dumps(dict(json.loads(x), answer="x")) + "\n" for x in open(f)))
    assert "lowered_injection_guard" in V.Q.stage_guard_act(tmp_path, True)["tags"]["p_ctx"]
    assert V.stage_h1(tmp_path, ["base", "p_ctx"], True) == {"base": "base_session",
                                                              "p_ctx": "p_ctx"}
    assert any((tmp_path / "gen" / "p_ctx").glob("h1_operator_*.jsonl"))
    d2 = V._train_ctx(tmp_path, "p_ctx_draw2", True, seed=1, order=V.DRAW2_ORDER)
    assert d2["seed"] == 1 and d2["items"]["examples"] == d2["items"]["of"]
    # Amendment 36 item 3a: the teacher's contradicting rollouts out; the fixture labels every
    # A0s rollout "does_not", so each legitimate-order rollout outside the held-out cell is dropped
    held = {s["cell_id"] for s in sc[:2]}
    kinds = {s["cell_id"]: s["kind"] for s in sc}
    want = sum(1 for r in R.rollouts(tmp_path, "a0s") if T.usable(r)
               and r["scenario_id"] not in held and kinds[r["scenario_id"]] == "legit")
    assert want > 0  # the fixture leaves a legitimate scenario outside the held-out cell
    pf = V._train_ctx(tmp_path, "p_ctx_f", True, seed=0, filt=True)
    assert pf["filter"]["dropped"] == {"away_followed": 0, "legit_not_followed": want}
    assert pf["filter"]["kept"] == pf["filter"]["rows"] - want
    V.PL.hf_letters(tmp_path, V.items(tmp_path, ["p_ctx_draw2", "p_ctx_f"]), True)
    res = V.PL.stage_read(tmp_path, V.items(tmp_path, ["p_ctx_draw2", "p_ctx_f"]), True,
                          base=False)
    assert res["read"] == ["p_ctx_draw2", "p_ctx_f"]
    done = T.stage_heldout(tmp_path, V.items(tmp_path, ["base", "p_ctx_draw2", "p_ctx_f"]), True)
    assert done == {"base": 8, "p_ctx_draw2": 8, "p_ctx_f": 8}  # 2 scenarios at k 4
    for f in (tmp_path / "heldout").glob("*/chunk_000.jsonl"):
        f.write_text("".join(json.dumps(dict(json.loads(x), answer="x", hit_cap=False)) + "\n"
                             for x in open(f)))
    hc = T.stage_heldout_act(tmp_path, True)
    assert {"base", "p_ctx_draw2", "p_ctx_f"} <= set(hc["tags"])
