"""CST pressure line, sessions 2-ctx and 2-dpo (Amendments 25 C4, 27, 28 item 3):
papers/cst_constitutional_steering/scripts/cst_pressure_train_pod.py."""

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

import cst_pressure_pod as R  # noqa: E402
import cst_pressure_train_pod as T  # noqa: E402


class _Render:
    def __call__(self, s):
        return [7, 7, 7]


def _fixture(tmp_path, n_away=3, a0s_label="does_not", capped=()):
    """A tiny 2-roll output: scenarios, rollouts (A0 k 4, A0s k 2, A0n k 2) and labels."""
    sc = [{"cell_id": f"PZ-A{i:03d}", "kind": "away", "writer": "subject"} for i in range(n_away)]
    sc += [{"cell_id": "PZ-L000", "kind": "legit", "writer": "claude"},
           {"cell_id": "PZ-N000", "kind": "none", "writer": "subject"}]
    (tmp_path / "scenarios.json").write_text(json.dumps({"scenarios": sc}))
    labels = []
    for arm, k in R.K.items():
        d = tmp_path / "roll" / arm
        d.mkdir(parents=True)
        rows = []
        for s in sc:
            for j in range(k):
                rid = f"{arm}|{s['cell_id']}|{j}"
                rows.append({"rollout_id": rid, "arm": arm, "scenario_id": s["cell_id"], "k": j,
                             "kind": s["kind"], "writer": s["writer"],
                             "gen_ids": list(range(10 + 3 * j)), "answer": "x",
                             "hit_cap": rid in capped, "trace_tokens": 5, "text": "t</think>x"})
                if s["kind"] != "none":
                    lab = {"a0": "follows", "a0s": a0s_label, "a0n": "follows"}[arm]
                    labels.append({"rollout_id": rid, "main": lab})
        (d / "chunk_000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "act.jsonl").write_text("".join(json.dumps(x) + "\n" for x in labels))
    (tmp_path / "act.json").write_text(json.dumps({"retest_kappa": 1.0}))
    return sc


def test_dpo_pairs_are_rollout_level_capped_at_four_and_follow_the_labels(tmp_path):
    # most probable failure (Amendment 27 item 2): pairs on scenarios the P-dpo gate did not pass,
    # more than 4 per scenario, a chosen rollout that followed the order, or a capped rollout
    _fixture(tmp_path, capped={"a0|PZ-A000|0"})
    R.stage_contrast(tmp_path)
    pairs = T.dpo_pairs(tmp_path, _Render())
    by = {}
    for p in pairs:
        by.setdefault(p["chosen_id"].split("|")[1], []).append(p)
    assert set(by) == {"PZ-A000", "PZ-A001", "PZ-A002"} and all(len(v) == 4 for v in by.values())
    assert all(p["chosen_id"].startswith("a0s|") and p["rejected_id"].startswith("a0|")
               for p in pairs)
    assert "a0|PZ-A000|0" not in {p["rejected_id"] for p in pairs}
    assert len(T.retain_items(tmp_path, _Render())) == 4  # A0s on the legit and none scenarios


def test_ctx_rungs_never_train_on_the_held_out_cell(tmp_path, monkeypatch):
    # most probable failure (Amendment 29 item 3): P-ctx or P-ctx-n trains on a held-out scenario,
    # so the held-out read is not held out
    _fixture(tmp_path)
    monkeypatch.setattr(T, "heldout_ids", lambda: {"PZ-A001", "PZ-L000"})
    ex = T.ctx_examples(tmp_path, "a0s", _Render())
    assert len(ex) == 2 * 3  # 5 scenarios at k 2, less the 2 held out
    assert len(T.ctx_examples(tmp_path, "a0n", _Render())) == 6
    assert len(json.loads((ROOT / "papers/cst_constitutional_steering/data/"
                           "pressure_heldout_v1.json").read_text())["cell_ids"]) == 177


def test_long_pairs_choose_the_longest_against_the_shortest(tmp_path):
    # failure: P-long pairs the wrong ends of A0's lengths, so it trains toward short answers
    _fixture(tmp_path)
    pairs = T.long_pairs(tmp_path, _Render())
    assert len(pairs) == 5 and all(len(p["chosen"]) == 19 and len(p["rejected"]) == 10
                                   for p in pairs)


def test_p_dpo_is_not_trained_below_its_gate_and_the_line_gate_bails(tmp_path):
    # failure: P-dpo trains on too few disagreeing scenarios, or a pod trains on a failed line gate
    _fixture(tmp_path, n_away=3)
    R.stage_contrast(tmp_path)  # 3 disagreeing, under the 60 bar
    rec = T.stage_train(tmp_path, "p_dpo", True)
    assert rec["trained"] is False and rec["disagreeing"] == 3
    c = json.loads((tmp_path / "contrast.json").read_text())
    c["line_gate"]["passes"] = False
    (tmp_path / "contrast.json").write_text(json.dumps(c))
    with pytest.raises(SystemExit):
        T.gate_ok(tmp_path)


@pytest.mark.slow
def test_dry_sessions_smoke_train_parity_and_read(tmp_path, monkeypatch):
    # most probable failure (Amendments 28 item 3, 31): a later stage's entry path breaks on the
    # pod after hours of training; the smoke pass must run the data builders, one step of each
    # loss, the structural save check on each one-step adapter, the fixed adapter (its dry
    # stand-in) through the served gate and a 2-prompt read, and the dry sessions must reach reads
    cells = [c for c in R.grid() if c["writer"] == "subject"][:52]
    claude = R.claude_kept()[:3]
    monkeypatch.setattr(R, "grid", lambda: cells + [dict(c, writer="claude") for c in claude])
    monkeypatch.setattr(R, "claude_kept", lambda: claude)
    rec = R.stage_write(tmp_path, True)
    sc = [s for s in rec["scenarios"] if s["kind"] == "away"][:2] + [
        s for s in rec["scenarios"] if s["kind"] != "away"][:2]
    monkeypatch.setattr(R, "scen_list", lambda out: sc)
    for arm in R.K:
        R.stage_roll(tmp_path, arm, True)
    labels = []
    for arm in R.K:  # the stand-in's 8-token replies never close a trace: clear the caps
        rows = [dict(r, hit_cap=False, answer=r["answer"] or "x",
                     gen_ids=r["gen_ids"] + [1] * r["k"])  # lengths vary, as real ones do
                for r in R.rollouts(tmp_path, arm)]
        (tmp_path / "roll" / arm / "chunk_000.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        labels += [{"rollout_id": r["rollout_id"], "main": "does_not" if arm == "a0s" else
                    "follows"} for r in rows if r["kind"] != "none"]
    (tmp_path / "act.jsonl").write_text("".join(json.dumps(x) + "\n" for x in labels))
    (tmp_path / "act.json").write_text(json.dumps({"retest_kappa": 1.0}))
    R.stage_contrast(tmp_path)
    c = json.loads((tmp_path / "contrast.json").read_text())
    c["p_dpo_gate"]["passes"] = True  # 2 scenarios stand in for 60
    (tmp_path / "contrast.json").write_text(json.dumps(c))
    h = T.stage_smoke_hf(tmp_path, list(T.RUNGS), True)
    # Amendment 31: every rung's one-step adapter through the structural check (p_dpo's too)
    assert {Path(x["adapter"]).parts[-2] for x in h["structural"]} >= {"p_ctx", "p_long"}
    assert all(x["passes"] and x["keys_checked"] for x in h["structural"])
    assert T.stage_smoke_serve(tmp_path, True)["parity_passes"]
    for rung in T.RUNGS:
        assert T.stage_train(tmp_path, rung, True)["trained"]
    T.stage_parity_hf(tmp_path, list(T.RUNGS), True)
    res = T.stage_read(tmp_path, ["p_ctx", "p_dpo", "p_long"], True)
    assert set(res["parity"]) == {"p_ctx", "p_dpo", "p_long"} and res["base_tag"] == "base_session"
    T.stage_legit(tmp_path, "a0s", True)
    assert any((tmp_path / "gen" / "a0s_legit").glob("*.jsonl"))
    monkeypatch.setattr(T, "heldout_ids", lambda: {s["cell_id"] for s in sc[:3]})
    done = T.stage_heldout(tmp_path, [("base", None), ("p_ctx", T.adapter_dir(tmp_path, "p_ctx")),
                                      ("p_long", T.adapter_dir(tmp_path, "p_long"))], True)
    assert done == {"base": 6, "p_ctx": 6, "p_long": 6}
    for f in (tmp_path / "heldout").glob("*/chunk_000.jsonl"):  # the stand-in never closes a trace
        f.write_text("".join(json.dumps(dict(json.loads(x), answer="x", hit_cap=False)) + "\n"
                             for x in open(f)))
    hc = T.stage_heldout_act(tmp_path, True)
    assert set(hc["tags"]) == {"base", "p_ctx", "p_long"} and "note_p_long" in hc
    assert "vs_base" in hc["tags"]["p_ctx"]["away"]
