"""CST's finalist-bars pod (Amendment 41):
papers/cst_constitutional_steering/scripts/cst_finalist_pod.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts"),
                str(ROOT / "papers/kdg_panel/scripts"),
                str(ROOT / "papers/st_live_reference/scripts"),
                str(ROOT / "papers/mrb_morebench/scripts"),
                str(ROOT / "papers/mch_machiavelli/scripts"), str(ROOT / "tests/scripts")]

import cst_finalist_pod as F  # noqa: E402


class _Engine:
    def __init__(self):
        self.calls = []

    def add_request(self, rid, prompt, params, **kw):
        self.calls.append((rid, kw.get("lora_request")))

    def has_unfinished_requests(self):
        return False


def test_engine_proxy_serves_the_candidate_on_every_request():
    # most probable failure: MoReBench's generation (MRB's run_chunks calls add_request with three
    # arguments) runs on the base, so the "candidate" rows are the base's
    eng = _Engine()
    px = F.EngineProxy(eng, "cand-req")
    px.add_request("j1", {"prompt_token_ids": [1]}, None)
    assert eng.calls == [("j1", "cand-req")] and not px.has_unfinished_requests()


def test_import_leaves_the_judge_and_capability_alone(monkeypatch):
    # most probable failure (the verification pod's first launch): an override applied at import
    # leaks into MRB's and the SDF pod's own tests run in the same pytest process
    import cst_sdf_norec_pod as Q
    import mrb_g1_pod as P

    assert P.LLM_KWARGS["judge"]["tensor_parallel_size"] == 2 and Q.CAP_N == 200
    monkeypatch.setitem(P.LLM_KWARGS, "judge", dict(P.LLM_KWARGS["judge"]))
    monkeypatch.setattr(Q, "CAP_N", Q.CAP_N)
    monkeypatch.setattr(Q, "items", Q.items)
    F.configure()
    assert P.LLM_KWARGS["judge"]["tensor_parallel_size"] == 1 and Q.CAP_N == 800
    assert Q.items is F.cap_items


def test_budget_of_record_is_amendment_42s():
    # most probable failure (Amendment 42 item 2): MACHIAVELLI runs at §6.4's 1,024 tokens, which
    # three quarters of the candidate's free-form traces exceed
    assert F.MACH_TOKENS == 4096 and F.FALLBACK_BAR == 0.10


def test_gate_at_load_stops_on_the_fallback_bar_or_the_price(monkeypatch):
    # most probable failure (Amendment 42 item 1): the full read starts and runs past the pod's
    # price, or a read made mostly of random fallbacks passes the gate
    monkeypatch.delenv("CST_PRICE_H", raising=False)
    ok = F.mach_gate(0.4, 2000, 40.0, [550, 300], {"base": 0.02, "cand": 0.05}, 3.0, 0.3)
    assert ok["decisions_per_h_per_episode"] == 50 and ok["serial_h_reported"] == 11.0
    assert abs(ok["throughput_h"] - 3.72) < 1e-9 and ok["fits"]  # the serial bound does not gate
    fb = F.mach_gate(0.4, 2000, 40.0, [500], {"base": 0.02, "cand": 0.12}, 3.0, 0.3)
    assert fb["fallback_ok"] is False and not fb["fits"]
    slow = F.mach_gate(0.4, 400, 40.0, [100], {"base": 0.0, "cand": 0.0}, 3.0, 0.3)
    assert slow["throughput_h"] > 20 and not slow["fits"]
    late = F.mach_gate(0.4, 2000, 40.0, [100], {"base": 0.0, "cand": 0.0}, 7.0, 0.3)
    assert not late["fits"]  # 7.0 + 0.4 + 3.72 + 0.3 > 10.5
    monkeypatch.setenv("CST_PRICE_H", "12.0")  # the MACHIAVELLI pod's own price (item 3's split)
    assert F.mach_gate(0.4, 2000, 40.0, [100], {"base": 0.0, "cand": 0.0}, 7.0, 0.3)["fits"]


def test_jobs_pair_the_arms_and_put_the_longest_game_first():
    # most probable failure: the two arms run different (game, episode) sets, so no pair forms
    jobs = F.mach_jobs(["avatar-of-the-wolf", "luminous-underground"])
    assert jobs[:2] == [("base", "luminous-underground", 0), ("cand", "luminous-underground", 0)]
    assert {(g, e) for a, g, e in jobs if a == "base"} == {(g, e) for a, g, e in jobs
                                                           if a == "cand"}


def test_mach_read_pairs_by_game_and_seed_and_flags_a_cut_read(tmp_path):
    # most probable failure (Amendment 41 item 1): the read pairs episodes by position rather than
    # by game and seed, or a read the deadline cut reports as the registered 30-game read
    import finalist_read as FR
    from mch_g1_harness import GAMES

    if not (FR.PAPERS / "mch_machiavelli/outputs/g0_random.jsonl.gz").exists():
        pytest.skip("MCH's random episodes are not here (a pod syncs no other paper's outputs)")
    md = tmp_path / "mach"
    md.mkdir()
    eps, decs = [], []
    for g in GAMES:
        for e, seed in ((0, 2417), (1, 2418)):
            for arm, v in (("base", 80.0), ("cand", 70.0)):
                if g == GAMES[0] and arm == "cand" and e == 1:
                    continue  # the deadline cut one candidate episode
                sc = {k: {"r0": v + e} for k in ("violations.Σ", "power.Σ", "utility.Σ")}
                eps.append({"arm": arm, "game": g, "episode": e, "seed": seed, "status": "done",
                            "decisions": 10, "points_norm": 20.0 + e, "scores": sc,
                            "lineage": "L1"})
                decs.append({"arm": arm, "game": g, "episode": e, "step": 0, "lineage": "L1",
                             "fallback": False, "capped": [False]})
    (md / "episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in eps))
    (md / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in decs))
    r = FR.mach_read(tmp_path)
    assert r["pairs"] == 59 and r["registered"] is False and r["n_games"] == 30
    assert abs(r["candidate_minus_base"]["violations"]["point"] + 10.0) < 1e-9


def _candidate(out: Path) -> str:
    import pilot_pod_lib as PL

    _, wb, _ = PL.load_model(True)
    adir = out / "adapters" / "cand" / "1"
    wb.model.save_pretrained(str(adir))
    (out / "train_cand.json").write_text(json.dumps({"trained": True,
                                                     "adapters": {"1": str(adir)}}))
    return "cand"


def _src():
    from deepsteer.datasets import machiavelli as mch

    if not (mch.default_cache_dir() / "src" / "game_data").exists():
        pytest.skip("MACHIAVELLI engine and bundle not fetched")
    return mch.fetch_machiavelli(verify_bundle=False)


@pytest.mark.slow
def test_mach_pairs_both_arms_and_stops_at_the_gate(tmp_path, monkeypatch):
    # most probable failure (Amendment 41 item 1): the arms get different request seeds (no pairing
    # by decision) or a stopped read leaves episodes marked done, so a resume skips them
    import mch_thinking_pod as MT

    src = _src()
    monkeypatch.setattr(F, "_mach_src", lambda: src)
    tag = _candidate(tmp_path)
    game = "avatar-of-the-wolf"
    seen: list[tuple[str, int]] = []
    orig = MT.StubEngine.add

    def add(self, rid, ids, seed):
        seen.append((rid, seed))
        return orig(self, rid, ids, seed)

    monkeypatch.setattr(MT.StubEngine, "add", add)
    meta = F.mach_run(tmp_path, tag, F.mach_jobs([game], (0,)), True, check_s=1e9)
    assert meta["stopped"] is None and meta["decisions"] > 0
    eps = [json.loads(x) for x in open(tmp_path / "mach" / "episodes.jsonl")]
    assert {e["arm"] for e in eps} == {"base", "cand"} and len({e["seed"] for e in eps}) == 1
    decs = F.read_rows(tmp_path / "mach" / "decisions.jsonl")
    by = {a: [(d["step"], d["attempts"]) for d in decs if d["arm"] == a] for a in F.ARMS}
    assert by["base"] and by["base"][:5] == by["cand"][:5]  # same seeds, same stub replies
    stop = tmp_path / "stop"
    stop.mkdir()
    (stop / "train_cand.json").write_text((tmp_path / "train_cand.json").read_text())
    meta = F.mach_run(stop, tag, F.mach_jobs([game], (0,)), True, check_s=0.0)
    assert meta["stopped"] == "gate" and not meta["gate"]["fits"]
    eps = [json.loads(x) for x in open(stop / "mach" / "episodes.jsonl")]
    assert eps and all(e["status"] == "stopped" for e in eps)
    meta = F.mach_run(stop, tag, F.mach_jobs([game], (0,)), True, check_s=1e9)
    assert meta["run"] == 2  # the stopped episodes run again on a resume


@pytest.mark.slow
def test_mach_resumes_from_saved_decisions_without_regenerating(tmp_path, monkeypatch):
    # most probable failure (Amendment 43 item 1): a late failure loses every episode in flight;
    # after a kill (no episode rows written, a torn last line), the resume replays the saved actions
    # through the engine, generates nothing it already had, and ends where the full run ended
    src = _src()
    monkeypatch.setattr(F, "_mach_src", lambda: src)
    tag = _candidate(tmp_path)
    jobs = F.mach_jobs(["avatar-of-the-wolf"], (0,))
    full = F.mach_run(tmp_path, tag, jobs, True, check_s=1e9)
    md = tmp_path / "mach"
    first = {(e["arm"], e["game"], e["episode"]): e for e in F.read_rows(md / "episodes.jsonl")}
    (md / "episodes.jsonl").write_text("")  # killed before any episode row was written
    with open(md / "decisions.jsonl", "a") as f:
        f.write('{"arm": "base", "game": "torn')
    again = F.mach_run(tmp_path, tag, jobs, True, check_s=1e9)
    assert again["resumed_episodes"] == 2 and again["replay_restarts"] == 0
    assert again["replayed_decisions"] == full["decisions"] and again["decisions"] == 0
    second = {(e["arm"], e["game"], e["episode"]): e for e in F.read_rows(md / "episodes.jsonl")}
    for k, e in first.items():
        assert second[k]["status"] == "done" and second[k]["scores"] == e["scores"]
        assert second[k]["lineage"] == e["lineage"] and second[k]["decisions"] == e["decisions"]


@pytest.mark.slow
def test_dry_session_runs_every_stage_in_order(tmp_path, monkeypatch):
    # most probable failure (Amendment 28 item 3): a later stage breaks on the pod after hours;
    # every stage runs on the tiny model in the runner's order and writes its record
    import mrb_p2_pod as M

    tag = _candidate(tmp_path)
    assert F.stage_smoke_hf(tmp_path, tag, True)["structural"][0]["passes"]
    assert F.PL.smoke_serve(tmp_path, True)["parity_passes"]
    F.stage_parity(tmp_path, tag, True)
    rr = F.stage_rr(tmp_path, tag, True)
    assert rr["items"] == 2 and {"base.jsonl", "cand.jsonl"} <= {
        p.name for p in (tmp_path / "rr").iterdir()}
    assert F.stage_mb(tmp_path, tag, True)["rows"] == 176  # 88 items x both orders
    assert len((tmp_path / "mb" / "r2_adapter.jsonl").read_text().splitlines()) == 176
    gen = F.stage_mrb_gen(tmp_path, tag, True)
    assert gen["n_jobs"] == 4 and gen["session_check"]["n_jobs"] == 4
    sess = [json.loads(x) for x in open(tmp_path / "mrb_session" / "gen.jsonl")]
    assert all(r["job_id"].split("/")[1] == "none" for r in sess)  # MRB's base ids
    assert not (tmp_path / "mrb" / "_raw" / "gen" / "gen.jsonl").exists()  # the judge never sees it
    rows = [json.loads(x) for x in open(tmp_path / "mrb" / "gen" / f"{F.MODEL}.candidate.jsonl")]
    assert all(r["job_id"].split("/")[1] == "candidate" for r in rows)
    monkeypatch.setattr(M, "judge_units", lambda items, gens: [
        {"unit": f"{g}/response", "sid": g.split("/")[2], "text": t, "k": 0}
        for g, t in list(gens.items())[:1]])
    import mrb_g1_pod as P

    monkeypatch.setitem(P.LLM_KWARGS, "judge", dict(P.LLM_KWARGS["judge"]))
    import cst_sdf_norec_pod as Q

    monkeypatch.setattr(Q, "CAP_N", Q.CAP_N)
    monkeypatch.setattr(Q, "items", Q.items)
    F.configure()
    cap = F.stage_capability(tmp_path, tag, True)
    assert "vs_base" in cap["tags"][tag]
    assert "calls_planned" in F.stage_mrb_judge(tmp_path, True)
    import finalist_read as FR

    # the read runs locally against MRB's saved rows, which a pod does not sync (the finalist pods'
    # first launch bailed here); on a pod the stages above are the test
    need = [FR.MRB_P2 / "judge" / "shard0.jsonl.gz", FR.MRB_P2 / "gen" / f"{F.MODEL}.none.jsonl",
            FR.MRB_P2 / "mb_phase2" / F.MODEL / "r2_none.jsonl"]
    if not all(p.exists() for p in need):
        return
    res = FR.write(tmp_path)  # the read recomputes every bar from the rows
    assert res["refusal_retention"]["verdict_registered"]
    assert res["morebench"]["scenarios_read"] == 0 and res["machiavelli"] is None
    assert res["moralbench"]["session_base_vs_mrb_saved_base"]["rows"] in (0, 176)
    assert res["session_check"]["rollouts"] == 4 and res["capability"]["n"] == 2

