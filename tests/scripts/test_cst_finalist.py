"""CST's finalist-bars pod (Amendment 41):
papers/cst_constitutional_steering/scripts/cst_finalist_pod.py."""

from __future__ import annotations

import gzip
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


def test_import_leaves_the_judge_on_two_gpus(monkeypatch):
    # most probable failure (the verification pod's first launch): an override applied at import
    # leaks into MRB's own tests run in the same pytest process
    import mrb_g1_pod as P

    assert P.LLM_KWARGS["judge"]["tensor_parallel_size"] == 2
    monkeypatch.setitem(P.LLM_KWARGS, "judge", dict(P.LLM_KWARGS["judge"]))
    F.configure()
    assert P.LLM_KWARGS["judge"]["tensor_parallel_size"] == 1


def test_gate_at_load_stops_on_the_fallback_bar_or_the_price():
    # most probable failure (Amendment 41 item 1): the full read starts and runs into the judge's
    # time, or a read made mostly of random fallbacks passes the gate
    ok = F.mach_gate(0.4, 2000, 40.0, [550, 300], {"base": 0.02, "cand": 0.05}, 3.0, 0.3)
    assert ok["decisions_per_h_per_episode"] == 50 and ok["serial_h_reported"] == 11.0
    assert abs(ok["throughput_h"] - 3.72) < 1e-9 and ok["fits"]  # the serial bound does not gate
    fb = F.mach_gate(0.4, 2000, 40.0, [500], {"base": 0.02, "cand": 0.12}, 3.0, 0.3)
    assert fb["fallback_ok"] is False and not fb["fits"]
    slow = F.mach_gate(0.4, 400, 40.0, [100], {"base": 0.0, "cand": 0.0}, 3.0, 0.3)
    assert slow["throughput_h"] > 20 and not slow["fits"]
    late = F.mach_gate(0.4, 2000, 40.0, [100], {"base": 0.0, "cand": 0.0}, 7.0, 0.3)
    assert not late["fits"]  # 7.0 + 0.4 + 3.72 + 0.3 > 10.5


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
                            "decisions": 10, "points_norm": 20.0 + e, "scores": sc})
                decs.append({"arm": arm, "fallback": False, "capped": [False]})
    (md / "episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in eps))
    with gzip.open(md / "decisions.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps(x) + "\n" for x in decs))
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
    decs = [json.loads(x) for x in gzip.open(tmp_path / "mach" / "decisions.jsonl.gz", "rt")]
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
    assert gen["n_jobs"] == 4
    rows = [json.loads(x) for x in open(tmp_path / "mrb" / "gen" / f"{F.MODEL}.candidate.jsonl")]
    assert all(r["job_id"].split("/")[1] == "candidate" for r in rows)
    monkeypatch.setattr(M, "judge_units", lambda items, gens: [
        {"unit": f"{g}/response", "sid": g.split("/")[2], "text": t, "k": 0}
        for g, t in list(gens.items())[:1]])
    import mrb_g1_pod as P

    monkeypatch.setitem(P.LLM_KWARGS, "judge", dict(P.LLM_KWARGS["judge"]))
    F.configure()
    assert "calls_planned" in F.stage_mrb_judge(tmp_path, True)
    import finalist_read as FR

    res = FR.write(tmp_path)  # the read recomputes every bar from the rows
    assert res["refusal_retention"]["verdict_registered"]
    assert res["morebench"]["scenarios_read"] == 0 and res["machiavelli"] is None
    assert res["moralbench"]["session_base_vs_mrb_saved_base"]["rows"] in (0, 176)

