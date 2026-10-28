"""The frontier exhibit's client and F0 reads (ST_PLAN Amendment 56; scripts/frontier_client.py).
Recorded-response mocks only: no test makes a live API call or reads a key."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "papers" / "st_live_reference" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
import frontier_client as F  # noqa: E402


class _Rec:
    """A recorded-response stand-in for an SDK resource: returns canned responses in order and
    keeps every call's keyword arguments."""

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def create(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


def _anthropic(text="Answer: B", out=1200, thinking=900, stop="end_turn", cache_read=0):
    usage = NS(input_tokens=400, cache_read_input_tokens=cache_read,
               cache_creation_input_tokens=0, output_tokens=out,
               output_tokens_details=NS(thinking_tokens=thinking))
    return NS(content=[NS(type="thinking", thinking=""), NS(type="text", text=text)],
              usage=usage, stop_reason=stop, model="claude-opus-5-5", _request_id="req_x")


def _openai(text="Answer: A", out=800, reasoning=600, refusal=False, cached=0):
    part = (NS(type="refusal", refusal="I can't help with that.") if refusal
            else NS(type="output_text", text=text))
    usage = NS(input_tokens=420, input_tokens_details=NS(cached_tokens=cached),
               output_tokens=out, output_tokens_details=NS(reasoning_tokens=reasoning))
    return NS(output=[NS(type="reasoning"), NS(type="message", content=[part])], usage=usage,
              status="completed", incomplete_details=None, model="gpt-6.1-sol",
              _request_id="req_y")


def _scenario():
    import pod_st_screen as P

    from deepsteer.kdg.schema import load_scenario_dir

    return load_scenario_dir([P.H1_SET])[0][0]


def _job(arm="default", k=0):
    from deepsteer.kdg.schema import assign_letters

    s = _scenario()
    return {"scenario": s, "k": k, "arm": arm, "order": assign_letters(s, k),
            "messages": [{"role": "user", "content": "..."}]}


def test_anthropic_call_sends_effort_and_no_temperature():
    # most probable failure: a temperature or a disabled-thinking setting is sent (both are a 400
    # on Opus 5.5), or the arm's effort is not the one recorded
    rec = _Rec([_anthropic()])
    c = F.AnthropicAdapter(NS(messages=rec)).call("claude-opus-5-5", [], "low")
    kw = rec.calls[0]
    assert kw["output_config"] == {"effort": "low"} and kw["thinking"] == {"type": "adaptive"}
    assert "temperature" not in kw and "fallbacks" not in kw
    assert c.text == "Answer: B" and c.reasoning_tokens == 900 and not c.refused


def test_openai_call_and_refusal():
    # most probable failure: a refusal part is read as an empty answer instead of a refusal, or
    # the request is stored server-side
    rec = _Rec([_openai(refusal=True)])
    c = F.OpenAIAdapter(NS(responses=rec)).call("gpt-6.1-sol", [], "medium")
    kw = rec.calls[0]
    assert kw["reasoning"] == {"effort": "medium"} and kw["store"] is False
    assert "temperature" not in kw
    assert c.refused and c.text == "" and c.reasoning_tokens == 600


def test_cost_counts_cached_input_once():
    # most probable failure: Anthropic's input_tokens (which excludes cache reads) and OpenAI's
    # (which includes them) are costed the same way, double-counting or dropping cached tokens
    a = F.AnthropicAdapter(NS(messages=_Rec([_anthropic(cache_read=100)]))).call(
        "claude-opus-5-5", [], "low")
    assert a.input_tokens == 500 and a.cached_input_tokens == 100
    assert F.cost_usd("claude-opus-5-5", a) == pytest.approx(
        (400 * 4.0 + 100 * 0.20 + 1200 * 20.0) / 1e6)
    o = F.OpenAIAdapter(NS(responses=_Rec([_openai(cached=120)]))).call("gpt-6.1-sol", [], "low")
    assert o.input_tokens == 420
    assert F.cost_usd("gpt-6.1-sol", o) == pytest.approx(
        (300 * 2.0 + 120 * 0.10 + 800 * 10.0) / 1e6)


def test_cap_stops_before_the_call_that_would_pass_it(tmp_path):
    # most probable failure: the cap is checked after the call, so the run ends above it
    per = F.cost_usd("gpt-6.1-sol", F.OpenAIAdapter(NS(responses=_Rec([_openai()]))).call(
        "gpt-6.1-sol", [], "low"))
    rec = _Rec([_openai() for _ in range(10)])
    r = F.Runner("gpt-6.1-sol", F.OpenAIAdapter(NS(responses=rec)), tmp_path, cap_usd=per * 3.5,
                 prior_call_usd=per)
    res = F.run(r, [_job(k=i) for i in range(10)])
    assert res["stopped"] == "cap" and len(rec.calls) == 3
    assert r.spent <= per * 3.5


def test_resume_skips_saved_rows_and_keeps_spend(tmp_path):
    # most probable failure: a rerun repeats saved calls (paying twice) or restarts its spend at 0,
    # so the cap no longer bounds the whole stage
    rec = _Rec([_openai() for _ in range(2)])
    r = F.Runner("gpt-6.1-sol", F.OpenAIAdapter(NS(responses=rec)), tmp_path, 1.0, 0.01)
    F.run(r, [_job("default"), _job("minimal")])
    rec2 = _Rec([_openai()])
    r2 = F.Runner("gpt-6.1-sol", F.OpenAIAdapter(NS(responses=rec2)), tmp_path, 1.0, 0.01)
    assert r2.spent == pytest.approx(r.spent)
    res = F.run(r2, [_job("default"), _job("minimal"), _job("default", k=1)])
    assert len(rec2.calls) == 1 and res["calls_this_run"] == 1
    rows = [json.loads(x) for x in (tmp_path / "gpt-6.1-sol.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and {r_["arm"] for r_ in rows} == {"default", "minimal"}


def test_rows_parse_with_the_shared_harness(tmp_path):
    # most probable failure: a private parser maps the reply, so the frontier rows drift from the
    # open models' readout (harness parity)
    job = _job()
    viol = next(L for L, o in job["order"] if o.norm_status == "violating")
    rec = _Rec([_anthropic(text=f"I would protect them. Answer: {viol}")])
    r = F.Runner("claude-opus-5-5", F.AnthropicAdapter(NS(messages=rec)), tmp_path, 1.0, 0.1)
    row = r.run_one(job)
    assert row["norm_status"] == "violating" and row["parse_method"] == "answer_line"
    assert row["temperature"] == "provider default (not sent)" and row["effort"] == "medium"
    assert row["harness_version"] == "1.0.0" and row["model_returned"] == "claude-opus-5-5"


def test_unknown_model_has_no_adapter_or_price():
    # most probable failure: an unpinned or unpriced model runs, so the cap has no rate to use
    with pytest.raises(SystemExit):
        F.adapter_for("some-model")
    with pytest.raises(SystemExit):
        F.Runner("gpt-6.1", None, Path("/tmp/none"), 1.0, 0.1)


def test_screen_jobs_are_the_60_screen_scenarios_interleaved():
    # most probable failure: a different scenario draw than the open screens', or the arms run
    # one after the other so a cap stop leaves only one arm
    jobs = F.screen_jobs(2)
    ids = {j["scenario"].id for j in jobs}
    assert len(ids) == 60 and len(jobs) == 60 * 2 * 2
    assert [j["arm"] for j in jobs[:4]] == ["default", "minimal", "default", "minimal"]
    assert jobs[0]["k"] == 0 and jobs[-1]["k"] == 1
    assert len(jobs[0]["messages"]) == 1 and jobs[0]["messages"][0]["role"] == "user"


def test_continuous_readout_renormalizes_over_shown_letters():
    # most probable failure: P(protect) is read from the raw log-prob (unnormalized, or over all
    # five letters), so the continuous reference drifts from the sampled share's frame
    import frontier_f0 as F0

    r = {"option_logps": {"A": np.log(0.2), "B": np.log(0.6), "C": np.log(0.1)},
         "order": {"A": "o1", "B": "o2", "C": "o3"}}
    roles = {"o1": "consistent", "o2": "violating", "o3": "neutral"}
    assert F0.p_protect_continuous(r, roles) == pytest.approx(0.6 / 0.9)


def test_subsample_band_and_agreement():
    # most probable failure: the band is taken over scenarios instead of over subsample draws of
    # the scenario mean, or the agreement compares against the subsample itself
    import frontier_f0 as F0

    by = {f"s{i}": [{"norm_status": "violating" if j < 8 else "consistent"} for j in range(16)]
          for i in range(30)}
    ref = {s: 0.5 for s in by}
    out = F0.subsample_read(by, ref, 16)
    assert out["band95"] == [0.0, 0.0] and out["agreement"] == 1.0
    out4 = F0.subsample_read(by, ref, 4)
    assert out4["band95"][0] < 0 < out4["band95"][1] and out4["n_scenarios"] == 30


def test_rate_limits_wait_and_retry_other_errors_raise():
    # most probable failure: a 429 ends the run (or skips the call), or a real error is retried
    class RateLimitError(Exception):
        pass

    class Flaky:
        def __init__(self, fails, exc):
            self.fails, self.exc, self.n = fails, exc, 0

        def call(self, *a):
            self.n += 1
            if self.n <= self.fails:
                raise self.exc
            return "ok"
    slept = []
    out, waits = F.call_paced(Flaky(2, RateLimitError()), "m", [], "low", sleep=slept.append)
    assert out == "ok" and waits == 2 and slept == [F.RATE_WAIT_S] * 2
    with pytest.raises(ValueError):
        F.call_paced(Flaky(1, ValueError("bad request")), "m", [], "low", sleep=slept.append)


def _f1_rows(p_def, p_min, defer_def=0.0, n_scen=30, k=8):
    rows = []
    for i in range(n_scen):
        for arm, p, dfr in (("default", p_def, defer_def), ("minimal", p_min, 0.0)):
            for j in range(k):
                x = j / k
                st = "violating" if x < p else ("neutral" if x < p + dfr else "consistent")
                rows.append({"scenario_id": f"S{i:02d}", "rollout": j, "arm": arm,
                             "norm_status": st, "refused": False, "cost_usd": 0.01,
                             "output_tokens": 500, "input_tokens": 400, "model": "gpt-6.1-sol"})
    return rows


def test_f1_branches(tmp_path):
    # most probable failure (Amendment 56 item 2): a protect drop that goes to deferral reads as
    # a brake into the rule, or the brake is called without the engaged criterion
    import analyze_frontier_f1 as A1

    f = tmp_path / "r.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in _f1_rows(0.25, 0.75)))
    assert A1.read(f)["branch"].startswith("brake")
    f.write_text("\n".join(json.dumps(r) for r in _f1_rows(0.25, 0.75, defer_def=0.5)))
    assert A1.read(f)["branch"].startswith("deferral-dominated")
    f.write_text("\n".join(json.dumps(r) for r in _f1_rows(0.5, 0.5)))
    res = A1.read(f)
    assert res["branch"].startswith("no brake") and res["brake_B"]["mean"] == 0.0
    proj = A1.project(res, "claude-opus-5-5")
    assert proj["central"]["total_usd"] == pytest.approx(
        2 * (400 * 4.0 + 500 * 20.0) / 1e6 * 8 * 60)


def test_high_arm_branch_needs_forced_deliberation():
    # most probable failure (the addition, item 3): a flat protect share at a high effort that
    # barely reasons more reads as "judgment survives deliberation"
    import analyze_frontier_f1 as A1

    flat = {"mean": 0.0, "ci95": [-0.03, 0.03]}
    assert A1.high_branch(flat, 400, 39).startswith("judgment survives")
    assert A1.high_branch(flat, 120, 39).startswith("deliberation not forced")
    assert A1.high_branch({"mean": -0.2, "ci95": [-0.3, -0.1]}, 60, 39).startswith("brake")


def _ladder_rows(p_by_arm, tok_by_arm, n_scen=20, k=8, zero_floor_scen=True):
    rows = []
    for i in range(n_scen):
        for arm, p in p_by_arm.items():
            for j in range(k):
                t = tok_by_arm[arm]
                if arm == "minimal" and zero_floor_scen and i == 0:
                    t = 0  # a scenario whose floor mean is 0: a per-scenario log is undefined
                rows.append({"scenario_id": f"S{i:02d}", "rollout": j, "arm": arm,
                             "norm_status": "violating" if j / k < p else "consistent",
                             "refused": False, "cost_usd": 0.01, "output_tokens": t + 7,
                             "input_tokens": 274, "reasoning_tokens": t, "stop": "completed",
                             "model": "gpt-6.1-sol"})
    return rows


def test_ladder_slope_sign_and_x_rule(tmp_path):
    # most probable failure (the third addition, item 2): x is taken per scenario, so a scenario
    # with zero floor reasoning gives log(0) and the read crashes or drops it, or the slope's
    # sign is read against the registered sign
    import analyze_frontier_f1 as A1

    tok = {"minimal": 20, "default": 40, "high": 110, "xhigh": 300}
    up = {"minimal": 0.5, "default": 0.625, "high": 0.75, "xhigh": 0.875}
    ld = A1.ladder(_ladder_rows(up, tok))
    assert ld["n_scenarios"] == 20 and ld["slope"]["mean"] > 0
    assert ld["branch"].startswith("deliberation helps")
    x = np.log([20 * 19 / 20, 40, 110, 300])  # the floor's arm mean includes the zero scenario
    assert ld["arm_mean_reasoning_tokens"]["minimal"] == pytest.approx(19.0)
    assert ld["slope"]["mean"] == pytest.approx(np.polyfit(x, list(up.values()), 1)[0])
    down = dict(zip(up, reversed(list(up.values()))))
    assert A1.ladder(_ladder_rows(down, tok))["branch"].startswith("brake")
    flat = {a: 0.5 for a in up}
    assert A1.ladder(_ladder_rows(flat, tok))["branch"].startswith("flat")
    assert A1.ladder([r for r in _ladder_rows(up, tok) if r["arm"] != "xhigh"]) is None
    f = tmp_path / "r.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in _ladder_rows(up, tok)))
    res = A1.read(f)
    assert res["ladder"]["branch"].startswith("deliberation helps")
    assert res["xhigh"]["incomplete_rate"] == 0.0


def _a34_rows(move_by_scen, tok_by_scen, k=8):
    rows = []
    for s, (base, moved) in move_by_scen.items():
        for arm, p, t in (("default", base, 200), ("forced", moved, tok_by_scen[s])):
            for j in range(k):
                rows.append({"scenario_id": s, "rollout": j, "arm": arm,
                             "norm_status": "violating" if j / k < p else "neutral",
                             "reasoning_tokens": t})
    return rows


def test_a34_rule():
    # most probable failure (ST-A34's fixed rule): a move carried by the longest-thinking quarter
    # reads as broad, or the top quartile is taken by the shortest thinking (sort order)
    import analyze_frontier_f1 as A1

    sids = [f"S{i:02d}" for i in range(40)]
    tok = {s: 1000 + 100 * i for i, s in enumerate(sids)}
    tail = {s: (0.5, 1.0 if i >= 30 else 0.5) for i, s in enumerate(sids)}  # the top 10 move
    r = A1.a34(_a34_rows(tail, tok))
    assert r["up"] == 10 and r["top_quartile_share_of_sum"] == pytest.approx(1.0)
    assert r["branch"].startswith("R_tail")
    rng = np.random.default_rng(1)
    broad = {s: (0.5, 0.75 if rng.random() < 0.6 else 0.5) for s in sids}
    r = A1.a34(_a34_rows(broad, tok))
    assert not r["branch"].startswith("R_tail")  # moves spread with no length order
    assert r["defer_change_mean"] == pytest.approx(-r["d_mean"])  # protect gained from defer


def test_jobs_for_the_xhigh_arm_only():
    # most probable failure: the xhigh arm is sent at another effort, or reruns the other arms
    jobs = F.screen_jobs(1, ("xhigh",))
    assert {j["arm"] for j in jobs} == {"xhigh"} and F.ARMS["xhigh"] == "xhigh"


def test_jobs_for_the_high_arm_only():
    # most probable failure: the addition reruns the F1 arms (paying twice) instead of high alone
    jobs = F.screen_jobs(1, ("high",))
    assert {j["arm"] for j in jobs} == {"high"} and len(jobs) == 60
    assert F.ARMS["high"] == "high"


def test_max_effort_streams_with_the_long_ceiling():
    # most probable failure: the max-effort arm is sent non-streaming at 16k, so long thinking is
    # cut off (stop max_tokens) and the reply reads as unparsed
    class _Stream:
        def __init__(self, resp, log):
            self.resp, self.log = resp, log

        def __call__(self, **kw):
            self.log.append(kw)
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return self.resp
    log = []
    client = NS(messages=NS(stream=_Stream(_anthropic(out=9000, thinking=8800), log),
                            create=lambda **kw: (_ for _ in ()).throw(AssertionError("create"))))
    c = F.AnthropicAdapter(client).call("claude-opus-5-5", [], F.ARMS["forced"])
    assert log[0]["max_tokens"] == F.MAX_OUTPUT_STREAM and log[0]["output_config"] == {
        "effort": "max"}
    assert c.reasoning_tokens == 8800 and "temperature" not in log[0]


class _Blocking:
    """A recorded-response stand-in whose calls wait until released (calls held in flight)."""

    def __init__(self, n):
        import threading

        self.responses, self.calls = [_openai() for _ in range(n)], []
        self.lock, self.release = threading.Lock(), threading.Event()

    def create(self, **kw):
        with self.lock:
            self.calls.append(kw)
            r = self.responses.pop(0)
        self.release.wait(5)
        return r


def test_inflight_calls_count_in_the_cap(tmp_path):
    # most probable failure (Amendment 58 item 2): with 8 workers each sees spend 0 and starts its
    # call, so 8 calls run against a cap that allows 3
    import threading

    per = F.cost_usd("gpt-6.1-sol", F.OpenAIAdapter(NS(responses=_Rec([_openai()]))).call(
        "gpt-6.1-sol", [], "medium"))
    rec = _Blocking(20)
    r = F.Runner("gpt-6.1-sol", F.OpenAIAdapter(NS(responses=rec)), tmp_path, cap_usd=per * 3.5,
                 prior_call_usd=per)
    out = {}
    t = threading.Thread(target=lambda: out.update(F.run(r, [_job(k=i) for i in range(20)],
                                                         workers=8)))
    t.start()
    import time as _t

    _t.sleep(0.5)  # every worker has claimed or stopped while the first calls are held
    assert len(rec.calls) == 3
    rec.release.set()
    t.join(10)
    assert out["stopped"] == "cap" and out["calls_this_run"] == 3
    assert r.spent <= per * 3.5 and not r.inflight


def test_concurrent_rows_are_saved_once_and_resume(tmp_path):
    # most probable failure: two workers take one job (a duplicate row, paid twice) or a row is
    # lost under concurrent appends, so the resume repeats calls
    rec = _Rec([_openai() for _ in range(40)])
    r = F.Runner("gpt-6.1-sol", F.OpenAIAdapter(NS(responses=rec)), tmp_path, 10.0, 0.01)
    res = F.run(r, [_job(k=i) for i in range(40)], workers=8)
    rows = [json.loads(x) for x in (tmp_path / "gpt-6.1-sol.jsonl").read_text().splitlines()]
    assert res["stopped"] == "complete" and len(rec.calls) == 40 and len(rows) == 40
    assert len({(x["scenario_id"], x["rollout"], x["arm"]) for x in rows}) == 40
    assert r.spent == pytest.approx(sum(x["cost_usd"] for x in rows))
    r2 = F.Runner("gpt-6.1-sol", F.OpenAIAdapter(NS(responses=_Rec([]))), tmp_path, 10.0, 0.01)
    assert F.run(r2, [_job(k=i) for i in range(40)], workers=8)["calls_this_run"] == 0


def test_a_worker_error_stops_the_run_and_frees_its_slot(tmp_path):
    # most probable failure: a failed call leaves its key in flight (the cap stays reserved) or
    # the other workers keep calling after an error
    import threading

    class _Fail(_Rec):
        lock = threading.Lock()

        def create(self, **kw):
            with self.lock:  # thread-safe: exactly the third call fails
                if len(self.calls) == 2:
                    self.calls.append(kw)
                    raise ValueError("bad request")
                return super().create(**kw)

    for workers, most in ((1, 3), (8, 3 + 7)):  # 8 workers: the 7 others' calls in flight end
        rec = _Fail([_openai() for _ in range(40)])
        d = tmp_path / str(workers)
        r = F.Runner("gpt-6.1-sol", F.OpenAIAdapter(NS(responses=rec)), d, 10.0, 0.01)
        with pytest.raises(ValueError):
            F.run(r, [_job(k=i) for i in range(40)], workers=workers)
        assert not r.inflight and 3 <= len(rec.calls) <= most


def test_panel_jobs():
    # most probable failure (Amendment 58 item 2): a cell is missing or sent at another effort,
    # the known-gap cell loses its operator prompt, or the jobs are not interleaved by rollout
    jobs = F.panel_jobs(2)
    assert len(jobs) == 586 * 5 * 2
    assert {j["k"] for j in jobs[: 586 * 5]} == {0}
    assert {j["cell"] for j in jobs} == {"dl_chat_neutral", "jl_chat_neutral",
                                         "dl_chat_neutral_pressure_removed",
                                         "jl_chat_neutral_pressure_removed", "dl_chat_known_gap"}
    assert {j["effort"] for j in jobs} == {"medium"}
    kg = [j for j in jobs if j["cell"] == "dl_chat_known_gap"]
    assert all(j["messages"][0]["role"] == "system" for j in kg)
    assert all([m["role"] for m in j["messages"]] == ["user"] for j in jobs
               if j["cell"] != "dl_chat_known_gap")


def _panel_rows(p_by_cell, n_scen=40, k=8, unparsed_cell=None):
    rows = []
    for i in range(n_scen):
        for cell, p in p_by_cell.items():
            pc = p(i) if callable(p) else p
            for j in range(k):
                st = "violating" if j / k < pc else "consistent"
                if cell == unparsed_cell and i == 0 and j < 5:
                    st = None  # 3 of 8 parsed: below the floor
                rows.append({"scenario_id": f"P{i:03d}", "rollout": j, "arm": cell,
                             "norm_status": st, "refused": False, "output_tokens": 46,
                             "cost_usd": 0.001, "model": "gpt-6.1-sol"})
    return rows


def test_panel_e_read_and_branches():
    # most probable failure (Amendment 58 items 3-4): E is reported though the known-gap control
    # failed, the floor admits a scenario with too few parsed replies, or the at-rest gap is not
    # subtracted (a baseline gap read as pressure-attributable)
    import analyze_frontier_panel as AP

    base = {AP.D: 0.5, AP.J: 0.25, AP.DN: 0.25, AP.JN: 0.25, AP.KG: 0.875}
    r = AP.read(_panel_rows(base))
    assert r["positive_control"]["passes"] and r["E"]["mean"] == pytest.approx(0.25)
    assert r["branch"].startswith("E > 0")
    at_rest = {AP.D: 0.5, AP.J: 0.25, AP.DN: 0.5, AP.JN: 0.25, AP.KG: 0.875}  # gap only at rest
    r = AP.read(_panel_rows(at_rest))
    assert r["E"]["mean"] == pytest.approx(0.0) and r["branch"].startswith("null")
    no_ctrl = {**base, AP.KG: 0.25}
    r = AP.read(_panel_rows(no_ctrl))
    assert r["branch"].startswith("positive control fails") and "withheld" in r["E"]
    r = AP.read(_panel_rows(base, unparsed_cell=AP.J))
    assert r["n_scenarios_E"] == 39
    neg = {AP.D: 0.25, AP.J: 0.25, AP.DN: 0.5, AP.JN: 0.25, AP.KG: 0.875}
    assert AP.read(_panel_rows(neg))["branch"].startswith("unregistered")


def test_panel_jobs_through_the_openai_adapter(tmp_path):
    # most probable failure: the known-gap cell's system prompt is dropped on the way to the API,
    # or the panel's calls go out at the F1 arm's effort instead of Amendment 58's medium
    import analyze_frontier_panel as AP

    jobs = F.panel_jobs(1)[:5]  # one scenario's five cells
    rec = _Rec([_openai() for _ in range(5)])
    r = F.Runner("gpt-6.1-sol", F.OpenAIAdapter(NS(responses=rec)), tmp_path, 1.0, 0.01)
    assert F.run(r, jobs, workers=8)["calls_this_run"] == 5
    assert {c["reasoning"]["effort"] for c in rec.calls} == {"medium"}
    kg = [c for c in rec.calls if c["input"][0]["role"] == "system"]
    assert len(kg) == 1 and "Standing instruction" in kg[0]["input"][0]["content"]
    rows = [json.loads(x) for x in (tmp_path / "gpt-6.1-sol.jsonl").read_text().splitlines()]
    assert {x["cell"] for x in rows} == set(AP.CELLS) and {x["effort"] for x in rows} == {"medium"}
