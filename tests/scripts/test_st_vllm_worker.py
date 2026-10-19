"""Amendment 7: the vLLM generation worker's job loop and envelope guard (gen_vllm.run_jobs) on a
fake engine (vLLM itself only runs on the pod)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "papers" / "st_live_reference" / "scripts"))
import gen_vllm as G  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeEngine:
    """Decodes ``per_step`` tokens for every live request per step; each step costs ``dt``."""

    def __init__(self, lengths: dict[str, int], clock: FakeClock, per_step=8, dt=1.0):
        self.lengths, self.clock, self.per_step, self.dt = lengths, clock, per_step, dt
        self.live: dict[str, int] = {}
        self.params: dict = {}
        self.aborted: list[str] = []

    def add_request(self, rid, prompt, params):
        assert prompt["prompt_token_ids"]  # token-id prompts, never text
        self.live[rid] = 0
        self.params[rid] = params

    def has_unfinished_requests(self):
        return bool(self.live)

    def step(self):
        self.clock.t += self.dt
        outs = []
        for rid in list(self.live):
            n = min(self.lengths[rid], self.live[rid] + self.per_step)
            self.live[rid] = n
            fin = n >= self.lengths[rid]
            outs.append(SimpleNamespace(request_id=rid, finished=fin, outputs=[SimpleNamespace(
                token_ids=list(range(n)), finish_reason="stop" if fin else None)]))
            if fin:
                del self.live[rid]
        return outs

    def abort_request(self, ids):
        self.aborted += ids
        for i in ids:
            self.live.pop(i, None)


def _jobs(n, cap=200):
    return [{"job_id": f"j{i}", "prompt_ids": [1, 2, 3], "max_tokens": cap, "seed": 100 + i}
            for i in range(n)]


def test_all_jobs_finish_with_their_ids_and_seeds():
    clock = FakeClock()
    lengths = {f"j{i}": 20 + 10 * i for i in range(10)}
    eng = FakeEngine(lengths, clock)
    rows, rec = G.run_jobs(eng, _jobs(10), lambda j: {"seed": j["seed"]}, clock=clock)
    assert all(rows[k]["n_tokens"] == v and rows[k]["status"] == "finished"
               for k, v in lengths.items())
    assert eng.params["j3"] == {"seed": 103}  # most probable failure: seeds not per request
    assert rec["decoded_tokens"] == sum(lengths.values()) and rec["stopped"] is None


def test_guard_aborts_when_the_projection_passes_the_deadline():
    # most probable failure: the guard never fires inside a concurrent run, so a slow engine
    # burns the envelope (the first T1.0 pod's lesson, Amendment 6 item 2)
    clock = FakeClock()
    lengths = {f"j{i}": (16 if i < 8 else 4000) for i in range(40)}
    eng = FakeEngine(lengths, clock, per_step=8, dt=1.0)
    # with the pilot's prior the projection sees the long tail before the deadline does
    rows, rec = G.run_jobs(eng, _jobs(40, cap=4000), lambda j: {}, deadline=60.0,
                           expected_len=3000, clock=clock)
    assert rec["stopped"] == "envelope" and rec["guard_checks"][-1]["stop"] == "envelope"
    assert clock.t < 60.0
    assert eng.aborted and all(rows[i]["status"] == "aborted" for i in eng.aborted)
    assert len(rows) == 40  # every job has a row, finished or aborted


def test_guard_lets_a_fitting_run_finish():
    clock = FakeClock()
    lengths = {f"j{i}": 40 for i in range(12)}
    eng = FakeEngine(lengths, clock, per_step=8, dt=1.0)
    rows, rec = G.run_jobs(eng, _jobs(12), lambda j: {}, deadline=1e6, clock=clock)
    assert rec["stopped"] is None and rec["n_finished"] == 12


def test_settings_hash_changes_with_engine_settings():
    a = G.settings_hash({"llm_kwargs": {"speculative_config": None}}, {"vllm": "0.31.0"})
    b = G.settings_hash({"llm_kwargs": {"speculative_config": {"method": "ngram_gpu"}}},
                        {"vllm": "0.31.0"})
    assert a != b


def test_without_a_prior_the_deadline_still_stops_the_run():
    # the pilot has no prior: early finishers are short, the projection underestimates, and the
    # hard deadline is the backstop
    clock = FakeClock()
    lengths = {f"j{i}": (16 if i < 8 else 4000) for i in range(40)}
    eng = FakeEngine(lengths, clock, per_step=8, dt=1.0)
    rows, rec = G.run_jobs(eng, _jobs(40, cap=4000), lambda j: {}, deadline=60.0, clock=clock)
    assert rec["stopped"] == "deadline" and clock.t <= 61.0


def test_worker_env_skips_flashinfer_sampler_jit_and_finds_venv_tools(monkeypatch):
    # most probable failure (third VALIDATE pod): engine warm-up JIT-compiles FlashInfer's sampler
    # and dies without ninja on PATH; the worker env turns that sampler off and puts the venv's
    # bin first
    import vllm_phase as V

    monkeypatch.setattr(V, "VLLM_PY", "/workspace/venv_vllm/bin/python")
    env = V.worker_env()
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["PATH"].split(":")[0] == "/workspace/venv_vllm/bin"


def test_first_logprobs_kept_for_the_requested_ids_only():
    # most probable failure (Amendment 8's descriptive column): a letter outside vLLM's top-k list
    # is imputed (or the row breaks) instead of being reported missing
    clock = FakeClock()
    eng = FakeEngine({"j0": 1, "j1": 1}, clock)
    real_step = eng.step

    def step():
        outs = real_step()
        for o in outs:
            o.outputs[0].logprobs = [{32: SimpleNamespace(logprob=-0.1),
                                      33: SimpleNamespace(logprob=-2.5)}]
        return outs
    eng.step = step
    jobs = _jobs(2, cap=1)
    jobs[0]["logprob_ids"] = [32, 33, 34]
    rows, _ = G.run_jobs(eng, jobs, lambda j: {"seed": j["seed"]}, clock=clock)
    assert rows["j0"]["first_logprobs"] == {"32": -0.1, "33": -2.5}  # 34 missing, not imputed
    assert "first_logprobs" not in rows["j1"]
    assert G.sampling_params(lambda **k: k, {**jobs[0], "seed": 1})["logprobs"] == 20
    assert G.sampling_params(lambda **k: k, {**jobs[1], "seed": 1})["logprobs"] is None
