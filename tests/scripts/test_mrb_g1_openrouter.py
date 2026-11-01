"""MRB-G1 OpenRouter cross-check (papers/mrb_morebench/scripts/mrb_g1_openrouter.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "papers" / "mrb_morebench" / "scripts"))

import mrb_g1_openrouter as O  # noqa: E402

from deepsteer.benchmarks.moral_reasoning.morebench.prompts import JUDGE_INSTRUCTION  # noqa: E402


class Recorder(O.StubClient):
    def __init__(self, contents=None):
        super().__init__()
        self.calls, self.contents = [], list(contents or [])

    def _create(self, model, messages, **kw):
        self.calls.append({"model": model, "messages": messages, **kw})
        r = super()._create(model, messages, **kw)
        if self.contents:
            r.choices[0].message.content = self.contents.pop(0)
        return r


def test_requests_are_upstreams_calls():
    # most probable failure: a request body that drifts from upstream's utils.py (the effort sent
    # as OpenRouter's reasoning object in the paper arm, or the judge at Table 4's top-p 0.95)
    cl = Recorder()
    O.call(cl, O.GEN_MODEL, "p", O.GEN_ARMS["paper"])
    O.call(cl, O.GEN_MODEL, "p", O.GEN_ARMS["reasoning_object"])
    O.call(cl, O.JUDGE_MODEL, "p", O.JUDGE_CALL)
    paper, obj, judge = cl.calls
    assert paper == {"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "p"}],
                     "temperature": 1.0, "top_p": 0.95, "max_tokens": 10_500,
                     "reasoning_effort": "high"}
    assert obj == {**paper, "extra_body": {"reasoning": {"effort": "high"}}}
    assert judge == {"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "p"}],
                     "temperature": 1, "top_p": 1, "max_tokens": 10_500,
                     "reasoning_effort": "high"}
    # the judge prompt is upstream's run_best_judge_on_responses.py line 78
    want = f"Reasoning Response:R\n\n{JUDGE_INSTRUCTION}\n\nRubric Criterion:C"
    assert O.judge_prompt("R", "C") == want


def test_empty_reply_is_retried_as_upstream_does():
    # most probable failure: a None content (OpenRouter's empty reply) recorded as a verdict-less
    # call on the first try, where upstream retries up to three times
    cl = Recorder(contents=[None, None, "yes"])
    assert O.call(cl, O.JUDGE_MODEL, "p", O.JUDGE_CALL)["content"] == "yes"
    assert len(cl.calls) == 3


def test_budget_stops_new_calls(tmp_path):
    # most probable failure: the run keeps calling past the token budget (the cost cap)
    b = O.Budget({"prompt_tokens": O.BUDGET["prompt_tokens"], "completion_tokens": 0})
    n = O.run_jobs([{"job_id": "a"}], lambda j: ({"prompt_tokens": 1}, {}), tmp_path / "o.jsonl",
                   tmp_path / "_raw" / "o.jsonl", 2, b)
    assert n == 0 and not (tmp_path / "o.jsonl").exists()


def _cached() -> bool:
    from deepsteer.datasets.morebench import default_cache_dir
    return default_cache_dir().exists() and (O.POD_OUT / "_raw" / "gen").exists()


@pytest.mark.skipif(not _cached(), reason="needs MoReBench cached and the pod's raw generations")
def test_dry_run_keeps_text_out_of_tracked_rows(tmp_path):
    # most probable failure: a tracked row carrying a generation, a trace or a judge reply (each
    # can quote the scenario; MRB_PLAN §9), or a judge unit missing for one model or field
    run = [sys.executable, str(REPO / "papers/mrb_morebench/scripts/mrb_g1_openrouter.py"),
           "all", "--out", str(tmp_path), "--dry-run", "--workers", "4"]
    rc = subprocess.run(run, capture_output=True, text=True, cwd=REPO,
                        env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"})
    assert rc.returncode == 0, rc.stderr[-2000:]
    gen = [json.loads(x) for x in (tmp_path / "gen.jsonl").read_text().splitlines()]
    judge = [json.loads(x) for x in (tmp_path / "judge.jsonl").read_text().splitlines()]
    assert len(gen) == 100 and not any(k in r for r in gen for k in ("trace", "response"))
    assert not any(k in r for r in judge for k in ("content", "reasoning", "prompt"))
    units = {r["unit"].rsplit("/", 1)[0].split("/")[0] + "/" + r["unit"].rsplit("/", 1)[1]
             for r in judge}
    assert units == {f"{m}/{f}" for m in ("gpt_oss_20b_or", "gpt_oss_20b", "qwen3_30b")
                     for f in ("trace", "response")}
    res = json.loads((tmp_path / "g1_openrouter_analysis.json").read_text())
    assert set(res["r3_near_misses"]) == {"gpt_oss_20b", "qwen3_30b"}
