"""Gate tests for the MCH thinking-on runner (MCH_PLAN.md §5, Amendment 10).

Run locally before the pod and as the pod's gate. Render and dry-run tests need the pinned
tokenizers and the fetched engine; they skip when those are absent.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "papers" / "mch_machiavelli" / "scripts"))

pod = pytest.importorskip("mch_thinking_pod")


def _tok(model: str):
    tr = pytest.importorskip("transformers")
    m = pod.MODELS[model]
    try:
        return tr.AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"])
    except OSError:
        pytest.skip(f"{m['repo']} tokenizer not available")


def _src():
    from deepsteer.datasets import machiavelli as mch

    if not (mch.default_cache_dir() / "src" / "game_data").exists():
        pytest.skip("MACHIAVELLI engine and bundle not fetched")
    return mch.fetch_machiavelli(verify_bundle=False)


def _state(n: int = 3):
    from deepsteer.benchmarks.agentic.machiavelli.prompts import SceneState

    return SceneState(title="T", short_summary="S", player_character="P",
                      achievements=(("a1", "First: do it", 10),), unlocked=frozenset(),
                      num_choices=n, choice_texts=tuple("xyzw"[:n]), obs="Scene.\n0: x\n1: y")


def test_agents_swap_only_the_json_paragraph_for_the_papers_integer_answer():
    # most probable failure: new wording slipping into the prompts, or the JSON format left in
    from deepsteer.benchmarks.agentic.machiavelli import prompts as P

    s = _state()
    for agent, base in pod.BASE.items():
        sys_, user = pod.think_prompts(agent, s)
        base_sys, base_user = P.chat_prompts(base, s)
        assert user == base_user
        assert sys_ == base_sys[: -len(P._CHAT_JSON)] + pod.INTEGER_ANSWER
        assert "JSON" not in sys_ and pod.INTEGER_ANSWER in P._CHAT_ACTION_INT
    assert "TARGET BEHAVIOR" in pod.think_prompts("ThinkGoodCoTAgent", s)[0]
    assert "TARGET BEHAVIOR" not in pod.think_prompts("ThinkCoTAgent", s)[0]


def test_integer_parse_is_the_papers():
    # upstream LMAgent: the first number in the stripped reply, if it is a label
    assert pod.parse_integer(" 2", 3) == 2
    assert pod.parse_integer("I pick 1, not 2", 3) == 1
    assert pod.parse_integer("Action 7", 3) is None
    assert pod.parse_integer("no number", 3) is None


def test_gpt_oss_render_thinks_at_medium_with_the_date_pinned_and_no_prefill():
    # most probable failure: a dose-0 prefill left in (which would suppress thinking) or the date
    tok = _tok("gpt_oss_20b")
    text = pod.ThinkRenderer("gpt_oss_20b", tok).text("ThinkCoTAgent", _state())
    assert "Reasoning: medium\n" in text and "Current date: 2026-10-04\n" in text
    assert text.endswith("<|start|>assistant") and "<|channel|>final" not in text


def test_qwen_render_thinks_natively_with_no_prefill():
    tok = _tok("qwen3_30b")
    text = pod.ThinkRenderer("qwen3_30b", tok).text("ThinkGoodCoTAgent", _state())
    assert text.endswith("<|im_start|>assistant\n<think>\n") and "</think>" not in text


def test_split_finds_the_answer_after_the_trace_and_none_when_it_never_closes():
    tok = _tok("gpt_oss_20b")
    r = pod.ThinkRenderer("gpt_oss_20b", tok)
    done = tok.encode("<|channel|>analysis<|message|>think<|end|><|start|>assistant"
                      "<|channel|>final<|message|>1<|return|>", add_special_tokens=False)
    assert r.split(done) == ("think", "1")
    cut = tok.encode("<|channel|>analysis<|message|>still thinking", add_special_tokens=False)
    assert r.split(cut)[1] is None
    q = pod.ThinkRenderer("qwen3_30b", _tok("qwen3_30b"))
    assert q.split(q.tok.encode("hmm</think>\n\n0<|im_end|>", add_special_tokens=False)) == (
        "hmm", "\n\n0")
    assert q.split(q.tok.encode("hmm, still", add_special_tokens=False))[1] is None


def _dry(tmp_path, mode: str):
    tok = _tok("qwen3_30b")
    src = _src()
    gen = pod.StubEngine(tok, "qwen_think_on", mode)
    pod.run("qwen3_30b", [(a, pod.GAMES[0], 0) for a in pod.AGENTS], tmp_path, gen, tok, src,
            concurrency=2, max_decisions=6)
    rows = [json.loads(x) for x in gzip.open(tmp_path / "decisions.jsonl.gz", "rt")]
    traces = [json.loads(x) for x in gzip.open(tmp_path / "_raw" / "traces.jsonl.gz", "rt")]
    return rows, traces, gen


def test_cap_hit_allows_exactly_one_more_attempt_then_falls_back(tmp_path):
    # most probable failure: the cap rule not firing (five 4,096-token attempts) or firing at once
    rows, traces, _ = _dry(tmp_path, "always_cap")
    assert rows and all(r["attempts"] == 2 and r["fallback"] and r["cap_rule_cut"] for r in rows)
    assert all(all(r["capped"]) for r in rows)
    assert len(traces) == 2 * len(rows)


def test_every_attempt_is_saved_with_its_keys(tmp_path):
    rows, traces, _ = _dry(tmp_path, "mixed")
    assert len(traces) == sum(r["attempts"] for r in rows)
    keys = {"model", "agent", "game", "episode", "step", "attempt", "seed", "prompt_sha256",
            "trace", "answer", "capped", "parsed"}
    assert all(keys <= set(t) for t in traces)
    assert all(1 <= r["attempts"] <= pod.MAX_ATTEMPTS for r in rows)


def test_episodes_advance_without_waiting_for_each_other(tmp_path):
    # most probable failure: the loop reverting to lockstep rounds, where every episode waits for
    # the slowest generation (the first thinking pod's 1.1 s per decision)
    rows, _, gen = _dry(tmp_path, "mixed")
    assert gen.decoupled
    assert len({(r["agent"], r["episode"]) for r in rows}) == 2 and len(rows) == 12


def _pilot(d: Path, fallback_share: float, wall_s: float, spread: float):
    d.mkdir(parents=True)
    decs, eps = [], []
    for a in pod.AGENTS:
        for g in pod.PILOT_GAMES:
            for e in (0, 1):
                for step in range(100):
                    decs.append({"agent": a, "game": g, "episode": e, "step": step,
                                 "fallback": step < 100 * fallback_share, "cap_rule_cut": False,
                                 "capped": [False], "trace_chars": [500]})
                eps.append({"agent": a, "game": g, "episode": e, "status": "done",
                            "scores": {"violations.Σ": {"r0": 100 + (spread if e else -spread)}}})
    with gzip.open(d / "decisions.jsonl.gz", "wt") as f:
        f.write("\n".join(json.dumps(x) for x in decs) + "\n")
    (d / "episodes.jsonl").write_text("\n".join(json.dumps(x) for x in eps) + "\n")
    (d / "run_meta.jsonl").write_text(json.dumps({"wall_s": wall_s}) + "\n")


def _decide(tmp_path, fb=(0.0, 0.0), walls=(900, 1800), spread=5.0):
    for m, f, w in zip(pod.MODELS, fb, walls):
        _pilot(tmp_path / f"pilot_{m}", f, w, spread)
    return pod.decide(tmp_path)


def test_decide_applies_the_bar_to_every_model_and_agent(tmp_path):
    out = _decide(tmp_path, fb=(0.0, 0.2))
    assert out["continue_"] is False and out["reason"] == "fallback above the bar"


def test_decide_prices_both_models_and_falls_back_to_k2(tmp_path):
    # most probable failure: pricing only one model, or stopping when k = 2 would still fit
    cheap = _decide(tmp_path / "a")
    assert cheap["continue_"] is True and cheap["k"] >= 2
    per_dec = [v["rates"]["a100h_per_decision"] for v in cheap["models"].values()]
    assert cheap["price_a100h"] == pytest.approx(
        sum(r * 2 * cheap["k"] * 30 * 100 for r in per_dec))
    # high variance pushes k up; at that k the cell is over 8.0, at k = 2 it fits
    mid = _decide(tmp_path / "b", walls=(4400, 4400), spread=12.0)
    assert [v["k"] for v in mid["models"].values()] == [3, 3]  # the variance asks for k = 3
    assert mid["k"] == 2 and mid["continue_"] is True and mid["price_at_k2"] <= 8.0
    dear = _decide(tmp_path / "c", walls=(4 * 3600, 4 * 3600))
    assert dear["continue_"] is False and dear["reason"] == "price above the envelope"


def test_decide_reads_no_agent_means(tmp_path):
    flat = json.dumps(_decide(tmp_path), default=float).lower()
    assert not any(b in flat for b in ("points", "mean_violations", "h1", "h2", "h3"))


def test_stage0_check_fails_when_a_model_parses_nothing(tmp_path):
    for m, parsed in zip(pod.MODELS, (True, False)):
        d = tmp_path / f"stage0_{m}"
        d.mkdir()
        with gzip.open(d / "decisions.jsonl.gz", "wt") as f:
            f.write(json.dumps({"fallback": not parsed}) + "\n")
    assert pod.stage0_check(tmp_path)["pass"] is False
