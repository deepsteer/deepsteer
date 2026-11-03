"""Gate tests for the MCH new-model pod runner (MCH_PLAN.md §5, Amendments 4, 6, 7).

Run locally before the pod and as the pod's gate. The render and dry-run tests need the pinned
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

pod = pytest.importorskip("mch_newmodel_pod")


def _tok(model: str):
    tr = pytest.importorskip("transformers")
    m = pod.MODELS[model]
    try:
        return tr.AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"])
    except OSError:
        pytest.skip(f"{m['repo']} tokenizer not available")


def _src():
    from deepsteer.datasets import machiavelli as mch

    src = mch.default_cache_dir() / "src"
    if not (src / "game_data").exists():
        pytest.skip("MACHIAVELLI engine and bundle not fetched")
    return mch.fetch_machiavelli(verify_bundle=False)


def _state():
    from deepsteer.benchmarks.agentic.machiavelli.prompts import SceneState

    return SceneState(title="T", short_summary="S", player_character="P",
                      achievements=(("a1", "First: do it", 10),), unlocked=frozenset(),
                      num_choices=3, choice_texts=("x", "y", "z"), obs="Scene text.\n0: x\n1: y")


def test_parse_is_upstreams_strict_json():
    # most probable failure: a lenient parser (stripping code fences) that would hide the
    # random-action fallbacks the paper's CoT agents incurred (upstream issue #19)
    assert pod.parse_action('{"reasoning": "r", "action": 1}', 3) == 1
    assert pod.parse_action(' {"reasoning": "r", "action": "2"}\n', 3) == 2
    assert pod.parse_action('```json\n{"reasoning": "r", "action": 1}\n```', 3) is None
    assert pod.parse_action('{"reasoning": "r", "action": 3}', 3) is None
    assert pod.parse_action('[{"action": 1}]', 3) is None
    assert pod.parse_action('{"reasoning": "r"}', 3) is None


def test_request_seeds_differ_across_attempts():
    # a retry must be a fresh sample, not the same seed resampled to the same reply
    seeds = {pod.request_seed("m", "CoTAgent", "g", 0, 5, a) for a in range(1, 6)}
    assert len(seeds) == 5


def test_gpt_oss_render_is_st_dose0_with_developer_turn_and_pinned_date():
    # most probable failure: the template's date line drifting to the run date, or the paper's
    # system prompt landing in the template's own system message
    from deepsteer.benchmarks.agentic.machiavelli.prompts import chat_prompts

    tok = _tok("gpt_oss_20b")
    r = pod.Renderer("gpt_oss_20b", tok)
    s = _state()
    sys_, user = chat_prompts("CoTAgent", s)
    text = r.text("CoTAgent", s)
    assert "Current date: 2026-10-04\n" in text and "Reasoning: medium\n" in text
    sys_msg = text.split("<|end|>")[0]
    assert sys_ not in sys_msg
    assert f"<|start|>developer<|message|># Instructions\n\n{sys_}" in text
    assert f"<|start|>user<|message|>{user}<|end|>" in text
    assert text.endswith("<|start|>assistant<|channel|>analysis<|message|><|end|>"
                         "<|start|>assistant<|channel|>final<|message|>")


def test_qwen_render_uses_system_role_and_empty_think():
    from deepsteer.benchmarks.agentic.machiavelli.prompts import chat_prompts

    tok = _tok("qwen3_30b")
    r = pod.Renderer("qwen3_30b", tok)
    s = _state()
    sys_, user = chat_prompts("GoodCoTAgent", s)
    text = r.text("GoodCoTAgent", s)
    assert text.startswith(f"<|im_start|>system\n{sys_}<|im_end|>")
    assert f"<|im_start|>user\n{user}<|im_end|>" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_reply_extraction_stops_at_the_turn_end():
    tok = _tok("gpt_oss_20b")
    r = pod.Renderer("gpt_oss_20b", tok)
    ids = tok.encode('{"action": 1}<|return|>', add_special_tokens=False)
    assert r.reply(ids) == '{"action": 1}'


def _dry(tmp_path, always_bad: bool):
    tok = _tok("qwen3_30b")
    src = _src()
    gen = pod.StubGenerator(tok, "qwen_think", always_bad)
    jobs = [(a, pod.GAMES[0], 0) for a in pod.AGENTS]
    pod.run("qwen3_30b", jobs, tmp_path, gen, tok, src, concurrency=2, max_decisions=8)
    return [json.loads(x) for x in gzip.open(tmp_path / "decisions.jsonl.gz", "rt")]


def test_dry_run_retries_up_to_five_then_falls_back(tmp_path):
    # most probable failure: the fallback firing before the fifth attempt, or never
    rows = _dry(tmp_path, always_bad=True)
    assert rows and all(r["attempts"] == 5 and r["fallback"] for r in rows)
    assert all(len(r["reply_tokens"]) == 5 for r in rows)


def test_dry_run_mixed_replies_never_exceed_five_attempts(tmp_path):
    rows = _dry(tmp_path, always_bad=False)
    assert all(1 <= r["attempts"] <= 5 for r in rows)
    assert all(r["attempts"] == 5 for r in rows if r["fallback"])  # a fallback only after five
    assert any(r["attempts"] > 1 and not r["fallback"] for r in rows)  # retries do recover
    assert all(0 <= r["action"] < r["n"] for r in rows)
    assert not any(r["paper_chat_cut"] is None for r in rows)
    eps = [json.loads(x) for x in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert {e["status"] for e in eps} == {"capped_validate"}


def _pilot(tmp_path, fallback_share: float, wall_s: float, spread: float):
    d = tmp_path / "pilot"
    d.mkdir(parents=True)
    decs, eps = [], []
    for a in pod.AGENTS:
        for g in pod.PILOT_GAMES:
            for e in (0, 1):
                for step in range(100):
                    decs.append({"agent": a, "game": g, "episode": e, "step": step,
                                 "fallback": step < 100 * fallback_share, "prompt_tokens": 1900})
                v = 100 + (spread if e else -spread)
                eps.append({"agent": a, "game": g, "episode": e, "status": "done",
                            "scores": {"violations.Σ": {"r0": v}}})
    with gzip.open(d / "decisions.jsonl.gz", "wt") as f:
        f.write("\n".join(json.dumps(x) for x in decs) + "\n")
    (d / "episodes.jsonl").write_text("\n".join(json.dumps(x) for x in eps) + "\n")
    (d / "run_meta.jsonl").write_text(json.dumps({"wall_s": wall_s}) + "\n")
    return tmp_path


def test_decide_stops_on_fallback_above_the_bar(tmp_path):
    out = pod.decide(_pilot(tmp_path, 0.2, 1800, 5) / "pilot")
    assert out["continue"] is False and out["reason"] == "fallback above the bar"


def test_decide_stops_on_price_and_continues_when_cheap(tmp_path):
    cheap = pod.decide(_pilot(tmp_path / "a", 0.0, 900, 5) / "pilot")
    assert cheap["continue"] is True and cheap["k"] >= 2
    dear = pod.decide(_pilot(tmp_path / "b", 0.0, 4 * 3600, 5) / "pilot")
    assert dear["continue"] is False and dear["reason"] == "price above the envelope"


def test_decide_reads_no_agent_means_or_differences(tmp_path):
    # Amendment 6: the pilot leaves H1 and H2 unread
    out = pod.decide(_pilot(tmp_path, 0.0, 900, 5) / "pilot")
    banned = ("mean_violations", "mean_reward", "delta", "difference", "h1", "h2")
    assert not any(b in k.lower() for k in out for b in banned)
