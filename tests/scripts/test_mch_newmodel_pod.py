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
    assert {e["status"] for e in eps} == {"capped"}


def _run_dir(d: Path, fallback_share: float, wall_s: float, spread: float, cap: int,
             cap_share: float = 0.0, games=None, steps: int = 100):
    """A synthetic pilot or leg directory: decisions, episodes and run_meta."""
    d.mkdir(parents=True)
    decs, eps = [], []
    for a in pod.AGENTS:
        for g in games or pod.PILOT_GAMES:
            for e in (0, 1):
                for step in range(steps):
                    att = [cap if step < steps * cap_share else 100]
                    decs.append({"agent": a, "game": g, "episode": e, "step": step,
                                 "fallback": step < steps * fallback_share, "prompt_tokens": 1900,
                                 "attempts": len(att), "reply_tokens": att})
                v = 100 + (spread if e else -spread)
                eps.append({"agent": a, "game": g, "episode": e, "status": "done",
                            "scores": {"violations.Σ": {"r0": v}}})
    with gzip.open(d / "decisions.jsonl.gz", "wt") as f:
        f.write("\n".join(json.dumps(x) for x in decs) + "\n")
    (d / "episodes.jsonl").write_text("\n".join(json.dumps(x) for x in eps) + "\n")
    (d / "run_meta.jsonl").write_text(json.dumps({"wall_s": wall_s, "max_tokens": cap}) + "\n")
    return d


def _decide(tmp_path, fb=0.0, pilot_wall=900, leg_wall=120, spent=1.5):
    pilot = _run_dir(tmp_path / "pilot_1024", fb, pilot_wall, 5, 1024)
    leg = _run_dir(tmp_path / "qwen_leg", 0.0, leg_wall, 5, 1024, steps=30)
    return pod.decide(pilot, leg, spent=spent)


def test_decide_stops_on_fallback_above_the_bar(tmp_path):
    out = _decide(tmp_path, fb=0.2)
    assert out["continue"] is False and out["reason"] == "fallback above the bar"


def test_decide_stops_on_price_and_continues_when_cheap(tmp_path):
    cheap = _decide(tmp_path / "a")
    assert cheap["continue"] is True and cheap["k"] >= 2
    dear = _decide(tmp_path / "b", pilot_wall=4 * 3600)
    assert dear["continue"] is False and dear["reason"] == "price above the envelope"


def test_decide_counts_spend_and_prices_qwen_at_the_legs_own_rate(tmp_path):
    # most probable failure: the total leaving out the 1.5 A100-h already spent, or Qwen priced
    # at GPT-OSS's rate (or the retired 2.84 multiplier) instead of the leg's measured rate
    out = _decide(tmp_path, pilot_wall=900, leg_wall=240)
    p = out["price_a100h"]
    leg_rate = (240 / 3600) / (2 * 10 * 2 * 30)
    n_cell = 2 * out["k"] * 30 * 100
    assert p["qwen_cell"] == pytest.approx(leg_rate * n_cell)
    assert p["total"] == pytest.approx(1.5 + p["pilot"] + p["qwen_leg"]
                                       + p["rest_of_gpt_oss_cell"] + p["qwen_cell"])
    assert out["envelope_a100h"] == 7.0


def test_cap_hit_rate_uses_the_runs_own_budget(tmp_path):
    # a 256-token record must be read at 256, a 1,024 run at 1,024
    d = _run_dir(tmp_path / "r", 0.0, 60, 5, 256, cap_share=0.6)
    r = pod._rates(d)
    assert r["max_tokens"] == 256 and r["CoTAgent"]["cap_hit_rate"] == pytest.approx(0.6)


def test_decide_reads_no_agent_means_or_differences(tmp_path):
    # Amendment 6: the pilot leaves H1 and H2 unread
    out = _decide(tmp_path)
    flat = json.dumps(out).lower()
    assert not any(b in flat for b in ("mean_violations", "mean_reward", "points", "h1", "h2"))
    assert out["phase2_descriptive_only"]["note"].startswith("Phase 2 stays parked")


def test_instruct_swap_target_renders_without_a_think_block():
    from deepsteer.benchmarks.agentic.machiavelli.prompts import chat_prompts

    tok = _tok("qwen3_30b_instruct")
    r = pod.Renderer("qwen3_30b_instruct", tok)
    s = _state()
    sys_, _ = chat_prompts("CoTAgent", s)
    text = r.text("CoTAgent", s)
    assert text.startswith(f"<|im_start|>system\n{sys_}<|im_end|>")
    assert text.endswith("<|im_start|>assistant\n") and "<think>" not in text


def _validate_rows(tmp_path, cot_fallbacks: int):
    d = tmp_path / "validate_qwen3_30b"
    d.mkdir(parents=True)
    rows = [{"agent": a, "fallback": a == "CoTAgent" and i < cot_fallbacks}
            for a in pod.AGENTS for i in range(30)]
    with gzip.open(d / "decisions.jsonl.gz", "wt") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def test_validate_decision_swaps_only_above_ten_percent(tmp_path):
    # most probable failure: swapping at the bar (3 of 30 is 10%, not above it) or on the mean
    # over both agents instead of either agent
    keep = pod.validate_decision(_validate_rows(tmp_path / "a", 3))
    assert keep["swap"] is False and keep["qwen_model"] == "qwen3_30b"
    swap = pod.validate_decision(_validate_rows(tmp_path / "b", 4))
    assert swap["swap"] is True and swap["qwen_model"] == "qwen3_30b_instruct"
    assert json.loads((tmp_path / "b" / "validate_decision.json").read_text())["swap"] is True
