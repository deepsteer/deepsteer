"""T0.11 format layer (reasoning_formats.py) against the pinned tokenizers of the screen models.

Loads from the local HF cache when present (download once; the pod downloads them anyway)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "papers" / "st_live_reference" / "scripts"))
import reasoning_formats as R  # noqa: E402

REG = yaml.safe_load((ROOT / "papers/kdg_panel/models.yaml").read_text())["st"]
DATE = "2026-10-18"
MSG = [{"role": "user", "content": "Pick one.\n\nA. rule\nB. protect\nC. defer"}]


def _tok(key: str):
    from transformers import AutoTokenizer

    spec = REG[key]
    try:
        return AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    except OSError as e:  # no cache and no network
        pytest.skip(f"tokenizer for {key} unavailable: {e}")


@pytest.fixture(scope="module")
def olmo():
    tok = _tok("olmo3_think")
    return tok, R.build_format("olmo3_think", tok)


@pytest.fixture(scope="module")
def qwen():
    tok = _tok("qwen38_27b")
    return tok, R.build_format("qwen38_27b", tok)


@pytest.fixture(scope="module")
def muse():
    tok = _tok("muse_glimmer_30b")
    return tok, R.build_format("muse_glimmer_30b", tok)


def _letters(tok):
    return R.letter_token_ids(tok)


# ---- OLMo-3-7B-Think --------------------------------------------------------------------------


def test_olmo_dose0_keeps_prompt_tokens_as_prefix(olmo):
    # most probable failure: the prefill is re-tokenized jointly with the prompt, merging the
    # rendered "<think>"'s final ">" with "</" so the prompt's own tokens change
    tok, f = olmo
    rendered = f.render(tok, MSG, None)
    assert rendered.endswith("<|im_start|>assistant\n<think>")
    p = f.prompt_ids(tok, rendered)
    d = f.dose0_ids(tok, MSG, None, f.dose0[0])
    assert d[: len(p)] == p
    assert d[len(p):] == tok.encode("</think>\n\n", add_special_tokens=False)
    assert tok.encode(rendered + "</think>\n\n", add_special_tokens=False)[: len(p)] != p


def test_olmo_parse_marker_split_across_tokens(olmo):
    # "</" "think" ">\n\n" are three tokens; the trace excludes the marker's own tokens
    tok, f = olmo
    body = tok.encode("\nShe should follow the rule.\n", add_special_tokens=False)
    gen = body + tok.encode("</think>\n\n", add_special_tokens=False) + tok.encode(
        "B", add_special_tokens=False)
    p = f.parse(tok, gen, False, _letters(tok))
    assert (p.status, p.token_identity, p.letter_strict, p.parse_mode) == (
        "completed", True, "B", "token")
    assert p.letter_record == "B"
    assert p.trace_len == len(body) and gen[p.answer_idx] == tok.encode("B")[0]


def test_olmo_parse_empty_trace_truncation_and_text_fallbacks(olmo):
    tok, f = olmo
    L = _letters(tok)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    assert f.parse(tok, enc("</think>\n\nC"), False, L).status == "no_trace"
    assert f.parse(tok, enc("still thinking about it"), True, L).status == "truncated"
    assert f.parse(tok, enc("still thinking"), False, L).status == "other"
    bold = f.parse(tok, enc("x</think>\n\n**B**"), False, L)
    assert (bold.letter_strict, bold.parse_mode, bold.token_identity) == (
        "B", "leading_text", False)
    assert bold.letter_record is None  # §40's parser: the first token is not a bare letter
    # most probable misparse: the article "A" read as option A (its token is the letter token)
    art = f.parse(tok, enc("x</think>\n\nA careful approach"), False, L)
    assert (art.letter_strict, art.parse_mode, art.token_identity) == (None, "prose", True)
    assert art.letter_record == "A"  # the parser of record keeps §40's rule; the columns differ
    for s in ("A.", "A\n\nBecause", "(A)", "A"):
        assert f.parse(tok, enc("x</think>\n\n" + s), False, L).letter_strict == "A", s


# ---- Qwen3.8-27B ------------------------------------------------------------------------------


def test_qwen_effort_render_and_ladder(qwen):
    tok, f = qwen
    assert f.effort_ladder[0] == REG["qwen38_27b"]["effort_default"] == "xhigh"
    assert "Reasoning effort is set to xhigh" in f.render(tok, MSG, "xhigh")
    assert "Reasoning effort" not in f.render(tok, MSG, "medium")
    assert f.render(tok, MSG, "xhigh").endswith("<|im_start|>assistant\n<think>\n")
    with pytest.raises(Exception):
        f.render(tok, MSG, "high")  # the template has no "high"


def test_qwen_dose0_and_roundtrip(qwen):
    tok, f = qwen
    d = f.dose0_ids(tok, MSG, "xhigh", f.dose0[0])
    assert tok.decode(d).endswith("<think>\n\n</think>\n\n")
    # a completed turn as the template itself renders it parses to its letter by token identity
    full = tok.apply_chat_template(MSG + [{"role": "assistant", "content": "B",
                                           "reasoning_content": "weighing it"}], tokenize=False)
    gp = f.render(tok, MSG, "xhigh")  # the template applies its xhigh default to a completed turn
    assert full.startswith(gp)
    p = f.parse(tok, tok.encode(full[len(gp):], add_special_tokens=False), False, _letters(tok))
    assert (p.status, p.letter_record, p.letter_strict, p.token_identity) == (
        "completed", "B", "B", True)


# ---- Muse Glimmer 30B -------------------------------------------------------------------------


def test_muse_render_pins_date_and_strength(muse):
    tok, f = muse
    with pytest.raises(ValueError):
        f.render(tok, MSG, "high")
    r = f.render(tok, MSG, "high", DATE)
    assert f"Current date: {DATE}." in r and "Reasoning strength: high." in r
    assert r.endswith("<|start|>assistant")


def test_muse_dose0_candidates_in_order(muse):
    tok, f = muse
    assert [c.name for c in f.dose0] == ["empty_reasoning", "direct_answer"]
    empty = f.dose0_ids(tok, MSG, "high", f.dose0[0], DATE)
    direct = f.dose0_ids(tok, MSG, "high", f.dose0[1], DATE)
    tail = list(f.reason_open) + [f.reason_end] + list(f.answer_header)
    assert empty[-len(tail):] == tail
    assert direct[-len(f.direct_open):] == list(f.direct_open)


def test_muse_parse_channels(muse):
    tok, f = muse
    L = _letters(tok)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    reason = list(f.reason_open) + enc("weighing the rule") + [f.reason_end] + list(f.answer_header)
    p = f.parse(tok, reason + enc("B"), False, L)
    assert (p.status, p.letter_record, p.letter_strict, p.token_identity) == (
        "completed", "B", "B", True)
    assert f.parse(tok, list(f.direct_open) + enc("A"), False, L).status == "no_trace"
    assert f.parse(tok, list(f.reason_open) + enc("long"), True, L).status == "truncated"
    tool = list(f.reason_open) + enc("x") + [f.reason_end] + enc("<|start|>assistant to=search")
    assert f.parse(tok, tool, False, L).status == "other"
    # no blank skipping on Muse: a space before the letter breaks token identity
    sp = f.parse(tok, reason + enc(" C"), False, L)
    assert sp.token_identity is False


def test_muse_roundtrip_completed_turn(muse):
    tok, f = muse
    full = tok.apply_chat_template(MSG + [{"role": "assistant", "content": "C",
                                           "reasoning_content": "weighing it"}],
                                   tokenize=False, current_date=DATE)
    gp = f.render(tok, MSG, "high", DATE)
    assert full.startswith(gp)
    p = f.parse(tok, tok.encode(full[len(gp):], add_special_tokens=False), False, _letters(tok))
    assert (p.status, p.letter_record, p.token_identity) == ("completed", "C", True)


def test_letters_single_tokens_everywhere(olmo, qwen, muse):
    for tok, _ in (olmo, qwen, muse):
        assert sorted(R.letter_token_ids(tok).values()) == list("ABCDE")
