"""Reasoning-model formats for the ST screen (ST_PLAN Amendments 2 and 4; T0.11).

One ``ReasoningFormat`` per screen model: how the KDG letter frame is rendered at an effort level,
how the dose-0 prompt is built (thinking suppressed), and how a generated trace is parsed into the
action letter and the decision position. The action of record is the letter generated at the start
of the answer, as in KDG_RESULTS §40; the forward readout reads the letter distribution at that
same position, so it is defined only on rows with token identity (the answer's first token is a
bare option letter).

Token-level facts, from the pinned tokenizers (registry ``st:`` entries, 2026-10-18):

- OLMo-3-7B-Think: the template hard-codes ``<think>`` after ``<|im_start|>assistant\\n``;
  neither ``<think>`` nor ``</think>`` is a special token, and ``>`` merges with what follows
  (``>\\n\\n`` is one token), so the trace end is found on the decoded text. The dose-0 prefill
  ``</think>\\n\\n`` is encoded on its own and appended as ids: re-tokenizing the rendered prompt
  with it appended merges the prompt's final ``>`` with ``</`` and changes the prompt's tokens.
- Qwen3.8-27B: ``</think>`` is special token 248069; ``enable_thinking=False`` renders
  ``<think>\\n\\n</think>\\n\\n`` after the assistant header; the effort level is a template kwarg.
- Muse Glimmer 30B: no think tokens; reasoning is the message
  ``<|start|>assistant to=self<|message|>...<|eom|>``, and the answer follows the header
  ``<|start|>assistant to=user<|message|>``; the generation prompt ends at ``<|start|>assistant``;
  effort is ``reasoning_strength``, and the date is pinned through ``current_date``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

PARSE_VERSION = 2  # bump on any change to what counts as an action (2: Amendment 5 columns)
LETTERS = "ABCDE"
# a leading option letter closed by end of text, newline, ")", ".", ":" or "**" (never a space, so
# the article in "A careful approach" is not read as option A)
_LEADING_LETTER = re.compile(r"^\s*(?:\*\*|\()?\s*([A-E])\s*(?:\*\*|\)|\.|:|\n|$)")


@dataclass(frozen=True)
class Dose0:
    """One dose-0 candidate: render kwargs (on top of the format's own) and a prefill appended as
    separately encoded ids after the rendered prompt."""

    name: str
    prefill: str = ""
    render_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Parsed:
    """A generated trace, parsed. ``status``: completed | truncated | no_trace | other.
    ``answer_idx`` indexes the generated ids at the answer's first token (the decision position of
    the post-reasoning readout; ``token_identity``: the token there is a bare option letter, which
    makes the row eligible for the readout).

    Two action columns (Amendment 5). ``letter_record`` is the parser of record, KDG_RESULTS §40's
    rule (harness parity): the answer's first token is a bare option letter, else no action.
    ``letter_strict`` is the second column: a leading option letter that ends the answer
    (``parse_mode`` "token" when it is the identity token, "leading_text" when it follows markup,
    "prose" for an identity token that runs on into words, which is no strict action)."""

    status: str
    token_identity: bool
    trace_len: int
    answer_idx: int | None
    letter_strict: str | None
    parse_mode: str | None
    letter_record: str | None = None


@dataclass(frozen=True)
class ReasoningFormat:
    """Rendering, dose-0 construction and trace parsing for one reasoning model."""

    key: str
    effort_kwarg: str | None
    effort_ladder: tuple[str | None, ...]  # shipped default first, then the step-down order
    dose0: tuple[Dose0, ...]
    date_kwarg: str | None = None

    # ---- rendering ----------------------------------------------------------------------------
    def render(self, tok, messages: list[dict[str, str]], effort: str | None,
               date_pin: str | None = None, **extra: Any) -> str:
        kw: dict[str, Any] = dict(extra)
        if self.effort_kwarg is not None and effort is not None:
            kw[self.effort_kwarg] = effort
        if self.date_kwarg is not None:
            if date_pin is None:
                raise ValueError(f"{self.key}: the template renders a date; pin it (date_pin)")
            kw[self.date_kwarg] = date_pin
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)

    def prompt_ids(self, tok, rendered: str) -> list[int]:
        """The rendered prompt's ids exactly as the batched sampler encodes it."""
        return tok.encode(rendered, add_special_tokens=False)

    def dose0_ids(self, tok, messages: list[dict[str, str]], effort: str | None, cand: Dose0,
                  date_pin: str | None = None) -> list[int]:
        """Prompt ids for a dose-0 candidate; its last id is the decision token."""
        rendered = self.render(tok, messages, effort, date_pin, **cand.render_kwargs)
        ids = self.prompt_ids(tok, rendered)
        if cand.prefill:
            ids = ids + tok.encode(cand.prefill, add_special_tokens=False)
        return ids

    # ---- parsing ------------------------------------------------------------------------------
    skip_blank: bool = True  # skip whitespace-only tokens between the reasoning end and the answer

    def parse(self, tok, gen: list[int], hit_cap: bool, letter_ids: dict[int, str]) -> Parsed:
        """Parse one generated trace. ``letter_ids`` maps each bare option-letter token id to its
        letter."""
        g = list(gen)
        loc = self._locate(tok, g)
        if loc is None:
            return Parsed("truncated" if hit_cap else "other", False, len(g), None, None, None)
        if isinstance(loc, str):  # reasoning closed, answer header not canonical
            return Parsed(loc, False, len(g), None, None, None)
        trace_len, j = loc
        if self.skip_blank:
            while j < len(g) and self._is_blank(tok, g[j]):
                j += 1
        status = "completed" if trace_len else "no_trace"
        if j >= len(g):
            return Parsed(status, False, trace_len, None, None, None)
        # the action needs a letter that ends the answer ("A", "A.", "**A**", "(A)"); a letter
        # token that runs on into prose ("A careful approach") is no action, never option A
        identity = g[j] in letter_ids
        record = letter_ids[g[j]] if identity else None
        m = _LEADING_LETTER.match(tok.decode(g[j:], skip_special_tokens=True))
        if m is None:
            return Parsed(status, identity, trace_len, j, None, "prose" if identity else None,
                          record)
        return Parsed(status, identity, trace_len, j, m.group(1),
                      "token" if identity else "leading_text", record)

    def _locate(self, tok, g: list[int]) -> tuple[int, int] | str | None:
        """(trace length in tokens, index just past the end-of-reasoning marker); None when the
        reasoning never closes; a status string when it closes into a non-canonical answer."""
        raise NotImplementedError

    @staticmethod
    def _is_blank(tok, i: int) -> bool:
        return tok.decode([i]).strip() == ""


@dataclass(frozen=True)
class ThinkTextFormat(ReasoningFormat):
    """``</think>`` found on decoded text (not a special token; it may span or share tokens); the
    generated ids start inside the think block because the template opens it. The trace is the
    tokens wholly before the marker's first character; it is empty when they decode to blank."""

    close: str = "</think>"

    def _locate(self, tok, g):
        text, ends = "", []
        for i in g:
            text += tok.decode([i])
            ends.append(len(text))
            at = text.find(self.close)
            if at >= 0:
                n_before = sum(1 for e in ends if e <= at)
                body = text[:at].strip()
                return (n_before if body else 0), len(ends)
        return None


@dataclass(frozen=True)
class ThinkTokenFormat(ReasoningFormat):
    """``</think>`` is a single special token; the template opens the think block."""

    close_id: int = -1

    def _locate(self, tok, g):
        if self.close_id not in g:
            return None
        k = g.index(self.close_id)
        return (k if tok.decode(g[:k]).strip() else 0), k + 1


@dataclass(frozen=True)
class ChannelFormat(ReasoningFormat):
    """Muse: the generated ids start after ``<|start|>assistant``. Either a reasoning message
    `` to=self<|message|>...<|eom|>`` followed by the canonical answer header ``<|start|>assistant
    to=user<|message|>``, or a direct answer opening `` to=user<|message|>``. The answer's first
    token follows the header immediately (no blank skipping)."""

    skip_blank: bool = False
    reason_open: tuple[int, ...] = ()
    reason_end: int = -1
    answer_header: tuple[int, ...] = ()
    direct_open: tuple[int, ...] = ()

    def _locate(self, tok, g):
        n = len(self.direct_open)
        if tuple(g[:n]) == self.direct_open:
            return 0, n
        m = len(self.reason_open)
        if tuple(g[:m]) != self.reason_open:
            return "other"
        if self.reason_end not in g[m:]:
            return None
        e = g.index(self.reason_end, m)
        h = len(self.answer_header)
        if tuple(g[e + 1 : e + 1 + h]) != self.answer_header:
            return "other"
        return (e - m if tok.decode(g[m:e]).strip() else 0), e + 1 + h


def _muse_ids(tok, text: str) -> tuple[int, ...]:
    return tuple(tok.encode(text, add_special_tokens=False))


def build_format(key: str, tok) -> ReasoningFormat:
    """The format of record for a ``st:`` registry key, built against its pinned tokenizer
    (token ids are read from the tokenizer, never hard-coded)."""
    if key == "olmo3_think":
        return ThinkTextFormat(
            key=key, effort_kwarg=None, effort_ladder=(None,),
            dose0=(Dose0("close_think", prefill="</think>\n\n"),),
        )
    if key == "qwen38_27b":
        (close_id,) = tok.encode("</think>", add_special_tokens=False)
        return ThinkTokenFormat(
            key=key, effort_kwarg="reasoning_effort", effort_ladder=("xhigh", "medium", "low"),
            dose0=(Dose0("enable_thinking_false", render_kwargs={"enable_thinking": False}),),
            close_id=close_id,
        )
    if key == "muse_glimmer_30b":
        (reason_end,) = tok.encode("<|eom|>", add_special_tokens=False)
        return ChannelFormat(
            key=key, effort_kwarg="reasoning_strength", effort_ladder=("high", "medium", "low"),
            dose0=(
                Dose0("empty_reasoning",
                      prefill=" to=self<|message|><|eom|><|start|>assistant to=user<|message|>"),
                Dose0("direct_answer", prefill=" to=user<|message|>"),
            ),
            date_kwarg="current_date",
            reason_open=_muse_ids(tok, " to=self<|message|>"),
            reason_end=reason_end,
            answer_header=_muse_ids(tok, "<|start|>assistant to=user<|message|>"),
            direct_open=_muse_ids(tok, " to=user<|message|>"),
        )
    raise KeyError(f"no reasoning format for {key!r}")


def letter_token_ids(tok) -> dict[int, str]:
    """Bare option-letter token ids; each letter must be a single token."""
    out: dict[int, str] = {}
    for L in LETTERS:
        ids = tok.encode(L, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"option letter {L!r} is not a single token: {ids}")
        out[ids[0]] = L
    return out
