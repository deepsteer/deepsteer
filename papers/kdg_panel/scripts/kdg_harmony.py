"""Harmony-format pieces of the KDG harness for GPT-OSS-20B (KDG_GPTOSS_SPEC v0.2).

Rendering (spec §2): the pinned template is rendered with ``reasoning_effort`` passed explicitly,
the template's ``Current date:`` line is replaced by the registered pin (exactly once), and the
``Knowledge cutoff:`` line is left byte-identical (G-A5). The rendered prompt ends with
``<|start|>assistant``; its last token is the decision token (FL's END_OF_PROMPT site), and the
letter readout appends a prefill after it (primary: an empty closed analysis turn, then an opened
final channel; construct check: the final channel directly).

C0 (spec §3, G-A1, G-A2): final-channel letter parsing, the completed/truncated label, the strict
majority over completed rollouts (ties count as non-agreement), and the Wilson interval printed
beside the unchanged 0.80 rule.

Nothing here loads weights; the pod driver (``pod_kdg_gptoss.py``) and the analysis
(``analyze_gptoss.py``) import it, and the local tests exercise it on the pinned tokenizer.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Any

DATE_LINE = re.compile(r"^Current date: \d{4}-\d{2}-\d{2}$", re.M)
FINAL_OPEN = "<|channel|>final<|message|>"
# a final message ends at <|return|> (end of turn) or <|end|>; generation may also stop at <|call|>
FINAL_CLOSE = re.compile(r"<\|return\|>|<\|end\|>|<\|call\|>")
# letter-only reply: optional markdown / brackets, one option letter, then nothing but punctuation
_LETTER = re.compile(r"^\W*\(?\**\s*([A-E])\s*\**\)?[\s.:)\]*]*$")
_LEADING_LETTER = re.compile(r"^\W*\(?\**\s*([A-E])\b(?![a-z'])")


@dataclasses.dataclass(frozen=True)
class HarmonyConfig:
    """The ``harmony`` block of a models.yaml entry."""

    date_pin: str
    cutoff_line: str
    reasoning_level: str
    reasoning_level_c0_generate: str
    prefill_primary: str
    prefill_direct_final: str
    prefill_dose_matched: str
    c0_max_new_tokens: int
    c0_rollouts: int

    @classmethod
    def from_registry(cls, spec: dict[str, Any]) -> HarmonyConfig:
        h = spec["harmony"]
        return cls(**{f.name: h[f.name] for f in dataclasses.fields(cls)})

    def prefill(self, name: str) -> str:
        return {"primary": self.prefill_primary, "direct_final": self.prefill_direct_final,
                "dose_matched": self.prefill_dose_matched}[name]


def pin_date(rendered: str, cfg: HarmonyConfig) -> str:
    """Replace the template's ``Current date:`` line with the registered pin.

    Raises unless the date line occurs exactly once and the cutoff line occurs exactly once, before
    and after the replacement (the pin must never touch the cutoff line; G-A5)."""
    n_date = len(DATE_LINE.findall(rendered))
    if n_date != 1:
        raise RuntimeError(f"harmony render: expected one 'Current date:' line, found {n_date}")
    if rendered.count(cfg.cutoff_line) != 1:
        raise RuntimeError(f"harmony render: expected {cfg.cutoff_line!r} exactly once")
    out = DATE_LINE.sub(cfg.date_pin, rendered)
    if out.count(cfg.cutoff_line) != 1 or out.count(cfg.date_pin) != 1:
        raise RuntimeError("harmony date pin disturbed the system message")
    return out


def render(tok, messages: list[dict[str, str]], cfg: HarmonyConfig, level: str) -> str:
    """Rendered prompt ending at ``<|start|>assistant`` (the decision token), date-pinned."""
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, reasoning_effort=level
    )
    if f"Reasoning: {level}\n" not in text:
        raise RuntimeError(f"harmony render: 'Reasoning: {level}' missing from the system message")
    if not text.endswith("<|start|>assistant"):
        raise RuntimeError("harmony render must end at <|start|>assistant (the decision token)")
    return pin_date(text, cfg)


def split_prefill(tok, rendered: str, prefill: str) -> tuple[list[int], list[int]]:
    """Token ids of the rendered prompt and of the prefill, asserting that tokenizing the
    concatenation is the concatenation of the two (no merge across the boundary), so the decision
    token's index is ``len(prompt_ids) - 1`` and every prefill token lies after it."""
    a = tok.encode(rendered, add_special_tokens=False)
    b = tok.encode(prefill, add_special_tokens=False) if prefill else []
    ab = tok.encode(rendered + prefill, add_special_tokens=False)
    if ab != a + b:
        raise RuntimeError("prefill tokenization merges across the decision-token boundary")
    return a, b


# ---- C0: generated final answers ------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FinalAnswer:
    """One C0c rollout read off the harmony stream (decoded with special tokens kept)."""

    reasoning_trace: str  # completed | truncated
    final_text: str | None
    letter: str | None
    parse_method: str  # letter_only | leading_letter | unparsed | truncated


def parse_final(full_text: str, hit_budget: bool) -> FinalAnswer:
    """Read the final-channel letter. ``completed`` iff the final channel opens in the generated
    text (spec §2); if the budget is hit before it opens the rollout is ``truncated`` and no letter
    is read. A completed rollout whose final text has no clean letter stays in the C0 denominator
    as unparsed (G-A1: it can only lower agreement)."""
    i = full_text.find(FINAL_OPEN)
    if i < 0 and hit_budget:
        return FinalAnswer("truncated", None, None, "truncated")
    if i < 0:  # stopped inside the trace without a final message: completed, nothing to read
        return FinalAnswer("completed", None, None, "unparsed")
    body = full_text[i + len(FINAL_OPEN):]
    m = FINAL_CLOSE.search(body)
    body = (body[: m.start()] if m else body).strip()
    if (mm := _LETTER.match(body)) is not None:
        return FinalAnswer("completed", body, mm.group(1), "letter_only")
    if (mm := _LEADING_LETTER.match(body)) is not None:
        return FinalAnswer("completed", body, mm.group(1), "leading_letter")
    return FinalAnswer("completed", body, None, "unparsed")


_OTHER_CHANNEL = re.compile(r"<\|channel\|>(analysis|commentary)")


def parse_dose_matched(full_text: str, hit_budget: bool) -> tuple[FinalAnswer, bool]:
    """G-A11 C0-dm: the generated text after the empty closed analysis turn. Returns the parsed
    final answer and ``redeliberated``: True when the model opens an analysis or commentary channel
    before its final channel (not dose 0); such a rollout counts as completed but non-matching."""
    i = full_text.find(FINAL_OPEN)
    head = full_text if i < 0 else full_text[:i]
    if _OTHER_CHANNEL.search(head):
        return FinalAnswer("completed", None, None, "redeliberated"), True
    return parse_final(full_text, hit_budget), False


def strict_majority(option_ids: list[str | None]) -> str | None:
    """G-A1: the option held by more than half of the completed rollouts, else None.

    ``option_ids`` has one entry per completed rollout (None = completed but unparsed)."""
    if not option_ids:
        return None
    counts: dict[str, int] = {}
    for o in option_ids:
        if o is not None:
            counts[o] = counts.get(o, 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1], default=(None, 0))
    return best[0] if best[1] * 2 > len(option_ids) else None


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for k successes in n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def c0_verdict(agree: list[bool], bar: float = 0.80) -> dict[str, Any]:
    """C0 rule (spec §3, unchanged) with G-A2's Wilson CI, SE and near-miss label."""
    n, k = len(agree), int(sum(agree))
    p = k / n if n else math.nan
    se = math.sqrt(p * (1 - p) / n) if n else math.nan
    passed = bool(n and p >= bar)
    near = bool(n and abs(p - bar) <= se)
    return {
        "n_scenarios": n,
        "agreement": p,
        "wilson_ci95": list(wilson(k, n)),
        "se": se,
        "bar": bar,
        "pass": passed,
        "near_miss": near,
        "verdict": ("pass" if passed else "fail") + (" (near-miss)" if near else ""),
    }


def project_hours(unit_seconds: float, n_letter_units: float, c0_batch_seconds: float,
                  c0_batches: int, load_seconds: float) -> float:
    """G-A4 timing projection for the whole run, in hours."""
    return (load_seconds + unit_seconds * n_letter_units + c0_batch_seconds * c0_batches) / 3600.0


# ---- dose-stated readout (KDG_GPTOSS_SPEC G-A15 / G-A16) -------------------------------------

ANALYSIS_OPEN = (200005, 35644, 200008)  # <|channel|>analysis<|message|>
# <|end|><|start|>assistant<|channel|>final<|message|>
CANON_HEADER = (200007, 200006, 173781, 200005, 17196, 200008)
DIRECT_FINAL = (200005, 17196, 200008)  # <|channel|>final<|message|>, no analysis turn
END = 200007


@dataclasses.dataclass(frozen=True)
class DoseStated:
    """One dose-stated rollout, read off its generated token ids.

    ``status``: completed (the trace closed and a final channel followed) | truncated (no
    ``<|end|>`` before the budget) | no_trace (the model answered without an analysis turn) | other
    (anything else after the trace). ``token_identity``: the continuation after the trace is exactly
    the canonical final header followed by an option letter. ``header_end``: index just past
    ``<|message|>`` (the forward readout's input ends there). ``assistant_idx``: index of the
    header's ``assistant`` token (the post-reasoning decision token)."""

    status: str
    token_identity: bool
    trace_len: int
    header_end: int | None
    assistant_idx: int | None
    letter_id: int | None


def parse_dose_stated(gen: list[int], hit_budget: bool, letter_ids: set[int]) -> DoseStated:
    g = list(gen)
    if tuple(g[:3]) == DIRECT_FINAL:
        return DoseStated("no_trace", False, 0, None, None, None)
    if tuple(g[:3]) != ANALYSIS_OPEN:
        return DoseStated("other", False, 0, None, None, None)
    try:
        e = g.index(END, 3)
    except ValueError:
        return DoseStated("truncated" if hit_budget else "other", False, len(g) - 3, None, None,
                          None)
    trace_len = e - 3
    head = tuple(g[e : e + len(CANON_HEADER)])
    if head != CANON_HEADER:
        return DoseStated("other", False, trace_len, None, None, None)
    end = e + len(CANON_HEADER)
    letter = g[end] if end < len(g) else None
    ok = letter is not None and letter in letter_ids
    return DoseStated("completed", ok, trace_len, end, e + 2, letter if ok else None)
