#!/usr/bin/env python3
"""t_inst vs t_post-inst token positions (Zhao et al. 2025, arXiv:2507.11878).

Zhao et al. show harmfulness is encoded at **t_inst** (last token of the user
INSTRUCTION) and refusal at **t_post-inst** (last token of the whole templated
prompt). Our existing extractor only reads the last templated token (t_post-inst,
the weak causal handle); this module locates BOTH positions so Phase 2c+ can
extract diff-of-means at the harmfulness site too.

Finding t_inst robustly, without per-model hardcoded token counts (Zhao Table 4):
a chat template appends a FIXED post-instruction suffix after the user content
(turn-end + assistant header + any reasoning opener), identical regardless of the
instruction. So the suffix = the longest common token SUFFIX across a few diverse
probe instructions; ``t_inst = len(ids) - len(suffix) - 1`` and
``t_post_inst = len(ids) - 1``. This is model-agnostic (works for the DeepSeek-R1
``<｜Assistant｜><think>`` opener and the GPT-OSS harmony channel alike) and avoids
the offset-mapping / re-tokenization-boundary fragility.
"""

from __future__ import annotations

# Diverse probes that end differently, so their common token suffix is exactly the
# template's fixed post-instruction tokens (not accidental shared instruction tail).
_PROBES = (
    "Tell me about the history of cats.",
    "Explain how photosynthesis works in plants.",
    "What is the capital of France?",
)


def _apply_ids(tokenizer, instruction: str) -> list[int]:
    """Templated input_ids with the assistant turn opened (the model's actual input).

    ``apply_chat_template(tokenize=True)`` returns a plain list on some transformers
    versions and a ``BatchEncoding`` on others; normalize to a flat ``list[int]``.
    """
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=True, add_generation_prompt=True,
    )
    if hasattr(out, "input_ids"):          # BatchEncoding
        out = out["input_ids"]
    if len(out) and isinstance(out[0], (list, tuple)):  # batched [[...]]
        out = out[0]
    return [int(t) for t in out]


def _common_suffix_len(a: list[int], b: list[int]) -> int:
    n = 0
    while n < len(a) and n < len(b) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def post_instruction_token_count(tokenizer, probes: tuple[str, ...] = _PROBES) -> int:
    """Number of FIXED template tokens after the user instruction content.

    The longest common token suffix across diverse probes; if it accidentally eats
    a shared trailing token it would only shift t_inst by 1 (harmless for a
    diff-of-means over many prompts), but the probes are chosen to end differently.
    """
    id_lists = [_apply_ids(tokenizer, p) for p in probes]
    n = min(_common_suffix_len(id_lists[0], id_lists[i]) for i in range(1, len(id_lists)))
    return n


def instruction_positions(tokenizer, instruction: str, post_count: int | None = None) -> dict:
    """``{t_inst, t_post_inst, n_tokens, ids, post_count}`` for one instruction.

    ``t_inst`` = index of the last instruction-content token; ``t_post_inst`` =
    index of the last templated token (= -1). ``post_count`` may be precomputed
    once per tokenizer (it is instruction-independent) and reused.
    """
    if post_count is None:
        post_count = post_instruction_token_count(tokenizer)
    ids = _apply_ids(tokenizer, instruction)
    n = len(ids)
    t_inst = max(0, n - post_count - 1)
    return {"t_inst": t_inst, "t_post_inst": n - 1, "n_tokens": n,
            "post_count": post_count, "ids": ids}


def describe(tokenizer, instruction: str, post_count: int | None = None) -> dict:
    """Human-readable check: the decoded instruction-side vs post-instruction-side
    text, so a wrong split is obvious at the gate."""
    pos = instruction_positions(tokenizer, instruction, post_count)
    ids = pos["ids"]
    inst_tok = tokenizer.decode([ids[pos["t_inst"]]])
    suffix = tokenizer.decode(ids[pos["t_inst"] + 1:])
    return {**{k: pos[k] for k in ("t_inst", "t_post_inst", "n_tokens", "post_count")},
            "t_inst_token": inst_tok, "post_instruction_suffix": suffix,
            "head": tokenizer.decode(ids[: pos["t_inst"] + 1])[-60:]}
