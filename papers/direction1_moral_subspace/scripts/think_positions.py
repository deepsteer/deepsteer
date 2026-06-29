#!/usr/bin/env python3
"""Locate the four pre-registered refusal positions on an OLMo-3-Think generation.

Pre-registered in PREREGISTRATION.md (2026-06-29 amendment), grounded in the Zhao et al.
keystone (`papers/7_reasoning/scripts/token_positions.py`):

  P0  t_inst         -- last instruction-content token (harmfulness / comprehension site)
  P1  pre-trace gate -- the `<think>` opener (last templated token; Base/Instruct A/B analog)
  P2  in-trace       -- last reasoning token before `</think>` (the coupling detector)
  P3  post-answer    -- last meaningful answer token after `</think>` (the decision site)

P0/P1 are prompt-side and, by causal masking, independent of what is generated afterwards, so
all four positions are read from a SINGLE full-sequence forward (one generate + one forward).
`</think>` is NOT a single token on this tokenizer: it is the 3-token subsequence
`[524, 27963, 29]` (`</`, `think`, `>`) -- verified. A generation that never closes `</think>`
yields P2 = P3 = None (excluded from those positions, counted; never silently dropped).

GPU module: imports torch lazily so the file is importable for unit-style checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "7_reasoning" / "scripts"))
from token_positions import post_instruction_token_count  # noqa: E402

THINK_CLOSE = [524, 27963, 29]  # </think> on allenai/Olmo-3-7B-Think (verified, see prereg)
# Terminators we do not want P3 to land on (measure the last MEANINGFUL answer token).
_TERMINATORS = {100265, 100257}  # <|im_end|>, <|endoftext|>-class; eos added at runtime


def _find_subseq(ids: list[int], sub: list[int], start: int = 0) -> int:
    m = len(sub)
    for i in range(start, len(ids) - m + 1):
        if ids[i:i + m] == sub:
            return i
    return -1


def _last_meaningful(ids: list[int], after: int, terminators: set[int]) -> int | None:
    """Last index > `after` whose token is not a terminator (the answer decision token)."""
    for i in range(len(ids) - 1, after, -1):
        if ids[i] not in terminators:
            return i
    return None


def _forward_capture(model, full_ids: list[int], layer: int):
    """Layer-`layer` residual activations for the full sequence: returns (seq, hidden)."""
    import torch
    cap: dict[str, object] = {}

    def hook(_m, _i, o):
        cap["h"] = (o[0] if isinstance(o, tuple) else o).detach()[0]

    h = model._get_layer_module(layer).register_forward_hook(hook)
    try:
        t = torch.tensor([full_ids], device=model._device)
        with torch.no_grad():
            model.model(input_ids=t, attention_mask=torch.ones_like(t))
    finally:
        h.remove()
    return cap["h"]


def generate_and_position(model, prompt: str, layer: int, max_new_tokens: int,
                          post_count: int) -> dict:
    """One prompt -> {p0,p1,p2,p3: np.ndarray|None, closed, n, plen}.

    Greedy decode (deterministic, reproducible). One generate + one full-sequence forward.
    """
    import torch
    tok = model.tokenizer
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model._device)
    plen = int(enc["input_ids"].shape[1])
    with torch.no_grad():
        gen = model.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                   num_beams=1, pad_token_id=tok.eos_token_id)
    full_ids = gen[0].tolist()
    acts = _forward_capture(model, full_ids, layer)

    terminators = set(_TERMINATORS)
    if tok.eos_token_id is not None:
        terminators.add(int(tok.eos_token_id))

    p0 = max(0, plen - post_count - 1)        # t_inst
    p1 = plen - 1                             # pre-trace gate (<think> opener, last templated)
    close = _find_subseq(full_ids, THINK_CLOSE, start=plen)  # </think> in the GENERATED region
    if close < 0:
        idx2 = idx3 = None
    else:
        idx2 = close - 1                                  # last reasoning token before </think>
        idx3 = _last_meaningful(full_ids, close + len(THINK_CLOSE) - 1, terminators)

    def take(i):
        return acts[i].float().cpu().numpy() if i is not None else None

    return {"p0": take(p0), "p1": take(p1), "p2": take(idx2), "p3": take(idx3),
            "closed": close >= 0, "has_answer": idx3 is not None,
            "n": len(full_ids), "plen": plen}


def refusal_directions(model, harmful: list[str], harmless: list[str], layer: int,
                       max_new_tokens: int) -> dict:
    """Harmful-minus-harmless diff-of-means refusal direction at each of P0..P3.

    Returns unit refusal vectors keyed 'p0'..'p3' (None if a position has no clean pairing),
    plus per-position counts + exclusion stats. P0/P1 use all prompts; P2/P3 use only
    generations that closed `</think>` and produced an answer (excluded counts reported).
    """
    post_count = post_instruction_token_count(model.tokenizer)

    def collect(prompts):
        rows = [generate_and_position(model, p, layer, max_new_tokens, post_count)
                for p in prompts]
        return rows

    h_rows = collect(harmful)
    s_rows = collect(harmless)

    out: dict[str, object] = {"layer": layer, "post_count": post_count,
                              "max_new_tokens": max_new_tokens, "positions": {}}
    for pos in ("p0", "p1", "p2", "p3"):
        hv = np.array([r[pos] for r in h_rows if r[pos] is not None], dtype=np.float64)
        sv = np.array([r[pos] for r in s_rows if r[pos] is not None], dtype=np.float64)
        if len(hv) == 0 or len(sv) == 0:
            out["positions"][pos] = {"vec": None, "n_harmful": len(hv), "n_harmless": len(sv)}
            continue
        r = hv.mean(0) - sv.mean(0)
        out["positions"][pos] = {"vec": r / (np.linalg.norm(r) + 1e-12),
                                 "n_harmful": int(len(hv)), "n_harmless": int(len(sv))}
    out["closed_rate_harmful"] = float(np.mean([r["closed"] for r in h_rows]))
    out["closed_rate_harmless"] = float(np.mean([r["closed"] for r in s_rows]))
    out["answer_rate_harmful"] = float(np.mean([r["has_answer"] for r in h_rows]))
    out["answer_rate_harmless"] = float(np.mean([r["has_answer"] for r in s_rows]))
    return out
