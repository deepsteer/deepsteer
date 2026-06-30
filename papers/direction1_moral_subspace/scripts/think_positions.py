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
`</think>` has NO fixed token form: it is not a special token, and under BPE the leading `</`
merges with the preceding char (`.</`=4005, ` </`, `\n</`, standalone `</`=524, ...), so a fixed
subsequence misses almost every real close. The boundary is found on the STABLE suffix
`[27963, 29]` (`think`,`>`) with a per-occurrence `</`-prefix check (`_find_think_close`). A
generation that never closes `</think>` yields P2 = P3 = None (excluded, counted; never dropped).

GPU module: imports torch lazily so the file is importable for unit-style checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "7_reasoning" / "scripts"))
from token_positions import post_instruction_token_count  # noqa: E402

# `</think>` is NOT a fixed token sequence: the leading `</` merges with the preceding char
# under BPE (`.</`=4005, ` </`, standalone `</`=524, ...). Anchor on the stable suffix instead
# and validate the `</` prefix per-occurrence -- see _find_think_close. (`<think>`/`</think>`
# are not special tokens; convert_tokens_to_ids returns the unk id for both.)
# Terminators we do not want P3 to land on (measure the last MEANINGFUL answer token).
_TERMINATORS = {100265, 100257}  # <|im_end|>, <|endoftext|>-class; eos added at runtime


def _find_subseq(ids: list[int], sub: list[int], start: int = 0) -> int:
    m = len(sub)
    for i in range(start, len(ids) - m + 1):
        if ids[i:i + m] == sub:
            return i
    return -1


THINK_SUFFIX = [27963, 29]  # 'think','>' -- the STABLE tail of both <think> and </think>


def _find_think_close(model, ids: list[int], start: int) -> tuple[int, int]:
    """First `</think>` close in ids[start:]. Returns (close_start_idx, gt_idx) or (-1, -1).

    `</think>` is NOT a single/fixed token: under BPE the leading `</` merges with the
    preceding char (`.</`=4005, ` </`, `\\n</`, standalone `</`=524, ...), so a fixed 3-token
    subsequence misses almost every real close. The stable invariant is the `[27963, 29]`
    (`think`,`>`) suffix; a CLOSE is one whose preceding token decodes to something containing
    `</` (an OPEN `<think>` has a preceding `<` without the slash). The opener lives in the
    prompt (< start), so the first `</`-validated suffix in the generated region is the close.
    """
    tok = model.tokenizer
    i = start
    while True:
        j = _find_subseq(ids, THINK_SUFFIX, i)  # index of 27963 ('think')
        if j < 0:
            return -1, -1
        if j - 1 >= 0 and "</" in tok.decode([ids[j - 1]]):
            return j - 1, j + 1            # close_start = the '</'-token; gt_idx = the '>' token
        i = j + 1


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
    close_start, gt_idx = _find_think_close(model, full_ids, plen)  # </think> in generated region
    if close_start < 0:
        idx2 = idx3 = None
    else:
        idx2 = close_start - 1                # last reasoning token before the '</'-token
        idx3 = _last_meaningful(full_ids, gt_idx, terminators)  # answer token after '>'

    def take(i):
        return acts[i].float().cpu().numpy() if i is not None and i >= 0 else None

    return {"p0": take(p0), "p1": take(p1), "p2": take(idx2), "p3": take(idx3),
            "closed": close_start >= 0, "has_answer": idx3 is not None,
            "n": len(full_ids), "plen": plen, "n_gen": len(full_ids) - plen,
            "tail": tok.decode(full_ids[plen:])[-300:],  # diagnostic: end of the generation
            "p2_tok": (tok.decode([full_ids[idx2]]) if idx2 is not None and idx2 >= 0 else None),
            "p3_tok": (tok.decode([full_ids[idx3]]) if idx3 is not None else None)}


def refusal_directions(model, harmful: list[str], harmless: list[str], layer: int,
                       max_new_tokens: int) -> dict:
    """Harmful-minus-harmless diff-of-means refusal direction at each of P0..P3.

    Returns unit refusal vectors keyed 'p0'..'p3' (None if a position has no clean pairing),
    plus per-position counts + exclusion stats. P0/P1 use all prompts; P2/P3 use only
    generations that closed `</think>` and produced an answer (excluded counts reported).
    """
    post_count = post_instruction_token_count(model.tokenizer)

    def collect(prompts, label):
        # Serial GPU generation (NOT parallel_map -- one model). Print progress every 25
        # prompts (and at the end) so the long run is visible; closed-so-far is the </think>
        # adequacy signal the pilot validates. flush=True -> reaches session.log promptly.
        rows, closed = [], 0
        n = len(prompts)
        for k, p in enumerate(prompts, 1):
            row = generate_and_position(model, p, layer, max_new_tokens, post_count)
            rows.append(row)
            closed += int(row["closed"])
            if k % 25 == 0 or k == n:
                print(f"  {k}/{n} ({label}) done, closed-so-far={closed / k:.2f}", flush=True)
        return rows

    h_rows = collect(harmful, "harmful")
    s_rows = collect(harmless, "harmless")

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
    # Diagnostics: generation-length distribution tells us whether non-closes are the cap
    # (n_gen at max_new_tokens) vs a detection miss (short n_gen but closed=False).
    gen = [r["n_gen"] for r in h_rows + s_rows]
    out["gen_len"] = {"mean": float(np.mean(gen)), "max": int(max(gen)),
                      "p90": float(np.percentile(gen, 90)),
                      "hit_cap_frac": float(np.mean([g >= max_new_tokens for g in gen]))}
    # A few decoded samples (end-of-generation + detected P2/P3 tokens) so a re-failure or a
    # cap-limited closed_rate is diagnosable from the artifacts alone.
    def sample(rows, label):
        return [{"set": label, "closed": r["closed"], "n_gen": r["n_gen"],
                 "p2_tok": r["p2_tok"], "p3_tok": r["p3_tok"], "tail": r["tail"]}
                for r in rows[:4]]
    out["samples"] = sample(h_rows, "harmful") + sample(s_rows, "harmless")
    return out
