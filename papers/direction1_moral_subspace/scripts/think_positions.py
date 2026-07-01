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
The `</think>` boundary (for P2/P3) is located by the SHARED `deepsteer.reasoning.think_io`
helpers (`cot_token_boundary` / `split_rollout`): string-match the delimiter in the decoded
text, then binary-search back to the token index -- robust to BPE token-merge artifacts on the
delimiter (which broke an earlier fixed-token-subsequence matcher). A generation that never
closes `</think>` yields P2 = P3 = None (excluded, counted; never dropped).

GPU module: imports torch lazily so the file is importable for unit-style checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "7_reasoning" / "scripts"))
from token_positions import post_instruction_token_count  # noqa: E402

# The `</think>` boundary is located by the SHARED, robust helper (string-match in the decoded
# text + binary-search back to the token index) -- immune to the BPE merges that broke a
# fixed-token-subsequence matcher. Reused, not reimplemented (Paper 7 solved this in think_io).
from deepsteer.reasoning import think_io as ti  # noqa: E402

_FMT = ti.CoTFormat.THINK_TAGS
# Terminators we do not want P3 to land on (measure the last MEANINGFUL answer token).
_TERMINATORS = {100265, 100257}  # <|im_end|>, <|endoftext|>-class; eos added at runtime


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


def prompt_only_positions(model, prompt: str, layer: int, post_count: int) -> dict:
    """P0 (t_inst) + P1 (pre-trace gate) from a PROMPT-ONLY forward -- NO generation.

    P0/P1 depend only on the prompt tokens (causal masking), so they are identical whether or
    not a trace is generated afterwards. Extracting them prompt-only lets them use ALL prompts
    cheaply while the expensive generation (P2/P3) runs on a subset.
    """
    tok = model.tokenizer
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text)["input_ids"]
    plen = len(ids)
    acts = _forward_capture(model, ids, layer)
    p0 = max(0, plen - post_count - 1)        # t_inst
    p1 = plen - 1                             # pre-trace gate (<think> opener, last templated)
    return {"p0": acts[p0].float().cpu().numpy(), "p1": acts[p1].float().cpu().numpy()}


def generate_and_position(model, prompt: str, layer: int, max_new_tokens: int,
                          post_count: int, gen_cfg: dict | None = None,
                          cot_window_n: int = 256, fmt=_FMT) -> dict:
    """One prompt -> P2 (in-trace) span-means + P3 (post-answer), plus diagnostics.

    One generate + one full-sequence forward. `gen_cfg` selects decode (default greedy). `fmt`
    (CoTFormat) selects the trace parser: THINK_TAGS (OLMo/R1: `<think>` opens in the prompt) or
    HARMONY_ANALYSIS (GPT-OSS: the analysis channel opens DURING generation). The in-trace anchor
    is re-derived per format via `ti.cot_content_start` -- the one thing that does not transfer.

    **P2 (in-trace) is a SYMMETRIC window** to avoid a span-length confound: the reasoning span is
    short for harmful (early close) but full for harmless (never closes), so a full-span mean would
    be a short-vs-long contrast correlated with the label. Primary **p2 = mean over the first
    `cot_window_n` reasoning tokens FROM THE ANCHOR** (reasoning-content start), SAME N both sides.
    `p2_full` = whole-reasoning-span mean (confounded) is a ROBUSTNESS check only.
    """
    import torch
    tok = model.tokenizer
    cfg = gen_cfg or {"do_sample": False}
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model._device)
    plen = int(enc["input_ids"].shape[1])
    kw = {"max_new_tokens": max_new_tokens, "num_beams": 1, "pad_token_id": tok.eos_token_id}
    if cfg.get("do_sample"):
        kw.update(do_sample=True, temperature=cfg.get("temperature", 0.6),
                  top_p=cfg.get("top_p", 0.95))
    else:
        kw["do_sample"] = False
    with torch.no_grad():
        gen = model.model.generate(**enc, **kw)
    full_ids = gen[0].tolist()
    acts = _forward_capture(model, full_ids, layer)
    gen_ids = full_ids[plen:]                  # the generated region (reasoning + answer)
    gen_text = ti.decode_rollout(tok, gen_ids, fmt)

    # Format-agnostic terminators: all special/added tokens (channel markers, eos) so P3 lands on
    # the last MEANINGFUL answer token for both `</think>` and harmony channel formats.
    terminators = set(getattr(tok, "all_special_ids", []) or []) | set(_TERMINATORS)
    if tok.eos_token_id is not None:
        terminators.add(int(tok.eos_token_id))

    # Reasoning span in gen_ids = [content_start, boundary): content_start re-derived per format
    # (THINK_TAGS 0; HARMONY after the generated <|channel|>analysis<|message|>); boundary =
    # reasoning end (== len(gen_ids) if never closed).
    _, answer = ti.split_rollout(gen_text, fmt)
    content_start = ti.cot_content_start(tok, gen_ids, fmt)
    boundary = ti.cot_token_boundary(tok, gen_ids, fmt)
    closed = boundary < len(gen_ids)
    anchor = plen + content_start                        # full_ids-space reasoning-content start
    reason_end = plen + (boundary if closed else len(gen_ids))
    reason_len = reason_end - anchor                     # reasoning tokens from the anchor

    def span_mean(a, b):
        return acts[a:b].mean(0).float().cpu().numpy() if b > a else None

    # P2 symmetric window: first cot_window_n reasoning tokens from the anchor (excluded if
    # reasoning < N, so the window is always PURE reasoning -- never spills into the answer).
    p2_window = span_mean(anchor, anchor + cot_window_n) if reason_len >= cot_window_n else None
    p2_full = span_mean(anchor, reason_end) if reason_len >= 1 else None   # robustness only
    # P3 post-answer: needs a closed trace AND a non-empty answer (benign side may not reach it).
    idx3 = (_last_meaningful(full_ids, plen + boundary, terminators)
            if closed and answer.strip() else None)

    return {"p0": acts[max(0, plen - post_count - 1)].float().cpu().numpy(),
            "p1": acts[plen - 1].float().cpu().numpy(),
            "p2": p2_window, "p2_full": p2_full,
            "p3": acts[idx3].float().cpu().numpy() if idx3 is not None else None,
            "closed": closed, "has_answer": idx3 is not None, "win_ok": p2_window is not None,
            "has_close_str": "</think>" in gen_text, "reason_len": reason_len,
            "n": len(full_ids), "plen": plen, "n_gen": len(full_ids) - plen,
            "tail": gen_text[-300:], "gen_text": gen_text,  # gen_text kept only for samples
            "p3_tok": (tok.decode([full_ids[idx3]]) if idx3 is not None else None)}


def refusal_directions(model, harmful: list[str], harmless: list[str], layer: int,
                       max_new_tokens: int, gen_cfg: dict | None = None,
                       p23_n: int | None = None, cot_window_n: int = 256, fmt=_FMT) -> dict:
    """Harmful-minus-harmless diff-of-means refusal direction at P0, P1, P2(+robustness), P3.

    Split for cost: **P0/P1 are prompt-side** (no generation) so they use ALL prompts cheaply;
    **P2/P3 require generation** so they use a SUBSET of `p23_n` prompts/side (None = all).
    **P2 (in-trace)** = symmetric first-`cot_window_n`-reasoning-token span-mean (same N both
    sides -> no span-length confound); `p2_full` (full-span mean) is reported as a robustness
    check only. **P3 (post-answer)** needs a closed trace + answer, which the benign side does not
    reach within budget -> P3 is UNMEASURED there (not measured-and-null).
    """
    post_count = post_instruction_token_count(model.tokenizer)

    def collect_prompt(prompts, label):  # P0/P1 -- cheap prompt-only forward, ALL prompts
        rows, n = [], len(prompts)
        for k, p in enumerate(prompts, 1):
            rows.append(prompt_only_positions(model, p, layer, post_count))
            if k % 50 == 0 or k == n:
                print(f"  {k}/{n} ({label}) prompt-side done", flush=True)
        return rows

    def collect_gen(prompts, label):     # P2/P3 -- generation, SUBSET
        rows, closed, win, n = [], 0, 0, len(prompts)
        for k, p in enumerate(prompts, 1):
            row = generate_and_position(model, p, layer, max_new_tokens, post_count,
                                        gen_cfg, cot_window_n, fmt)
            rows.append(row)
            closed += int(row["closed"])
            win += int(row["win_ok"])
            if k % 25 == 0 or k == n:
                print(f"  {k}/{n} ({label}) gen done, closed={closed / k:.2f} "
                      f"window-ok={win / k:.2f}", flush=True)
        return rows

    h_prompt = collect_prompt(harmful, "harmful")
    s_prompt = collect_prompt(harmless, "harmless")
    gh = harmful[:p23_n] if p23_n else harmful
    gs = harmless[:p23_n] if p23_n else harmless
    h_gen = collect_gen(gh, "harmful")
    s_gen = collect_gen(gs, "harmless")

    out: dict[str, object] = {"layer": layer, "post_count": post_count,
                              "max_new_tokens": max_new_tokens, "cot_window_n": cot_window_n,
                              "n_prompt_side": len(harmful), "n_gen_side": len(gh),
                              "positions": {}}
    # p2 = symmetric window (PRIMARY), p2_full = full-span (ROBUSTNESS), p3 = post-answer.
    src = {"p0": (h_prompt, s_prompt), "p1": (h_prompt, s_prompt),
           "p2": (h_gen, s_gen), "p2_full": (h_gen, s_gen), "p3": (h_gen, s_gen)}
    for pos, (hr, sr) in src.items():
        hv = np.array([r[pos] for r in hr if r[pos] is not None], dtype=np.float64)
        sv = np.array([r[pos] for r in sr if r[pos] is not None], dtype=np.float64)
        if len(hv) == 0 or len(sv) == 0:
            out["positions"][pos] = {"vec": None, "n_harmful": len(hv), "n_harmless": len(sv)}
            continue
        r = hv.mean(0) - sv.mean(0)
        out["positions"][pos] = {"vec": r / (np.linalg.norm(r) + 1e-12),
                                 "n_harmful": int(len(hv)), "n_harmless": int(len(sv))}
    # closed-rate / window-rate / has_close_str / gen_len from the GEN rows (P2/P3 subset) only.
    out["closed_rate_harmful"] = float(np.mean([r["closed"] for r in h_gen]))
    out["closed_rate_harmless"] = float(np.mean([r["closed"] for r in s_gen]))
    out["window_ok_harmful"] = float(np.mean([r["win_ok"] for r in h_gen]))
    out["window_ok_harmless"] = float(np.mean([r["win_ok"] for r in s_gen]))
    out["answer_rate_harmful"] = float(np.mean([r["has_answer"] for r in h_gen]))
    out["answer_rate_harmless"] = float(np.mean([r["has_answer"] for r in s_gen]))
    out["has_close_str_rate"] = float(np.mean([r["has_close_str"] for r in h_gen + s_gen]))
    rl = [r["reason_len"] for r in h_gen + s_gen]
    out["reason_len"] = {"harmful_mean": float(np.mean([r["reason_len"] for r in h_gen])),
                         "harmless_mean": float(np.mean([r["reason_len"] for r in s_gen])),
                         "min": int(min(rl))}
    gen = [r["n_gen"] for r in h_gen + s_gen]
    out["gen_len"] = {"mean": float(np.mean(gen)), "max": int(max(gen)),
                      "p90": float(np.percentile(gen, 90)),
                      "hit_cap_frac": float(np.mean([g >= max_new_tokens for g in gen]))}
    # A few decoded samples (full trace + detected P2/P3 tokens) so a re-failure or a cap-limited
    # closed_rate is diagnosable from the artifacts alone.
    def sample(rows, label):
        return [{"set": label, "closed": r["closed"], "has_close_str": r["has_close_str"],
                 "n_gen": r["n_gen"], "reason_len": r["reason_len"], "win_ok": r["win_ok"],
                 "p3_tok": r["p3_tok"],
                 "gen_text": r["gen_text"][:6000]}  # full trace (capped) to see the transition
                for r in rows[:3]]
    out["samples"] = sample(h_gen, "harmful") + sample(s_gen, "harmless")
    return out
