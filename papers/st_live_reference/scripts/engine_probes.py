"""Amendment 7 item 4: the HF memory-and-throughput probe (the fallback engine's pricing input).

Generating 1,536 tokens per batch size would cost ~36 min per size on GPT-OSS-20B at §40's
1.37 s/step, so the probe measures the parts and projects the rest:

- prefill B real prompts (left-padded, mask-derived positions) and decode ``decode_steps`` tokens
  with KDG sampler v2; record peak GPU memory and the seconds per decoding step at batch B;
- project the peak at the cap of record: measured peak + the KV cache the remaining
  (cap - decode_steps) tokens add, from the config's attention geometry (full-attention layers
  grow with length; sliding-window layers stop at their window);
- ``fits`` iff the projected peak stays under ``limit_fraction`` of the card's memory.

Nothing is generated past ``decode_steps``; the per-step time at B is what the fallback re-price
uses (``gen_hours`` with the measured step at that batch).
"""

from __future__ import annotations

import time

import numpy as np


def kv_bytes_per_token(config, length: int) -> float:
    """KV-cache bytes one sequence holds at ``length`` tokens (bf16), from the text config."""
    t = getattr(config, "text_config", config)
    kvh = t.num_key_value_heads
    hd = getattr(t, "head_dim", None) or t.hidden_size // t.num_attention_heads
    types = getattr(t, "layer_types", None) or ["full_attention"] * t.num_hidden_layers
    window = getattr(t, "sliding_window", None) or length
    per_layer = 2 * kvh * hd * 2  # K and V, bf16
    total = 0.0
    for ty in types:
        if ty == "full_attention":
            total += per_layer * length
        elif ty == "sliding_attention":
            total += per_layer * min(length, window)
        # linear-attention (DeltaNet) layers hold a fixed-size state, not a per-token cache
    return total


def hf_batch_probe(model, prompt_ids: list[list[int]], batch: int, *, cap: int,
                   decode_steps: int = 32, temperature: float = 0.7, seed: int = 0,
                   limit_fraction: float = 0.92) -> dict:
    """One probe at ``batch``: real prompts cycled to fill the batch."""
    torch = model.torch
    rows = [prompt_ids[i % len(prompt_ids)] for i in range(batch)]
    pad = model.tok.pad_token_id
    L = max(len(x) for x in rows)
    ids = torch.full((batch, L), pad, dtype=torch.long)
    mask = torch.zeros((batch, L), dtype=torch.long)
    for r, x in enumerate(rows):
        ids[r, L - len(x):] = torch.tensor(x)
        mask[r, L - len(x):] = 1
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    out: dict = {"batch": batch, "decode_steps": decode_steps, "prompt_len_max": L, "cap": cap,
                 "warmed": True}
    ids, mask = ids.to(model.device), mask.to(model.device)

    def timed(n: int) -> float:
        torch.manual_seed(seed)
        t0 = time.time()
        with torch.no_grad():
            model.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=n,
                                 min_new_tokens=n, do_sample=True, temperature=temperature,
                                 top_p=1.0, top_k=0, pad_token_id=pad)
        if cuda:
            torch.cuda.synchronize()
        return time.time() - t0

    try:
        # untimed warm-up at the same shapes: first-call costs (kernel selection, a compile at a
        # new batch shape) otherwise land in t_short and shrink or flip the slope (local test,
        # 2026-10-19: t_short 0.160 s vs t_long 0.056 s)
        timed(decode_steps)
        t_short = timed(decode_steps)
        t_long = timed(2 * decode_steps)
    except torch.cuda.OutOfMemoryError as e:  # the probe's answer, not a crash
        torch.cuda.empty_cache()
        return {**out, "fits": False, "oom": str(e)[:200]}
    # prefill cancels in the difference of two decode lengths: the per-step slope at batch B
    t_prefill = max(0.0, t_short - decode_steps * (t_long - t_short) / decode_steps)
    step = (t_long - t_short) / decode_steps
    timing = {"prefill_seconds_est": t_prefill, "seconds_per_step": step, "oom": None,
              "t_decode_n": t_short, "t_decode_2n": t_long}
    if not cuda:  # local test path: timing only
        return {**out, **timing, "fits": None, "peak_gb_measured": None}
    peak = torch.cuda.max_memory_allocated() / 1e9
    card = torch.cuda.get_device_properties(0).total_memory / 1e9
    cfg = model.model.config
    extra_kv = batch * (kv_bytes_per_token(cfg, L + cap)
                        - kv_bytes_per_token(cfg, L + decode_steps))
    projected = peak + extra_kv / 1e9
    return {**out, **timing, "peak_gb_measured": peak, "card_gb": card,
            "projected_peak_gb_at_cap": projected, "limit_gb": limit_fraction * card,
            "fits": bool(projected <= limit_fraction * card)}


def summarize(probes: list[dict]) -> dict:
    ok = [p for p in probes if p.get("fits")]
    best = max(ok, key=lambda p: p["batch"]) if ok else None
    return {"probes": probes, "largest_fitting_batch": best["batch"] if best else None,
            "seconds_per_step_at_largest": best["seconds_per_step"] if best else None,
            "step_ratio_vs_smallest": (best["seconds_per_step"] / ok[0]["seconds_per_step"]
                                       if best and ok else None),
            "mean_step_seconds": float(np.mean([p["seconds_per_step"] for p in ok])) if ok
            else None}
