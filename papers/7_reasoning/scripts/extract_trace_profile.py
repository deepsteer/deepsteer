#!/usr/bin/env python3
"""Phase 2a: position-resolved trace activations for length disentanglement.

The Phase-1 trace-distribution signal (GPT-OSS carries the most distributed moral
content) is confounded with trace length: GPT-OSS harmful traces are short and
decisive (~54 tok) while the distills ramble (700+). Before that axis can carry
any weight, the moral content must be measured at MATCHED length / matched
fractional position.

This captures, per prompt at the headline layer, the reasoning-trace residual
activations summarized two ways:

  * **matched-prefix means** — mean over the first N trace tokens for a grid of N
    (NaN when the trace is shorter than N), so the harmful-harmless mean-diff and
    its moral fraction can be recomputed at a length matched ACROSS models.
  * **fractional-position bins** — mean over each of K equal fractional slices of
    the trace, so moral content can be plotted vs relative position (0=start,
    1=end) independent of absolute length.

Saves a single ``trace_profile.npz``; all length-controlled analysis runs LOCALLY
(no GPU) in ``trace_length_disentangle.py``. GPT-OSS loads bf16-dequantized
automatically.

Usage (driven by run_phase2a; same conventions as Phase 1):
    python papers/7_reasoning/scripts/extract_trace_profile.py --key gpt_oss_20b \
        --prompts papers/5_moral_alignment/refusal_prompts.json \
        --layer 12 --n 64 --max-new-tokens 768 --output-dir .../gpt_oss_20b
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "5_moral_alignment" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "6_cross_model" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_registry as reg  # noqa: E402
import think_io  # noqa: E402
from extract_two_site import _acts_from_ids  # noqa: E402  (reuse the id-aligned forward)

logger = logging.getLogger(__name__)

PREFIX_GRID = [16, 32, 64, 128, 256, 512]  # matched-length prefixes
K_BINS = 10                                # fractional-position bins


def _profile_prompt(model, prompt_text, layer, max_new_tokens, cot_format):
    """Return (prefix_means (G,hidden), bin_means (K,hidden), trace_len, coherent, closed).

    ``closed`` = the reasoning trace ended (``</think>`` / harmony final reached)
    WITHIN the budget, so the fractional bins span a COMPLETE deliberation
    (start->decision). For an unclosed trace the boundary is the budget itself, so
    its fractional bins are a truncated prefix, not the true full-trace shape; the
    length-disentanglement verdict therefore reads only closed traces.
    """
    tok = model.tokenizer
    prompt = think_io.think_prompt(tok, prompt_text)
    enc = tok(prompt, return_tensors="pt")
    plen = enc["input_ids"].shape[1]
    gen = model.model.generate(enc["input_ids"].to(model.model.device),
                               attention_mask=enc["attention_mask"].to(model.model.device),
                               max_new_tokens=max_new_tokens, do_sample=False)
    full_ids = gen[0].detach().cpu()
    generated = full_ids[plen:]
    boundary = think_io.cot_token_boundary(tok, generated, cot_format)
    closed = boundary < len(generated)  # marker found within the budget
    rollout = think_io.decode_rollout(tok, generated, cot_format)
    reasoning, _final = think_io.split_rollout(rollout, cot_format)
    coherent = boundary >= 1 and not think_io.looks_degenerate(reasoning or rollout)

    acts = _acts_from_ids(model, full_ids.unsqueeze(0), [layer])[layer]  # (seq, hidden)
    trace = acts[plen:plen + boundary]  # (boundary, hidden)
    H = acts.shape[1]
    prefix = np.full((len(PREFIX_GRID), H), np.nan, dtype=np.float32)
    bins = np.full((K_BINS, H), np.nan, dtype=np.float32)
    if boundary >= 1:
        for i, N in enumerate(PREFIX_GRID):
            if boundary >= N:
                prefix[i] = trace[:N].mean(0)
        for j in range(K_BINS):
            lo = (j * boundary) // K_BINS
            hi = ((j + 1) * boundary) // K_BINS
            if hi > lo:
                bins[j] = trace[lo:hi].mean(0)
    return prefix, bins, int(boundary), bool(coherent), bool(closed)


def _collect(model, prompts, layer, max_new_tokens, cot_format):
    pre, bn, tl, coh, cls = [], [], [], [], []
    for p in prompts:
        a, b, t, c, cl = _profile_prompt(model, p, layer, max_new_tokens, cot_format)
        pre.append(a); bn.append(b); tl.append(t); coh.append(c); cls.append(cl)
    return (np.stack(pre), np.stack(bn), np.asarray(tl, np.int64),
            np.asarray(coh, bool), np.asarray(cls, bool))


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2a position-resolved trace capture")
    ap.add_argument("--key", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--layer", type=int, required=True, help="headline layer")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    spec = reg.get(args.key)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ps = json.load(open(args.prompts))
    harmful, harmless = ps["harmful"][: args.n], ps["harmless"][: args.n]

    t0 = time.time()
    model = WhiteBoxModel(spec.reasoning_repo, device=args.device,
                          torch_dtype=torch.bfloat16, access_tier=AccessTier.WEIGHTS)
    spec.assert_matches_model(model.info.n_layers,
                              getattr(model.model.config, "hidden_size", None),
                              model_type_live=getattr(model.model.config, "model_type", None))
    # Record the realized float precision so the cross-model comparison can assert
    # all three models were profiled at the SAME effective precision (bf16) — a
    # GPT-OSS profiled at mxfp4 vs bf16 distills would bake a precision confound
    # into the moral-fraction comparison alongside length.
    fdtypes = sorted({str(p.dtype) for p in model.model.parameters() if p.is_floating_point()})
    print(f"Loaded {spec.reasoning_repo} ({model.info.n_layers}L) in {time.time()-t0:.0f}s; "
          f"headline L{args.layer}; cot={spec.cot_format.value}; float dtypes={fdtypes}; "
          f"{len(harmful)} harmful / {len(harmless)} harmless @ {args.max_new_tokens} tok")

    print(">> harmful trace profiles ...")
    Hp, Hb, Htl, Hc, Hcl = _collect(model, harmful, args.layer, args.max_new_tokens, spec.cot_format)
    print(f"   harmful coherent {int(Hc.sum())}/{len(harmful)}, "
          f"closed {int(Hcl.sum())}/{len(harmful)}, trace_len median {int(np.median(Htl))}")
    print(">> harmless trace profiles ...")
    Sp, Sb, Stl, Sc, Scl = _collect(model, harmless, args.layer, args.max_new_tokens, spec.cot_format)
    print(f"   harmless coherent {int(Sc.sum())}/{len(harmless)}, "
          f"closed {int(Scl.sum())}/{len(harmless)}, trace_len median {int(np.median(Stl))}")
    model.release()

    np.savez(out / "trace_profile.npz",
             key=spec.key, headline=np.int64(args.layer),
             prefix_grid=np.asarray(PREFIX_GRID, np.int64), k_bins=np.int64(K_BINS),
             float_dtypes=np.asarray(fdtypes), max_new_tokens=np.int64(args.max_new_tokens),
             H_prefix=Hp, H_bins=Hb, H_trace_len=Htl, H_coherent=Hc, H_closed=Hcl,
             S_prefix=Sp, S_bins=Sb, S_trace_len=Stl, S_coherent=Sc, S_closed=Scl)
    print(f"Wrote {out}/trace_profile.npz  (prefix grid {PREFIX_GRID}, {K_BINS} fractional "
          f"bins over CLOSED traces, headline L{args.layer}, dtypes={fdtypes})")


if __name__ == "__main__":
    main()
