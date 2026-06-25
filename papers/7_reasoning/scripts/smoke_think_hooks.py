#!/usr/bin/env python3
"""Phase 0c smoke: confirm activation hooks work on the reasoning generation.

The cheap pre-flight that must pass before any full Paper 7 pass. On the lightest
distill (one layer, a handful of prompts) it checks the three plumbing facts the
rest of the paper depends on:

  1. END_OF_PROMPT capture — ``get_activations`` on the ``think``-format prompt
     fires the layer hook and the last-input-token vector is finite (this is the
     Paper-6 refusal position, reused here as the reflexive site).
  2. The model actually emits a reasoning trace — a short generation shows the
     expected delimiters (``<think>...</think>`` for the distills; the harmony
     analysis channel for GPT-OSS).
  3. COT capture — re-running ``get_activations`` over prompt+rollout fires the
     hook and yields finite activations at token positions INSIDE the trace (the
     deliberative site Phase 2 localizes to decision sentences).

Writes a small JSON report; exits non-zero if any check fails so the RunPod
runner can stop before spending on a full pass.

Usage:
    python papers/7_reasoning/scripts/smoke_think_hooks.py --key ds_r1_llama8b \
        --output papers/7_reasoning/outputs/ds_r1_llama8b/smoke.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as reg  # noqa: E402
import think_io  # noqa: E402

logger = logging.getLogger(__name__)


def _finite_2d(t: torch.Tensor) -> bool:
    return t.ndim == 3 and t.shape[0] == 1 and bool(torch.isfinite(t).all())


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 0c reasoning-hook smoke")
    ap.add_argument("--key", default="ds_r1_llama8b", help="registry key (lightest distill)")
    ap.add_argument("--n-prompts", type=int, default=3, help="prompts per smoke pass")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="rollout budget; must be large enough to reach </think>")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    spec = reg.get(args.key)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  [{'ok  ' if cond else 'FAIL'}] {msg}")
        if not cond:
            fails.append(msg)

    t0 = time.time()
    model = WhiteBoxModel(spec.reasoning_repo, device=args.device, access_tier=AccessTier.WEIGHTS)
    cfg = model.model.config
    n_experts_live = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts", None)
    spec.assert_matches_model(
        model.info.n_layers, getattr(cfg, "hidden_size", None),
        model_type_live=getattr(cfg, "model_type", None), n_experts_live=n_experts_live,
    )
    L = spec.primary_layer
    print(f"Loaded {spec.reasoning_repo} ({model.info.n_layers}L) in {time.time()-t0:.1f}s; "
          f"smoke layer L{L}; cot_format={spec.cot_format.value}")

    prompts = (think_io.SMOKE_HARMFUL + think_io.SMOKE_HARMLESS)[: args.n_prompts]

    # --- (1) END_OF_PROMPT capture on the think-format prompt --------------------
    print("\n[1] END_OF_PROMPT activation capture")
    check(bool(getattr(model.tokenizer, "chat_template", None)),
          "tokenizer ships a chat template (think format available)")
    eop_prompt = think_io.think_prompt(model.tokenizer, prompts[0])
    eop_acts = model.get_activations(eop_prompt, layers=[L])
    check(L in eop_acts and _finite_2d(eop_acts[L]), f"L{L} hook fired, finite (1,seq,hidden)")
    eop_vec = eop_acts[L][0, -1, :].float().numpy() if L in eop_acts else None
    check(eop_vec is not None and np.isfinite(eop_vec).all() and eop_vec.shape[0] == spec.hidden,
          f"last-input-token vector finite, dim=={spec.hidden}")

    # --- (2) the model emits a reasoning trace with a detectable boundary -------
    # Use an EASY prompt: the trace closes (</think>) within the budget, so we can
    # verify CoT->final boundary detection (a refusal prompt can reason past the
    # budget without closing). The R1 template puts <think> in the PROMPT, so the
    # continuation contains the reasoning then </think> then the final answer.
    print("\n[2] reasoning-trace generation + CoT/final boundary")
    easy_prompt = think_io.think_prompt(model.tokenizer, think_io.SMOKE_EASY[0])
    check(think_io.prompt_opened_trace(easy_prompt, spec.cot_format),
          "think template opened the trace in the prompt (<think> present)")
    inputs = model.tokenizer(easy_prompt, return_tensors="pt").to(model.model.device)
    plen = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen_ids = gen[0, plen:]
    rollout = think_io.decode_rollout(model.tokenizer, gen_ids, spec.cot_format)  # clean
    closed = "</think>" in rollout if spec.cot_format.value == "think_tags" else \
        think_io.has_reasoning_structure(rollout, spec.cot_format)
    check(closed, f"reasoning trace closed within {args.max_new_tokens} tokens "
                  f"(boundary detectable; raise --max-new-tokens if not)")
    reasoning, final = think_io.split_rollout(rollout, spec.cot_format)
    check(len(reasoning) > 0, "CoT (reasoning) span is non-empty")
    check(len(final) > 0, "final-answer span is non-empty")
    # This tokenizer's decode leaks GPT-2 byte placeholders; decode_rollout must
    # reconstruct them so the refusal classifier / StrongREJECT see real spaces.
    check(not any(c in rollout for c in "ĠĊĉ"),
          "decoded rollout is clean (byte-level Ġ/Ċ reconstructed, classifier-ready)")
    print(f"     reasoning chars={len(reasoning)}  final chars={len(final)}")
    print(f"     first gen tokens: {model.tokenizer.convert_ids_to_tokens(gen_ids[:12].tolist())}")
    print(f"     clean rollout head: {rollout[:160]!r}")
    print(f"     final answer head:  {final[:120]!r}")

    # --- (3) COT-position capture over prompt+rollout ---------------------------
    print("\n[3] COT-position activation capture")
    full_text = easy_prompt + rollout
    full_acts = model.get_activations(full_text, layers=[L])
    seq_len = full_acts[L].shape[1] if L in full_acts else 0
    check(L in full_acts and _finite_2d(full_acts[L]) and seq_len > plen,
          f"L{L} hook fired over prompt+rollout (seq {seq_len} > prompt {plen})")
    if L in full_acts and seq_len > plen:
        cot_slice = full_acts[L][0, plen:, :].float().numpy()
        check(np.isfinite(cot_slice).all() and cot_slice.shape[0] >= 1,
              f"CoT-position activations finite ({cot_slice.shape[0]} positions, dim {cot_slice.shape[1]})")

    payload = {
        "analysis": "phase0c_smoke", "key": spec.key, "model": spec.reasoning_repo,
        "n_layers": model.info.n_layers, "smoke_layer": L, "cot_format": spec.cot_format.value,
        "n_prompts": len(prompts), "max_new_tokens": args.max_new_tokens,
        "reasoning_chars": len(reasoning), "final_chars": len(final),
        "trace_closed": bool(closed),
        "rollout_head": rollout[:240], "final_head": final[:240],
        "passed": not fails, "failures": fails, "elapsed_s": round(time.time() - t0, 1),
    }
    out.write_text(json.dumps(payload, indent=2))
    model.release()

    print(f"\nWrote {out}")
    if fails:
        print(f"SMOKE FAILED ({len(fails)}): " + "; ".join(fails))
        sys.exit(1)
    print("SMOKE PASSED — hooks work on the reasoning generation structure.")


if __name__ == "__main__":
    main()
