#!/usr/bin/env python3
"""Path A step 1: is the CoT (generation-time) refusal direction CAUSAL?

The n=32 keystone showed GPT-OSS refusal is robust to ablating the EOP (input-
boundary) refusal direction. Phase 1 found EOP_refusal perp CoT_refusal (cos 0.13)
for GPT-OSS, so generated refusal is plausibly carried by the CoT direction, not
the EOP one. Before building the keystone yardstick on the CoT direction, VERIFY
it is causal (decodable != causal — the EOP lesson): compute the CoT refusal
direction FRESH from generation-time activations and test whether ablating it
flips refusal, with generated-text dumps. Cheap and decisive in one run.

Computes three fresh directions at the headline layer and ablates each at all
layers on test prompts:
  * eop_fresh   — last input token mean-diff (the n=32 yardstick that did NOT fire)
  * cot_mean    — mean over the reasoning-trace tokens, harmful-harmless mean-diff
  * cot_last    — last reasoning-trace token, harmful-harmless mean-diff
Reports pairwise cosines (reproduce Phase 1 EOP-CoT) and, per test prompt, the
clean vs ablated rollout + refusal flag, so the causal direction (if any) is named.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "5_moral_alignment" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_registry as reg  # noqa: E402
import think_io  # noqa: E402
from deepsteer.directions import extraction as du  # noqa: E402
from causal_ablation import ablate_subspace, _orthonormal  # noqa: E402
from extract_two_site import _acts_from_ids  # noqa: E402
from gpt_oss_precision_gate import _normalize, _REFUSAL, _looks_degenerate  # noqa: E402


_unit = du.unit_vector  # shared: deepsteer.directions.extraction.unit_vector


def collect_cot(model, prompts, layer, cot_format, max_new_tokens):
    """Per-prompt (mean-over-trace, last-trace-token) residuals at ``layer``."""
    tok = model.tokenizer
    means, lasts = [], []
    for p in prompts:
        enc = tok(think_io.think_prompt(tok, p), return_tensors="pt").to(model.model.device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        full = gen[0].detach().cpu()
        boundary = think_io.cot_token_boundary(tok, full[plen:], cot_format)
        acts = _acts_from_ids(model, full.unsqueeze(0), [layer])[layer]  # (seq, hidden)
        cot = acts[plen:plen + boundary]
        if cot.shape[0] >= 1:
            means.append(cot.mean(0))
            lasts.append(cot[-1])
    return np.stack(means), np.stack(lasts)


def gen_text(model, prompt_text, cot_format, max_new_tokens):
    tok = model.tokenizer
    enc = tok(think_io.think_prompt(tok, prompt_text), return_tensors="pt").to(model.model.device)
    plen = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    roll = think_io.decode_rollout(tok, out[0, plen:], cot_format)
    coherent = not _looks_degenerate(roll)
    return bool(_REFUSAL.search(_normalize(roll))), coherent, roll


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="gpt_oss_20b")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=8, help="prompts/class for the direction estimate.")
    ap.add_argument("--n-test", type=int, default=3, help="test prompts for the ablation dump.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    spec = reg.get(args.key)
    L = spec.primary_layer
    ps = json.load(open(args.prompts))
    harmful, harmless = ps["harmful"][: args.n], ps["harmless"][: args.n]
    model = WhiteBoxModel(spec.reasoning_repo, device=args.device,
                          torch_dtype=torch.bfloat16, access_tier=AccessTier.WEIGHTS)
    nL = model.info.n_layers
    print(f"== CoT-direction causality debug {spec.key} L{L}, {nL} layers, {args.max_new_tokens} tok ==")

    # --- fresh directions at the headline layer ---------------------------------
    from gpt_oss_precision_gate import refusal_direction
    eop = refusal_direction(model, harmful, harmless, L)
    Hm, Hl = collect_cot(model, harmful, L, spec.cot_format, args.max_new_tokens)
    Sm, Sl = collect_cot(model, harmless, L, spec.cot_format, args.max_new_tokens)
    cot_mean = _unit(Hm.mean(0) - Sm.mean(0))
    cot_last = _unit(Hl.mean(0) - Sl.mean(0))
    print(f"cos(eop, cot_mean)={du.cosine(eop, cot_mean):.3f}  "
          f"cos(eop, cot_last)={du.cosine(eop, cot_last):.3f}  "
          f"cos(cot_mean, cot_last)={du.cosine(cot_mean, cot_last):.3f}   (Phase 1: EOP-CoT ~0.13)")

    dirs = {"eop_fresh": eop, "cot_mean": cot_mean, "cot_last": cot_last}
    # A CLEAN flip = refusal removed AND the ablated rollout stays COHERENT. An
    # incoherent "flip" (refusal language gone only because generation degenerated)
    # is over-ablation, NOT a usable yardstick (user gate).
    clean_flip = {name: 0 for name in dirs}
    incoh_flip = {name: 0 for name in dirs}
    n_test = 0
    for p in harmful[: args.n_test]:
        n_test += 1
        print(f"\nprompt: {p[:78]}")
        r0, c0, t0 = gen_text(model, p, spec.cot_format, args.max_new_tokens)
        print(f"  [clean]    refused={r0} coherent={c0} | {t0[:120]!r}")
        for name, d in dirs.items():
            with ablate_subspace(model, _orthonormal(d[None, :]), list(range(nL))):
                r, c, t = gen_text(model, p, spec.cot_format, args.max_new_tokens)
            removed = r0 and not r
            if removed and c:
                clean_flip[name] += 1
            elif removed and not c:
                incoh_flip[name] += 1
            tag = "CLEAN-FLIP" if (removed and c) else ("incoherent" if removed else "still-refuses")
            print(f"  [{name:9}] refused={r} coherent={c} {tag} | {t[:100]!r}")

    print(f"\nDIAGNOSIS (over {n_test} test prompts):")
    causal = []
    for name in dirs:
        verdict = ("CLEANLY CAUSAL" if clean_flip[name] > 0
                   else ("over-ablation only (incoherent flips, NOT usable)"
                         if incoh_flip[name] > 0 else "not causal at this site"))
        if clean_flip[name] > 0:
            causal.append(name)
        print(f"  {name:9}: clean_flips={clean_flip[name]} incoherent_flips={incoh_flip[name]} -> {verdict}")
    print()
    if causal:
        print(f"GATE PASSED: {causal} cleanly flip refusal (coherent) -> build the keystone")
        print("yardstick on it (Path A); run moral/random/persona three-way against it.")
    else:
        print("GATE NOT PASSED: no direction cleanly flips refusal with coherence preserved.")
        print("FINDING: GPT-OSS refusal is distributed across both EOP and CoT sites, no")
        print("single-direction bottleneck. STOP — do NOT interpret a moral null against a")
        print("yardstick that did not fire. Rest on convergent Phases 1+2a+2b-robustness.")
    model.release()


if __name__ == "__main__":
    main()
