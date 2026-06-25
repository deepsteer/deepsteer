#!/usr/bin/env python3
"""One-shot diagnostic: why does causal_ablation's intervention do nothing on GPT-OSS?

Runs, on 2 harmful prompts, every combination of {direction source} x {ablation
machinery} and PRINTS THE GENERATED TEXT, so the no-op is isolated to direction
vs machinery vs harness:

  * fresh refusal direction (precision-gate refusal_direction) vs npz eop[headline]
    -> cosine, and whether they ablate the same.
  * precision-gate ablate_all_layers (PROVEN to fire) vs causal ablate_subspace
    -> same direction through both; if precision fires and subspace doesn't, the
    bug is in ablate_subspace; if neither fires, the bug is the harness/budget.

Prints clean vs ablated rollout heads + refusal flags. Definitive in one run.
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
import direction_utils as du  # noqa: E402
from causal_ablation import ablate_subspace, _orthonormal  # noqa: E402
from gpt_oss_precision_gate import (refusal_direction, ablate_all_layers,  # noqa: E402
                                    _normalize, _REFUSAL)


def gen_text(model, prompt_text, cot_format, max_new_tokens):
    tok = model.tokenizer
    enc = tok(think_io.think_prompt(tok, prompt_text), return_tensors="pt").to(model.model.device)
    plen = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    roll = think_io.decode_rollout(tok, out[0, plen:], cot_format)
    refused = bool(_REFUSAL.search(_normalize(roll)))
    return refused, roll


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="gpt_oss_20b")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    spec = reg.get(args.key)
    L = spec.primary_layer
    ps = json.load(open(args.prompts))
    harmful = ps["harmful"][: args.n]
    harmless = ps["harmless"][: args.n]
    model = WhiteBoxModel(spec.reasoning_repo, device=args.device,
                          torch_dtype=torch.bfloat16, access_tier=AccessTier.WEIGHTS)
    nL = model.info.n_layers
    print(f"== debug {spec.key} L{L}, {nL} layers, {args.max_new_tokens} tok ==")

    fresh = refusal_direction(model, harmful, harmless, L)            # precision-gate fresh dir
    rd_npz = du.load_directions(
        f"papers/7_reasoning/outputs/{spec.key}/two_site_refusal_directions.npz")["eop"][L]
    cos = float(fresh @ rd_npz / (np.linalg.norm(fresh) * np.linalg.norm(rd_npz) + 1e-12))
    print(f"cosine(fresh refusal, npz eop[{L}]) = {cos:.4f}")

    p = harmful[0]
    print(f"\nprompt: {p[:80]}")
    r0, t0 = gen_text(model, p, spec.cot_format, args.max_new_tokens)
    print(f"[clean]                      refused={r0}  | {t0[:140]!r}")

    B_fresh = _orthonormal(fresh[None, :])
    with ablate_subspace(model, B_fresh, list(range(nL))):
        r1, t1 = gen_text(model, p, spec.cot_format, args.max_new_tokens)
    print(f"[fresh + ablate_subspace]    refused={r1}  | {t1[:140]!r}")

    with ablate_all_layers(model, fresh):
        r2, t2 = gen_text(model, p, spec.cot_format, args.max_new_tokens)
    print(f"[fresh + ablate_all_layers]  refused={r2}  | {t2[:140]!r}")

    B_npz = _orthonormal(rd_npz[None, :])
    with ablate_subspace(model, B_npz, list(range(nL))):
        r3, t3 = gen_text(model, p, spec.cot_format, args.max_new_tokens)
    print(f"[npz + ablate_subspace]      refused={r3}  | {t3[:140]!r}")

    print("\nDIAGNOSIS:")
    print(f"  text changed under fresh+subspace : {t1 != t0}")
    print(f"  text changed under fresh+all_layers: {t2 != t0}")
    print(f"  text changed under npz+subspace    : {t3 != t0}")
    model.release()


if __name__ == "__main__":
    main()
