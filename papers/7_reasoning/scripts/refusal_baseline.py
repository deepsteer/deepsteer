#!/usr/bin/env python3
"""Phase 2b precondition: measure each model's CLEAN behavioral refusal rate.

The causal load-bearing test (causal_ablation.py) can only run where the model
actually refuses at baseline — there must be a refusal to ablate. The smoke showed
ds_r1_llama8b at clean refusal 0.0 (at a short budget), which makes the whole test
UNDERPOWERED. This diagnostic establishes, per model, the precondition:

  * clean refusal rate, classified TWO ways:
      - whole-rollout (refusal phrase anywhere; catches GPT-OSS's analysis-channel
        "this is disallowed, I must refuse")
      - final-answer only (post-</think> / harmony final; catches a distill that
        reasons at length then declines in its answer)
  * reached_final / closed counts and trace lengths, so a low rate that is really
    "budget too short to reach the decision" is distinguishable from genuine
    compliance;
  * dumped samples (rollout head + final) so the behavior is inspectable rather
    than guessed.

Cheap (no ablation loop): one generous budget, per model, loop the panel. GPT-OSS
auto-dequantizes to bf16.

Usage:
    python papers/7_reasoning/scripts/refusal_baseline.py --models all \
        --prompts papers/5_moral_alignment/refusal_prompts.json \
        --n 24 --max-new-tokens 1024 --output-dir papers/7_reasoning/outputs
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_registry as reg  # noqa: E402
import think_io  # noqa: E402
from gpt_oss_precision_gate import _normalize, _REFUSAL, _looks_degenerate  # noqa: E402

logger = logging.getLogger(__name__)


def measure(model, harmful, cot_format, max_new_tokens, n_dump=4) -> dict:
    tok = model.tokenizer
    rows, samples = [], []
    for i, p in enumerate(harmful):
        prompt = think_io.think_prompt(tok, p)
        enc = tok(prompt, return_tensors="pt").to(model.model.device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        gen_ids = gen[0, plen:]
        rollout = think_io.decode_rollout(tok, gen_ids, cot_format)
        reasoning, final = think_io.split_rollout(rollout, cot_format)
        boundary = think_io.cot_token_boundary(tok, gen_ids, cot_format)
        closed = boundary < len(gen_ids)
        coherent = not _looks_degenerate(rollout)
        ref_whole = bool(_REFUSAL.search(_normalize(rollout))) if coherent else None
        ref_final = bool(_REFUSAL.search(_normalize(final))) if (coherent and final.strip()) else None
        rows.append({"closed": bool(closed), "coherent": coherent,
                     "refused_whole": ref_whole, "refused_final": ref_final,
                     "trace_len": int(boundary), "final_chars": len(final)})
        if i < n_dump:
            samples.append({"prompt": p[:90], "closed": bool(closed),
                            "refused_whole": ref_whole, "refused_final": ref_final,
                            "rollout_head": rollout[:280], "final": final[:200]})

    coh = [r for r in rows if r["coherent"]]
    wholes = [r["refused_whole"] for r in coh if r["refused_whole"] is not None]
    finals = [r["refused_final"] for r in coh if r["refused_final"] is not None]
    return {
        "n": len(rows), "n_coherent": len(coh),
        "n_closed": sum(1 for r in rows if r["closed"]),
        "refusal_rate_whole": round(float(np.mean(wholes)), 4) if wholes else None,
        "refusal_rate_final": round(float(np.mean(finals)), 4) if finals else None,
        "n_with_final": len(finals),
        "trace_len_median": int(np.median([r["trace_len"] for r in rows])),
        "rows": rows, "samples": samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2b clean-refusal baseline diagnostic")
    ap.add_argument("--models", default="all", help="'all' or comma list: " + ",".join(reg.PANEL_ORDER))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    ps = json.load(open(args.prompts))
    harmful = ps["harmful"][: args.n]
    keys = reg.PANEL_ORDER if args.models == "all" else [k.strip() for k in args.models.split(",")]
    out = Path(args.output_dir)

    summary = {}
    for key in keys:
        spec = reg.get(key)
        t0 = time.time()
        model = WhiteBoxModel(spec.reasoning_repo, device=args.device,
                              torch_dtype=torch.bfloat16, access_tier=AccessTier.WEIGHTS)
        spec.assert_matches_model(model.info.n_layers,
                                  getattr(model.model.config, "hidden_size", None),
                                  model_type_live=getattr(model.model.config, "model_type", None))
        print(f"\n== {key} ({spec.provenance.value}) @ {args.max_new_tokens} tok, "
              f"{len(harmful)} harmful ==")
        r = measure(model, harmful, spec.cot_format, args.max_new_tokens)
        model.release()
        r["key"] = key; r["provenance"] = spec.provenance.value
        (out / spec.out).mkdir(parents=True, exist_ok=True)
        (out / spec.out / "refusal_baseline.json").write_text(json.dumps(r, indent=2))
        summary[key] = {k: r[k] for k in ("refusal_rate_whole", "refusal_rate_final",
                                          "n_coherent", "n_closed", "n_with_final",
                                          "trace_len_median", "provenance")}
        print(f"   refusal_rate whole={r['refusal_rate_whole']} final={r['refusal_rate_final']} "
              f"| closed {r['n_closed']}/{r['n']} | with_final {r['n_with_final']} "
              f"| trace_len med {r['trace_len_median']} | {time.time()-t0:.0f}s")
        if r["samples"]:
            s = r["samples"][0]
            print(f"   [sample] closed={s['closed']} refused_whole={s['refused_whole']} "
                  f"refused_final={s['refused_final']}")
            print(f"     rollout: {s['rollout_head'][:180]!r}")
            print(f"     final  : {s['final'][:140]!r}")

    (out / "refusal_baseline_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== clean refusal baseline (precondition for Phase 2b causal test) ===")
    print(f"  {'model':14} {'prov':16} {'whole':>6} {'final':>6} {'closed':>8} {'trace_med':>9}")
    for k in keys:
        s = summary[k]
        print(f"  {k:14} {s['provenance']:16} {str(s['refusal_rate_whole']):>6} "
              f"{str(s['refusal_rate_final']):>6} {s['n_closed']:>8} {s['trace_len_median']:>9}")
    print("\n  Causal test needs a non-trivial clean refusal (>=~0.5) to have signal to ablate.")
    print(f"  Wrote {out}/refusal_baseline_summary.json + per-model refusal_baseline.json")


if __name__ == "__main__":
    main()
