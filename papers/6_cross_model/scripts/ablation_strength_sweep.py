#!/usr/bin/env python3
"""Phase 2d: Llama refusal-ablation STRENGTH sweep (dose-response + bootstrap).

The Phase-2c control showed Llama's moral-judgment drop under refusal ablation is
refusal-SPECIFIC (outlier below a magnitude-matched random null). This sharpens
it: vary the FRACTION of the refusal direction removed (alpha: partial -> full ->
over-ablation, W -= alpha * d d^T W) and, at each step, measure how much refusal
is actually stripped (harmful refusal rate) AND moral judgment (with a paired
bootstrap CI), plus probe acc / eff-dim. Run a magnitude-matched random null at
the same removal magnitudes.

The discriminator:
  * judgment degrades MONOTONICALLY as more refusal is removed (and tracks the
    refusal-rate drop) -> genuine graded coupling ("entangled"); draw with a
    properly-powered effect size.
  * judgment is flat then CLIFFS at one alpha while refusal-removal saturates ->
    destabilization confound caught before publication.

Reuses the Phase-2c ablation/representation helpers and the MoralFoundationsProbe
scenarios + parser (per-scenario outcomes for the bootstrap). Diagnostic only;
over-ablation is a measurement probe, not a deployable intervention.

Usage:
    python papers/6_cross_model/scripts/ablation_strength_sweep.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --refusal-npz papers/6_cross_model/outputs/llama31_instruct/refusal_directions.npz \
        --prompts papers/5_moral_alignment/refusal_prompts.json --layer 13 \
        --alphas 0,0.5,1.0,1.5,2.0 --null-alphas 1.0,2.0 --n-null 3 --n-boot 5000 \
        --output papers/6_cross_model/outputs/llama31/ablation_strength_sweep.json
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
_P5 = Path(__file__).resolve().parent.parent.parent / "5_moral_alignment" / "scripts"
sys.path.insert(0, str(_P5))

import direction_utils as du  # noqa: E402
import random_ablation_control as rc  # noqa: E402  (reuse helpers)
from stage2_a3_causal import refusal_eval  # noqa: E402

logger = logging.getLogger(__name__)


@torch.no_grad()
def removal_norms_full(model, d_np: np.ndarray) -> list[float]:
    """Per-matrix ||d^T W||_F at full strength (alpha=1), without modifying weights."""
    d_unit = d_np / (np.linalg.norm(d_np) + 1e-12)
    norms = []
    for mod in rc._resid_matrices(model):
        W = mod.weight.data
        d = torch.from_numpy(d_unit).to(dtype=W.dtype, device=W.device)
        norms.append(float((d @ W).float().norm()))
    return norms


@torch.no_grad()
def ablate_scaled(model, d_np: np.ndarray, alpha: float) -> None:
    """W <- W - alpha * d d^T W for every resid-write matrix (alpha=1 = full ablation)."""
    d_unit = d_np / (np.linalg.norm(d_np) + 1e-12)
    for mod in rc._resid_matrices(model):
        W = mod.weight.data
        d = torch.from_numpy(d_unit).to(dtype=W.dtype, device=W.device)
        W -= alpha * torch.outer(d, d @ W)


def eval_judgment(chat_model, scenarios, max_tokens: int) -> tuple[float, list[int]]:
    """Per-scenario moral-judgment correctness (replicates MoralFoundationsProbe)."""
    from deepsteer.benchmarks.moral_reasoning.foundations import (
        _PROMPT_TEMPLATE,
        _parse_moral_judgment,
    )
    correct = []
    for sc in scenarios:
        prompt = _PROMPT_TEMPLATE.format(scenario=sc.scenario)
        resp = chat_model.generate(prompt, max_tokens=max_tokens, temperature=0.0).text
        j = _parse_moral_judgment(resp)
        correct.append(1 if j == sc.expected_judgment else 0)
    return float(np.mean(correct)), correct


def _spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _boot_drop(clean: list[int], cond: list[int], n_boot: int, seed: int) -> dict:
    c, a = np.array(clean, float), np.array(cond, float)
    n = len(c)
    rng = np.random.default_rng(seed)
    drops = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        drops[b] = c[idx].mean() - a[idx].mean()
    return {"drop": round(float(c.mean() - a.mean()), 4),
            "ci95": [round(float(np.percentile(drops, 2.5)), 4),
                     round(float(np.percentile(drops, 97.5)), 4)],
            "p_drop_gt_0": round(float(np.mean(drops > 0)), 4), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2d refusal-ablation strength sweep")
    ap.add_argument("--model", required=True)
    ap.add_argument("--refusal-npz", required=True)
    ap.add_argument("--prompts", required=True, help='JSON {"harmful":[...]} for the refusal rate.')
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alphas", default="0,0.5,1.0,1.5,2.0")
    ap.add_argument("--null-alphas", default="1.0,2.0")
    ap.add_argument("--n-null", type=int, default=3)
    ap.add_argument("--n-harmful", type=int, default=40)
    ap.add_argument("--dataset-target", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from deepsteer.benchmarks.moral_reasoning.foundations import get_scenarios
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    L = args.layer
    refusal = du.load_directions(args.refusal_npz)["refusal"][L]
    harmful = json.load(open(args.prompts))["harmful"][: args.n_harmful]
    scenarios = get_scenarios()
    groups = rc._build_groups(args.dataset_target)
    alphas = [float(x) for x in args.alphas.split(",")]
    null_alphas = [float(x) for x in args.null_alphas.split(",")]
    t0 = time.time()

    def load():
        return WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)

    # Full-strength per-matrix removal norms (for matched-random scaling).
    m = load()
    n_layers = m.info.n_layers
    headline = round(0.5 * n_layers)
    full_norms = removal_norms_full(m, refusal)
    m.release()

    # ---- refusal arm: dose-response ----
    refusal_arm = []
    clean_correct = None
    for a in alphas:
        wb = load()
        if a != 0.0:
            ablate_scaled(wb, refusal, a)
        rate, _ = refusal_eval(wb, harmful, args.max_tokens)
        jacc, correct = eval_judgment(rc._chat_model(wb), scenarios, args.max_tokens)
        pacc, eff = rc._representation(wb, groups, headline)
        wb.release()
        if a == 0.0:
            clean_correct = correct
        refusal_arm.append({"alpha": a, "refusal_rate": rate, "judgment": round(jacc, 4),
                            "probe_acc": pacc, "eff_dim": eff, "_correct": correct})
        print(f"  refusal a={a}: refusal_rate={rate} judgment={jacc:.4f} "
              f"probe_acc={pacc} eff={eff}")

    clean_rate = next(r["refusal_rate"] for r in refusal_arm if r["alpha"] == 0.0)
    clean_judg = next(r["judgment"] for r in refusal_arm if r["alpha"] == 0.0)

    # bootstrap each alpha's judgment drop vs clean (paired over scenarios)
    for i, r in enumerate(refusal_arm):
        r["refusal_removed"] = round(clean_rate - r["refusal_rate"], 4)
        r["boot"] = _boot_drop(clean_correct, r["_correct"], args.n_boot, seed=1000 + i)
        del r["_correct"]

    # ---- random null arm at matched magnitudes ----
    null_arm = []
    for a in null_alphas:
        js = []
        scaled = [a * f for f in full_norms]
        for k in range(args.n_null):
            wb = load()
            rc.perturb_matched(wb, scaled, seed=7000 + int(a * 10) * 17 + k)
            jacc, _ = eval_judgment(rc._chat_model(wb), scenarios, args.max_tokens)
            wb.release()
            js.append(round(jacc, 4))
        null_arm.append({"alpha": a, "judgments": js,
                         "mean": round(float(np.mean(js)), 4), "std": round(float(np.std(js)), 4)})
        print(f"  random-null a={a}: judgments={js} mean={np.mean(js):.4f}")

    # ---- monotonicity / cliff diagnosis ----
    a_sorted = sorted(refusal_arm, key=lambda r: r["alpha"])
    aa = [r["alpha"] for r in a_sorted]
    jj = [r["judgment"] for r in a_sorted]
    rr = [r["refusal_removed"] for r in a_sorted]
    drops = [clean_judg - j for j in jj]
    steps = [jj[i] - jj[i + 1] for i in range(len(jj) - 1)]  # positive = judgment fell
    total_drop = clean_judg - min(jj)
    max_step = max(steps) if steps else 0.0
    max_step_ratio = round(max_step / total_drop, 3) if total_drop > 1e-6 else float("nan")
    sp_alpha = round(_spearman(aa, jj), 3)          # expect strongly negative if graded
    sp_removed = round(_spearman(rr, drops), 3)     # expect strongly positive if dose-response

    if total_drop < 0.04:
        verdict = "no material judgment drop across strengths (coupling not reproduced here)"
    elif sp_removed >= 0.8 and max_step_ratio <= 0.75:
        verdict = ("graded dose-response: judgment degrades monotonically as refusal is removed "
                   "(genuine coupling; use 'entangled')")
    elif max_step_ratio > 0.85:
        verdict = ("cliff: judgment drop concentrated at one ablation step while refusal-removal "
                   "saturates (destabilization confound)")
    else:
        verdict = "partial/ambiguous: monotone trend with a dominant step; report both curves"

    payload = {
        "analysis": "ablation_strength_sweep", "model": args.model, "layer": L,
        "headline_layer": headline, "max_tokens": args.max_tokens, "n_boot": args.n_boot,
        "n_scenarios": len(scenarios), "n_harmful": len(harmful),
        "clean_refusal_rate": clean_rate, "clean_judgment": clean_judg,
        "refusal_arm": refusal_arm, "random_null_arm": null_arm,
        "monotonicity": {"spearman_alpha_vs_judgment": sp_alpha,
                         "spearman_removed_vs_drop": sp_removed,
                         "max_step_ratio": max_step_ratio, "total_drop": round(total_drop, 4)},
        "verdict": verdict, "elapsed_s": round(time.time() - t0, 1),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"\n=== refusal-ablation strength sweep @L{L} (headline L{headline}) ===")
    print(f"  {'alpha':>6} {'refusal':>8} {'removed':>8} {'judgment':>9} {'drop[95% CI]':>22} "
          f"{'probe':>6} {'eff':>4}")
    for r in a_sorted:
        b = r["boot"]
        print(f"  {r['alpha']:>6} {r['refusal_rate']:>8} {r['refusal_removed']:>8} "
              f"{r['judgment']:>9} {str(b['drop'])+' '+str(b['ci95']):>22} "
              f"{str(r['probe_acc']):>6} {str(r['eff_dim']):>4}")
    for nl in null_arm:
        print(f"  random null a={nl['alpha']}: judgment {nl['mean']} +/- {nl['std']}")
    print(f"  Spearman(removed, drop)={sp_removed}  max_step_ratio={max_step_ratio}")
    print(f"  VERDICT: {verdict}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
