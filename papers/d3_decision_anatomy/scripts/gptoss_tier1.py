#!/usr/bin/env python3
"""GPT-OSS-20B Tier-1 session (Amendment 5 finalization): the reasoning-model commitment axis.

One modest 20B session. Order (compute-ordering + the frozen A5 rule):
  1. END_OF_PROMPT refusal direction (fresh, forward-pass diff-of-means) + decision-channel act-sample.
  2. POSITION GATE (runs first): post-standardization PR of the decision channel + the refusal
     spare-channel null. position_valid iff the channel is the D2 low-dim bottleneck; else
     behavioral_primary -> the projection-based commitment curve is reported descriptively, not as a
     causal read.
  3. Psychometric curve (harmony generation, FINAL-channel refusal) -> operating / boundary bands.
  4. Prefill deliberation cells: engage=inculpating prefill, disengage=exculpating prefill; asymmetry A.
  5. Trace-position commitment curve (gated on position_valid).
  6. Frozen-branch verdict (reversible-reader / early-commit-in-trace / harm-keyed-deliberation).

VALIDATE=1 -> tiny model + a few items (plumbing smoke; a non-harmony tiny model degrades gracefully:
no analysis channel -> the whole rollout is classified, cells still produce numbers).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))
sys.path.insert(0, str(HERE.parents[1] / "6_cross_model" / "scripts"))
sys.path.insert(0, str(HERE.parents[1] / "7_reasoning" / "scripts"))

from deepsteer.directions import extraction as du  # noqa: E402
import stage2_content as s2  # noqa: E402
import reasoning_cells as rc  # noqa: E402
from stage1_attribution import _unit  # noqa: E402
import think_io  # noqa: E402
from extract_two_site import _acts_from_ids  # noqa: E402
from gpt_oss_precision_gate import _normalize, _REFUSAL  # noqa: E402


def endofprompt_acts(model, prompts, layer, cot_format):
    """Last-input-token residual at `layer` for each prompt, in the reasoning (think) prompt format —
    the END_OF_PROMPT decision site (the reflexive refusal position, D2 `final_pre_assistant` analog).
    Returns (n, hidden)."""
    tok = model.tokenizer
    rows = []
    for p in prompts:
        text = think_io.think_prompt(tok, p)
        ids = tok(text, return_tensors="pt")["input_ids"]
        acts = _acts_from_ids(model, ids, [layer])[layer]     # (seq, hidden)
        rows.append(acts[-1])
    return np.stack(rows)


def _pr_cov(X):
    """Participation ratio from covariance eigenvalues (A1: NOT the diagonal-variance PR)."""
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)[:min(X.shape)] ** 2
    return float((s.sum() ** 2) / ((s ** 2).sum() + 1e-12))


def _refusal_null_ratio(X, refusal, rng, k=200):
    """Refusal's projected-variance vs the covariance-matched random null q95 (D1 A3 spare-channel).
    <=1 -> refusal is a low-variance control direction at this position (below null)."""
    Xc = X - X.mean(0)
    pv = float(np.var(Xc @ _unit(refusal)))
    rnd = [float(np.var(Xc @ _unit(rng.standard_normal(X.shape[1])))) for _ in range(k)]
    return pv / (float(np.percentile(rnd, 95)) + 1e-12)


def frozen_verdict(deliberation, commitment, position_valid) -> dict:
    """Amendment 5 frozen branches (all publishable). reversible-reader: engage & disengage both move,
    low |A|. early-commit-in-trace: disengage fails (A high positive) OR commitment fraction low.
    harm-keyed-deliberation: engage moves refusal but the deliberation effect is carried by harm framing
    (left as the reading when both prefills move only via the inculpating/harm cue)."""
    A = (deliberation.get("asymmetry_A") or {}).get("A")
    e = deliberation.get("engage_mean_shift")
    d = deliberation.get("disengage_mean_shift")
    cf = (commitment.get("commitment") or {}).get("commitment_fraction") if commitment else None
    moves = lambda x: x is not None and abs(x) >= 0.15
    if A is None:
        return {"verdict": "insufficient", "reason": "deliberation cell underpowered"}
    if moves(e) and moves(d) and abs(A) <= 0.34:
        v = "reversible_reader"
    elif (A >= 0.34) or (position_valid and cf is not None and cf <= 0.4):
        v = "early_commit_in_trace"
    else:
        v = "harm_keyed_or_indeterminate"
    return {"verdict": v, "A": A, "engage_shift": e, "disengage_shift": d,
            "commitment_fraction": cf, "position_valid": position_valid,
            "note": "reversible_reader = deliberative training yields a reversible reader; "
                    "early_commit_in_trace = decision set before deliberation completes"}


def main() -> None:
    ap = argparse.ArgumentParser(description="GPT-OSS-20B Tier-1 commitment-axis session.")
    ap.add_argument("--key", default="gpt_oss_20b")
    ap.add_argument("--layer", type=int, default=None, help="override; default = registry primary_layer")
    ap.add_argument("--refusal-prompts",
                    default=str(HERE.parents[1] / "5_moral_alignment" / "refusal_prompts.json"))
    ap.add_argument("--out", default=str(HERE.parent / "outputs"))
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    validate = os.environ.get("VALIDATE") == "1"
    import model_registry as reg
    if validate:
        model_id, layer, cot_format, key = "allenai/OLMo-2-0425-1B-Instruct", 8, think_io.CoTFormat.THINK_TAGS, "gpt_oss_20b"
    else:
        spec = reg.get(args.key)
        model_id, cot_format, key = spec.reasoning_repo, spec.cot_format, spec.key
        layer = args.layer if args.layer is not None else spec.primary_layer
    mnt = 64 if validate else args.max_new_tokens
    rng = np.random.default_rng(0)

    rp = json.loads(Path(args.refusal_prompts).read_text())
    n_dir = 12 if validate else 64
    harmful, harmless = rp["harmful"][:n_dir], rp["harmless"][:n_dir]

    model = du.load_whitebox(model_id)
    if not validate:
        spec.assert_matches_model(model.info.n_layers, getattr(model.model.config, "hidden_size", None),
                                  model_type_live=getattr(model.model.config, "model_type", None))

    # --- refusal direction + decision-channel act-sample at END_OF_PROMPT (forward pass, no generation) ---
    Xh = endofprompt_acts(model, harmful, layer, cot_format)
    Xs = endofprompt_acts(model, harmless, layer, cot_format)
    refusal = _unit(Xh.mean(0) - Xs.mean(0))
    channel_act = np.concatenate([Xh, Xs], 0)

    # --- POSITION GATE (first): post-std PR + refusal spare-channel null ---
    var = channel_act.var(0); sig = np.sqrt(var); sig = np.where(sig > 1e-8, sig, 1.0)
    pr_std = _pr_cov(channel_act / sig)
    null_ratio = _refusal_null_ratio(channel_act, refusal, rng)
    gate = rc.position_gate_verdict(pr_std, null_ratio)
    position_valid = gate["position_valid"]

    # --- behavioral cells (always run; position-independent) ---
    is_ref = lambda rollout: rc._refusal_final(rollout, cot_format, _normalize, _REFUSAL)
    from deepsteer.datasets import get_severity_twins, get_boundary_twins
    boundary_mode = os.environ.get("BOUNDARY") == "1"
    twins = (get_boundary_twins() if boundary_mode else get_severity_twins())
    psycho = rc.psychometric_harmony(model, twins, cot_format, is_ref, max_new_tokens=mnt, validate=validate)
    sel_band = psycho["boundary_band"] if boundary_mode else psycho["operating_band"]
    band_pairs = [(foll, viol) for (_f, lvl, foll, viol) in twins if lvl in sel_band]
    if not band_pairs:                                        # fall back so the cell still runs
        band_pairs = [(foll, viol) for (_f, _l, foll, viol) in twins[:(3 if validate else 10)]]
    deliberation = rc.prefill_deliberation(model, band_pairs, cot_format, is_ref, rng,
                                           max_new_tokens=mnt, validate=validate)

    # --- projection cell: commitment curve (gated on position_valid; else descriptive) ---
    commitment = rc.trace_commitment_curve(model, harmful, harmless, layer, refusal, cot_format,
                                           is_ref, max_new_tokens=mnt, validate=validate)
    if not position_valid and commitment.get("commitment"):
        commitment["commitment"]["note"] = ("DESCRIPTIVE ONLY: position gate failed (channel not a "
                                             "bottleneck), projection read is not position-valid")

    verdict = frozen_verdict(deliberation, commitment, position_valid)

    result = {"model": model_id, "key": key, "layer": layer, "cot_format": cot_format.value,
              "n_refusal_dir": {"harmful": len(harmful), "harmless": len(harmless)},
              "position_gate": gate, "psychometric": psycho,
              "band_mode": "boundary" if boundary_mode else "operating", "n_band_pairs": len(band_pairs),
              "deliberation": deliberation, "commitment": commitment, "verdict": verdict,
              "note": "Tier-1 commitment axis: behavioral cells (psychometric, prefill deliberation) are "
                      "position-independent and always valid; the commitment curve is a projection read "
                      "gated on the position check. Tier-2 (causal C1-MoE) is held."}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"tier1_inputs_{key}.npz", refusal=refusal, channel_act=channel_act, layer=layer,
             pr_std=pr_std, refusal_null_ratio=null_ratio)
    (out / f"tier1_session_{key}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    model.release()


if __name__ == "__main__":
    main()
