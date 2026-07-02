#!/usr/bin/env python3
"""C1 one-model session orchestrator: extract inputs (refusal, t_inst harm dir, mean_content V_moral,
decision-token channel act-sample) -> pilot screen -> Stage 1 (per-head write) -> Stage 2 (content
transport) -> causal cells, in a single process (one model load). Pre-registered in
../PREREGISTRATION.md + Amendment 1; per-unit saves (compute-ordering).

VALIDATE=1 -> tiny model + a few items: the full-integration smoke that the pod runs BEFORE the real
pass (feedback_test_gates_before_gpu). Real run: OLMo first, then Llama (comparative prediction), Qwen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_P5 = HERE.parents[1] / "5_moral_alignment" / "scripts"
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(_P5))
sys.path.insert(0, str(HERE.parents[1] / "6_cross_model" / "scripts"))
sys.path.insert(0, str(HERE.parents[1] / "d2_decision_coupling" / "scripts"))

from deepsteer.directions import extraction as du  # noqa: E402
import stage1_attribution as s1  # noqa: E402
import stage2_content as s2  # noqa: E402
import causal_cells as cc  # noqa: E402
import patch_stimuli as ps  # noqa: E402
from informat_ladder import extract_positions, pooled_chat_actsample, find_content_span, POS_CLASSES  # noqa: E402
from heretic_ablation import last_token_means  # noqa: E402

_unit = s1._unit


def spans_fn(model, templated_text):
    """Position classes for Stage 2: t_inst (last instruction-content tokens), content, template."""
    tok = model.tokenizer
    full = tok(templated_text, add_special_tokens=False)["input_ids"]
    # locate the user content span inside the templated ids (informat convention)
    # (approximate: the content is the longest run before the assistant tail)
    n = len(full)
    return {"t_inst": list(range(max(0, n - 6), n - 3)), "content": list(range(1, max(1, n - 6))),
            "template": [0] + list(range(n - 3, n))}


def extract_inputs(model, layer, moral_pairs, refusal_prompts, n_cap):
    """refusal (last-input-token chat diff @ layer), harm (mean_content diff @ layer), V_moral
    mean_content basis, decision-token channel act-sample."""
    harmful = refusal_prompts["harmful"][:n_cap] if n_cap else refusal_prompts["harmful"]
    harmless = refusal_prompts["harmless"][:n_cap] if n_cap else refusal_prompts["harmless"]
    hm = last_token_means(model, harmful, "chat", [layer])[layer]
    sm = last_token_means(model, harmless, "chat", [layer])[layer]
    refusal = _unit(hm - sm)
    # harm direction at content position (t_inst analog): mean_content diff of harmful vs harmless
    hpos = extract_positions(model, [(h, s) for h, s in zip(harmful, harmless)], [layer])
    harm = _unit(hpos["mean_content"][layer][0])
    # V_moral at mean_content (the position-valid content subspace)
    src = {k: extract_positions(model, v, [layer]) for k, v in moral_pairs.items()}
    Vbasis = s2._ortho([_unit(src[k]["mean_content"][layer][0]) for k in moral_pairs])
    # decision-token (final_pre_assistant) channel act-sample
    texts = [t for v in moral_pairs.values() for pair in v[:20] for t in pair]
    act = pooled_chat_actsample(model, texts, [layer])["final_pre_assistant"][layer]
    return {"refusal": refusal, "harm": harm, "Vbasis": Vbasis, "channel_act": act}


def main() -> None:
    ap = argparse.ArgumentParser(description="C1 one-model session (Stage 1+2 + cells).")
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--key", default="olmo3")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--dataset", default=str(HERE.parents[2] / "deepsteer" / "datasets" / "d1_vmoral_v1.json"))
    ap.add_argument("--fables", default=str(HERE.parents[1] / "d1_moral_subspace" / "outputs" / "full" / "fables_train_full.json"))
    ap.add_argument("--refusal-prompts", default=str(_P5.parent / "refusal_prompts.json"))  # paper5 root
    ap.add_argument("--out", default=str(HERE.parent / "outputs"))
    args = ap.parse_args()

    validate = os.environ.get("VALIDATE") == "1"
    if validate:
        args.model, args.layer = "allenai/OLMo-2-0425-1B", 8
    n_cap = 12 if validate else None

    from informat_ladder import load_moral_pairs
    moral = load_moral_pairs(args.dataset, args.fables, n_cap)
    rp = json.loads(Path(args.refusal_prompts).read_text())
    model = du.load_whitebox(args.model)

    inp = extract_inputs(model, args.layer, moral, rp, n_cap or 60)

    # pilot screen (behavioral-discrimination) on the typed stimuli
    manifest = ps.build_manifest()
    screened = ps.screen(model, manifest)

    # Stage 1: per-head write attribution at the decision token, on a harmful request.
    Qc = s1.channel_control_basis(inp["channel_act"], inp["refusal"])
    probe = (screened["request_twins"] or manifest["request_twins"]["pairs"])[0]["violating"]
    st1 = s1.attribute(model, probe, inp["refusal"], args.layer)
    spec = s1.head_specificity(st1["per_head_contrib"], inp["refusal"], Qc)
    ks = s1.kselect(spec)
    mf = s1.mlp_fraction(list(st1["head_writes"].values()), list(st1["mlp_writes"].values()))
    top_heads = [tuple(h) for h in ks["ranked_heads"][:ks["k"]]]

    # Stage 2: what those heads read (skip if the Jacobian branch fired).
    st2 = {}
    if not mf["jacobian_branch"]:
        rng = np.random.default_rng(0)
        null95 = s2.value_null_q95(inp["channel_act"] - inp["channel_act"].mean(0), inp["Vbasis"], rng)
        band_min = min(s2._frac(inp["Vbasis"], inp["Vbasis"][:, j]) for j in range(inp["Vbasis"].shape[1]))
        s2pass = s2.stage2_pass(model, probe, [h for h in top_heads if h[0] == args.layer],
                                args.layer, spans_fn)
        for h, r in s2pass.items():
            vs = s2.value_side(r["read_vec"], inp["Vbasis"], inp["harm"])
            st2[str(h)] = {"source_dist": r["source_dist"], **vs,
                           "classify": s2.classify_head(r["source_dist"], vs, band_min, null95)}

    # ---- causal cells (a) + (b) + transport positive control (Amendment 1; projection readouts) ----
    rng2 = np.random.default_rng(0)
    tw = (screened["compositional_twins"] or manifest["compositional_twins"]["pairs"])
    tw_pairs = [(p["moral"], p["neutral_or_violating"]) for p in tw[:(6 if validate else 40)]]
    # judgment-decision direction = twin moral-status contrast at the decision token (transport readout)
    jdir = (_unit(extract_positions(model, tw_pairs, [args.layer])["final_pre_assistant"][args.layer][0])
            if tw_pairs else inp["harm"])
    rt = (screened["request_twins"] or manifest["request_twins"]["pairs"])[:(6 if validate else None)]
    full_d, restr_d, tc_d = [], [], []
    for p in rt:
        try:
            sp, tp = cc.patch_positions(model, p["following"], p["violating"])
        except ValueError:
            continue
        base = cc.baseline_proj(model, p["violating"], args.layer, inp["refusal"])
        full_d.append(cc.interchange(model, p["following"], p["violating"], sp, tp, args.layer,
                                     inp["refusal"]) - base)                              # cell (a)/full
        restr_d.append(cc.interchange(model, p["following"], p["violating"], sp, tp, args.layer,
                                      inp["refusal"], restrict_Q=inp["Vbasis"]) - base)   # cell (b) restricted
    for p in tw_pairs[:(4 if validate else 20)]:                                          # transport control
        try:
            sp, tp = cc.patch_positions(model, p[0], p[1])
        except ValueError:
            continue
        jbase = cc.baseline_proj(model, p[1], args.layer, jdir)
        tc_d.append(cc.interchange(model, p[0], p[1], sp, tp, args.layer, jdir,
                                   restrict_Q=inp["Vbasis"]) - jbase)
    mde_ref = cc.mde_bootstrap(full_d + restr_d, rng2) if len(full_d + restr_d) > 2 else float("inf")
    mde_jud = cc.mde_bootstrap(tc_d, rng2) if len(tc_d) > 2 else float("inf")
    cells = {"n_request_twins": len(full_d), "n_twins_transport": len(tc_d),
             "cell_a_full_refusal_delta_mean": round(float(np.mean(full_d)), 4) if full_d else None,
             "cell_b_restricted_refusal_delta_mean": round(float(np.mean(restr_d)), 4) if restr_d else None,
             "transport_control_judgment_delta_mean": round(float(np.mean(tc_d)), 4) if tc_d else None,
             "mde_refusal": round(mde_ref, 4), "mde_judgment": round(mde_jud, 4),
             "cell_b_verdict": cc.cell_b_verdict(float(np.mean(full_d)) if full_d else 0.0,
                                                 float(np.mean(restr_d)) if restr_d else 0.0,
                                                 float(np.mean(tc_d)) if tc_d else 0.0,
                                                 mde_ref, mde_jud)}

    result = {"model": args.model, "key": args.key, "layer": args.layer,
              "reconstruction": st1["reconstruction"], "reconstruction_ok": st1["reconstruction_ok"],
              "mlp": mf, "k": ks["k"], "channel_dim": int(Qc.shape[1]),
              "top_heads": [{"head": list(h), **spec[h]} for h in top_heads],
              "sparsity_curve": ks["sparsity_curve"], "stage2": st2,
              "screen_counts": screened["counts"], "cells": cells,
              "note": "cells (c) XSTest generalization + (d) mean/resample head ablation deferred to "
                      "the analysis follow-up; (a)/(b)/transport-control run here (projection readouts)"}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"c1_inputs_{args.key}.npz", refusal=inp["refusal"], harm=inp["harm"],
             Vbasis=inp["Vbasis"], channel_act=inp["channel_act"], layer=args.layer,
             cell_full_deltas=np.array(full_d), cell_restricted_deltas=np.array(restr_d),
             transport_control_deltas=np.array(tc_d))  # per-unit saves (compute-ordering)
    (out / f"c1_session_{args.key}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("sparsity_curve",)}, indent=2))
    if not st1["reconstruction_ok"]:
        print(f"WARNING: reconstruction {st1['reconstruction']} < {s1.RECON_FLOOR} -> escalate LN")


if __name__ == "__main__":
    main()
