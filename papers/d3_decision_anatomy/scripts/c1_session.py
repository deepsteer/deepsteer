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
import sweep as sw  # noqa: E402
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
    refusal_raw = np.asarray(inp["refusal"], float).copy()   # attribute()/fold/reconstruction stay RAW

    # STANDARDIZE (ANOMALIES A1): rebind geometry directions to per-dim-standardized (zscore) or
    # top-k-projection-out (projout) space so the massive-activation outlier dims cannot dominate the
    # V_moral restriction, the channel null, or the readout on Llama/Qwen/GPT-OSS. `factor` (per-dim
    # multiplier) standardizes vectors; `sigma` is threaded to the cells (patch restriction in the
    # standardized frame + standardized readout). No-op on OLMo up to the #17 invariance.
    standardize = os.environ.get("STANDARDIZE") == "1"
    robustify = os.environ.get("ROBUSTIFY", "zscore")
    factor = None; sigma = None; std_meta = None
    if standardize:
        var = inp["channel_act"].var(0)
        if robustify == "projout":
            big = var / (var.sum() + 1e-12) > 0.05
            factor = np.where(big, 0.0, 1.0)                 # zero the >5%-variance dims
        else:
            sig = np.sqrt(var); sig = np.where(sig > 1e-8, sig, 1.0)
            factor = 1.0 / sig; sigma = sig                  # zscore; sigma threads to the cells
        _oc = lambda M: np.linalg.qr(M)[0][:, :M.shape[1]]   # re-orthonormalize columns in the new frame
        q95_raw = s2.value_null_q95(inp["channel_act"] - inp["channel_act"].mean(0), inp["Vbasis"],
                                    np.random.default_rng(0))
        inp["refusal"] = inp["refusal"] * factor
        inp["harm"] = inp["harm"] * factor
        inp["Vbasis"] = _oc(inp["Vbasis"] * factor[:, None])
        inp["channel_act"] = inp["channel_act"] * factor
        q95_std = s2.value_null_q95(inp["channel_act"] - inp["channel_act"].mean(0), inp["Vbasis"],
                                    np.random.default_rng(0))
        ev = inp["channel_act"].var(0)
        std_meta = {"robustify": robustify, "null_q95_raw": round(float(q95_raw), 4),
                    "null_q95_std": round(float(q95_std), 4),
                    "post_std_participation_ratio": round(float((ev.sum() ** 2) / (ev ** 2).sum()), 2),
                    "n_dims_projected_out": int((factor == 0).sum()) if robustify == "projout" else 0}
    _std = (lambda v: np.asarray(v, float) * factor) if standardize else (lambda v: v)

    # pilot screen (behavioral-discrimination) on the typed stimuli
    manifest = ps.build_manifest()
    screened = ps.screen(model, manifest)

    # Stage 1: per-head write attribution at the decision token, on a harmful request.
    Qc = s1.channel_control_basis(inp["channel_act"], inp["refusal"])
    probe = (screened["request_twins"] or manifest["request_twins"]["pairs"])[0]["violating"]
    st1 = s1.attribute(model, probe, refusal_raw, args.layer)   # RAW: fold + reconstruction in raw space
    contribs_eff = {h: _std(c) for h, c in st1["per_head_contrib"].items()}
    spec = s1.head_specificity(contribs_eff, inp["refusal"], Qc)   # standardized specificity
    ks = s1.kselect(spec)
    mf = s1.mlp_fraction(list(st1["head_writes"].values()), list(st1["mlp_writes"].values()))
    top_heads = [tuple(h) for h in ks["ranked_heads"][:ks["k"]]]

    # Stage 2: what those heads read, at EACH writer's own layer (Amendment 3 per-layer coverage;
    # V_moral/harm are extracted at args.layer, so earlier-layer reads are compared in the shared
    # residual-stream basis -- a cross-layer approximation, flagged in the result note).
    st2 = {}
    if not mf["jacobian_branch"]:
        rng = np.random.default_rng(0)
        null95 = s2.value_null_q95(inp["channel_act"] - inp["channel_act"].mean(0), inp["Vbasis"], rng)
        band_min = min(s2._frac(inp["Vbasis"], inp["Vbasis"][:, j]) for j in range(inp["Vbasis"].shape[1]))
        s2pass = s2.stage2_pass(model, probe, top_heads, args.layer, spans_fn)
        for h, r in s2pass.items():
            vs = s2.value_side(_std(r["read_vec"]), inp["Vbasis"], inp["harm"])   # standardized read
            st2[str(h)] = {"source_dist": r["source_dist"], **vs, "at_read_layer": bool(h[0] == args.layer),
                           "classify": s2.classify_head(r["source_dist"], vs, band_min, null95)}

    # ---- causal cells: refusal (request-twins) + judgment (compositional twins) interchange patches ----
    # Amendment 3: full/restricted/COMPLEMENT/harm-restricted/random-control on refusal; full->judgment
    # (RIDER 0) + restricted->judgment (transport) on judgment; ratio-of-ratios verdict; behavioral
    # generate-under-patch; L15 H15 anti-refusal discriminator.
    from b1_judgment_direction import is_refusal  # noqa: E402
    rng2 = np.random.default_rng(0)
    hidden = inp["refusal"].shape[0]
    Qrand = cc.random_ortho_basis(hidden, inp["Vbasis"].shape[1], rng2, exclude_Q=inp["Vbasis"])
    harmQ = _unit(inp["harm"])[:, None]                     # unit harm as a rank-1 basis (std or raw frame)
    Qhp = sw.harm_partial_basis(inp["Vbasis"], inp["harm"])   # V_moral with d_harm projected out (Amendment 4)
    tw = (screened["compositional_twins"] or manifest["compositional_twins"]["pairs"])
    tw_cap = (6 if validate else 120)                       # use all screened twins (>=2x transport headroom)
    tw_pairs = [(p["moral"], p["neutral_or_violating"]) for p in tw[:tw_cap]]
    jdir = (_unit(_std(extract_positions(model, tw_pairs, [args.layer])["final_pre_assistant"][args.layer][0]))
            if tw_pairs else _unit(inp["harm"]))            # judgment readout in the standardized frame
    rt = (screened["request_twins"] or manifest["request_twins"]["pairs"])[:(6 if validate else None)]

    # Amendment 4 rank sweep (SWEEP=1 or validate): nested moral-contrast PCA basis + per-rank restrict.
    KS = [1, 3, 8, 16]
    run_sweep = os.environ.get("SWEEP") == "1" or validate
    sweep_bases, sweep_rand_bases, sweep_meta = None, None, None
    if run_sweep:
        all_pairs = [pr for v in moral.values() for pr in v]
        contrasts = _std(extract_positions(model, all_pairs, [args.layer])["mean_content"][args.layer][1])
        sweep_bases = sw.nested_pca_basis(contrasts, KS)     # PCA basis in the standardized frame
        sweep_rand_bases = {k: cc.random_ortho_basis(hidden, sweep_bases[k].shape[1], rng2,
                                                     exclude_Q=inp["Vbasis"]) for k in KS}
        sweep_meta = {"purity_k": {k: round(sw.subspace_purity(sweep_bases[k], contrasts.mean(0)), 4)
                                   for k in KS},
                      "cos_harm_pc": sw.cos_harm_components(sweep_bases[max(KS)], inp["harm"], n=8),
                      "ranks": {k: int(sweep_bases[k].shape[1]) for k in KS}}
    sweep_ref = {k: [] for k in KS}; sweep_rand = {k: [] for k in KS}; sweep_jud = {k: [] for k in KS}

    full_d, restr_d, compl_d, harm_d, rand_d, hp_d = [], [], [], [], [], []   # all read refusal, request-twins
    for p in rt:
        try:
            sp, tp = cc.patch_positions(model, p["following"], p["violating"])
            L, r = args.layer, inp["refusal"]
            base = cc.baseline_proj(model, p["violating"], L, r, sigma=sigma)
            ic = lambda **kw: cc.interchange(model, p["following"], p["violating"], sp, tp, L, r,
                                             sigma=sigma, **kw) - base
            full_d.append(ic())
            restr_d.append(ic(restrict_Q=inp["Vbasis"]))
            compl_d.append(ic(restrict_Q=inp["Vbasis"], restrict_mode="complement"))
            harm_d.append(ic(restrict_Q=harmQ))
            rand_d.append(ic(restrict_Q=Qrand))
            hp_d.append(ic(restrict_Q=Qhp))                                  # V_moral perp harm
            if run_sweep:
                for k in KS:
                    sweep_ref[k].append(ic(restrict_Q=sweep_bases[k]))
                    sweep_rand[k].append(ic(restrict_Q=sweep_rand_bases[k]))
        except (ValueError, RuntimeError, IndexError):
            continue

    full_jud_d, tc_d = [], []                              # full->judgment (rider 0) + restricted->judgment
    for p in tw_pairs:
        try:
            sp, tp = cc.patch_positions(model, p[0], p[1])
            jbase = cc.baseline_proj(model, p[1], args.layer, jdir, sigma=sigma)
            jc = lambda **kw: cc.interchange(model, p[0], p[1], sp, tp, args.layer, jdir,
                                             sigma=sigma, **kw) - jbase
            full_jud_d.append(jc())
            tc_d.append(jc(restrict_Q=inp["Vbasis"]))
            if run_sweep:
                for k in KS:
                    sweep_jud[k].append(jc(restrict_Q=sweep_bases[k]))
        except (ValueError, RuntimeError, IndexError):
            continue

    # Amendment 4 severity ladder: psychometric curve + operating band -> behavioral out of the floor.
    from deepsteer.datasets import get_severity_twins  # noqa: E402
    sev = get_severity_twins()[:(10 if validate else None)]
    psycho = cc.severity_psychometric(model, sev, is_refusal)
    band_pairs = [(foll, viol) for (_f, lvl, foll, viol) in sev if lvl in psycho["operating_band"]]
    behav_src = band_pairs or [(p["following"], p["violating"]) for p in rt]   # fall back if band empty

    # behavioral generate-under-patch: does the full patch flip a baseline refusal to compliance?
    behav = {"n": 0, "base_refusals": 0, "flips_to_comply": 0,
             "source": "operating_band" if band_pairs else "request_twins_fallback"}
    for foll, viol in behav_src[:(3 if validate else 10)]:
        try:
            sp, tp = cc.patch_positions(model, foll, viol)
            base_ref = is_refusal(cc.generate_plain(model, viol))
            patched_ref = is_refusal(cc.generate_under_patch(model, foll, viol, sp, tp, args.layer))
            behav["n"] += 1
            behav["base_refusals"] += int(base_ref)
            behav["flips_to_comply"] += int(base_ref and not patched_ref)
        except (ValueError, RuntimeError, IndexError):
            continue

    # L15 H15-style anti-refusal discriminator: mean-ablate the most-negative-specificity top head;
    # on benign (norm-following) requests it should INCREASE over-refusal (needs out-of-floor benign).
    anti = min(top_heads, key=lambda h: spec[h]["specificity"])
    disc = None
    if spec[anti]["specificity"] < 0:
        benign = [foll for foll, _ in behav_src][:(4 if validate else 12)]
        disc = cc.head_mean_ablation_refusal_rate(model, benign, anti[0], anti[1], is_refusal)

    mde_ref = cc.mde_bootstrap(full_d + restr_d, rng2) if len(full_d + restr_d) > 2 else float("inf")
    mde_jud = cc.mde_bootstrap(full_jud_d + tc_d, rng2) if len(full_jud_d + tc_d) > 2 else float("inf")
    _m = lambda a: round(float(np.mean(a)), 4) if a else None
    ror = (cc.ratio_of_ratios(full_d, restr_d, full_jud_d, tc_d, rng2)
           if len(full_d) > 2 and len(full_jud_d) > 2 else None)
    additivity = sw.additivity_ratio(full_d, restr_d, compl_d, rng2) if len(full_d) > 2 else None
    harm_ci = None
    if len(harm_d) > 2:
        ha = np.asarray(harm_d)
        hb = [ha[rng2.integers(0, len(ha), len(ha))].mean() for _ in range(2000)]
        harm_ci = [round(float(np.percentile(hb, 2.5)), 4), round(float(np.percentile(hb, 97.5)), 4)]
    sweep_result = None
    if run_sweep and len(full_d) > 2 and len(full_jud_d) > 2:
        fdm, fjm = float(np.mean(full_d)), float(np.mean(full_jud_d))
        Rref = {k: float(np.mean(sweep_ref[k]) / fdm) for k in KS}
        Rjud = {k: float(np.mean(sweep_jud[k]) / fjm) for k in KS}
        Rrand = {k: float(np.mean(sweep_rand[k]) / fdm) for k in KS}
        harm_r1 = float(np.mean(harm_d) / fdm) if harm_d else 0.0
        sweep_result = {"ks": KS, "R_refusal_k": {k: round(Rref[k], 4) for k in KS},
                        "R_judgment_k": {k: round(Rjud[k], 4) for k in KS},
                        "random_null_k": {k: round(Rrand[k], 4) for k in KS},
                        "harm_rank1_R": round(harm_r1, 4), **sweep_meta,
                        "shape_verdict": sw.shape_verdict(Rref, Rjud, harm_r1, KS)}
    cells = {"n_request_twins": len(full_d), "n_twins_transport": len(tc_d),
             "cell_a_full_refusal_delta_mean": _m(full_d),
             "cell_b_restricted_refusal_delta_mean": _m(restr_d),
             "complement_refusal_delta_mean": _m(compl_d),          # expect: moves refusal
             "harm_restricted_refusal_delta_mean": _m(harm_d),      # expect: moves refusal if reads harm
             "harm_partialed_refusal_delta_mean": _m(hp_d),         # V_moral perp harm; ~0 if V_moral effect IS harm
             "random_rank3_refusal_delta_mean": _m(rand_d),         # control: expect ~0
             "full_judgment_delta_mean": _m(full_jud_d),            # RIDER 0
             "transport_control_judgment_delta_mean": _m(tc_d),
             "mde_refusal": round(mde_ref, 4), "mde_judgment": round(mde_jud, 4),
             "ratio_of_ratios": ror,                                # Amendment 3 PRIMARY verdict
             "additivity": additivity, "harm_rank1_ci": harm_ci,   # Amendment 4 identification
             "sweep": sweep_result,                                 # Amendment 4 rank sweep + shape verdict
             "cell_b_verdict_absolute": cc.cell_b_verdict(         # the Amendment-1 absolute gate (continuity)
                 float(np.mean(full_d)) if full_d else 0.0, float(np.mean(restr_d)) if restr_d else 0.0,
                 float(np.mean(tc_d)) if tc_d else 0.0, mde_ref, mde_jud),
             "severity_psychometric": psycho,                      # Amendment 4 dose-response + operating band
             "behavioral_generate_under_patch": behav,
             "anti_refusal_discriminator": disc}

    result = {"model": args.model, "key": args.key, "layer": args.layer,
              "reconstruction": st1["reconstruction"], "reconstruction_ok": st1["reconstruction_ok"],
              "reordered_norm": st1.get("reordered_norm", False),
              "standardize": std_meta,                               # A1 null de-saturation (if STANDARDIZE=1)
              "mlp": mf, "k": ks["k"], "channel_dim": int(Qc.shape[1]),
              "top_heads": [{"head": list(h), **spec[h]} for h in top_heads],
              "sparsity_curve": ks["sparsity_curve"], "stage2": st2,
              "screen_counts": screened["counts"], "cells": cells,
              "note": "Amendment 3: ratio_of_ratios is the PRIMARY verdict (full->judgment = rider 0, now "
                      "logged); complement/harm/random cells + behavioral flips + anti-refusal "
                      "discriminator run here (projection readouts + generate-under-patch). Stage-2 "
                      "earlier-layer reads use the shared residual basis (cross-layer approximation)."}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ph_keys = list(st1["per_head_contrib"].keys())          # per-head arrays for the standardized rider
    ph_arr = np.stack([st1["per_head_contrib"][k] for k in ph_keys]) if ph_keys else np.zeros((0, hidden))
    save = dict(refusal=inp["refusal"], harm=inp["harm"], Vbasis=inp["Vbasis"],
                channel_act=inp["channel_act"], layer=args.layer,
                cell_full_deltas=np.array(full_d), cell_restricted_deltas=np.array(restr_d),
                cell_complement_deltas=np.array(compl_d), cell_harm_deltas=np.array(harm_d),
                cell_harm_partial_deltas=np.array(hp_d), cell_random_deltas=np.array(rand_d),
                full_judgment_deltas=np.array(full_jud_d), transport_control_deltas=np.array(tc_d),
                per_head_contribs=ph_arr, per_head_keys=np.array(ph_keys))  # standardized-invariance rider
    if run_sweep:                                            # per-rank paired deltas (rows = KS)
        save["sweep_ks"] = np.array(KS)
        save["sweep_refusal"] = np.array([sweep_ref[k] for k in KS])
        save["sweep_judgment"] = np.array([sweep_jud[k] for k in KS])
        save["sweep_random"] = np.array([sweep_rand[k] for k in KS])
    np.savez(out / f"c1_inputs_{args.key}.npz", **save)
    (out / f"c1_session_{args.key}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("sparsity_curve",)}, indent=2))
    if not st1["reconstruction_ok"]:
        print(f"WARNING: reconstruction {st1['reconstruction']} outside "
              f"[{s1.RECON_FLOOR}, {s1.RECON_CEIL}] (reordered_norm={st1.get('reordered_norm')}) "
              f"-> Stage-1/2 anatomy is un-folded; verdict cell is unaffected")


if __name__ == "__main__":
    main()
