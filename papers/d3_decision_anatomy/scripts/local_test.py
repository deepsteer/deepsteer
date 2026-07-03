#!/usr/bin/env python3
"""Local gate for C1 typing prep (zero-GPU): request-twins, manifest typing + alignment, and the
behavioral-discrimination screen's gap-logic (via a stub model). No model load.
Run: python papers/d3_decision_anatomy/scripts/local_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1] / "d2_decision_coupling" / "scripts"))

import numpy as np  # noqa: E402

import patch_stimuli as ps  # noqa: E402
import stage1_attribution as s1  # noqa: E402
import stage2_content as s2  # noqa: E402
import causal_cells as cc  # noqa: E402
from deepsteer.datasets import get_request_twins  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


class StubModel:
    """Returns a scripted completion per prompt substring (for the screen gap-logic)."""
    def __init__(self, rules):
        self.rules = rules

    def generate(self, prompt, **kw):
        text = next((v for k, v in self.rules.items() if k in prompt), "Sure, here is how.")
        return type("R", (), {"text": text})()


def test_request_twins():
    print("request-twins:")
    ts = get_request_twins()
    check("count >= 20", len(ts) >= 20, f"n={len(ts)}")
    check("all triples (foundation, following, violating)", all(len(t) == 3 for t in ts))
    check("following != violating", all(a != b for _, a, b in ts))
    check("nontrivial shared prefix (>=20 chars) each pair",
          all(ps._align(a, b)["shared_prefix_chars"] >= 20 for _, a, b in ts))
    from collections import Counter
    fnd = Counter(f for f, _, _ in ts)
    check("spans all 6 MFT foundations", len(fnd) == 6, str(dict(fnd)))


def test_manifest():
    print("typed manifest (construct-audit type blocks + alignment):")
    m = ps.build_manifest()
    tw, rq, xs = m["compositional_twins"], m["request_twins"], m["xstest_generalization"]
    check("twins outcome=judgment + refusal EXPECTED-FLAT",
          tw["type_block"]["outcome_variable"] == "judgment-decision readout"
          and "EXPECTED-FLAT" in tw["type_block"]["role"])
    check("request-twins outcome=refusal + MINIMAL-PAIR role",
          rq["type_block"]["outcome_variable"].startswith("refusal")
          and "MINIMAL-PAIR" in rq["type_block"]["role"])
    check("xstest typed as GENERALIZATION (not minimal pair)",
          "GENERALIZATION" in xs["type_block"]["role"])
    check("every type block carries source_dataset + extraction_commit",
          all(m[k]["type_block"].get("source_dataset") and m[k]["type_block"].get("extraction_commit")
              for k in m))
    al = rq["pairs"][0]["align"]
    check("alignment: shared prefix + distinct flipped spans",
          al["shared_prefix_chars"] > 10
          and al["flipped_span_following"] != al["flipped_span_violating"])


def test_screen_logic():
    print("behavioral-discrimination screen gap-logic (stub model):")
    from b1_judgment_direction import build_prompt  # noqa: F401 (import path sanity)
    # judgment flips: model says wrong for the moral action, acceptable for the neutral one.
    jm = StubModel({"protect": "acceptable", "humiliate": "wrong"})
    check("judgment flip detected across a twin",
          ps._judgment_flips(jm, "She lied to protect", "She lied to humiliate"))
    jm_flat = StubModel({"protect": "acceptable", "humiliate": "acceptable"})
    check("no judgment flip -> pair dropped", not ps._judgment_flips(jm_flat,
          "She lied to protect", "She lied to humiliate"))
    # refusal differs: comply on following, refuse on violating.
    rm = StubModel({"supported": "Sure, here is how.", "worse": "I'm sorry, I can't help with that."})
    check("refusal difference detected across a request-twin",
          ps._refusal_differs(rm, "help ... feels supported", "help ... feels worse"))
    rm_flat = StubModel({"supported": "Sure!", "worse": "Sure!"})
    check("no refusal difference -> pair dropped",
          not ps._refusal_differs(rm_flat, "help ... supported", "help ... worse"))


def test_stage1_math():
    print("Stage-1 attribution math (synthetic linear decomposition):")
    rng = np.random.default_rng(0)
    d = 64
    r = s1._unit(rng.standard_normal(d))
    # a ~10-dim channel subspace, then activations living in it
    B, _ = np.linalg.qr(rng.standard_normal((d, 10)))
    X = (rng.standard_normal((50, 10)) @ B.T) * 3.0
    Qc = s1.channel_control_basis(X, r)
    check("channel basis orthonormal", np.allclose(Qc.T @ Qc, np.eye(Qc.shape[1]), atol=1e-6))
    check("refusal dir projected out of channel basis (|Qc^T r|~0)",
          float(np.linalg.norm(Qc.T @ r)) < 1e-6, f"{float(np.linalg.norm(Qc.T @ r)):.2e}")

    # per-head contributions: 2 heads aligned with r_hat (strong writers), 6 broadly-in-channel,
    # plus embed + mlp; residual is their exact linear sum.
    contribs = {}
    contribs[(0, 0)] = 2.0 * r
    contribs[(0, 1)] = 1.5 * r
    for h in range(6):
        contribs[(1, h)] = 0.8 * B[:, h]                      # channel writers (not r_hat)
    mlp = {5: 0.3 * B[:, 0]}
    embed = 0.1 * rng.standard_normal(d)
    resid = sum(contribs.values()) + sum(mlp.values()) + embed
    target = float(resid @ r)
    writes = [float(c @ r) for c in contribs.values()] + [float(m @ r) for m in mlp.values()] + [float(embed @ r)]
    check("reconstruction ~ 1.0 (residual is linear)",
          abs(s1.reconstruction_ratio(writes, target) - 1.0) < 1e-9,
          f"{s1.reconstruction_ratio(writes, target):.6f}")

    spec = s1.head_specificity(contribs, r, Qc)
    check("r_hat-aligned head has top specificity",
          spec[(0, 0)]["specificity"] > spec[(1, 0)]["specificity"]
          and spec[(0, 0)]["specificity"] > spec[(1, 3)]["specificity"])
    check("channel-writer specificity is small (penalized by channel_mean)",
          abs(spec[(1, 0)]["specificity"]) < spec[(0, 0)]["specificity"])
    ks = s1.kselect(spec)
    check("k-selection returns the two strong writers first + caps",
          ks["ranked_heads"][0] == (0, 0) and ks["ranked_heads"][1] == (0, 1) and ks["k"] <= s1.K_CAP)

    mf = s1.mlp_fraction([2.0, 1.5], [0.1])
    check("mlp_fraction low -> no jacobian branch", not mf["jacobian_branch"])
    mf2 = s1.mlp_fraction([0.2], [1.0, 1.0])
    check("mlp_fraction > 0.5 -> jacobian branch", mf2["jacobian_branch"])


def test_reordered_norm_fold():
    print("Reordered-norm fold + two-sided reconstruction gate (A3):")
    # two-sided gate: an un-folded block norm shows up as OVERSHOOT, which must fail.
    check("gate accepts recon ~1.0", s1.reconstruction_ok(1.0) and s1.reconstruction_ok(0.95))
    check("gate rejects 3.05 overshoot (the OLMo un-folded case)", not s1.reconstruction_ok(3.05))
    check("gate rejects undershoot 0.5", not s1.reconstruction_ok(0.5))

    rng = np.random.default_rng(7)
    d = 64
    r = s1._unit(rng.standard_normal(d))
    # per-head pre-norm writes whose sum is the attention output A (the RMSNorm input).
    contribs = {h: rng.standard_normal(d) * (2.0 if h < 2 else 0.5) for h in range(8)}
    A = np.sum(list(contribs.values()), axis=0)
    weight, eps = rng.standard_normal(d) * 0.5 + 1.0, 1e-6
    normed = weight * A / np.sqrt(np.mean(A ** 2) + eps)          # reference RMSNorm(A) = true write
    g = s1.rms_gain(A, weight, eps)
    folded_sum_write = sum(float((c * g) @ r) for c in contribs.values())
    check("raw per-head sum OVERSHOOTS the normed write",
          abs(sum(float(c @ r) for c in contribs.values())) > 1.3 * abs(float(normed @ r)))
    check("folded per-head writes reconstruct <norm(A), r> exactly",
          abs(folded_sum_write - float(normed @ r)) < 1e-9,
          f"{folded_sum_write:.6f} vs {float(normed @ r):.6f}")


def test_stage2_math():
    print("Stage-2 content-transport math (synthetic):")
    rng = np.random.default_rng(1)
    d = 48
    # spans: 3 t_inst positions, 4 content, 2 template
    spans = {"t_inst": [0, 1, 2], "content": [3, 4, 5, 6], "template": [7, 8]}
    attn = np.zeros(9); attn[[0, 1, 2]] = 0.8 / 3; attn[[3, 4, 5, 6]] = 0.15 / 4; attn[[7, 8]] = 0.05 / 2
    sd = s2.source_distribution(attn, spans)
    check("source dist sums ~1 + t_inst plurality", abs(sum(sd[c] for c in spans) - 1.0) < 1e-6
          and sd["plurality"] == "t_inst", str(sd))
    Vb, _ = np.linalg.qr(rng.standard_normal((d, 3)))
    harm = s1._unit(rng.standard_normal(d))
    vs_moral = s2.value_side(Vb @ rng.standard_normal(3), Vb, harm)      # read vec IN V_moral
    vs_harm = s2.value_side(harm * 2.0, Vb, harm)                        # read vec = harm dir
    check("read-in-Vmoral loads Vmoral high", vs_moral["vmoral_frac"] > 0.9)
    check("read=harm loads harm high, Vmoral low", vs_harm["harm_abs_cos"] > 0.99
          and vs_harm["vmoral_frac"] < 0.3)
    cl = s2.classify_head({"plurality": "t_inst"}, {"vmoral_frac": 0.10, "harm_abs_cos": 0.60},
                          band_min=0.5, null_q95=0.3)
    check("copy-head-for-harm classified", cl["copy_head_for_harm"] and cl["label"] == "copy-head-for-harm")
    cl2 = s2.classify_head({"plurality": "content"}, {"vmoral_frac": 0.62, "harm_abs_cos": 0.1},
                           band_min=0.5, null_q95=0.3)
    check("moral-content-reading classified", cl2["moral_content_reading"]
          and cl2["label"] == "moral-content-reading")


def test_causal_logic():
    print("Causal-cell decision logic (Amendment 1):")
    rng = np.random.default_rng(2)
    # cell (b) four branches, mde_refusal=0.05, mde_judgment=0.05
    v1 = cc.cell_b_verdict(0.02, 0.0, 0.2, 0.05, 0.05)   # full doesn't move
    check("no_content_effect when full < MDE", v1["verdict"] == "no_content_effect")
    v2 = cc.cell_b_verdict(0.3, 0.0, 0.01, 0.05, 0.05)   # full moves, restricted can't move judgment
    check("uninformative when transport control fails", v2["verdict"] == "uninformative"
          and not v2["transport_control_passed"])
    v3 = cc.cell_b_verdict(0.3, 0.25, 0.2, 0.05, 0.05)   # restricted moves refusal
    check("vmoral_is_read_substrate when restricted moves refusal",
          v3["verdict"] == "vmoral_is_read_substrate" and v3["transport_control_passed"])
    v4 = cc.cell_b_verdict(0.3, 0.0, 0.2, 0.05, 0.05)    # full moves, restricted moves judgment not refusal
    check("reads_non_vmoral_features (the program-null-explainer)",
          v4["verdict"] == "reads_non_vmoral_features" and v4["transport_control_passed"])
    mde = cc.mde_bootstrap([0.1, -0.05, 0.2, 0.0, 0.15, -0.1], rng)
    check("mde_bootstrap positive + reasonable", 0 < mde < 0.5, f"mde={mde:.3f}")
    ao = cc.ablation_outlier(-0.6, [-0.1, -0.05, 0.0, -0.15, 0.02])
    check("ablation outlier below random floor -> head-specific", ao["head_specific"])
    ao2 = cc.ablation_outlier(-0.08, [-0.6, -0.5, -0.4, -0.3, -0.2])
    check("ablation within floor -> not head-specific", not ao2["head_specific"])


def test_amendment3_cells():
    print("Amendment 3 cells (ratio-of-ratios verdict + control bases):")
    rng = np.random.default_rng(3)

    def arrs(Rr, Rj, n=40, noise=0.01):
        # full effects ~ -0.1 (refusal) / +0.1 (judgment); restricted = R * full + small noise
        fr = -0.1 + noise * rng.standard_normal(n)
        rr = Rr * fr + noise * rng.standard_normal(n)
        fj = 0.1 + noise * rng.standard_normal(n)
        rj = Rj * fj + noise * rng.standard_normal(n)
        return fr, rr, fj, rj

    fr, rr, fj, rj = arrs(0.23, 0.80)                      # V_moral keeps 80% of judgment, 23% of refusal
    v = cc.ratio_of_ratios(fr, rr, fj, rj, rng)
    check("reads_non_vmoral_features when R_judgment >> R_refusal",
          v["verdict"] == "reads_non_vmoral_features" and v["stands"], str(v["verdict"]))
    fr, rr, fj, rj = arrs(0.50, 0.50)                      # equal transfer fractions
    v = cc.ratio_of_ratios(fr, rr, fj, rj, rng)
    check("under_transfer when the ratios are comparable (CI includes 0)",
          v["verdict"] == "under_transfer" and not v["ci_excludes_0"], str(v["verdict"]))
    fr, rr, fj, rj = arrs(0.80, 0.20)                      # refusal keeps MORE V_moral than judgment
    v = cc.ratio_of_ratios(fr, rr, fj, rj, rng)
    check("reads_vmoral_more when R_judgment < R_refusal", v["verdict"] == "reads_vmoral_more", str(v["verdict"]))
    fr, rr, fj, rj = arrs(0.30, 0.40)                      # positive but small (< M_ratio 0.15)
    v = cc.ratio_of_ratios(fr, rr, fj, rj, rng)
    check("small-margin branch when CI excludes 0 but diff < M_ratio",
          v["verdict"] == "reads_non_vmoral_features_small_margin" and not v["stands"], str(v["verdict"]))

    # control bases
    Qr = cc.random_ortho_basis(64, 3, rng)
    check("random_ortho_basis orthonormal rank-3", Qr.shape == (64, 3)
          and np.allclose(Qr.T @ Qr, np.eye(3), atol=1e-6))
    Vb, _ = np.linalg.qr(rng.standard_normal((64, 3)))
    Qx = cc.random_ortho_basis(64, 3, rng, exclude_Q=Vb)
    check("random_ortho_basis excludes the V_moral subspace", float(np.linalg.norm(Vb.T @ Qx)) < 1e-6,
          f"{float(np.linalg.norm(Vb.T @ Qx)):.2e}")
    srcs = [s1._unit(rng.standard_normal(64)) for _ in range(3)]
    check("rankk_moral_basis caps at available source rank",
          cc.rankk_moral_basis(srcs, 2).shape == (64, 2) and cc.rankk_moral_basis(srcs, 8).shape[1] == 3)


def test_sweep():
    print("Amendment 4 rank sweep + harm identification (pure math):")
    import sweep as sw
    rng = np.random.default_rng(5)
    d = 64
    # contrasts with a dominant direction d0 + weaker structure -> PC1 ~ d0, nested bases
    d0 = s1._unit(rng.standard_normal(d))
    d1 = s1._unit(rng.standard_normal(d))
    contrasts = (rng.standard_normal((40, 1)) * 3.0) * d0 + (rng.standard_normal((40, 1))) * d1 \
        + 0.1 * rng.standard_normal((40, d))
    bases = sw.nested_pca_basis(contrasts, [1, 3, 8, 16])
    check("nested_pca_basis shapes + caps", bases[1].shape == (d, 1) and bases[16].shape == (d, 16))
    check("rank-1 basis lies inside rank-3 (nested)",
          float(np.linalg.norm(bases[3].T @ bases[1])) > 0.999)
    check("PC1 aligns with the dominant contrast direction", abs(float(bases[1][:, 0] @ d0)) > 0.9)
    mean_c = contrasts.mean(0)
    p1, p8 = sw.subspace_purity(bases[1], mean_c), sw.subspace_purity(bases[8], mean_c)
    check("purity rises with rank", p8 >= p1 and p8 <= 1.0 + 1e-9, f"{p1:.3f}->{p8:.3f}")
    ch = sw.cos_harm_components(bases[16], d0, n=3)
    check("cos_harm_components returns |cos| in [0,1]", len(ch) == 3 and all(0 <= c <= 1 for c in ch))

    # harm-partialed basis is orthogonal to the harm direction
    V, _ = np.linalg.qr(rng.standard_normal((d, 3)))
    harm = s1._unit(V @ np.array([1.0, 0.5, 0.2]))          # harm inside span(V)
    Qp = sw.harm_partial_basis(V, harm)
    check("harm_partial_basis is orthogonal to d_harm", float(np.linalg.norm(harm @ Qp)) < 1e-6,
          f"{float(np.linalg.norm(harm @ Qp)):.2e}")

    full = list(-0.1 + 0.005 * rng.standard_normal(30))
    add1 = sw.additivity_ratio(full, [0.5 * x for x in full], [0.5 * x for x in full], rng)
    check("additivity ~1 when restricted+complement = full", add1["additive"], str(add1["additivity_ratio"]))
    add2 = sw.additivity_ratio(full, [0.6 * x for x in full], [0.6 * x for x in full], rng)
    check("additivity >1 flagged non-additive (overlap)", not add2["additive"], str(add2["additivity_ratio"]))

    ks = [1, 3, 8, 16]
    v_broad = sw.shape_verdict({1: 0.30, 3: 0.45, 8: 0.65, 16: 0.80}, {1: .4, 3: .6, 8: .8, 16: 0.85}, 0.34, ks)
    check("shape verdict: broad_moral", v_broad["verdict"] == "broad_moral", v_broad["verdict"])
    v_harm = sw.shape_verdict({1: 0.33, 3: 0.34, 8: 0.35, 16: 0.36}, {1: .4, 3: .55, 8: .68, 16: 0.72}, 0.34, ks)
    check("shape verdict: harm_saturating", v_harm["verdict"] == "harm_saturating", v_harm["verdict"])
    v_ceil = sw.shape_verdict({1: 0.20, 3: 0.28, 8: 0.31, 16: 0.33}, {1: .2, 3: .3, 8: .34, 16: 0.36}, 0.30, ks)
    check("shape verdict: instrument_ceiling", v_ceil["verdict"] == "instrument_ceiling", v_ceil["verdict"])


def test_standardize():
    print("STANDARDIZE / A1-robustification (pure math):")
    rng = np.random.default_rng(9)
    d, n = 48, 300
    X = rng.standard_normal((n, d)); X[:, 0] *= 9.0          # dim 0 = massive-activation outlier
    var = X.var(0)
    # projout: factor zeros exactly the >5%-variance dims
    big = var / var.sum() > 0.05
    f_proj = np.where(big, 0.0, 1.0)
    check("projout zeros the outlier dim (dim 0)", f_proj[0] == 0.0 and f_proj[1:].sum() == d - 1,
          f"{int((f_proj == 0).sum())} dim(s) out")
    # zscore de-saturates the covariance null when V_moral loads on the outlier
    Vb, _ = np.linalg.qr(np.column_stack([X[:5].mean(0), rng.standard_normal(d), rng.standard_normal(d)]))
    Vb = Vb[:, :3]
    q_raw = s2.value_null_q95(X - X.mean(0), Vb, np.random.default_rng(0))
    sig = np.sqrt(var); fz = 1.0 / sig
    Xs = X * fz; Vbs = np.linalg.qr(Vb * fz[:, None])[0][:, :3]
    q_std = s2.value_null_q95(Xs - Xs.mean(0), Vbs, np.random.default_rng(0))
    check("zscore de-saturates the covariance null (q95 drops)", q_std < q_raw, f"{q_raw:.3f} -> {q_std:.3f}")
    # standardized-frame subspace patch round-trips: off-subspace RAW component preserved
    sigv = sig
    Q = Vbs[:, :2]                                           # std-frame basis
    cur = rng.standard_normal(d) * sigv; src = rng.standard_normal(d) * sigv
    cur_w, src_w = cur / sigv, src / sigv
    new_w = cur_w - (cur_w @ Q) @ Q.T + (src_w @ Q) @ Q.T    # subspace swap in std frame
    new = new_w * sigv                                       # map back to raw
    off = cur_w - (cur_w @ Q) @ Q.T
    check("std-frame subspace patch preserves the tgt off-subspace part",
          np.allclose((new / sigv) - (src_w @ Q) @ Q.T, off, atol=1e-9))

    # regression parity: c1-style standardized Stage-1 lead head == #17 invariance lead head
    npz = HERE.parent / "outputs" / "c1_inputs_olmo3.npz"
    if npz.exists():
        z = np.load(npz)
        C = z["per_head_contribs"].astype(float); keys = [tuple(int(x) for x in k) for k in z["per_head_keys"]]
        Xc = z["channel_act"].astype(float); r = z["refusal"].astype(float)
        s = np.sqrt(Xc.var(0)); s = np.where(s > 1e-8, s, 1.0); fac = 1.0 / s
        Qc = s1.channel_control_basis(Xc * fac, r * fac)
        spec = s1.head_specificity({h: C[i] * fac for i, h in enumerate(keys)}, r * fac, Qc)
        lead = max(keys, key=lambda h: abs(spec[h]["specificity"]))
        inv = json.loads((HERE.parent / "outputs" / "standardized_invariance_olmo3.json").read_text()) \
            if (HERE.parent / "outputs" / "standardized_invariance_olmo3.json").exists() else None
        if inv:
            check("regression parity: c1 std lead head == #17 std lead head",
                  list(lead) == inv["lead_head_std"], f"{list(lead)} vs {inv['lead_head_std']}")
    else:
        print("  [skip] OLMo npz not present -> regression parity checked on the pod")


def main():
    print("=== C1 typing-prep local gate ===\n")
    test_request_twins()
    test_manifest()
    test_screen_logic()
    test_stage1_math()
    test_reordered_norm_fold()
    test_stage2_math()
    test_causal_logic()
    test_amendment3_cells()
    test_sweep()
    test_standardize()
    print()
    if FAILS:
        print(f"FAILED: {FAILS}"); sys.exit(1)
    print("ALL LOCAL CHECKS PASSED. Behavioral screen runs on MPS/pod: patch_stimuli.py --screen")


if __name__ == "__main__":
    main()
