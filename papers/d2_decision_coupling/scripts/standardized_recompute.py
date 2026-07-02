#!/usr/bin/env python3
"""Zero-GPU standardized recompute of R2/R3/R5 (D2 Amendment 1, committed 5a7f63a, before this ran).

Massive-activation outlier dims (Qwen dim458=59%, Llama dim788=32%) saturate the raw
covariance-matched null. Recompute in a per-dimension-standardized space with format/position-
MATCHED sigma (rider a): the chat-decision-site R3 cell uses sigma from the judgment decision-site
activations (acts_headline); the raw-content R5/band cells use sigma from the raw act_sample. Report
alongside:
  * raw (committed) values, so the OLMo raw->standardized INVARIANCE check (rider b) is visible;
  * a top-k projection-out ROBUSTNESS variant (rider c): drop dims individually > 5% of variance.

Operates entirely on the chunk-1 artifacts already rsync'd back. R2 (judgment x V_moral) stays
FORMAT-CONFOUNDED (chat judgment vs raw V_moral) even standardized -> reported with the caveat; the
valid R2 needs the chat-format in-format ladder (GPU chunk, Amendment 1 §7).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE.parent / "outputs"
K = 2000
SEED = 0
MARGIN_M = 0.05
MODELS = [("olmo3", 16), ("qwen25", 14), ("llama31", 16)]
SRC = ["moral_stories", "fables", "ethics"]
CONTROLS = ["syntax", "register", "sentiment"]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def ortho(dirs):
    Q, _ = np.linalg.qr(np.stack(dirs, axis=1))
    return Q


def frac(Q, v):
    return float(np.linalg.norm(Q.T @ unit(v)))  # unit v -> in-subspace norm fraction


def std_unit(d, sigma):
    return unit(np.asarray(d, np.float64) / sigma)


def topk_mask(var, thresh=0.05):
    """Boolean keep-mask dropping dims individually > thresh of total variance."""
    return var / var.sum() <= thresh


def load(key, layer):
    d = OUT / key
    jd = np.load(d / f"b1_judgment_dir_{key}.npz")
    vm = np.load(d / "vmoral_sources.npz")
    a = {
        "layer": layer,
        "X_raw": np.load(d / "act_sample.npz")["X"].astype(np.float64),
        "acts_chat": jd["acts_headline"].astype(np.float64),   # chat decision-site sample
        "judgment": jd["judgment_dir"].astype(np.float64),
        "label_contrast": jd["label_contrast"].astype(np.float64),
        "refusal": np.load(d / "refusal.npz")["refusal"].astype(np.float64),
        "persona": np.load(d / "persona_direction.npz")[f"persona_layer{layer}"].astype(np.float64),
        "sources": {s: vm[f"{s}_layer{layer}"].astype(np.float64) for s in SRC},
        "controls": {c: np.load(d / f"b3_{c}_dir_{key}.npz")[f"{c}_layer{layer}"].astype(np.float64)
                     for c in CONTROLS},
    }
    return a


def pairwise_null_q95(Ac, rng, k=K):
    n = Ac.shape[0]
    cs = [abs(float(unit(Ac.T @ rng.standard_normal(n)) @ unit(Ac.T @ rng.standard_normal(n))))
          for _ in range(k)]
    return float(np.percentile(cs, 95))


def null_q95_on(Q, Ac, rng, k=K):
    n = Ac.shape[0]
    fr = [frac(Q, Ac.T @ rng.standard_normal(n)) for _ in range(k)]
    return float(np.percentile(fr, 95))


def band(sources_map):
    hv = {}
    for s in SRC:
        others = [sources_map[o] for o in SRC if o != s]
        hv[s] = frac(ortho(others), sources_map[s])
    return [round(min(hv.values()), 4), round(max(hv.values()), 4)], {k: round(v, 4) for k, v in hv.items()}


def r3(refusal, judgment, sigma, cov_sample, rng):
    """|cos(refusal, judgment)| + pairwise null in the (standardized) chat space defined by sigma."""
    rs, js = std_unit(refusal, sigma), std_unit(judgment, sigma)
    A = (cov_sample - cov_sample.mean(0)) / sigma
    Ac = A - A.mean(0)
    q95 = pairwise_null_q95(Ac, rng)
    cos = abs(float(rs @ js))
    return {"abs_cos": round(cos, 4), "pairwise_null_q95": round(q95, 4),
            "coupling_detected": bool(cos > q95 + MARGIN_M),
            "verdict": "coupling" if cos > q95 + MARGIN_M else "dissociation"}


def r5_band(sources_map, controls_map, persona, refusal, judgment, label_contrast, sigma, cov_sample, rng):
    """Raw-content cell: standardized V_moral, band, controls, refusal (format-confounded), R2."""
    S = {s: std_unit(v, sigma) for s, v in sources_map.items()}
    Q = ortho([S[s] for s in SRC])
    A = (cov_sample - cov_sample.mean(0)) / sigma
    Ac = A - A.mean(0)
    q95 = null_q95_on(Q, Ac, rng)
    bnd, hv = band(S)
    c = {ck: round(frac(Q, std_unit(cv, sigma)), 4) for ck, cv in controls_map.items()}
    c["persona"] = round(frac(Q, std_unit(persona, sigma)), 4)
    p_ref = round(frac(Q, std_unit(refusal, sigma)), 4)          # R5/G3 (format-confounded: chat refusal)
    p_jud = round(frac(Q, std_unit(judgment, sigma)), 4)          # R2 (format-confounded)
    p_lab = round(frac(Q, std_unit(label_contrast, sigma)), 4)    # content ref (format-confounded)
    floor = min(c["syntax"], c["register"])
    return {"null_q95": round(q95, 4), "moral_family_band": bnd, "heldout": hv, "c_controls": c,
            "refusal_p": p_ref, "min_c_syntax_register": round(floor, 4),
            "strong_form_holds": bool(p_ref <= floor + MARGIN_M),
            "R2_judgment_p_FORMAT_CONFOUNDED": p_jud, "label_contrast_ref": p_lab}


def analyse(key, layer, rng):
    a = load(key, layer)
    raw_sig = a["X_raw"].std(0) + 1e-12
    chat_sig = a["acts_chat"].std(0) + 1e-12
    out = {"key": key, "layer": layer,
           "sigma_provenance": {"R3": "chat decision-site (acts_headline), sink-free (last-token)",
                                "R5_band_R2": "raw pooled act_sample"}}

    # R3 (chat class): raw sigma-of-ones (=identity => raw) vs standardized chat sigma.
    ones = np.ones_like(chat_sig)
    out["R3_raw"] = r3(a["refusal"], a["judgment"], ones, a["acts_chat"], np.random.default_rng(SEED))
    out["R3_standardized"] = r3(a["refusal"], a["judgment"], chat_sig, a["acts_chat"],
                                np.random.default_rng(SEED))
    # top-k robustness: drop >5%-var dims (chat variance), then raw-cos on the kept dims.
    keep = topk_mask(a["acts_chat"].var(0))
    if keep.sum() < len(keep):
        rr, jj = unit(a["refusal"][keep]), unit(a["judgment"][keep])
        Ak = a["acts_chat"][:, keep]
        out["R3_topk_projout"] = r3(a["refusal"][keep], a["judgment"][keep], np.ones(keep.sum()),
                                    Ak, np.random.default_rng(SEED))
        out["R3_topk_projout"]["dropped_dims"] = int((~keep).sum())

    # R5 / band / R2 (raw class): raw vs standardized.
    out["R5_raw"] = r5_band(a["sources"], a["controls"], a["persona"], a["refusal"], a["judgment"],
                            a["label_contrast"], np.ones_like(raw_sig), a["X_raw"],
                            np.random.default_rng(SEED))
    out["R5_standardized"] = r5_band(a["sources"], a["controls"], a["persona"], a["refusal"],
                                     a["judgment"], a["label_contrast"], raw_sig, a["X_raw"],
                                     np.random.default_rng(SEED))
    keepr = topk_mask(a["X_raw"].var(0))
    if keepr.sum() < len(keepr):
        srck = {s: v[keepr] for s, v in a["sources"].items()}
        ctrlk = {c: v[keepr] for c, v in a["controls"].items()}
        out["R5_topk_projout"] = r5_band(srck, ctrlk, a["persona"][keepr], a["refusal"][keepr],
                                         a["judgment"][keepr], a["label_contrast"][keepr],
                                         np.ones(int(keepr.sum())), a["X_raw"][:, keepr],
                                         np.random.default_rng(SEED))
        out["R5_topk_projout"]["dropped_dims"] = int((~keepr).sum())
    return out


def main():
    rng = np.random.default_rng(SEED)
    results = {}
    for key, layer in MODELS:
        results[key] = analyse(key, layer, rng)
    (OUT / "standardized_recompute.json").write_text(json.dumps(results, indent=2))

    print("=== R3 (refusal x judgment-decision; dissociation iff |cos| <= pairwise-null + M) ===")
    for key, _ in MODELS:
        r = results[key]
        line = f"  {key:8}"
        for tag in ("R3_raw", "R3_standardized", "R3_topk_projout"):
            if tag in r:
                v = r[tag]
                line += f" | {tag.split('_',1)[1]}: cos={v['abs_cos']} null={v['pairwise_null_q95']} {v['verdict']}"
        print(line)
    # OLMo invariance check (rider b)
    o = results["olmo3"]
    inv = o["R3_raw"]["verdict"] == o["R3_standardized"]["verdict"]
    print(f"\nOLMo R3 invariance (raw {o['R3_raw']['verdict']} == standardized "
          f"{o['R3_standardized']['verdict']}): {'PASS' if inv else 'FAIL -> transform suspect'}")

    print("\n=== R5 (refusal vs non-moral controls on V_moral; null q95) ===")
    for key, _ in MODELS:
        r = results[key]
        for tag in ("R5_raw", "R5_standardized"):
            v = r[tag]
            print(f"  {key:8} {tag.split('_',1)[1]:13}: null_q95={v['null_q95']:.3f} band={v['moral_family_band']} "
                  f"c_syn/reg/sent={v['c_controls']['syntax']}/{v['c_controls']['register']}/{v['c_controls']['sentiment']} "
                  f"refusal_p={v['refusal_p']} strong_form={v['strong_form_holds']}")
    print(f"\nwrote {OUT}/standardized_recompute.json")


if __name__ == "__main__":
    main()
