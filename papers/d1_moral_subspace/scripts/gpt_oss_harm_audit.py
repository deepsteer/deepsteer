#!/usr/bin/env python3
"""GPT-OSS refusal harm/⊥-harm decomposition across trace positions (D1 retro-audit under the D3
routing lens). Zero-GPU: reads only saved directions.

For each saved refusal position Pk (P0 = prompt t_inst, P2 = in-trace deliberation), project the
per-position refusal direction onto (a) the model's own harm percept `d_harm` = the Zhao
`harmfulness_t_inst` direction, and (b) the moral-subspace mean direction with harm projected out
(`V_moral ⊥ d_harm`). Report standardized cosines (GPT-OSS is a massive-activation A1 outlier, so σ
comes from the model's own content-position `act_sample` at the match layer, which is all that de-weighting the A1
massive-activation dims requires); raw is reported alongside.

Result of record (2026-07-02): P0 (prompt) is near-purely harm (std |cos|=0.977 vs 0.001), P2
(in-trace) stays harm-dominant (0.49 vs 0.13). The refusal read is HARM-LOADED at both the prompt and
in the trace: prompt→trace consistent. This is the correlational cross-model corroboration of the D3
`harm_saturating` verdict, in an independent 20B reasoning MoE via projection (not patching).
"""

from __future__ import annotations

import json
import os

import numpy as np

G = "papers/d1_moral_subspace/outputs/phase2/gpt_oss/"
POS = "papers/7_reasoning/outputs/gpt_oss_20b/position_directions.npz"


def _u(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def _nearest(layers: list[int], L: int) -> int:
    return L if L in layers else min(layers, key=lambda x: abs(x - L))


def decompose() -> dict:
    mor = np.load(G + "moral_directions.npz")
    pos = np.load(POS)
    ml = [int(k.split("layer")[1]) for k in mor.files if "moral_stories" in k]
    hl = [int(k.split("layer")[1]) for k in pos.files if "harmfulness_t_inst" in k]
    az = np.load(G + "act_sample.npz", allow_pickle=True)
    X = next(az[k].astype(np.float64) for k in az.files if getattr(az[k], "ndim", 0) == 2)
    sig = np.sqrt(X.var(0))
    sig = np.where(sig > 1e-8, sig, 1.0)

    out = {}
    for P in ("P0", "P1", "P2", "P3"):
        f = G + f"refusal_think_{P}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f)
        r = z["refusal"].astype(np.float64)
        L = int(z["layer"])
        Vm = mor[f"moral_stories_layer{_nearest(ml, L)}"].astype(np.float64)
        harm = pos[f"harmfulness_t_inst_layer{_nearest(hl, L)}"].astype(np.float64)
        rec = {"layer": L}
        for tag, scale in (("std", sig), ("raw", np.ones_like(sig))):
            p, h, v = _u(r / scale), _u(harm / scale), _u(Vm / scale)
            vp = _u(v - (v @ h) * h)                        # V_moral component ⊥ d_harm
            rec[tag] = {"cos_refusal_harm": round(abs(float(p @ h)), 4),
                        "cos_refusal_vmoral_perp_harm": round(abs(float(p @ vp)), 4),
                        "cos_vmoral_harm": round(abs(float(v @ h)), 4)}
        rec["verdict"] = ("harm_loaded"
                          if rec["std"]["cos_refusal_harm"] > rec["std"]["cos_refusal_vmoral_perp_harm"]
                          else "moral_loaded")
        out[P] = rec
    return out


if __name__ == "__main__":
    res = decompose()
    for P, rec in res.items():
        s = rec["std"]
        print(f"{P} (layer {rec['layer']}): STD |cos(refusal,harm)|={s['cos_refusal_harm']:.3f}  "
              f"|cos(refusal,Vmoral⊥harm)|={s['cos_refusal_vmoral_perp_harm']:.3f}  -> {rec['verdict']}")
    dst = G + "harm_audit.json"
    with open(dst, "w") as fh:
        json.dump(res, fh, indent=2)
    print(">> wrote", dst)
