#!/usr/bin/env python3
"""One-knob model of the C1 sweep (Amendment-4 rider). Refusal transfer = judgment transfer capped at
a single harm ceiling: R_refusal(k) ~= min(harm_ceiling, R_judgment(k)). One free parameter (the
ceiling ~ harm_rank1_R). If it fits, it is the flagship figure and the panel comparative becomes "does
the same one-knob model fit each model" (sharper than shape-vs-level). PC1 deviation -> nonlinearity.

Zero-GPU: reads the saved sweep JSON + inputs npz. Also fits the two harm-amplitude alternatives
(R ~ ceiling * harm_capture and ~ ceiling * harm_capture^2) to show they fail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def fit(session_json: Path, inputs_npz: Path) -> dict:
    d = json.loads(session_json.read_text())
    s = d["cells"]["sweep"]; c = d["cells"]
    Rref = {int(k): v for k, v in s["R_refusal_k"].items()}
    Rjud = {int(k): v for k, v in s["R_judgment_k"].items()}
    cos = s["cos_harm_pc"]                                   # per-PC |cos(harm, PC_i)| (len 8)
    ceiling = s["harm_rank1_R"]
    ks = [k for k in s["ks"] if k <= len(cos)]              # ranks with exact harm_capture
    hc = {k: float(np.sqrt(sum(cos[i] ** 2 for i in range(min(k, len(cos)))))) for k in s["ks"]}

    def resid(model):
        return {k: round(Rref[k] - model(k), 4) for k in ks}
    sat = lambda k: min(ceiling, Rjud[k])                  # the one-knob saturation model
    ampA = lambda k: ceiling * hc[k]
    ampB = lambda k: ceiling * hc[k] ** 2
    r_sat, r_A, r_B = resid(sat), resid(ampA), resid(ampB)

    pc1_meas, pc1_pred = Rref[1], sat(1)
    pc1_deviation = round(pc1_meas - pc1_pred, 4)
    # rmse over the plateau ranks (k>=3), where the ceiling binds
    plat = [k for k in ks if k >= 3]
    rmse = lambda rd: round(float(np.sqrt(np.mean([rd[k] ** 2 for k in plat]))), 4)
    hp_R = round(c["harm_partialed_refusal_delta_mean"] / c["cell_a_full_refusal_delta_mean"], 4)
    return {"ceiling_harm_rank1_R": ceiling, "ks_exact": ks,
            "R_refusal": {k: Rref[k] for k in s["ks"]}, "R_judgment": {k: Rjud[k] for k in s["ks"]},
            "harm_capture": {k: round(hc[k], 4) for k in s["ks"]},
            "saturation_model": {k: round(sat(k), 4) for k in ks}, "saturation_resid": r_sat,
            "saturation_rmse_plateau": rmse(r_sat),
            "amplitude_linear_resid": r_A, "amplitude_quad_resid": r_B,
            "pc1_measured": pc1_meas, "pc1_predicted": round(pc1_pred, 4),
            "pc1_deviation": pc1_deviation,
            "pc1_deviates": bool(abs(pc1_deviation) > 0.02),
            "harm_partialed_R": hp_R,
            "verdict": ("one-knob saturation fits the plateau; PC1 deviates (nonlinearity candidate)"
                        if rmse(r_sat) < 0.06 and abs(pc1_deviation) > 0.02 else
                        "one-knob saturation fits" if rmse(r_sat) < 0.06 else "one-knob model fails")}


def figure(res: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = sorted(int(k) for k in res["R_refusal"])
    rr = [res["R_refusal"][str(k) if str(k) in res["R_refusal"] else k] for k in ks]
    rj = [res["R_judgment"][str(k) if str(k) in res["R_judgment"] else k] for k in ks]
    ceil = res["ceiling_harm_rank1_R"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(ks, rj, "o-", color="#2c7fb8", label="$R_{judgment}(k)$ (reads broad subspace)")
    ax.plot(ks, rr, "s-", color="#d95f02", label="$R_{refusal}(k)$ (measured)")
    ax.plot(ks, [min(ceil, res["R_judgment"][str(k) if str(k) in res["R_judgment"] else k]) for k in ks],
            "--", color="#666", label="one-knob: min(harm ceiling, $R_{judgment}$)")
    ax.axhline(ceil, ls=":", color="#d95f02", alpha=0.6, label=f"harm ceiling {ceil:.2f}")
    ax.annotate("PC1 inert\n(deviates)", xy=(1, rr[0]), xytext=(1.5, 0.15),
                arrowprops=dict(arrowstyle="->", color="#999"), fontsize=8, color="#555")
    ax.set_xlabel("moral-basis rank $k$"); ax.set_ylabel("fraction of full-patch effect transferred")
    ax.set_xticks(ks); ax.set_title("Refusal transfer saturates at the harm ceiling (OLMo-3)")
    ax.legend(fontsize=7.5, loc="center right"); ax.set_ylim(-0.05, 0.72)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def main():
    out = HERE.parent / "outputs"
    key = sys.argv[1] if len(sys.argv) > 1 else "olmo3"
    res = fit(out / f"c1_session_{key}.json", out / f"c1_inputs_{key}.npz")
    (out / f"one_knob_{key}.json").write_text(json.dumps(res, indent=2))
    try:
        figure(res, out / f"one_knob_{key}.png")
        res["figure"] = str(out / f"one_knob_{key}.png")
    except Exception as e:
        res["figure_error"] = str(e)
    print(json.dumps({k: v for k, v in res.items() if k not in ("R_refusal", "R_judgment", "harm_capture")}, indent=2))


if __name__ == "__main__":
    main()
