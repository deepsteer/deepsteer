#!/usr/bin/env python3
"""A4 (calibration): statistics upgrade. Pre-registered in CALIBRATION_PREREG.md A4.

(1) Bootstrap CIs (resample pairs -> re-extract mean-diff direction -> recompute projection),
    B=2000, seed 0, for the held-one-out moral-family band and for refusal-p onto V_moral
    (refusal fixed, V_moral resampled) wherever the per-pair arrays + refusal vector exist.
    Missing per-pair arrays are logged to MISSING_ARTIFACTS.md, never silently regenerated.
(2) Cross-model combined P2 test (R7, EXPLORATORY / post-hoc): per-model null-exceedance
    p-value of P2 vs the covariance-matched null, combined via Fisher across Think + GPT-OSS.
    Per-model verdicts remain the pre-registered rule (R2/G3); this is an aggregate only.
(3) Signed cos(refusal, source-direction) alongside the magnitude fractions (diagnostic,
    basis-dependent), where the refusal vector is committed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deepsteer.directions import extraction as du  # noqa: E402
from calibration_a1_ladder import TAGS, _ortho, _frac  # noqa: E402

P2 = HERE.parent / "outputs" / "phase2"
OUT = P2 / "calibration"
B = 2000
K = 2000
SEED = 0
_unit = du.unit_vector

# Per-pair diff sources per tag: (moral_stories_file, fables_file, ethics_file) or None if a
# source's per-pair array is not committed.
PAIR_DIFFS = {
    "base": ("base/diffs_moral_stories.npz", "axis/axis_diffs_fables.npz",
             "axis/axis_diffs_ethics.npz"),
    "instruct": ("instruct/diffs_moral_stories.npz", None, None),
    "think": ("think/diffs_moral_stories.npz", "think_axis/axis_diffs_fables.npz",
              "think_axis/axis_diffs_ethics.npz"),
    "gpt_oss": ("gpt_oss/diffs_moral_stories.npz", "gpt_oss_axis/axis_diffs_fables.npz",
                "gpt_oss_axis/axis_diffs_ethics.npz"),
}
REFUSAL_VECS = {
    "base": {"P_A_proto": "refusal_base.npz"},
    "instruct": {"P_B_gate": "refusal_instruct.npz"},
    "think": {},
    "gpt_oss": {"P0": "gpt_oss/refusal_think_P0.npz", "P1": "gpt_oss/refusal_think_P1.npz",
                "P2": "gpt_oss/refusal_think_P2.npz", "P2_FULL": "gpt_oss/refusal_think_P2_FULL.npz",
                "P3": "gpt_oss/refusal_think_P3.npz"},
}
SRC = ["moral_stories", "fables", "ethics"]
# Points that get the full BCa treatment (jackknife-over-pairs); percentile is computed for all.
BCA_POINTS = {"gpt_oss": ["P2", "P2_FULL"], "base": ["P_A_proto"]}


def _ci(vals: list[float]) -> list[float]:
    a = np.asarray(vals)
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 50)), 4),
            round(float(np.percentile(a, 97.5)), 4)]


def _band_min_and_frac(dirs: dict[str, np.ndarray], v: np.ndarray) -> tuple[float, float]:
    """(held-one-out band-min, projection fraction of v onto the rank-3 span) for given dirs."""
    hv = {s: _frac(_ortho([dirs[o] for o in SRC if o != s]), dirs[s]) for s in SRC}
    return min(hv.values()), _frac(_ortho([dirs[s] for s in SRC]), v)


def _jackknife_delta(diffs: dict[str, np.ndarray], v: np.ndarray) -> np.ndarray:
    """Leave-one-pair-out jackknife of Δ = band-min − proj(v). Only the left-out source's
    direction changes per replicate; the other two stay at their full-sample mean-diff."""
    full = {s: _unit(diffs[s].mean(0)) for s in SRC}
    jack = []
    for s in SRC:
        D = diffs[s]
        tot = D.sum(0)
        n_s = D.shape[0]
        for j in range(n_s):
            dd = dict(full)
            dd[s] = _unit((tot - D[j]) / (n_s - 1))
            bmin, p = _band_min_and_frac(dd, v)
            jack.append(bmin - p)
    return np.asarray(jack)


def _bca_ci(draws: np.ndarray, theta_hat: float, jack: np.ndarray,
            alpha: float = 0.05) -> list[float]:
    """Bias-corrected-and-accelerated 95% CI. z0 from bootstrap bias, a from jackknife skew."""
    nd = NormalDist()
    draws = np.asarray(draws)
    frac = min(max((draws < theta_hat).mean(), 1e-6), 1 - 1e-6)
    z0 = nd.inv_cdf(frac)
    jbar = jack.mean()
    num = ((jbar - jack) ** 3).sum()
    den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
    a = float(num / den) if den != 0 else 0.0

    def adj(p: float) -> float:
        z = nd.inv_cdf(p)
        return nd.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lo = float(np.percentile(draws, 100 * adj(alpha / 2)))
    hi = float(np.percentile(draws, 100 * adj(1 - alpha / 2)))
    return [round(lo, 4), round(hi, 4)]


def _load_diffs(tag: str, layer: int) -> dict[str, np.ndarray] | None:
    files = PAIR_DIFFS[tag]
    if any(f is None or not (P2 / f).exists() for f in files):
        return None
    key = f"layer{layer}"
    return {s: np.load(P2 / f)[key].astype(np.float64) for s, f in zip(SRC, files)}


def bootstrap_tag(tag: str, layer: int, rng) -> dict:
    diffs = _load_diffs(tag, layer)
    refusals = {n: _unit(np.load(P2 / r)["refusal"]) for n, r in REFUSAL_VECS[tag].items()
                if (P2 / r).exists()}
    if diffs is None:
        return {"band_bootstrap": None,
                "reason": f"per-pair diffs incomplete for {tag} "
                          f"(fables/ethics missing) -> logged", "refusal_p_bootstrap": None}
    ns = {s: diffs[s].shape[0] for s in SRC}
    heldout_bs = {s: [] for s in SRC}
    band_min_bs, band_max_bs = [], []
    refusal_bs = {n: [] for n in refusals}
    delta_bs = {n: [] for n in refusals}  # paired Δ_i = band_min_i − P_i (same resampled V_moral)
    for _ in range(B):
        d = {}
        for s in SRC:
            idx = rng.integers(0, ns[s], ns[s])
            d[s] = _unit(diffs[s][idx].mean(0))
        Q = _ortho([d[s] for s in SRC])
        hv = {}
        for s in SRC:
            hv[s] = _frac(_ortho([d[o] for o in SRC if o != s]), d[s])
            heldout_bs[s].append(hv[s])
        bmin = min(hv.values())
        band_min_bs.append(bmin); band_max_bs.append(max(hv.values()))
        for n, rv in refusals.items():
            p = _frac(Q, rv)
            refusal_bs[n].append(p)
            delta_bs[n].append(bmin - p)

    # Paired Δ = band-min − P test (pre-registered A4 addendum, 2026-07-01).
    delta = None
    if refusals:
        full = {s: _unit(diffs[s].mean(0)) for s in SRC}
        delta = {}
        for n, rv in refusals.items():
            bmin_full, p_full = _band_min_and_frac(full, rv)
            dhat = bmin_full - p_full
            draws = np.asarray(delta_bs[n])
            plo, pmed, phi = (float(np.percentile(draws, q)) for q in (2.5, 50, 97.5))
            entry = {"delta_hat": round(dhat, 4),
                     "percentile_ci": [round(plo, 4), round(phi, 4)],
                     "percentile_median": round(pmed, 4),
                     "percentile_excludes_0": bool(plo > 0)}
            if n in BCA_POINTS.get(tag, []):
                bca = _bca_ci(draws, dhat, _jackknife_delta(diffs, rv))
                entry["bca_ci"] = bca
                entry["bca_excludes_0"] = bool(bca[0] > 0)
            delta[n] = entry

    return {
        "n_pairs": ns,
        "band_bootstrap": {
            "heldout_ci": {s: _ci(heldout_bs[s]) for s in SRC},
            "band_min_ci": _ci(band_min_bs), "band_max_ci": _ci(band_max_bs),
        },
        "refusal_p_bootstrap": ({n: {"ci": _ci(refusal_bs[n])} for n in refusals}
                                if refusals else None),
        "delta_band_min_minus_p": delta,
    }


def signed_cos(tag: str, layer: int, axis_sub: str) -> dict | None:
    from calibration_a1_ladder import source_dirs
    refs = {n: _unit(np.load(P2 / r)["refusal"]) for n, r in REFUSAL_VECS[tag].items()
            if (P2 / r).exists()}
    if not refs:
        return None
    dirs = source_dirs(tag, axis_sub, layer)
    persona = _unit(du.load_directions(P2 / tag / "persona_direction.npz")["persona"][layer])
    out = {}
    for n, rv in refs.items():
        out[n] = {**{s: round(float(du.cosine(rv, v)), 4) for s, v in dirs.items()},
                  "persona": round(float(du.cosine(rv, persona)), 4)}
    return out


def combined_p2(rng) -> dict:
    """Fisher across Think + GPT-OSS: per-model null-exceedance p of the in-trace P2."""
    from calibration_a1_ladder import source_dirs
    res = {}
    ps = {}
    for tag, g3 in [("think", "think_g3_result.json"), ("gpt_oss", "gpt_oss_g3_result.json")]:
        layer, axis_sub, _m, _g = TAGS[tag]
        dirs = source_dirs(tag, axis_sub, layer)
        Q = _ortho([dirs[s] for s in SRC if s in dirs])
        X = np.load(P2 / tag / "act_sample.npz")["X"].astype(np.float64)
        Xc = X - X.mean(0, keepdims=True)
        n = Xc.shape[0]
        null = np.array([_frac(Q, Xc.T @ rng.standard_normal(n)) for _ in range(K)])
        p2 = float(json.load(open(P2 / g3))["positions"]["P2"]["refusal_p"])
        p_val = (1 + int((null >= p2).sum())) / (K + 1)
        ps[tag] = p_val
        res[tag] = {"P2": round(p2, 4), "null_exceedance_p": round(p_val, 5)}
    chi2 = -2.0 * (np.log(ps["think"]) + np.log(ps["gpt_oss"]))
    # Fisher combination, df = 2k = 4; closed-form survival for chi-square df=4.
    combined = float(np.exp(-chi2 / 2.0) * (1.0 + chi2 / 2.0))
    res["fisher"] = {"chi2_df4": round(float(chi2), 4), "combined_p": round(combined, 6),
                     "label": "EXPLORATORY / post-hoc aggregate; per-model verdicts remain the "
                              "pre-registered rule (R2/G3)"}
    return res


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    out = {}
    logged_missing = []
    for tag, (layer, axis_sub, _mft, _g3) in TAGS.items():
        bs = bootstrap_tag(tag, layer, rng)
        sc = signed_cos(tag, layer, axis_sub)
        out[tag] = {**bs, "signed_cos_refusal_diagnostic": sc}
        if bs["band_bootstrap"] is None:
            logged_missing.append(tag)
        print(f"[{tag}]")
        if bs["band_bootstrap"]:
            hc = bs["band_bootstrap"]["heldout_ci"]
            print("   held-one-out 95% CI: " + "  ".join(f"{s}={hc[s]}" for s in SRC))
            print(f"   band_min CI {bs['band_bootstrap']['band_min_ci']}  "
                  f"band_max CI {bs['band_bootstrap']['band_max_ci']}")
            if bs["refusal_p_bootstrap"]:
                for n, v in bs["refusal_p_bootstrap"].items():
                    print(f"   refusal {n} p CI {v['ci']}")
            if bs.get("delta_band_min_minus_p"):
                for n, dv in bs["delta_band_min_minus_p"].items():
                    bca = f" | BCa {dv['bca_ci']} excl0={dv['bca_excludes_0']}" if "bca_ci" in dv else ""
                    print(f"   Δ(band_min−{n}) hat={dv['delta_hat']} pct-CI {dv['percentile_ci']} "
                          f"excl0={dv['percentile_excludes_0']}{bca}")
        else:
            print(f"   band bootstrap SKIPPED: {bs['reason']}")
        if sc:
            for n, v in sc.items():
                print(f"   signed cos(refusal_{n}, .): {v}")

    out["combined_P2"] = combined_p2(rng)
    f = out["combined_P2"]["fisher"]
    print(f"\ncombined P2 (Fisher, EXPLORATORY): think p={out['combined_P2']['think']['null_exceedance_p']} "
          f"gpt_oss p={out['combined_P2']['gpt_oss']['null_exceedance_p']} "
          f"-> chi2_df4={f['chi2_df4']} combined_p={f['combined_p']}")

    (OUT / "a4_bootstrap.json").write_text(json.dumps(out, indent=2))
    if logged_missing:
        with open(P2.parent / "MISSING_ARTIFACTS.md", "a") as fh:
            fh.write("\n## A4 (2026-07-01): per-pair bootstrap gaps\n\n")
            for t in logged_missing:
                fh.write(f"- `{t}`: fables/ethics per-pair diff arrays not committed "
                         f"(`axis_{t}/axis_diffs_*.npz` absent) -> held-one-out band + "
                         f"refusal-p bootstrap CIs not computable. Re-extract with per-pair "
                         f"saves in B3.\n")


if __name__ == "__main__":
    main()
