#!/usr/bin/env python3
"""A1 (calibration): held-one-out moral positive control + MFT<->V_moral mutual projection
+ the calibrated ladder. Zero-GPU, numpy on committed artifacts. Pre-registered in
CALIBRATION_PREREG.md (committed at a513466, after the frozen D1 spine).

For each tag at its headline layer:
  * held-one-out p(d_src | span{other two}) for the 3 source moral directions -> moral-family
    band [min,max] (rule R1), the yardstick for "moral-adjacent" statements;
  * MFT<->V_moral mutual projection (both asymmetry directions) where MFT dirs are committed
    (base, instruct); missing tags (think, gpt_oss) logged to MISSING_ARTIFACTS.md;
  * calibrated ladder rungs: iso floor -> null q50/q95 (recomputed on the rank-3 span, frozen
    recipe) -> committed refusal points -> moral band -> persona c.

Conventions reused verbatim from phase2_g3_respec.py: _ortho (QR), _frac (in-subspace norm
ratio), covariance-matched null (Xc.T @ N(0,I_n)), K=1000, seed 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deepsteer.directions import extraction as du  # noqa: E402

P2 = HERE.parent / "outputs" / "phase2"
OUT = P2 / "calibration"
MISSING = P2.parent / "MISSING_ARTIFACTS.md"
K = 1000
SEED = 0
SOURCES = ["moral_stories", "fables", "ethics"]
MFT_FOUNDATIONS = ["care_harm", "fairness_cheating", "liberty_oppression",
                   "loyalty_betrayal", "authority_subversion", "sanctity_degradation"]

_unit = du.unit_vector


def _ortho(dirs: list[np.ndarray]) -> np.ndarray:
    """Orthonormal basis (hidden, r) spanning the source directions (QR, as G3)."""
    Q, _ = np.linalg.qr(np.stack(dirs, axis=1))
    return Q


def _frac(Q: np.ndarray, v: np.ndarray) -> float:
    """In-subspace norm fraction ||Q^T v|| / ||v|| (the G3 projection-fraction convention)."""
    return float(np.linalg.norm(Q.T @ v) / (np.linalg.norm(v) + 1e-12))


# Per-tag layout: (headline layer, axis subdir, mft file or None, committed g3 result file).
TAGS = {
    "base":     (16, "axis",         "base/mft_directions.npz",     "g3_respec_result.json"),
    "instruct": (16, "axis_instruct", "instruct/mft_directions.npz", "g3_respec_result.json"),
    "think":    (16, "think_axis",    None,                          "think_g3_result.json"),
    "gpt_oss":  (12, "gpt_oss_axis",  None,                          "gpt_oss_g3_result.json"),
}


def source_dirs(tag: str, axis_sub: str, layer: int) -> dict[str, np.ndarray]:
    d = {"moral_stories": _unit(du.load_directions(P2 / tag / "moral_directions.npz")
                                ["moral_stories"][layer])}
    ad = du.load_directions(P2 / axis_sub / "axis_directions.npz")
    for s in ("fables", "ethics"):
        if s in ad:
            d[s] = _unit(ad[s][layer])
    return d


def refusal_points(tag: str, g3_file: str) -> dict[str, float]:
    """Committed refusal projection(s) for this tag, read from its frozen G3 result."""
    r = json.load(open(P2 / g3_file))
    if tag == "base":
        return {"P_A_proto": r["pointA_base_proto"]["headline_rank3"]["refusal_p"]}
    if tag == "instruct":
        return {"P_B_gate": r["pointB_instruct_gate"]["headline_rank3"]["refusal_p"]}
    pts = {}
    for pk, pv in r.get("positions", {}).items():
        if pv.get("available") and pv.get("refusal_p") is not None:
            pts[pk] = float(pv["refusal_p"])
    return pts


def mft_mutual(tag: str, mft_file: str | None, vmoral_dirs: dict[str, np.ndarray],
               Q_vmoral: np.ndarray, layer: int) -> dict | None:
    if mft_file is None or not (P2 / mft_file).exists():
        return None
    mft = du.load_directions(P2 / mft_file)
    fdirs = {f: _unit(mft[f][layer]) for f in MFT_FOUNDATIONS if f in mft}
    Q_mft = _ortho(list(fdirs.values()))
    mft_onto_vm = {f: round(_frac(Q_vmoral, v), 4) for f, v in fdirs.items()}
    vm_onto_mft = {s: round(_frac(Q_mft, v), 4) for s, v in vmoral_dirs.items()}
    return {
        "mft_rank": int(Q_mft.shape[1]),
        "mft_onto_vmoral": mft_onto_vm,
        "mft_onto_vmoral_mean": round(float(np.mean(list(mft_onto_vm.values()))), 4),
        "vmoral_onto_mft": vm_onto_mft,
        "vmoral_onto_mft_mean": round(float(np.mean(list(vm_onto_mft.values()))), 4),
        "note": "asymmetry reported both directions; content-vs-content, closes Paper 7 Phase 2f "
                "at the subspace level",
    }


def measure(tag: str, rng) -> dict:
    layer, axis_sub, mft_file, g3_file = TAGS[tag]
    dirs = source_dirs(tag, axis_sub, layer)
    present = [s for s in SOURCES if s in dirs]
    Q = _ortho([dirs[s] for s in present])
    d = Q.shape[0]

    # Held-one-out positive control (R1): project each source onto the span of the other two.
    heldout = {}
    for s in present:
        others = [dirs[o] for o in present if o != s]
        heldout[s] = round(_frac(_ortho(others), dirs[s]), 4)
    band = [round(min(heldout.values()), 4), round(max(heldout.values()), 4)]

    # Null on THIS rank-3 span (frozen recipe: covariance-matched randoms), q50 + q95.
    X = np.load(P2 / tag / "act_sample.npz")["X"].astype(np.float64)
    Xc = X - X.mean(0, keepdims=True)
    n = Xc.shape[0]
    fr = np.array([_frac(Q, Xc.T @ rng.standard_normal(n)) for _ in range(K)])
    null_q50, null_q95 = float(np.percentile(fr, 50)), float(np.percentile(fr, 95))

    persona = _unit(du.load_directions(P2 / tag / "persona_direction.npz")["persona"][layer])
    c = round(_frac(Q, persona), 4)

    return {
        "tag": tag, "layer": layer, "hidden": d, "rank": int(Q.shape[1]), "sources": present,
        "heldout_positive_control": heldout,
        "moral_family_band": band,            # rule R1
        "iso_floor": round(float(np.sqrt(Q.shape[1] / d)), 4),
        "null_q50": round(null_q50, 4), "null_q95": round(null_q95, 4),
        "persona_c": c,
        "refusal_points": {k: round(v, 4) for k, v in refusal_points(tag, g3_file).items()},
        "mft_mutual_projection": mft_mutual(tag, mft_file, dirs, Q, layer),
    }


def log_missing(tags_without_mft: list[str]) -> None:
    lines = ["", "## A1 (2026-07-01): MFT directions not committed for reasoning tags", ""]
    for t in tags_without_mft:
        lines.append(f"- `outputs/phase2/{t}/mft_directions.npz` absent -> MFT<->V_moral mutual "
                     f"projection not computable for `{t}`. Queue MFT extraction into B3 if the "
                     f"reasoning-tag subspace comparison is wanted (base/instruct are covered).")
    header = "" if MISSING.exists() else "# Missing artifacts ledger\n"
    with open(MISSING, "a") as fh:
        fh.write(header + "\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = {}
    missing = []
    for tag in TAGS:
        res = measure(tag, rng)
        results[tag] = res
        (OUT / f"a1_ladder_{tag}.json").write_text(json.dumps(res, indent=2))
        if res["mft_mutual_projection"] is None:
            missing.append(tag)
        print(f"=== {tag} (L{res['layer']}) ===")
        print(f"  held-one-out: " + "  ".join(f"{s}={v}" for s, v in
                                              res["heldout_positive_control"].items()))
        print(f"  moral-family band (R1): {res['moral_family_band']}")
        print(f"  ladder: iso {res['iso_floor']} | null q50 {res['null_q50']} q95 "
              f"{res['null_q95']} | refusal {res['refusal_points']} | band "
              f"{res['moral_family_band']} | persona {res['persona_c']}")
        if res["mft_mutual_projection"]:
            m = res["mft_mutual_projection"]
            print(f"  MFT->V_moral mean {m['mft_onto_vmoral_mean']} | "
                  f"V_moral->MFT mean {m['vmoral_onto_mft_mean']}")
    (OUT / "a1_summary.json").write_text(json.dumps(results, indent=2))
    if missing:
        log_missing(missing)
        print(f"\nlogged missing MFT for {missing} -> {MISSING.name}")


if __name__ == "__main__":
    main()
