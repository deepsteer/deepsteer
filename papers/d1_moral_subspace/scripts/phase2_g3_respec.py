#!/usr/bin/env python3
"""Direction 1: G3 on the re-spec'd rank-3 source-direction V_moral (numpy, local; no GPU).

V_moral = orthonormalized span of the source moral mean-diff directions {d_moral, d_fables,
d_ethics} (rank 3) -- the finding-driven correction (PREREGISTRATION re-spec amendment):
the moral structure lives in the source directions, not the content-dominated pooled diffs.
Constructed exactly like the MFT subspace (span of moral directions) -> directly comparable
to Paper 5's 0.1044 (refusal onto a moral-direction span).

Two same-model points: Point A = base proto-refusal x base rank-3 V_moral; Point B = instruct
gate x instruct rank-3 V_moral. The rank-matched null (covariance-matched randoms onto the
span) + persona control are RECOMPUTED on this span (different subspace -> different null;
two-step: computed before the refusal projection). Reports the rank-3 span point estimate
(order-invariant, headline) + a rank-sweep (1->2->3). Refusal vectors come from
phase2_g3_respec_extract.py (saved this time).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))
from deepsteer.directions import extraction as du  # noqa: E402

P2 = HERE.parent / "outputs" / "phase2"
MARGIN_M = 0.05
K = 1000
SEED = 0
SWEEP = ["moral_stories", "fables", "ethics"]  # rank-sweep order (span is order-invariant)


_unit = du.unit_vector  # shared: deepsteer.directions.extraction.unit_vector


def _ortho(dirs: list[np.ndarray]) -> np.ndarray:
    """Orthonormal basis (hidden, r) spanning the source directions."""
    Q, _ = np.linalg.qr(np.stack(dirs, axis=1))
    return Q


def _frac(Q: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(Q.T @ v) / (np.linalg.norm(v) + 1e-12))


def source_dirs(tag: str, axis_dir: Path, layer: int) -> dict[str, np.ndarray]:
    base = P2 / tag
    d = {"moral_stories": _unit(du.load_directions(base / "moral_directions.npz")
                               ["moral_stories"][layer])}
    ad = du.load_directions(axis_dir / "axis_directions.npz")
    for s in ("fables", "ethics"):
        if s in ad:
            d[s] = _unit(ad[s][layer])
    return d


def measure(tag: str, axis_dir: Path, refusal_path: Path, layer: int, rng) -> dict:
    dirs = source_dirs(tag, axis_dir, layer)
    persona = _unit(du.load_directions(P2 / tag / "persona_direction.npz")["persona"][layer])
    refusal = _unit(np.load(refusal_path)["refusal"])
    X = np.load(P2 / tag / "act_sample.npz")["X"].astype(np.float64)
    Xc = X - X.mean(0, keepdims=True)
    n = Xc.shape[0]

    rows = []
    for r in range(1, len(SWEEP) + 1):
        present = [s for s in SWEEP[:r] if s in dirs]
        Q = _ortho([dirs[s] for s in present])
        fr = np.array([_frac(Q, Xc.T @ rng.standard_normal(n)) for _ in range(K)])
        q95 = float(np.percentile(fr, 95))
        c = _frac(Q, persona)
        p = _frac(Q, refusal)
        rows.append({"rank": Q.shape[1], "sources": present, "refusal_p": round(p, 4),
                     "null_q95": round(q95, 4), "control_c": round(c, 4),
                     "iso_floor": round(np.sqrt(Q.shape[1] / X.shape[1]), 4),
                     "clears": bool(p > q95 + MARGIN_M and p > c + MARGIN_M)})
    full = rows[-1]
    return {"tag": tag, "headline_rank3": full, "rank_sweep": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="G3 on rank-3 source-direction V_moral.")
    ap.add_argument("--axis-base", default=str(P2 / "axis"))
    ap.add_argument("--axis-instruct", default=str(P2 / "axis_instruct"))
    args = ap.parse_args()

    layer = int(json.load(open(P2 / "base" / "extract_meta.json"))["match_layer"])
    rng = np.random.default_rng(SEED)
    ptA = measure("base", Path(args.axis_base), P2 / "refusal_base.npz", layer, rng)
    ptB = measure("instruct", Path(args.axis_instruct), P2 / "refusal_instruct.npz", layer, rng)

    a, b = ptA["headline_rank3"], ptB["headline_rank3"]
    positive = a["clears"] and b["clears"]
    result = {
        "design": "rank-3 source-direction V_moral (span of d_moral,d_fables,d_ethics); "
                  "two same-model points; null+control recomputed on this span",
        "margin_M": MARGIN_M, "layer": layer,
        "pointA_base_proto": ptA, "pointB_instruct_gate": ptB,
        "headline_g3": "POSITIVE" if positive else "NULL",
        "pointB_vs_0_1044": "instruct refusal onto rank-3 instruct V_moral (cf. Paper-5 0.1044 "
                            "onto the rank-4 MFT foundation span)",
        "split_result": (None if a["clears"] == b["clears"]
                         else "A,B disagree -> NULL + flag"),
        "note": "rank-3 span point estimate is the headline (order-invariant); rank-sweep is "
                "illustrative; per-axis alignment is basis-dependent / diagnostic only",
    }
    (P2 / "g3_respec_result.json").write_text(json.dumps(result, indent=2))
    print("=== G3 (rank-3 source-direction V_moral) ===")
    for pt in (ptA, ptB):
        h = pt["headline_rank3"]
        print(f"  {pt['tag']:8} rank-3: refusal p={h['refusal_p']} vs q95+M="
              f"{h['null_q95']+MARGIN_M:.4f} c+M={h['control_c']+MARGIN_M:.4f} "
              f"-> {'clears' if h['clears'] else 'NULL'}")
        for row in pt["rank_sweep"]:
            print(f"     sweep r{row['rank']} ({'+'.join(row['sources'])}): p={row['refusal_p']} "
                  f"q95={row['null_q95']} c={row['control_c']}")
    print(f"  HEADLINE G3 = {result['headline_g3']}")


if __name__ == "__main__":
    main()
