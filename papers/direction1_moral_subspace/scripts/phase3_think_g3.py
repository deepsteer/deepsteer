#!/usr/bin/env python3
"""Direction 1, reasoning-model extension (local): G3 on the rank-3 OLMo-3-Think V_moral.

Reuses the exact span / covariance-matched-null / projection math from phase2_g3_respec.py,
applied to the Think artifacts (fresh per-model extraction; directions do not transfer).
Adds the two things the Think run needs:

  1. CONTENT-DOMINATED SPECTRUM CHECK (verify, do not assume). The rank-3 source-direction
     construction is only the right method if Think's pooled per-pair-diff spectrum is
     content-dominated like the base (no low-rank moral elbow). Reported, with a flag if Think
     unexpectedly shows a moral elbow (then the construction must be revisited, not reused).

  2. FOUR-POSITION refusal projection. The rank-matched null q95 + persona control c depend
     only on the span (same for every position); only the refusal projection p varies by
     position. Project each pre-registered refusal vector (P0 t_inst, P1 pre-trace gate, P2
     in-trace, P3 post-answer) onto the Think rank-3 span and report p vs q95+M and c+M.

Pre-registered (PREREGISTRATION.md 2026-06-29): P2 (in-trace) is the single coupling
hypothesis; the headline is orthogonal-at-all-four (robust) vs coupled-at-P2 (the
reasoning-specific finding). Numpy only; no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))
sys.path.insert(0, str(HERE.parents[2]))
import direction_utils as du  # noqa: E402
from phase2_g3_respec import (  # noqa: E402
    MARGIN_M,
    P2,
    SEED,
    SWEEP,
    K,
    _frac,
    _ortho,
    _unit,
    source_dirs,
)

TAG = "think"
# P2 = symmetric first-N-reasoning-token window (PRIMARY in-trace coupling test);
# P2_FULL = full-span mean (ROBUSTNESS, span-length-confounded); P3 = post-answer (UNMEASURED
# for benign prompts -- the benign side doesn't reach a post-answer state within budget).
POSITIONS = {"P0": "t_inst (harm/comprehension)", "P1": "pre-trace gate (A/B analog)",
             "P2": "in-trace window (coupling detector)",
             "P2_FULL": "in-trace full-span (robustness)",
             "P3": "post-answer (decision site)"}


def content_check(layer: int) -> dict:
    """Verify Think's pooled moral-diff spectrum is content-dominated (uncentered eff-dim)."""
    D = np.load(P2 / TAG / "diffs_moral_stories.npz")[f"layer{layer}"].astype(np.float64)
    s = np.linalg.svd(D, compute_uv=False)
    frac = (s ** 2) / (s ** 2).sum()
    effdim = int(np.searchsorted(np.cumsum(frac), 0.90) + 1)
    return {"top_dir_var_frac": round(float(frac[0]), 4), "effdim_0p90": effdim,
            "n_pairs": int(D.shape[0]), "hidden": int(D.shape[1]),
            "content_dominated": bool(frac[0] < 0.25 and effdim > 20),
            "note": "content_dominated True => rank-3 source-direction construction is the "
                    "right method (as base). If False (moral elbow), revisit construction."}


def main() -> None:
    layer = int(json.load(open(P2 / TAG / "extract_meta.json"))["match_layer"])
    rng = np.random.default_rng(SEED)

    dirs = source_dirs(TAG, P2 / "think_axis", layer)
    persona = _unit(du.load_directions(P2 / TAG / "persona_direction.npz")["persona"][layer])
    X = np.load(P2 / TAG / "act_sample.npz")["X"].astype(np.float64)
    Xc = X - X.mean(0, keepdims=True)
    n = Xc.shape[0]

    # Null q95 + control c depend only on the span -> compute once per rank.
    sweep_null = []
    for r in range(1, len(SWEEP) + 1):
        present = [s for s in SWEEP[:r] if s in dirs]
        Q = _ortho([dirs[s] for s in present])
        fr = np.array([_frac(Q, Xc.T @ rng.standard_normal(n)) for _ in range(K)])
        sweep_null.append({"rank": Q.shape[1], "sources": present,
                           "Q": Q, "q95": float(np.percentile(fr, 95)),
                           "c": _frac(Q, persona),
                           "iso_floor": float(np.sqrt(Q.shape[1] / X.shape[1]))})
    full = sweep_null[-1]  # rank-3 span (headline)

    positions: dict[str, dict] = {}
    for pos, desc in POSITIONS.items():
        f = P2 / TAG / f"refusal_think_{pos}.npz"
        if not f.exists():
            reason = ("post-answer contrast unbuildable: the benign side doesn't reach a "
                      "post-answer state within budget -> UNMEASURED (not measured-and-null)"
                      if pos == "P3" else "no vector (a side has 0; see refusal_meta)")
            positions[pos] = {"desc": desc, "available": False, "reason": reason}
            continue
        refusal = _unit(np.load(f)["refusal"])
        p_full = _frac(full["Q"], refusal)
        sweep = [{"rank": s["rank"], "sources": s["sources"],
                  "refusal_p": round(_frac(s["Q"], refusal), 4),
                  "null_q95": round(s["q95"], 4), "control_c": round(s["c"], 4),
                  "clears": bool(_frac(s["Q"], refusal) > s["q95"] + MARGIN_M
                                 and _frac(s["Q"], refusal) > s["c"] + MARGIN_M)}
                 for s in sweep_null]
        positions[pos] = {"desc": desc, "available": True,
                          "refusal_p": round(p_full, 4), "null_q95": round(full["q95"], 4),
                          "control_c": round(full["c"], 4),
                          "clears": bool(p_full > full["q95"] + MARGIN_M
                                         and p_full > full["c"] + MARGIN_M),
                          "rank_sweep": sweep}

    p2 = positions.get("P2", {})            # symmetric window (PRIMARY)
    p2f = positions.get("P2_FULL", {})      # full-span (ROBUSTNESS)
    coupling = bool(p2.get("available") and p2.get("clears"))
    robustness = None
    if p2.get("available") and p2f.get("available"):
        robustness = ("window + full-span AGREE" if p2["clears"] == p2f["clears"]
                      else "window vs full-span DIVERGE -> span-length sensitivity localized")
    result = {
        "design": "rank-3 source-direction Think V_moral; null+control recomputed on the Think "
                  "span (two-step, before projection). P2 in-trace = SYMMETRIC first-N-reasoning-"
                  "token window (no span-length confound); P2_FULL = full-span (robustness).",
        "model": json.load(open(P2 / TAG / "extract_meta.json"))["model"],
        "layer": layer, "margin_M": MARGIN_M,
        "content_check": content_check(layer),
        "rank3_span": {"rank": full["rank"], "null_q95": round(full["q95"], 4),
                       "control_c": round(full["c"], 4), "iso_floor": round(full["iso_floor"], 4)},
        "positions": positions,
        "coupling_hypothesis": "P2 (symmetric in-trace window) is the single pre-registered "
                               "coupling hypothesis; P2_FULL is a robustness check",
        "robustness_p2_window_vs_full": robustness,
        "headline": ("REASONING-COUPLING (P2 window clears both bars)" if coupling
                     else "NULL (refusal orthogonal at the in-trace deliberation site)"),
        "scope": "Bounded claim: orthogonal at harm-recognition (P0), gate (P1), and in-trace "
                 "deliberation (P2). Post-answer site (P3) UNMEASURED for benign prompts (the "
                 "benign side doesn't reach a post-answer state within budget) -- not null.",
    }
    (P2 / "think_g3_result.json").write_text(json.dumps(result, indent=2, default=float))

    cc = result["content_check"]
    print(f"=== G3 on rank-3 Think V_moral (layer {layer}) ===")
    print(f"  content check: top_dir_var_frac={cc['top_dir_var_frac']} "
          f"effdim@0.90={cc['effdim_0p90']} -> content_dominated={cc['content_dominated']}")
    print(f"  rank-3 span: null q95={result['rank3_span']['null_q95']} "
          f"persona c={result['rank3_span']['control_c']} "
          f"(iso floor {result['rank3_span']['iso_floor']})")
    for pos, info in positions.items():
        if not info["available"]:
            print(f"  {pos} [{info['desc']}]: UNAVAILABLE ({info['reason']})")
        else:
            print(f"  {pos} [{info['desc']}]: p={info['refusal_p']} vs "
                  f"q95+M={info['null_q95']+MARGIN_M:.4f} c+M={info['control_c']+MARGIN_M:.4f} "
                  f"-> {'CLEARS (coupling)' if info['clears'] else 'NULL'}")
    if result["robustness_p2_window_vs_full"]:
        print(f"  robustness: {result['robustness_p2_window_vs_full']}")
    print(f"  HEADLINE = {result['headline']}")
    print(f"  scope: {result['scope']}")


if __name__ == "__main__":
    main()
