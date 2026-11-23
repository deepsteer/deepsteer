#!/usr/bin/env python3
"""Amendment 6's power (DRAFT; GPC_PLAN.md §14): the root cell, the depth scenarios rendered as single decisions in
the conduct grid's format, read by letter. Inputs from the grid's measured values: GPC-1's held-out cell (base 0.57,
the clause 0.25, C1 0.27 at k 1; 100 slots) and the pilot's within-slot correlation (0.83 at k 2; 87 slots). Each
render's base is drawn so the between-render spread gives that correlation (beta-binomial); the clause's relative
reduction 0.56 (the held-out cell's); C1's ratio to the clause 1.0 (it matches) or 0.13 (the depth cell's). The
read: (base - C1) / (base - C_conduct) on pooled rates, a bootstrap over renders within scenario and variant.

    python z_gpc4_a6_power.py   # writes outputs/gpc4_prep/a6_power.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "outputs/gpc4_prep/a6_power.json"
BASE, REL, ICC, SIMS, BOOT = 0.57, 0.56, 0.83, 600, 400


def beta_params(mean: float, icc: float) -> tuple[float, float]:
    s = (1 - icc) / icc  # alpha + beta for a beta-binomial with this intra-class correlation
    return mean * s, (1 - mean) * s


def one(rng, renders: int, k: int, ratio: float) -> tuple[float, float]:
    p_b = rng.beta(*beta_params(BASE, ICC), renders)
    p_c = np.clip(p_b * (1 - REL), 0, 1)
    p_a = np.clip(p_b * (1 - REL * ratio), 0, 1)
    b, c, a = (rng.binomial(k, p, renders) / k for p in (p_b, p_c, p_a))
    rb, rc, ra = b.mean(), c.mean(), a.mean()
    point = (rb - ra) / (rb - rc) if rb > rc else -np.inf
    idx = rng.integers(0, renders, (BOOT, renders))
    bb, cc, aa = b[idx].mean(1), c[idx].mean(1), a[idx].mean(1)
    rs = np.where(bb > cc, (bb - aa) / np.where(bb > cc, bb - cc, 1), -np.inf)
    return point, float(np.percentile(rs, 2.5))


def main() -> None:
    rng = np.random.default_rng(20261123)
    rec = {"rule": "Amendment 6 (DRAFT), the root cell's power; GPC_PLAN.md §14",
           "inputs": {"base": BASE, "clause_relative_reduction": REL, "within_render_icc": ICC}, "grid": {}}
    for renders in (24, 48, 96):
        for k in (2, 4):
            pas = [one(rng, renders, k, 1.0) for _ in range(SIMS)]
            fail = [one(rng, renders, k, 0.13) for _ in range(SIMS)]
            rec["grid"][f"{renders}x{k}"] = {
                "renders": renders, "k": k, "decisions_an_arm": renders * k,
                "matches_reads_lb_ge_0.5": float(np.mean([lb >= 0.5 for _, lb in pas])),
                "does_not_reads_point_lt_0.5": float(np.mean([pt < 0.5 for pt, _ in fail])),
                "misread_match_as_does_not": float(np.mean([pt < 0.5 for pt, _ in pas]))}
            print(renders, k, rec["grid"][f"{renders}x{k}"], flush=True)
    OUT.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
