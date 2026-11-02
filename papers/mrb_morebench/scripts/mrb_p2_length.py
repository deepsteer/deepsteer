"""MRB Phase 2 length-matched read (MRB_PLAN v0.9.3 item 1; MRB-A13), zero GPU -> p2_length_read.json.

    mrb_p2_length.py [--out papers/mrb_morebench/outputs/p2]

Per model: β_m, the ``none`` arm's within-scenario length coupling (OLS through the origin of score
k1 − k0 on length k1 − k0, points per 1,000 characters); the length-adjusted cost Δ_adj = Δ − β_m ×
Δlength and the share of the cost that length explains, all three re-estimated together in each
scenario-bootstrap resample; beside it, the cross-scenario regression of Δscore on Δlength (its
intercept is the cost at matched length). The reading by v0.9.3's rule: R_length, R_quality, or part
length, part reasoning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mrb_g1_analyze as A  # noqa: E402

MODELS = ("muse_glimmer_30b", "qwen38_27b")
N_BOOT, SEED = 2000, 20261102


def arrays(us, gens, model, sids):
    """Per scenario and arm: score and length of k0 and k1."""
    out = {}
    for arm in ("none", "adapter"):
        sc = np.array([[us[f"{model}/{arm}/{s}/k{k}/response"]["credit_substring"] for k in (0, 1)]
                       for s in sids])
        ln = np.array([[gens[f"{model}/{arm}/{s}/k{k}"]["response_chars"] for k in (0, 1)]
                       for s in sids], dtype=float) / 1000.0  # thousands of characters
        out[arm] = (sc, ln)
    return out


def estimates(d, idx=None):
    (s0, l0), (s1, l1) = d["none"], d["adapter"]
    if idx is not None:
        s0, l0, s1, l1 = s0[idx], l0[idx], s1[idx], l1[idx]
    ds, dl = s0[:, 1] - s0[:, 0], l0[:, 1] - l0[:, 0]
    beta = float((ds * dl).sum() / (dl * dl).sum())
    delta = float((s1.mean(1) - s0.mean(1)).mean())
    dlen = float((l1.mean(1) - l0.mean(1)).mean())
    adj = delta - beta * dlen
    x, y = l1.mean(1) - l0.mean(1), s1.mean(1) - s0.mean(1)
    slope, icept = np.polyfit(x, y, 1)
    return {"beta": beta, "delta": delta, "delta_length_k": dlen, "delta_adj": adj,
            "share_explained": beta * dlen / delta if delta else float("nan"),
            "cross_slope": float(slope), "cross_intercept": float(icept)}


def ci(v):
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def reading(adj_ci, beta, share_ci):
    """v0.9.3's rule: a cost remains at matched length when Δ_adj's CI lies below 0 (part length,
    part reasoning if length explains a share above 0; R_quality if not); otherwise R_length, given
    a positive length coupling."""
    if adj_ci[1] < 0:
        return ("part length, part reasoning: a cost remains at matched length" if share_ci[0] > 0
                else "R_quality: the cost remains at matched length")
    if beta > 0:
        return "R_length: no cost remains at matched length"
    return "no cost remains at matched length, but the length coupling is not positive"


def main_from(out: Path) -> dict:
    from deepsteer.datasets.morebench import load_morebench
    items = load_morebench(subset="public")
    weights = {it.scenario_id: {c.criterion_id: c.weight for c in it.criteria} for it in items}
    us = A.unit_scores(A.read_jsonl(out / "judge" / "shard0.jsonl"), weights)
    gens = {}
    for f in sorted((out / "gen").glob("*.jsonl")):
        gens.update({r["job_id"]: r for r in A.read_jsonl(f)})
    sids = [it.scenario_id for it in items]
    rng = np.random.default_rng(SEED)
    res = {"rule": "MRB_PLAN v0.9.3 item 1", "n_boot": N_BOOT, "seed": SEED,
           "units": "score points; length in thousands of characters"}
    for m in MODELS:
        d = arrays(us, gens, m, sids)
        point = estimates(d)
        boots = [estimates(d, rng.integers(0, len(sids), len(sids))) for _ in range(N_BOOT)]
        cis = {k: ci([b[k] for b in boots]) for k in point}
        res[m] = {k: {"est": round(v, 4), "ci95": cis[k]} for k, v in point.items()}
        res[m]["reading"] = reading(cis["delta_adj"], point["beta"], cis["share_explained"])
    (out / "p2_length_read.json").write_text(json.dumps(res, indent=1) + "\n")
    for m in MODELS:
        r = res[m]
        print(f"{m}: β {r['beta']['est']:+.3f}/1k {r['beta']['ci95']} | Δ {r['delta']['est']:+.2f} "
              f"| Δlen {r['delta_length_k']['est']:+.3f}k | Δ_adj {r['delta_adj']['est']:+.2f} "
              f"{r['delta_adj']['ci95']} | share {r['share_explained']['est']:.2f} "
              f"{r['share_explained']['ci95']} | cross intercept {r['cross_intercept']['est']:+.2f} "
              f"{r['cross_intercept']['ci95']} | {r['reading']}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=A.REPO / "papers/mrb_morebench/outputs/p2")
    main_from(ap.parse_args().out)
