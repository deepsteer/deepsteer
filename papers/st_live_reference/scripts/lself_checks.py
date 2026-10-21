#!/usr/bin/env python3
"""Amendment 18 item 2: the three zero-GPU checks on L-self's objective and its rule (T1.1's bank).

    python3 papers/st_live_reference/scripts/lself_checks.py \
        papers/st_live_reference/outputs/t11/gpt_oss_20b

Over the ``clause_end`` positions at block 12 of the rollouts L-self trained on (H1_train minus the
reserved 10, Amendment 17), with each scenario's judgment-frame target t and ``chat_sigma`` σ (the
T1.4 inputs):

(i)  σ-dominance: the self loss at initialization, Σ_d ((h_d − t_d) / σ_d)², summed per dimension
     over positions; the share carried by the top 1% of dimensions (29 of 2,880);
(ii) target norm: the median ‖t‖ against the 5th–95th percentile band of ‖h‖ at those positions;
(iii) descriptive (ST-A14 R_c): each target's position on the protect − rule mean-difference
     direction at those positions, 0 at the rule mean and 1 at the protect mean.

Rule: (i) > 0.5 or (ii) outside the band → instrument failure. Writes
``data/t18_lself_checks_v1.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import constancy_read as CR  # noqa: E402

INP = HERE.parent / "outputs" / "t14_inputs" / "gpt_oss_20b"
OUT = HERE.parent / "data" / "t18_lself_checks_v1.json"
TOP_FRAC, SHARE_BAR = 0.01, 0.5


def main() -> None:
    d = Path(sys.argv[1])
    rollouts = [json.loads(x) for x in (d / "gen" / "h1_rollouts_read.jsonl").open()]
    meta = json.loads((INP / "t14_inputs.json").read_text())
    z = np.load(INP / "t14_inputs.npz")
    sigma = z["chat_sigma"].astype(np.float64)
    tgt = {str(s): z["self_src"][i].astype(np.float64) for i, s in enumerate(z["self_ids"])}
    train = set(json.loads((HERE.parent / "split_v1.json").read_text())["H1_train"])
    train -= set(meta["reserved"])
    info = {i: (r["scenario_id"], CR.ACTION.get(r.get("norm_status")))
            for i, r in enumerate(rollouts) if r["scenario_id"] in train}
    code = CR.CLASSES["clause_end"]
    per_dim = np.zeros(sigma.shape[0])
    norms: list[float] = []
    sums = {"protect": np.zeros_like(sigma), "rule": np.zeros_like(sigma)}
    counts = {"protect": 0, "rule": 0}
    n_pos = 0
    for f in sorted((d / "bank").glob("bank_*.npz")):
        b = np.load(f)
        sel = (b["cls"] & code) > 0
        h = b["resid"][sel, CR.L_STAR_SLOT].astype(np.float64)
        rows = b["row"][sel]
        for hv, r in zip(h, rows):
            if int(r) not in info:
                continue
            sid, act = info[int(r)]
            per_dim += ((hv - tgt[sid]) / sigma) ** 2
            norms.append(float(np.linalg.norm(hv)))
            n_pos += 1
            if act in sums:
                sums[act] += hv
                counts[act] += 1
    k = max(1, int(round(TOP_FRAC * sigma.shape[0])))
    order = np.argsort(per_dim)[::-1]
    share = float(per_dim[order[:k]].sum() / per_dim.sum())
    lo, hi = (float(x) for x in np.percentile(norms, [5, 95]))
    tn = np.array([np.linalg.norm(tgt[s]) for s in sorted(train)])
    med = float(np.median(tn))
    mp, mr = sums["protect"] / counts["protect"], sums["rule"] / counts["rule"]
    u = (mp - mr) / np.linalg.norm(mp - mr)
    span = float((mp - mr) @ u)
    pos = np.array([float((tgt[s] - mr) @ u) / span for s in sorted(train)])
    sigma_top = sigma[order[:k]]
    fail_i, fail_ii = share > SHARE_BAR, not (lo <= med <= hi)
    out = {"rule": "ST_PLAN Amendment 18 item 2", "n_positions": n_pos,
           "n_scenarios": len(train),
           "i_sigma_dominance": {"top_k": k, "share_top_1pct": share, "bar": SHARE_BAR,
                                 "fails": fail_i, "top_dims": [int(x) for x in order[:k]],
                                 "sigma_of_top_dims_median": float(np.median(sigma_top)),
                                 "sigma_median_all": float(np.median(sigma)),
                                 "loss_per_position": float(per_dim.sum() / max(n_pos, 1))},
           "ii_target_norm": {"median_target_norm": med, "band_p5_p95": [lo, hi],
                              "fails": fail_ii,
                              "target_norm_range": [float(tn.min()), float(tn.max())]},
           "iii_placement": {"median": float(np.median(pos)),
                             "iqr": [float(x) for x in np.percentile(pos, [25, 75])],
                             "note": "0 = the rule mean, 1 = the protect mean (descriptive)"},
           "instrument_failure": bool(fail_i or fail_ii),
           "consequence": ("H1_held generation descriptive only, no panel, sigma-floored retrain "
                           "deferred" if (fail_i or fail_ii) else "evaluate as trained"),
           "inputs": {"t14_inputs.npz": hashlib.sha256(
               (INP / "t14_inputs.npz").read_bytes()).hexdigest()}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f">> L-self checks: (i) top-1% share {share:.3f} (bar {SHARE_BAR}); (ii) median target "
          f"norm {med:.0f} vs band [{lo:.0f}, {hi:.0f}]; (iii) placement median "
          f"{np.median(pos):+.2f}; instrument failure {out['instrument_failure']}")


if __name__ == "__main__":
    main()
