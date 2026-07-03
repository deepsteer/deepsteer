#!/usr/bin/env python3
"""#17 — Stage-1 anatomy raw -> standardized invariance check (A1 legitimacy proof).

The massive-activation degeneracy (ANOMALIES A1) forces per-dim standardization for the Llama/Qwen
panel. Standardization is only legitimate if it leaves the CLEAN instrument's verdict unchanged: on
OLMo-3 (top-dim variance ~1.4%), the per-head refusal-write specificity ranking must be the same in
raw and per-dim-standardized space. This runs entirely on the saved per-head contribution arrays +
the decision-token channel sample (zero-GPU). If OLMo is invariant, the standardized Stage-1
extraction is trusted for the panel.

Standardization: z-score each residual dim by sigma from the decision-token channel sample (the
format/position-matched, sink-free act sample). Compare raw vs standardized head specificity
(ranking + top-k membership + the lead head).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage1_attribution as s1  # noqa: E402

SPEARMAN_MIN = 0.85     # pre-declared invariance thresholds
TOPK_JACCARD_MIN = 0.70


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def invariance(npz_path: Path, k: int = 10) -> dict:
    z = np.load(npz_path)
    contribs = z["per_head_contribs"].astype(np.float64)          # (N, hidden) folded residual writes
    keys = [tuple(int(x) for x in kk) for kk in z["per_head_keys"]]
    X = z["channel_act"].astype(np.float64)                       # (n, hidden) decision-token sample
    r = z["refusal"].astype(np.float64)
    sigma = X.std(0)
    sigma = np.where(sigma > 1e-8, sigma, 1.0)

    def spec_map(C, rr, Xc):
        Qc = s1.channel_control_basis(Xc, rr)
        cdict = {h: C[i] for i, h in enumerate(keys)}
        return s1.head_specificity(cdict, rr, Qc), Qc.shape[1]

    raw, ch_raw = spec_map(contribs, r, X)                        # raw space
    std, ch_std = spec_map(contribs / sigma, r / sigma, X / sigma)  # per-dim standardized

    raw_s = np.array([raw[h]["specificity"] for h in keys])
    std_s = np.array([std[h]["specificity"] for h in keys])
    rho = _spearman(raw_s, std_s)
    raw_top = [keys[i] for i in np.argsort(-np.abs(raw_s))[:k]]
    std_top = [keys[i] for i in np.argsort(-np.abs(std_s))[:k]]
    jac = len(set(raw_top) & set(std_top)) / len(set(raw_top) | set(std_top))
    lead_match = raw_top[0] == std_top[0]
    top_dim_share = float((X.var(0).max()) / (X.var(0).sum() + 1e-12))
    invariant = bool(rho >= SPEARMAN_MIN and jac >= TOPK_JACCARD_MIN and lead_match)
    return {"model": "olmo3", "n_heads": len(keys), "channel_dim_raw": ch_raw, "channel_dim_std": ch_std,
            "top_dim_variance_share": round(top_dim_share, 4),
            "spearman_raw_std": round(rho, 4), f"top{k}_jaccard": round(jac, 4),
            "lead_head_raw": list(raw_top[0]), "lead_head_std": list(std_top[0]),
            "lead_match": lead_match, "invariant": invariant,
            "thresholds": {"spearman_min": SPEARMAN_MIN, "topk_jaccard_min": TOPK_JACCARD_MIN},
            "raw_top": [list(h) for h in raw_top], "std_top": [list(h) for h in std_top]}


def main():
    out = HERE.parent / "outputs"
    res = invariance(out / "c1_inputs_olmo3.npz")
    (out / "standardized_invariance_olmo3.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\n{'PASS' if res['invariant'] else 'FAIL'}: OLMo Stage-1 anatomy is "
          f"{'raw->standardized INVARIANT -> standardized extraction licensed for the panel' if res['invariant'] else 'NOT invariant -> investigate before the panel'}")


if __name__ == "__main__":
    main()
