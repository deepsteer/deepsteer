#!/usr/bin/env python3
"""Instruct-gate MFT-null: the tightest judged-vs-judged reproduction of Paper 5's 0.1044.

Paper 5 reported 0.1044 RAW (instruct Heretic gate onto the 6-foundation MFT span, no null;
heretic_ablation.py:190). This judges the SAME instruct gate against each subspace's OWN
rank-matched null -- rank-3 instruct V_moral vs the 6-foundation instruct MFT span -- so the
"orthogonal to both" claim is judged-vs-judged, not raw-vs-null. Needs `instruct/mft_directions.npz`
(extracted on the GPT-OSS pod, payload 1); everything else is committed. Numpy only; no GPU.

Run after the GPT-OSS pod rsyncs back:  python compute_instruct_mft_null.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))
import direction_utils as du  # noqa: E402
from phase2_g3_respec import MARGIN_M, P2, SEED, K, _frac, _ortho, source_dirs  # noqa: E402

TAG = "instruct"


def main() -> None:
    layer = int(json.load(open(P2 / TAG / "extract_meta.json"))["match_layer"])
    mft_path = P2 / TAG / "mft_directions.npz"
    if not mft_path.exists():
        raise SystemExit(f"{mft_path} missing -- run the GPT-OSS pod (payload 1: instruct --mft)")

    refusal = np.load(P2 / "refusal_instruct.npz")["refusal"]
    refusal = refusal / (np.linalg.norm(refusal) + 1e-12)
    X = np.load(P2 / TAG / "act_sample.npz")["X"].astype(np.float64)
    Xc = X - X.mean(0, keepdims=True)
    n = Xc.shape[0]

    vdirs = source_dirs(TAG, P2 / "axis_instruct", layer)
    Qv = _ortho([vdirs[s] for s in ("moral_stories", "fables", "ethics")])
    mft = du.load_directions(mft_path)
    Qm = _ortho([mft[f][layer] / (np.linalg.norm(mft[f][layer]) + 1e-12) for f in mft])

    rng = np.random.default_rng(SEED)

    def judge(Q, name):
        fr = np.array([_frac(Q, Xc.T @ rng.standard_normal(n)) for _ in range(K)])
        q95, p = float(np.percentile(fr, 95)), _frac(Q, refusal)
        return {"subspace": name, "rank": int(Q.shape[1]), "refusal_p": round(p, 4),
                "null_q95": round(q95, 4), "clears": bool(p > q95 + MARGIN_M),
                "verdict": "clears (coupling)" if p > q95 + MARGIN_M else "NULL (orthogonal)"}

    res = {"design": "instruct Heretic gate judged vs each subspace's own rank-matched null "
                     "(tightest same-object reproduction of Paper 5's raw 0.1044)",
           "layer": layer, "margin_M": MARGIN_M,
           "rank3_v_moral": judge(Qv, "rank-3 V_moral"),
           "mft_6_foundation": judge(Qm, "6-foundation MFT span"),
           "paper5_raw_reference": 0.1044}
    (P2 / "instruct_mft_null_result.json").write_text(json.dumps(res, indent=2))
    print("=== instruct-gate judged-vs-judged (each vs its own rank-matched null) ===")
    for k in ("rank3_v_moral", "mft_6_foundation"):
        r = res[k]
        print(f"  {r['subspace']:22} p={r['refusal_p']} null q95={r['null_q95']} -> {r['verdict']}")
    print("  (Paper 5 raw reference: 0.1044 onto the 6-foundation MFT span, no null)")


if __name__ == "__main__":
    main()
