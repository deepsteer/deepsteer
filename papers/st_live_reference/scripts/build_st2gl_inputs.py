#!/usr/bin/env python3
"""The GPT-OSS ladder's fill draws (ST_PLAN Amendment 50 and its additions): ten covariance-matched
unit draws at block 12's ``clause_end`` (hidden index 13), the recipe of Cell 1's five
(``build_t13_inputs.py``: T1.1's banked residuals at the picked class, a 4,000-row sample,
``steering_gate.fill_draws``) with a new seed (2), so the ten are fresh draws of the same kind.
Local; no model.

    python3 papers/st_live_reference/scripts/build_st2gl_inputs.py <outputs/t11/gpt_oss_20b>
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
import steering_gate as SG  # noqa: E402

DATA = HERE.parent / "data"
ST2 = HERE.parent / "outputs" / "st2_inputs" / "gpt_oss_20b" / "st2_inputs.npz"
OUT = HERE.parent / "outputs" / "st2gl_inputs" / "gpt_oss_20b"
N_FILL, SEED, N_SAMPLE = 10, 2, 4000


def main() -> None:
    d = Path(sys.argv[1])
    con = json.loads((DATA / "t12_constancy_v1.json").read_text())
    code = CR.CLASSES[con["pick"]["class"]]
    parts = []
    for f in sorted((d / "bank").glob("bank_*.npz")):
        z = np.load(f)
        parts.append(z["resid"][(z["cls"] & code) > 0, CR.L_STAR_SLOT].astype(np.float32))
    xs = np.concatenate(parts)
    rng = np.random.default_rng(SEED)
    samp = xs[rng.choice(len(xs), min(len(xs), N_SAMPLE), replace=False)]
    fill = SG.fill_draws(samp, 1.0, N_FILL, SEED).astype(np.float32)
    old = np.load(ST2)["fill_unit"].astype(np.float64)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "fill10.npz", fill_unit=fill)
    meta = {"rule": "ST_PLAN Amendment 50: ten covariance-matched unit draws (seed 2) at "
                    "clause_end, block 12, Cell 1's recipe", "class": con["pick"]["class"],
            "seed": SEED, "n_draws": N_FILL, "sample_n": int(len(samp)),
            "max_abs_cos_to_cell1_draws": float(np.abs(fill.astype(np.float64) @ old.T).max()),
            "median_norm": con["median_norm_at_l_star"][con["pick"]["class"]],
            "sha256": {"fill10.npz": hashlib.sha256((OUT / "fill10.npz").read_bytes())
                       .hexdigest()}}
    (OUT / "fill10.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f">> fill10: {N_FILL} draws from {len(samp)} rows; max |cos| to Cell 1's draws "
          f"{meta['max_abs_cos_to_cell1_draws']:.3f}; median norm {meta['median_norm']:.1f}")


if __name__ == "__main__":
    main()
