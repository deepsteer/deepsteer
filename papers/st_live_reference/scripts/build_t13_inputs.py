#!/usr/bin/env python3
"""The T1.3 pod's inputs, built locally from T1.1's outputs after the constancy read (zero GPU).

    python3 papers/st_live_reference/scripts/build_t13_inputs.py \
        papers/st_live_reference/outputs/t11/gpt_oss_20b

Reads ``data/t12_constancy_v1.json`` (the picked class, the median residual norm at L* and class,
the alpha grid) and ``data/t13_draw_v1.json`` (the gate's 20, the sweep's 5), and writes
``outputs/t13_inputs/gpt_oss_20b/t13_inputs.npz`` + ``.json`` (synced to the pod):

- ``axis``: the unit anchor axis (``anchor_axis_t_inst``, block 12), the care vector's direction;
- ``fill_unit``: 5 fill directions, Gaussian in whitened space mapped back through T1.1's banked
  residuals at L* at the picked class (covariance-matched; ``steering_gate.fill_draws``, seed 0),
  unit norm; the pod scales them to the gate's alpha (norm-matched to ``care``);
- ``self_src``: the judgment-frame residual at L* (END_OF_PROMPT before the prefill, T1.1's
  ``jl_chat_neutral`` read) for each gate and sweep scenario, the ``self`` mix's source.
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
OUT = HERE.parent / "outputs" / "t13_inputs" / "gpt_oss_20b"
N_FILL, SEED, N_SAMPLE = 5, 0, 4000


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    d = Path(sys.argv[1])
    con = json.loads((DATA / "t12_constancy_v1.json").read_text())
    draw = json.loads((DATA / "t13_draw_v1.json").read_text())
    cls = con["pick"]["class"]
    code = CR.CLASSES[cls]
    zd = np.load(CR.DIRS)
    axis = zd[con["axis_of_record"]].astype(np.float64)
    axis = (axis / np.linalg.norm(axis)).astype(np.float32)
    parts = []
    for f in sorted((d / "bank").glob("bank_*.npz")):
        z = np.load(f)
        parts.append(z["resid"][(z["cls"] & code) > 0, CR.L_STAR_SLOT].astype(np.float32))
    xs = np.concatenate(parts)
    rng = np.random.default_rng(SEED)
    samp = xs[rng.choice(len(xs), min(len(xs), N_SAMPLE), replace=False)]
    fill = SG.fill_draws(samp, 1.0, N_FILL, SEED)
    jz = np.load(d / "readout" / "judgment_h1_train.npz")
    jrows = [json.loads(x)["scenario_id"] for x in (d / "readout" / "judgment_h1_train.jsonl")
             .open()]
    ids = list(draw["gate"]) + list(draw["sweep"])
    missing = sorted(set(ids) - set(jrows))
    if missing:
        raise SystemExit(f"no judgment-frame residual for {missing}")
    src = np.stack([jz["resid_eop"][jrows.index(s), CR.L_STAR_SLOT] for s in ids]).astype(
        np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "t13_inputs.npz", axis=axis, fill_unit=fill, self_ids=np.array(ids),
             self_src=src)
    meta = {"rule": "ST_PLAN Amendment 12", "class": cls, "l_star_hidden_index": 13,
            "median_norm": con["median_norm_at_l_star"][cls],
            "alphas": con["alphas"][cls], "alpha_factors": [0.05, 0.1, 0.2, 0.4],
            "gate": draw["gate"], "sweep": draw["sweep"], "fill_seed": SEED,
            "fill_sample_n": int(len(samp)), "axis": con["axis_of_record"],
            "sha256": {"t12_constancy_v1.json": sha(DATA / "t12_constancy_v1.json"),
                       "t13_draw_v1.json": sha(DATA / "t13_draw_v1.json"),
                       "directions": sha(CR.DIRS), "t13_inputs.npz": sha(OUT / "t13_inputs.npz")}}
    (OUT / "t13_inputs.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f">> t13 inputs: class {cls}, alphas {[round(x, 1) for x in meta['alphas']]}, "
          f"fill {fill.shape}, self {src.shape}")


if __name__ == "__main__":
    main()
