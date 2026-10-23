"""The ``care_harm`` replicate at 0.3x on Muse (ST_PLAN Amendment 32; ST-A20's candidate). One HF
process on the ST stack, on Cell 4's derived quantities (``--cell4``), with fresh fill draws
(``--inputs``, built locally by ``--build-inputs``: five covariance-matched unit draws, seed 1, from
Cell 4's block-26 ``clause_end`` bank):

- ``spec_none`` (parity with Cell 4's baseline), ``spec_care_03`` (parity with the ladder's read:
  deterministic), ``spec_fill_03`` (the fresh draws, draw = item index mod 5), all before any arm;
- ``arm_none``, ``arm_fill_03`` (the fresh draws cycled by rollout) and ``arm_care_03`` at
  ``clause_end``, block 26, alpha 0.3x the median norm, on Cell 4's 45 engaged x 16 with fresh
  seeds (stage ``st2mr``) shared across the three arms.

The analysis of record is local (``analyze_st2mr.py``).

    python3 st2mr_driver.py --build-inputs --cell4 <Cell 4 dir> --inputs <dir>     # local, zero GPU
    python3 st2mr_driver.py --out <dir> --cell4 <Cell 4 dir> --inputs <dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "kdg_panel" / "scripts"))
import screen_lib as L  # noqa: E402
import st2_driver as S2  # noqa: E402
import st2m_driver as M  # noqa: E402
import st2ml_driver as ML  # noqa: E402
import steering_gate as SG  # noqa: E402

RUNG = "03"
R = 16
STAGE = "st2mr"
FILL_SEED = 1
PHASES = ["spec_none", "spec_care_03", "spec_fill_03", "arm_none", "arm_fill_03", "arm_care_03"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_inputs(cell4: Path, out: Path) -> dict:
    """Amendment 32 item 1: five fresh covariance-matched unit draws (seed 1) from Cell 4's bank."""
    b = M.BLOCK
    z = np.load(cell4 / "bank" / "clause_end.npz")
    s = z[f"sample_b{b}"].astype(np.float64)
    fu = SG.fill_draws(s, 1.0, 5, FILL_SEED).astype(np.float32)
    old = np.load(cell4 / "derive.npz")["fill_unit"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "fill_seed1.npz", fill_unit=fu)
    meta = {"rule": "ST_PLAN Amendment 32 item 1", "seed": FILL_SEED, "block": b,
            "n_samples": int(len(s)), "max_abs_cos_to_cell4_draws": float(np.abs(fu @ old.T).max()),
            "sha256": {"clause_end.npz": hashlib.sha256((cell4 / "bank" / "clause_end.npz")
                                                        .read_bytes()).hexdigest(),
                       "fill_seed1.npz": hashlib.sha256((out / "fill_seed1.npz")
                                                        .read_bytes()).hexdigest()}}
    (out / "fill_seed1.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def context(a) -> dict:
    c = ML.context(a)
    c.update(record=c["out"] / "st2mr_record.json", r=R,
             inp={"fill_unit": np.load(a.inputs / "fill_seed1.npz")["fill_unit"]})
    if a.dry_run:  # the dry Cell 4's width: fresh draws at that width
        h = c["dirs"]["care_harm"].shape[0]
        rng = np.random.default_rng(FILL_SEED)
        c["inp"] = {"fill_unit": SG.fill_draws(rng.standard_normal((64, h)), 1.0, 5,
                                               FILL_SEED)}
        c["r"] = 2
    return c


def phase_arm(c, sc, model, name: str) -> dict:
    al = ML.alpha(c, RUNG)
    jobs = L.h1_jobs(sc, c["H"], c["engaged"], c["r"])
    if name == "none":
        iv = None
    elif name == "fill_03":
        fu = c["inp"]["fill_unit"]
        iv = SG.Add(np.stack([al * fu[j["k"] % len(fu)] for j in jobs]).astype(np.float32), 1.0)
    else:
        iv = SG.Add(c["dirs"]["care_harm"], al)
    rows = S2.run_arm(c, sc, model, STAGE, f"arm_{name}", jobs, iv,
                      extra=lambda i: {"alpha": 0.0 if name == "none" else al, "axis": name})
    return {"status": "ok", "n": len(rows), "alpha": 0.0 if name == "none" else al,
            "parse_rate": float(np.mean([r["letter"] is not None for r in rows]))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cell4", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--build-inputs", action="store_true")
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--date-pin", default=None, help="unused: Cell 4's rows set the pin")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    a = ap.parse_args()
    if a.build_inputs:
        m = build_inputs(a.cell4, a.inputs)
        print(f">> st2mr inputs: {json.dumps({k: m[k] for k in ('seed', 'n_samples')})}")
        return 0
    c = context(a)
    rec = M.load_record(c)
    todo = [p for p in a.phases.split(",") if not (rec["phases"].get(p) or {}).get("finished")]
    model, sc, per = None, None, []
    for name in todo:
        n = len(c["engaged"]) * c["r"] if name.startswith("arm_") else 0
        est = float(np.mean(per)) * n if per and n else 0.0
        if a.deadline and time.time() + est > a.deadline:
            print(f">> st2mr {name}: not started (would end past the deadline)", flush=True)
            return 3
        t0 = time.time()
        if model is None:
            model = M.load_model(c, a)
            sc = M.screen(c, model, c["pin"], a.dry_run)
        if name == "spec_none":
            res = ML.phase_spec_none(c, sc, model)
        elif name == "spec_care_03":
            res = ML.phase_spec_care(c, sc, model, RUNG)
        elif name == "spec_fill_03":
            res = ML.phase_spec_fill(c, sc, model, RUNG)
        else:
            res = phase_arm(c, sc, model, name[len("arm_"):])
            per.append((time.time() - t0) / n)
        rec = M.load_record(c)
        rec["phases"][name] = {**res, "seconds": time.time() - t0, "finished": _now()}
        M.save_record(c, rec)
        brief = {k: res[k] for k in ("n", "rows", "alpha", "parse_rate", "parity") if k in res}
        print(f">> st2mr {name}: done ({json.dumps(brief, default=str)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
