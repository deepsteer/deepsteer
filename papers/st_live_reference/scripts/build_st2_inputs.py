#!/usr/bin/env python3
"""ST Phase 2's inputs (Amendment 22; zero GPU), built locally from committed and T1.x artifacts.

    python3 papers/st_live_reference/scripts/build_st2_inputs.py engaged \
        papers/st_live_reference/outputs/t11/gpt_oss_20b
    python3 papers/st_live_reference/scripts/build_st2_inputs.py cell1

- ``engaged``: ``data/st2_engaged_train_v1.json``, T1.1's 45 brake-engaged H1_train ids copied from
  ``select/engaged.json`` (no new draw).
- ``cell1``: ``outputs/st2_inputs/gpt_oss_20b/st2_inputs.npz`` + ``.json`` (synced to the pod):
  the unit anchor axis, ``care_harm`` and ``V_harm`` at block 12 (exactly the T1.4 inputs' vectors)
  and the T1.3 gate's five unit fill draws, with the sources' sha256.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTS = HERE.parent / "outputs"
DATA = HERE.parent / "data"
T14 = OUTS / "t14_inputs" / "gpt_oss_20b" / "t14_inputs.npz"
T13 = OUTS / "t13_inputs" / "gpt_oss_20b" / "t13_inputs.npz"
OUT = OUTS / "st2_inputs" / "gpt_oss_20b"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def engaged(t11: Path) -> dict:
    f = t11 / "select" / "engaged.json"
    t = json.loads(f.read_text())["train"]
    if len(t["engaged_ids"]) != t["n_engaged"]:
        raise SystemExit("engaged.json is inconsistent")
    split = json.loads((HERE.parent / "split_v1.json").read_text())
    if not set(t["engaged_ids"]) <= set(split["H1_train"]):
        raise SystemExit("an engaged id is not in H1_train")
    return {"rule": f"T1.1's H1_train brake-engaged subset ({t['rule']}); ST_PLAN Amendment 22",
            "n": t["n_engaged"], "ids": sorted(t["engaged_ids"]), "source_sha256": sha(f)}


def cell1() -> dict:
    a, f = np.load(T14), np.load(T13)
    vecs = {k: a[k].astype(np.float32) for k in ("anchor_axis", "care_harm", "V_harm")}
    for k, v in vecs.items():
        if abs(float(np.linalg.norm(v)) - 1.0) > 1e-5:
            raise SystemExit(f"{k} is not unit")
    if not np.allclose(vecs["anchor_axis"], f["axis"], atol=1e-6):
        raise SystemExit("the T1.4 anchor axis differs from the T1.3 gate's")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "st2_inputs.npz", **vecs, fill_unit=f["fill_unit"].astype(np.float32))
    meta = {"rule": "ST_PLAN Amendment 22, Cell 1", "layer": "block 12 (hidden index 13)",
            "class": "clause_end", "alpha": 267.301171875,
            "vectors": {"anchor_axis": "t14_inputs anchor_axis (= the T1.3 gate's axis)",
                        "care_harm": "t14_inputs care_harm (W4 14.4, unit)",
                        "V_harm": "t14_inputs V_harm (t_inst, unit)",
                        "fill_unit": "t13_inputs fill_unit (5 covariance-matched unit draws)"},
            "cos": {"care_harm.V_harm": float(vecs["care_harm"] @ vecs["V_harm"]),
                    "care_harm.anchor_axis": float(vecs["care_harm"] @ vecs["anchor_axis"])},
            "sha256": {"t14_inputs.npz": sha(T14), "t13_inputs.npz": sha(T13),
                       "st2_inputs.npz": sha(OUT / "st2_inputs.npz")}}
    (OUT / "st2_inputs.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> None:
    if sys.argv[1:2] == ["engaged"]:
        e = engaged(Path(sys.argv[2]))
        (DATA / "st2_engaged_train_v1.json").write_text(json.dumps(e, indent=1) + "\n")
        print(f">> engaged H1_train: {e['n']}")
    elif sys.argv[1:2] == ["cell1"]:
        m = cell1()
        print(f">> st2 inputs (cell 1): cos {m['cos']}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
