#!/usr/bin/env python3
"""The T1.4 pod's inputs, built locally from T1.1's outputs (zero GPU; Amendments 14 and 16).

    python3 papers/st_live_reference/scripts/build_t14_inputs.py \
        papers/st_live_reference/outputs/t11/gpt_oss_20b

Writes ``outputs/t14_inputs/gpt_oss_20b/t14_inputs.npz`` + ``.json`` (synced to the pod):

- the three unit axes at block 12: ``anchor_axis`` (``anchor_axis_t_inst``), ``care_harm`` (W4 14.4)
  and ``V_harm`` (``t_inst``);
- τ per axis by Amendment 14's rule (the 75th percentile of the per-position margin over the
  H1_train protect rollouts' ``clause_end`` positions at block 12); the anchor axis's value must
  equal ``data/t14_tau_v1.json``'s (0.0589), the other two are Amendment 16's;
- ``chat_sigma`` at hidden index 13 and the ``self`` targets (the judgment-frame residual at
  END_OF_PROMPT, block 12, per H1_train scenario), with their provenance;
- the gate's 20 scenarios (``data/t13_draw_v1.json``).
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
import st_train as ST  # noqa: E402

DATA = HERE.parent / "data"
MFT = (HERE.parents[1] / "d3_decision_anatomy" / "outputs" / "w4" / "gpt_oss_20b"
       / "mft_directions.npz")
OUT = HERE.parent / "outputs" / "t14_inputs" / "gpt_oss_20b"


def unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


def taus(d: Path, rollouts: list[dict], axes: dict[str, np.ndarray]) -> dict[str, float]:
    act = {i: CR.ACTION.get(r.get("norm_status")) for i, r in enumerate(rollouts)
           if r.get("split") == "train"}
    code = CR.CLASSES["clause_end"]
    pos: dict[str, list[float]] = {k: [] for k in axes}
    for f in sorted((d / "bank").glob("bank_*.npz")):
        z = np.load(f)
        sel = (z["cls"] & code) > 0
        h = z["resid"][sel, CR.L_STAR_SLOT].astype(np.float32)
        nrm = np.linalg.norm(h, axis=-1).clip(1e-6)
        prot = np.array([act.get(int(r)) == "protect" for r in z["row"][sel]])
        for k, a in axes.items():
            pos[k] += list(((h @ a) / nrm)[prot])
    return {k: ST.compute_tau(np.array(v)) for k, v in pos.items()}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    d = Path(sys.argv[1])
    rollouts = [json.loads(x) for x in (d / "gen" / "h1_rollouts_read.jsonl").open()]
    zd, zm = np.load(CR.DIRS), np.load(MFT)
    axes = {"anchor_axis": unit(zd["anchor_axis_t_inst"]),
            "care_harm": unit(zm["care_harm_layer12"]), "V_harm": unit(zd["V_harm_t_inst"])}
    tau = taus(d, rollouts, axes)
    ref = json.loads((DATA / "t14_tau_v1.json").read_text())["tau"]
    if abs(tau["anchor_axis"] - ref) > 1e-6:
        raise SystemExit(f"anchor-axis tau {tau['anchor_axis']} != committed {ref}")
    jz = np.load(d / "readout" / "judgment_h1_train.npz")
    jrows = [json.loads(x)["scenario_id"] for x in (d / "readout" / "judgment_h1_train.jsonl")
             .open()]
    sigma = np.load(d / "readout" / "chat_sigma_h13.npy").astype(np.float32)
    rec = json.loads((d / "t11_record.json").read_text())
    draw = json.loads((DATA / "t13_draw_v1.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "t14_inputs.npz", **axes, chat_sigma=sigma, self_ids=np.array(jrows),
             self_src=jz["resid_eop"][:, CR.L_STAR_SLOT].astype(np.float32))
    meta = {"rule": "ST_PLAN Amendments 14 and 16", "layer": "block 12 (hidden index 13)",
            "class": "clause_end", "tau": tau, "gate": draw["gate"],
            "chat_sigma_provenance": rec["phases"]["h"]["chat_sigma"]["13"],
            "self_target": "judgment-frame residual, END_OF_PROMPT before the prefill, block 12, "
                           "jl_chat_neutral, one per H1_train scenario (T1.1)",
            "sha256": {"directions": sha(CR.DIRS), "mft_directions": sha(MFT),
                       "t14_tau_v1.json": sha(DATA / "t14_tau_v1.json"),
                       "t14_inputs.npz": sha(OUT / "t14_inputs.npz")}}
    (OUT / "t14_inputs.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f">> t14 inputs: tau {json.dumps({k: round(v, 5) for k, v in tau.items()})}; "
          f"self targets {len(jrows)}")


if __name__ == "__main__":
    main()
