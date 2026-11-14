#!/usr/bin/env python3
"""Amendment 52's pod: validation, then the registered reads, then a descriptive strength sweep.

    cst_a52_pod.py --stage validate|sweep_prep|sweep|fits --out DIR [--next STAGE]

- ``validate`` (item 3a, after the runner's stack check, gate tests and fixed smoke check): the
  base (``base_session``) and P-ctx on one LoRA engine, the operator cell's 458 clear items at k 2
  and cell A′ at k 8, P-ctx's served parity first (``pilot_pod_lib.stage_read``); then the
  reproduction: P-ctx minus the same-session base on the operator cell's ordered share,
  scenario-paired (``sdf_reread_quick.paired``). **Passes when its CI contains the RunPod draws'
  range** (−0.286 to −0.272: P-ctx −0.275 and its second draw −0.286 on the verification pod, config
  1 −0.272 on the loop pod; Amendment 51 item D5). ``validation.json``; the runner stops on a fail.
- ``sweep_prep`` (item 3d): P-ctx's adapter copied at 0.25, 0.5 and 0.75 of its update (every
  ``lora_B`` matrix scaled, so the served update is that fraction), the structural check on each,
  the HF letters for the 0.5 copy (its parity is read); the 0.25 and 0.75 copies carry a parity
  record that says parity was not run (item 3d: parity on one) and are read.
- ``sweep``: the three copies on the operator cell and cell A′ (``stage_read``, no base: stage a's
  base and full-strength read are the end points).
- ``fits --next``: the cut rule (a stage that would end past 10.25 h is skipped from the end).

The MACHIAVELLI stages (3b ``a52_a0ngood``, 3c ``a49_pctx``) are ``cst_teacher_pod.py``'s. The read
is ``a52_read.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2]), str(HERE.parents[1] / "kdg_panel/scripts"),
                str(HERE.parents[1] / "st_live_reference/scripts"),
                str(HERE.parents[1] / "mrb_morebench/scripts"),
                str(HERE.parents[1] / "mch_machiavelli/scripts")]

RUNPOD_RANGE = (-0.286, -0.272)  # the RunPod draws of P-ctx's operator-cell contrast
STRENGTHS = (0.25, 0.5, 0.75)
PARITY_ON = 0.5  # item 3d: parity on one copy
PARTS = ("panel_operator", "forced_plain")
K = {"panel_operator": 2, "forced_plain": 8}
CUT_H = 10.25
FORECAST_H = {"a52_a0ngood": 1.94, "a49_pctx": 3.17, "sweep": 1.8}  # RunPod's measured rates


def tag_of(s: float) -> str:
    return f"p_ctx_s{int(round(s * 100)):03d}"


def scale_adapter(src: Path, dst: Path, s: float) -> dict:
    """A copy of the adapter whose served update is ``s`` of the original's: every ``lora_B``
    tensor times ``s`` (the update is B A scaled by alpha over rank, linear in B)."""
    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)
    w = load_file(str(src / "adapter_model.safetensors"))
    n = 0
    for k in list(w):
        if "lora_B" in k:
            w[k] = (w[k].float() * s).to(w[k].dtype)
            n += 1
    if not n:
        raise ValueError(f"{src}: no lora_B tensors to scale")
    save_file(w, str(dst / "adapter_model.safetensors"))
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")
    return {"strength": s, "scaled_tensors": n, "of": len(w)}


def reproduction(out: Path) -> dict:
    """Item 3a's read: P-ctx minus the same-session base, the operator cell's ordered share."""
    import s0a_analysis as A
    import sdf2_quick as Z
    import sdf_reread_quick as Q

    lab = A.labels()
    gen = out / "gen"
    acted = lambda r: not r["capped"] and r["option"] is not None  # noqa: E731
    u = {t: Q.per_scenario(Z.cell_rows(t, "operator", lab, gen), acted)
         for t in ("base_session", "p_ctx")}
    ids = sorted(set(u["base_session"]) & set(u["p_ctx"]))
    c = Q.paired(u["p_ctx"], u["base_session"], ids)
    lo, hi = c["ci95"]
    rec = {"rule": "Amendment 52 item 3a (Amendment 51 item D5)", "n_items": len(ids),
           "p_ctx_minus_base": c, "runpod_range": list(RUNPOD_RANGE),
           "passes": bool(lo <= RUNPOD_RANGE[0] and RUNPOD_RANGE[1] <= hi)}
    return rec


def stage_validate(out: Path, dry: bool) -> dict:
    import cst_finalist_pod as F
    import pilot_pod_lib as PL

    r = PL.stage_read(out, [("p_ctx", F.cand_dir(out, "p_ctx"))], dry, parts=PARTS, k_override=K,
                      base=True)
    rec = {"read": r, **(reproduction(out) if not dry and "p_ctx" in r["read"] else
                         {"passes": "p_ctx" in r["read"] and dry, "dry": dry})}
    F.T._w(out, "validation", rec)  # noqa: SLF001
    print(">> validation:", json.dumps({k: v for k, v in rec.items() if k != "read"}), flush=True)
    return rec


def stage_sweep_prep(out: Path, dry: bool) -> dict:
    import cst_finalist_pod as F
    import pilot_pod_lib as PL

    src = Path(F.cand_dir(out, "p_ctx"))
    path = F.S.V._model_path(dry)  # noqa: SLF001
    rec = {"rule": "Amendment 52 item 3d", "copies": {}}
    for s in STRENGTHS:
        tag = tag_of(s)
        d = out / "adapters" / tag / "1"
        info = scale_adapter(src, d, s)
        info["structural"] = PL.structural_check(d, path)
        if not info["structural"].get("passes", True):
            raise SystemExit(f"BAIL: the {tag} copy fails the structural check")
        (out / f"train_{tag}.json").write_text(json.dumps(
            {"rung": tag, "trained": True, "scaled_from": str(src), "strength": s,
             "adapters": {"1": str(d)}}) + "\n")
        rec["copies"][tag] = info
    PL.hf_letters(out, [(tag_of(PARITY_ON), str(out / "adapters" / tag_of(PARITY_ON) / "1"))],
                  dry)
    for s in STRENGTHS:
        if s != PARITY_ON:
            f = out / "parity" / f"{tag_of(s)}.json"
            if not f.exists():
                PL.write_once(f, {"tag": tag_of(s), "passes": None, "read": True,
                                  "parity_not_run": "Amendment 52 item 3d: parity on one copy"})
    F.T._w(out, "sweep_prep", rec)  # noqa: SLF001
    print(">> sweep_prep:", json.dumps(rec), flush=True)
    return rec


def stage_sweep(out: Path, dry: bool) -> dict:
    import cst_finalist_pod as F
    import pilot_pod_lib as PL

    tags = [(tag_of(s), F.cand_dir(out, tag_of(s))) for s in STRENGTHS]
    return PL.stage_read(out, tags, dry, parts=PARTS, k_override=K, base=False)


def fits(out: Path, nxt: str) -> bool:
    import os

    import cst_finalist_pod as F

    start = float(os.environ.get("CST_START_EPOCH", time.time()))
    elapsed = (time.time() - start) / 3600
    fc = FORECAST_H[nxt]
    rec = {"rule": "Amendment 52 item 3 (cut from the end past 10.5 h)", "stage": nxt,
           "elapsed_h": elapsed, "forecast_h": fc, "ends_h": elapsed + fc, "cut_h": CUT_H,
           "runs": elapsed + fc <= CUT_H}
    F.T._w(out, f"cut_{nxt}", rec)  # noqa: SLF001
    print(">> cut rule:", json.dumps(rec), flush=True)
    return rec["runs"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("validate", "sweep_prep", "sweep", "fits"))
    ap.add_argument("--next", choices=tuple(FORECAST_H))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.stage == "fits":
        sys.exit(0 if fits(a.out, a.next) else 1)
    import cst_finalist_pod as F

    F.configure()
    {"validate": stage_validate, "sweep_prep": stage_sweep_prep,
     "sweep": stage_sweep}[a.stage](a.out, a.dry_run)


if __name__ == "__main__":
    main()
