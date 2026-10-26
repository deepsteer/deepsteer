"""The `care_harm` ladder on GPT-OSS-20B (ST_PLAN Amendment 50 and its additions; ST-A29's
discriminator). One HF process on pod C's stack (model, screen, panel and specificity set),
resumable phase by phase.

- ``spec_none``: the specificity set (60 rule-right x 8, dose 0, single-row), the parity re-read;
- ``spec_fill10_<rung>``: each of ten fresh covariance-matched fill draws (``--fill``, seed 2) at
  every input position of block 12, on all 60 items (fully crossed; Amendment 38 item 2's rule);
- ``spec_care_<rung>``: ``care_harm`` at every input position;
- ``arm_care_<rung>``: ``care_harm`` at ``clause_end``, block 12, on Cell 1's 45 engaged x 8 with
  Cell 1's stage and seeds (rollouts 0-7), so the rows pair with Cell 1's ``none``; coherence
  against it (Amendment 13's relative bars).

0.4x the median norm (alpha 534.6) is the phase of record. If ``arm_care_04`` fails a coherence bar,
the driver appends the 0.3x phases (alpha 401.0) on the pod (Amendment 50, second addition, item 1).
The analysis of record is local (``analyze_st2gl.py``).

    python3 st2gl_driver.py --out <dir> --inputs <st2_inputs dir> --podc-inputs <st2c_inputs dir> \
        --c1 <pod A's gpt_oss_20b dir> --fill <st2gl_inputs dir> [--dry-run] [--deadline <epoch>]
"""

from __future__ import annotations

import argparse
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
import st2c_driver as C  # noqa: E402
import steering_gate as SG  # noqa: E402
import t13_driver as T13  # noqa: E402

RUNGS = {"04": 0.4, "03": 0.3}  # 0.4x of record; 0.3x only if 0.4x is incoherent
FIRST = ["spec_none", "spec_fill10_04", "spec_care_04", "arm_care_04"]
SECOND = ["spec_fill10_03", "spec_care_03", "arm_care_03"]
R = 8  # Cell 1's rollouts 0-7


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def context(a) -> dict:
    c = C.context(a)
    c["record"] = c["out"] / "st2gl_record.json"
    c["fill10"] = np.load(a.fill / "fill10.npz")["fill_unit"]
    c["median_norm"] = S2.ALPHA / 0.2  # Cell 1's alpha is 0.2x the median norm (1,336.5)
    return c


def alpha(c: dict, rung: str) -> float:
    return c["alpha"] / 0.2 * RUNGS[rung] if c.get("dry") else RUNGS[rung] * c["median_norm"]


def phase_spec_fill10(c, sc, model, rung: str) -> dict:
    """Each of the ten draws on all 60 items (fully crossed), merged with the draw recorded."""
    al, fu = alpha(c, rung), c["fill10"]
    items = [c["panel"][i] for i in c["spec_ids"]]
    rows: list[dict] = []
    for g in range(len(fu)):
        with C.add_everywhere(model, c["layer"], al * fu[g]):
            L.run_letter_cell(sc, items, "dl_chat_neutral", f"spec/fill10_{rung}/draw{g}")
        part = c["out"] / "spec" / f"fill10_{rung}" / f"draw{g}" / "dl_chat_neutral.jsonl"
        rows += [{**json.loads(x), "fill_draw": g, "alpha": al} for x in part.open()]
    sc.save_jsonl(f"spec/fill10_{rung}/dl_chat_neutral.jsonl", rows)
    return {"status": "ok", "rows": len(rows), "alpha": al, "draws": len(fu),
            "rule": "ten draws, each on all 60 items (Amendment 50; Amendment 38 item 2)"}


def phase_spec_care(c, sc, model, rung: str) -> dict:
    al = alpha(c, rung)
    with C.add_everywhere(model, c["layer"], al * c["inp"]["care_harm"]):
        res = L.run_letter_cell(sc, [c["panel"][i] for i in c["spec_ids"]], "dl_chat_neutral",
                                f"spec/care_{rung}")
    return {"status": "ok", "alpha": al, **res}


def phase_arm(c, sc, model, rung: str) -> dict:
    al = alpha(c, rung)
    jobs = L.h1_jobs(sc, c["H"], c["engaged"], min(R, c["r1"]))  # Cell 1 holds 16 (dry: 2)
    rows = S2.run_arm(c, sc, model, "st2c1", f"gl_care_{rung}", jobs,
                      SG.Add(c["inp"]["care_harm"], al),
                      extra=lambda i: {"alpha": al, "axis": "care_harm", "rung": RUNGS[rung]})
    none = [json.loads(x) for x in c["c1_none"].open()]
    keys = {r["rollout_key"] for r in rows}
    none = [r for r in none if r["rollout_key"] in keys]
    if keys != {r["rollout_key"] for r in none}:
        raise SystemExit(f"gl_care_{rung} rows do not pair with Cell 1's none")
    coh = T13._coh(rows, none, relative=True)  # noqa: SLF001
    return {"status": "ok", "n": len(rows), "alpha": al, "coherence": coh,
            "coherent": bool(coh["passes"]),
            "parse_rate": float(np.mean([r["letter"] is not None for r in rows]))}


def then(name: str, done: dict, dry: bool) -> list[str]:
    """Amendment 50, second addition, item 1: the 0.3x phases follow ``arm_care_04`` only if 0.4x
    failed a coherence bar."""
    return SECOND if name == "arm_care_04" and not done.get("coherent") and not dry else []


def run(name: str, c, sc, model, a) -> dict:
    if name == "spec_none":
        return C.phase_spec(c, sc, model, None, a, "none")
    rung = name.rsplit("_", 1)[1]
    if name.startswith("spec_fill10_"):
        return phase_spec_fill10(c, sc, model, rung)
    if name.startswith("spec_care_"):
        return phase_spec_care(c, sc, model, rung)
    return phase_arm(c, sc, model, rung)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True, help="the st2_inputs dir")
    ap.add_argument("--podc-inputs", type=Path, required=True, help="the st2c_inputs dir")
    ap.add_argument("--c1", type=Path, required=True, help="pod A's output dir (Cell 1's none)")
    ap.add_argument("--fill", type=Path, required=True, help="the st2gl_inputs dir (fill10)")
    ap.add_argument("--date-pin", default=None, help="unused: Cell 1's none sets the pin")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    a = ap.parse_args()
    c = context(a)
    c["dry"] = a.dry_run
    pin = S2.c1_date_pin(c)
    model, sc = None, None
    todo = list(FIRST)
    while todo:
        name = todo.pop(0)
        rec = C.load_record(c)
        done = rec["phases"].get(name) or {}
        if not done.get("finished"):
            if a.deadline and name.startswith("arm_") and time.time() > a.deadline:
                print(f">> st2gl {name}: not started (past the deadline)", flush=True)
                return 3
            t0 = time.time()
            if model is None:
                model = C.load_model(c, a)
                if a.dry_run:  # the tiny width: fresh draws at that width
                    h = model.model.config.hidden_size
                    c["fill10"] = SG.fill_draws(np.random.default_rng(2).standard_normal((64, h)),
                                                1.0, 10, 2)
                sc = C.screen(c, model, a, pin)
            res = run(name, c, sc, model, a)
            rec = C.load_record(c)
            rec["phases"][name] = {**res, "seconds": time.time() - t0, "finished": _now()}
            C.save_record(c, rec)
            done = rec["phases"][name]
            brief = {k: res[k] for k in ("n", "rows", "alpha", "parse_rate", "coherent")
                     if k in res}
            print(f">> st2gl {name}: done ({json.dumps(brief, default=str)})", flush=True)
        nxt = then(name, done, a.dry_run)
        if nxt:
            print(">> st2gl: 0.4x failed a coherence bar; the 0.3x phases run (Amendment 50, "
                  "second addition)", flush=True)
            todo += nxt
    return 0


if __name__ == "__main__":
    sys.exit(main())
