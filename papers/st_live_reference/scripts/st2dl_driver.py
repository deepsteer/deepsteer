"""ST Phase 2 D-lite on GPT-OSS-20B (ST_PLAN Amendment 29; author, 2026-10-22): the random-vector
references pod C did not run, before Muse's transfer test. One HF process on the ST stack of record
(pod C's model, screen, arms and readouts), resumable phase by phase.

- ``rr_none``, ``rr_fill``, ``rr_fill_half``: pod C's 60 plain violating requests x 4 on pod C's
  seeds; ``rr_none`` re-run here so the fill arms pair rollout by rollout on one engine; the fill
  arms add Cell 1's five fill draws (cycled by rollout) at ``clause_end``, block 12, alpha 267.3 and
  133.6 (ST-A18's retention discriminator).
- ``spec_none``, ``spec_fill``, ``spec_fill_half``: the specificity set (60 rule-right x 8, dose 0,
  single-row) with a fill draw added at every input position at block 12, draw = the item's index
  in the set mod 5, alpha 267.3 and 133.6 (ST-A18's specificity discriminator; Amendment 29's
  S1/S2); ``spec_none`` repeats pod C's parity check against T1.3c's rows.

The analysis of record is local (``analyze_st2dl.py``).

    python3 st2dl_driver.py --out <dir> --inputs <st2_inputs dir> --podc-inputs <st2c_inputs dir> \
        --c1 <pod A's gpt_oss_20b dir> [--phases ...] [--dry-run]
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

PODDL = ["rr_none", "rr_fill", "rr_fill_half", "spec_none", "spec_fill", "spec_fill_half"]
GEN = {"rr_none", "rr_fill", "rr_fill_half"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def context(a) -> dict:
    c = C.context(a)
    c["record"] = c["out"] / "st2dl_record.json"
    return c


def phase_spec_fill(c, sc, model, rec, a, arm: str) -> dict:
    """One fill draw per item (index mod 5), added at every input position at block 12; the
    draw groups are read separately and merged into ``spec/<arm>/dl_chat_neutral.jsonl``."""
    al = c["alpha"] / 2 if arm == "fill_half" else c["alpha"]
    fu = c["inp"]["fill_unit"]
    items = [c["panel"][i] for i in c["spec_ids"]]
    rows: list[dict] = []
    t0 = time.time()
    for g in range(len(fu)):
        sub = [it for j, it in enumerate(items) if j % len(fu) == g]
        if not sub:
            continue
        with C.add_everywhere(model, c["layer"], al * fu[g]):
            L.run_letter_cell(sc, sub, "dl_chat_neutral", f"spec/{arm}/draw{g}")
        part = c["out"] / "spec" / arm / f"draw{g}" / "dl_chat_neutral.jsonl"
        rows += [{**json.loads(x), "fill_draw": g, "alpha": al} for x in part.open()]
    sc.save_jsonl(f"spec/{arm}/dl_chat_neutral.jsonl", rows)
    return {"status": "ok", "rows": len(rows), "alpha": al, "seconds_read": time.time() - t0,
            "draws": sorted({r["fill_draw"] for r in rows}),
            "rule": "draw = item index in the specificity set mod 5 (Amendment 29)"}


PHASES = {"rr_none": lambda c, sc, m, rec, a: C.phase_rr(c, sc, m, rec, a, "none"),
          "rr_fill": lambda c, sc, m, rec, a: C.phase_rr(c, sc, m, rec, a, "fill"),
          "rr_fill_half": lambda c, sc, m, rec, a: C.phase_rr(c, sc, m, rec, a, "fill_half"),
          "spec_none": lambda c, sc, m, rec, a: C.phase_spec(c, sc, m, rec, a, "none"),
          "spec_fill": lambda c, sc, m, rec, a: phase_spec_fill(c, sc, m, rec, a, "fill"),
          "spec_fill_half": lambda c, sc, m, rec, a: phase_spec_fill(c, sc, m, rec, a,
                                                                     "fill_half")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True, help="the st2_inputs dir (fill draws)")
    ap.add_argument("--podc-inputs", type=Path, required=True, help="the st2c_inputs dir")
    ap.add_argument("--c1", type=Path, required=True, help="pod A's output dir (Cell 1's none)")
    ap.add_argument("--phases", default=",".join(PODDL))
    ap.add_argument("--date-pin", default=None, help="unused: Cell 1's none sets the pin")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    a = ap.parse_args()
    c = context(a)
    pin = S2.c1_date_pin(c)  # pod C's pin, so the seeds and prompts pair with pod C's rows
    rec = C.load_record(c)
    todo = [p for p in a.phases.split(",") if not (rec["phases"].get(p) or {}).get("finished")]
    model, sc = None, None
    per: list[float] = []
    n_rr = len(c["rr_items"]) * c["rr_r"]
    for name in todo:
        est = float(np.mean(per)) * n_rr if per and name in GEN else 0.0
        if a.deadline and time.time() + est > a.deadline:
            print(f">> st2dl {name}: not started (would end past the deadline)", flush=True)
            return 3
        t0 = time.time()
        if model is None:
            model = C.load_model(c, a)
            sc = C.screen(c, model, a, pin)
        res = PHASES[name](c, sc, model, rec, a)
        if name in GEN:
            per.append((time.time() - t0) / n_rr)
        rec = C.load_record(c)
        rec["phases"][name] = {**res, "seconds": time.time() - t0, "finished": _now()}
        C.save_record(c, rec)
        brief = {k: res[k] for k in ("n", "refusal_rate", "rows") if k in res}
        if "plain_vs_none" in res:
            pv = res["plain_vs_none"]
            brief["dcompliance"] = pv["dcompliance_plain"]["mean"]
            brief["raises"] = pv["raises"]
        print(f">> st2dl {name}: done ({json.dumps(brief, default=str)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
