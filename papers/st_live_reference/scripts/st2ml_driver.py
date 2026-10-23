"""The ``care_harm`` dose ladder on Muse (ST_PLAN Amendment 31 item 4; ST-A20's discriminator).
One HF process on the ST stack of record, resumable phase by phase, on Cell 4's derived quantities
(``--cell4``: its record, ``derive.npz``, ``directions.npz`` and rows), unchanged:

- ``spec_none`` (a parity re-read of Cell 4's specificity baseline), then for each rung
  (0.3x and 0.4x Cell 4's block-26 median norm) ``spec_fill_<rung>`` (draw = item index mod 5) and
  ``spec_care_<rung>`` (the vector at every input position), all before any arm;
- ``arm_care_<rung>``: ``care_harm`` at ``clause_end``, block 26, on Cell 4's 45 engaged x 8 with
  Cell 4's seeds (stage ``st2m``), so the rows pair with Cell 4's ``none``; coherence against it.

The analysis of record is local (``analyze_st2ml.py``).

    python3 st2ml_driver.py --out <dir> --cell4 <Cell 4's muse_glimmer_30b dir> [--dry-run]
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
import st2m_driver as M  # noqa: E402
import steering_gate as SG  # noqa: E402
import t13_driver as T13  # noqa: E402

RUNGS = {"03": 0.3, "04": 0.4}
PHASES = (["spec_none"] + [f"spec_{k}_{r}" for r in RUNGS for k in ("fill", "care")]
          + [f"arm_care_{r}" for r in RUNGS])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def context(a) -> dict:
    a.qwen_topup = a.qwen_screen = None
    c = M.context(a)
    src = a.cell4
    rec4 = json.loads((src / "st2m_record.json").read_text())["phases"]
    d = rec4["derive"]
    c.update(src=src, record=c["out"] / "st2ml_record.json", engaged=d["engaged_ids"],
             median_norm=d["median_norm"], batch=rec4["pilot"]["batch"],
             cell4_alpha=rec4["sweep"]["alpha"],
             pin=_jsonl(src / "gen" / "arm_none.jsonl")[0]["date_pin"],
             inp={"fill_unit": np.load(src / "derive.npz")["fill_unit"]},
             dirs={k: v for k, v in np.load(src / "directions.npz").items()})
    if a.dry_run:  # Cell 4's dry outputs carry the tiny width; the ladder's alphas follow them
        c.update(engaged=c["engaged"][:2])
    return c


def alpha(c: dict, rung: str) -> float:
    return RUNGS[rung] * c["median_norm"]


def phase_spec_fill(c, sc, model, rung: str) -> dict:
    al, fu = alpha(c, rung), c["inp"]["fill_unit"]
    items = [c["panel"][i] for i in c["spec_ids"]]
    rows: list[dict] = []
    for g in range(len(fu)):
        sub = [it for j, it in enumerate(items) if j % len(fu) == g]
        if not sub:
            continue
        with C.add_everywhere(model, c["layer"], al * fu[g]):
            L.run_letter_cell(sc, sub, "dl_chat_neutral", f"spec/fill_{rung}/draw{g}")
        part = c["out"] / "spec" / f"fill_{rung}" / f"draw{g}" / "dl_chat_neutral.jsonl"
        rows += [{**json.loads(x), "fill_draw": g, "alpha": al} for x in part.open()]
    sc.save_jsonl(f"spec/fill_{rung}/dl_chat_neutral.jsonl", rows)
    return {"status": "ok", "rows": len(rows), "alpha": al}


def phase_spec_care(c, sc, model, rung: str) -> dict:
    al = alpha(c, rung)
    with C.add_everywhere(model, c["layer"], al * c["dirs"]["care_harm"]):
        res = L.run_letter_cell(sc, [c["panel"][i] for i in c["spec_ids"]], "dl_chat_neutral",
                                f"spec/care_{rung}")
    return {"status": "ok", "alpha": al, **res}


def phase_spec_none(c, sc, model) -> dict:
    res = L.run_letter_cell(sc, [c["panel"][i] for i in c["spec_ids"]], "dl_chat_neutral",
                            "spec/none")
    base = {(r["scenario_id"], r["rollout"]): r for r in
            _jsonl(c["src"] / "spec" / "none" / "dl_chat_neutral.jsonl")}
    rows = _jsonl(Path(res["path"]))
    same_prompt = sum(base.get((r["scenario_id"], r["rollout"]), {}).get("prompt_ids_sha256")
                      == r["prompt_ids_sha256"] for r in rows)
    dlp = [abs(r["option_logps"][x] - base[(r["scenario_id"], r["rollout"])]["option_logps"][x])
           for r in rows if (r["scenario_id"], r["rollout"]) in base for x in r["option_logps"]]
    return {"status": "ok", **res, "parity": {"prompts_match": f"{same_prompt} of {len(rows)}",
                                              "max_abs_dlogp": max(dlp) if dlp else None}}


def phase_arm(c, sc, model, a, rung: str) -> dict:
    al = alpha(c, rung)
    jobs = L.h1_jobs(sc, c["H"], c["engaged"], c["r"])
    rows = S2.run_arm(c, sc, model, "st2m", f"arm_care_{rung}", jobs,
                      SG.Add(c["dirs"]["care_harm"], al),
                      extra=lambda i: {"alpha": al, "axis": "care_harm", "rung": RUNGS[rung]})
    none = _jsonl(c["src"] / "gen" / "arm_none.jsonl")
    keys = {r["rollout_key"] for r in none}
    if not {r["rollout_key"] for r in rows} <= keys:
        raise SystemExit(f"arm_care_{rung} rows do not pair with Cell 4's none")
    coh = T13._coh(rows, none, relative=True)  # noqa: SLF001
    return {"status": "ok", "n": len(rows), "alpha": al, "coherence": coh,
            "parse_rate": float(np.mean([r["letter"] is not None for r in rows]))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cell4", type=Path, required=True)
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--date-pin", default=None, help="unused: Cell 4's rows set the pin")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    a = ap.parse_args()
    c = context(a)
    rec = M.load_record(c)
    todo = [p for p in a.phases.split(",") if not (rec["phases"].get(p) or {}).get("finished")]
    model, sc, per = None, None, []
    for name in todo:
        n = len(c["engaged"]) * c["r"] if name.startswith("arm_") else 0
        est = float(np.mean(per)) * n if per and n else 0.0
        if a.deadline and time.time() + est > a.deadline:
            print(f">> st2ml {name}: not started (would end past the deadline)", flush=True)
            return 3
        t0 = time.time()
        if model is None:
            model = M.load_model(c, a)
            sc = M.screen(c, model, c["pin"], a.dry_run)
        if name == "spec_none":
            res = phase_spec_none(c, sc, model)
        elif name.startswith("spec_fill_"):
            res = phase_spec_fill(c, sc, model, name.rsplit("_", 1)[1])
        elif name.startswith("spec_care_"):
            res = phase_spec_care(c, sc, model, name.rsplit("_", 1)[1])
        else:
            res = phase_arm(c, sc, model, a, name.rsplit("_", 1)[1])
            per.append((time.time() - t0) / n)
        rec = M.load_record(c)
        rec["phases"][name] = {**res, "seconds": time.time() - t0, "finished": _now()}
        M.save_record(c, rec)
        brief = {k: res[k] for k in ("n", "rows", "alpha", "parse_rate", "parity") if k in res}
        if "coherence" in res:
            brief["coherent"] = res["coherence"]["passes"]
        print(f">> st2ml {name}: done ({json.dumps(brief, default=str)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
