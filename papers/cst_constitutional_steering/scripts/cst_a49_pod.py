#!/usr/bin/env python3
"""Amendment 49's pod, the MoReBench parts and the cut rule (CST Amendment 49 items 2a, 2b).

    cst_a49_pod.py --stage <stage> --out DIR

- ``rejudge_prep`` (2a, zero GPU on the pod): MRB's saved base rollouts (p2, synced to
  ``outputs/a49_in``) for the first ``scenarios`` of the seeded order (the card writes the
  author's size to ``outputs/a49_in/rejudge.json``)
  (``data/a49_rejudge_order_v1.json``), both rollouts each, written as the arm ``rejudge`` so that
  MRB's judge draws fresh seeds (its seed is the job id's). Each response is checked against the
  saved row's sha256.
- ``a0n`` (2b): A0n on the 500 at k 2 under MRB's protocol (A0n's render, MRB's sampler, cap and
  seeds, so its rows pair with the saved base by job id), on A0s's engine settings.
- ``judge``: MRB's judge on every rollout not yet judged (resumable, to the deadline).
- ``fits --next a49_cgood|a49_pctx``: the cut rule; exit 0 if the stage's forecast ends by
  ``CUT_H``, else 1 (written to ``cut_<stage>.json``).

The MACHIAVELLI parts (2c, 2d, the citing cell) are ``cst_teacher_pod.py``'s ``a49_cgood`` and
``a49_pctx``. The read is ``a49_read.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel/scripts"),
                str(HERE.parents[1] / "mrb_morebench/scripts"),
                str(HERE.parents[1] / "st_live_reference/scripts"),
                str(HERE.parents[1] / "mch_machiavelli/scripts")]
import cst_finalist_pod as F  # noqa: E402
import cst_mrb2x2_pod as MB  # noqa: E402

CST = HERE.parent
MODEL = F.MODEL
ORDER = CST / "data/a49_rejudge_order_v1.json"
ORDER_SEED = "20261114|cst_a49_rejudge"
IN = CST / "outputs/a49_in"
BASE_GEN, BASE_RAW = IN / f"{MODEL}.none.gen.jsonl", IN / f"{MODEL}.none.raw.jsonl"
CUT_H = 10.25  # stages stop 0.25 h before the 10.5 h price


def build_order(sids: list[str]) -> dict:
    """The seeded scenario order the re-judge draws from (zero GPU; committed)."""
    ids = sorted(sids)
    random.Random(ORDER_SEED).shuffle(ids)
    return {"rule": "Amendment 49 item 2a", "seed": ORDER_SEED, "n": len(ids), "order": ids}


def rejudge_rows(n_scen: int, gen_rows: list[dict], raw_rows: list[dict],
                 order: list[str]) -> tuple[list[dict], list[dict]]:
    """The first ``n_scen`` scenarios' saved base rollouts, renamed to the arm ``rejudge``; the
    response text must hash to the saved row's ``response_sha256``."""
    keep = set(order[:n_scen])
    gen = {r["job_id"]: r for r in gen_rows}
    g_out, r_out = [], []
    for r in raw_rows:
        sid = r["job_id"].split("/")[2]
        if sid not in keep:
            continue
        g = gen[r["job_id"]]
        if hashlib.sha256(r["response"].encode()).hexdigest() != g["response_sha256"]:
            raise ValueError(f"{r['job_id']}: the synced response differs from the saved row")
        jid = r["job_id"].replace(f"{MODEL}/none/", f"{MODEL}/rejudge/")
        g_out.append(dict(g, job_id=jid))
        r_out.append(dict(r, job_id=jid))
    if len(g_out) != 2 * len(keep):
        raise ValueError(f"{len(g_out)} rollouts for {len(keep)} scenarios (want 2 each)")
    return g_out, r_out


def stage_rejudge_prep(out: Path, dry: bool) -> dict:
    n = int(json.loads((IN / "rejudge.json").read_text())["scenarios"]) if not dry else 2
    order = json.loads(ORDER.read_text())["order"]
    g, r = rejudge_rows(n, F.read_rows(BASE_GEN), F.read_rows(BASE_RAW), order)
    gd, rd = out / "mrb" / "gen", out / "mrb" / "_raw" / "gen"
    gd.mkdir(parents=True, exist_ok=True)
    rd.mkdir(parents=True, exist_ok=True)
    (gd / f"{MODEL}.rejudge.jsonl").write_text("".join(json.dumps(x) + "\n" for x in g))
    (rd / f"{MODEL}.rejudge.jsonl").write_text("".join(json.dumps(x) + "\n" for x in r))
    rec = {"rule": "Amendment 49 item 2a", "scenarios": n, "rollouts": len(g),
           "order_seed": ORDER_SEED}
    F.T._w(out, "rejudge_prep", rec)  # noqa: SLF001
    print(">> rejudge_prep:", json.dumps(rec), flush=True)
    return rec


def stage_a0n(out: Path, dry: bool) -> dict:
    tok, fmt = MB._tok_fmt(dry)  # noqa: SLF001
    jobs = MB.doc_jobs("a0n", tok, fmt, dry)
    engine, SP = MB.doc_engine("a0n", dry)  # noqa: N806
    name = f"{MODEL}.a0n.jsonl"
    t0 = time.time()
    rec = F._mrb_run(engine, SP, jobs, out / "mrb" / "gen" / name,  # noqa: SLF001
                     out / "mrb" / "_raw" / "gen" / name, fmt, tok)
    rec |= {"rule": "Amendment 49 item 2b (A0n)", "n_jobs": len(jobs),
            "max_model_len": MB.A0S_MAX_LEN, "wall_h": (time.time() - t0) / 3600}
    F.T._w(out, "gen_a0n", rec)  # noqa: SLF001
    return rec


def fits(out: Path, nxt: str) -> bool:
    import cst_teacher_pod as TP

    elapsed = (time.time() - MB.start_epoch()) / 3600
    fc = TP.cells_forecast_h(nxt)
    rec = {"rule": "Amendment 49 item 2 (cut from the end past 10.5 h)", "stage": nxt,
           "elapsed_h": elapsed, "forecast_h": fc, "ends_h": elapsed + fc, "cut_h": CUT_H,
           "runs": elapsed + fc <= CUT_H}
    F.T._w(out, f"cut_{nxt}", rec)  # noqa: SLF001
    print(">> cut rule:", json.dumps(rec), flush=True)
    return rec["runs"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("rejudge_prep", "a0n", "judge", "fits"))
    ap.add_argument("--next", choices=("a49_cgood", "a49_pctx"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.stage == "fits":
        sys.exit(0 if fits(a.out, a.next) else 1)
    F.configure()
    if a.stage == "rejudge_prep":
        stage_rejudge_prep(a.out, a.dry_run)
    elif a.stage == "a0n":
        stage_a0n(a.out, a.dry_run)
    else:
        F.stage_mrb_judge(a.out, a.dry_run)


if __name__ == "__main__":
    main()
