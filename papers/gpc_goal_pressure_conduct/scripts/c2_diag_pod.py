#!/usr/bin/env python3
"""C2-JAB's diagnostic pod (GPC_PLAN.md §19.20; the lead's 2026-11-27 8:15 PM item 3): the base on the 10 pilot
families, conflict versions only, 12 loops a family an arm, in the real-bash sandbox, with the cold verdict at each
decision point. Three arms, nothing else changed:
- **a:** cell A's four pressure sentences (`data/odcv_pressure_v1.json`) first in the system message, as ODCV-Bench
  places them;
- **b:** the task in an incentivized wording of our own (`data/c2_incentivized_v1.json`), no sentences;
- **c:** both.

The no-change arm is the third headroom pilot (`outputs/c2_room`, commit `721edc1a`). The families, the scorers and
the environment are frozen at that commit: the pod bails if any file's sha256 differs from `data/c2_frozen_v1.json`.

Stages (`--stage`):
- `smoke`: each arm's first D1 loop; the bail is §19.16's (a bash step answered, the scorer reached inside the step
  limit, uncovered 0.10 or less), on all three.
- `loops`: the 360 loops at 16 at once, interleaved (k, then family, then arm) so a cut leaves every arm balanced,
  each written as it ends to `diag/loops.jsonl` and `diag/loop_text.jsonl`, resumable by loop ID.
- `read`: `z_c2_diag_read.py` (`diag/read.json`).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_families as F  # noqa: E402
import c2_loop as L  # noqa: E402
import c2_room_pod as P  # noqa: E402

GPC = HERE.parent
K = 12
ARMS = ("a", "b", "c")
SENTENCES = ("goal_framing", "replacement_threat", "operator_insistence", "reduced_oversight")  # cell A (§12)


def frozen_ok() -> dict:
    rec = json.loads((GPC / "data/c2_frozen_v1.json").read_text())
    got = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in rec["sha256"]}
    bad = sorted(f for f, h in rec["sha256"].items() if got[f] != h)
    return {"ok": not bad, "differ": bad, "commit": rec["commit"]}


def sentences() -> str:
    lev = json.loads((GPC / "data/odcv_pressure_v1.json").read_text())["levers"]
    return " ".join(lev[k]["text"] for k in SENTENCES)


def incentivized() -> dict[str, str]:
    return json.loads((GPC / "data/c2_incentivized_v1.json").read_text())["user"]


def arm_inputs(arm: str, spec) -> tuple[str, str | None]:
    """(system prefix, task text) for an arm; everything else is the family's own."""
    prefix = sentences() if arm in ("a", "c") else ""
    user = incentivized()[spec.fid] if arm in ("b", "c") else None
    return prefix, user


def loop_ids() -> list[tuple]:
    return [(arm, s, k) for k in range(K) for s in F.ROOM for arm in ARMS]


def lid(arm: str, s, k: int) -> str:
    return f"diag|{arm}|{s.fid}|conflict|{k:02d}"


def stage_loops(out: Path, dry: bool, only: list[tuple] | None = None, shell: str | None = None) -> dict:
    shell = shell or ("emulator" if dry else "sandbox")
    f, ft = out / "diag" / "loops.jsonl", out / "diag" / "loop_text.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    have = {json.loads(x)["loop_id"] for x in f.read_text().splitlines()} if f.exists() else set()
    todo = [t for t in (only or loop_ids()) if lid(*t) not in have]
    lock = threading.Lock()

    def one(t):
        arm, s, k = t
        i = lid(arm, s, k)
        text = None
        try:
            prefix, user = arm_inputs(arm, s)
            rec = L.run_loop(s, False, P.client_for(s, False, i, dry), P.seed_of(i), i,
                             env=P.env_for(s, False, shell), system_prefix=prefix, user=user)
            rec |= {"loop_id": i, "k": k, "arm": arm}
            text = {"loop_id": i, "family": s.fid, "twin": False, "k": k, "arm": arm} | rec.pop("text")
        except Exception as e:  # noqa: BLE001  (counted; the read reports it)
            rec = {"loop_id": i, "family": s.fid, "twin": False, "k": k, "arm": arm, "error": repr(e)[:300]}
        with lock:
            if text is not None:
                with open(ft, "a") as w:
                    w.write(json.dumps(text) + "\n")
            with open(f, "a") as w:  # the loop record last, so a resumed run never has a record without its text
                w.write(json.dumps(rec) + "\n")
        return rec

    with cf.ThreadPoolExecutor(max_workers=1 if dry else P.CONCURRENCY) as ex:
        list(ex.map(one, todo))
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    rec = {"loops": len(rows), "errors": sum("error" in r for r in rows), "expected": len(loop_ids())}
    print(">> c2 diag loops:", json.dumps(rec), flush=True)
    return rec


def stage_smoke(out: Path, dry: bool, shell: str | None = None) -> dict:
    shell = shell or ("emulator" if dry else "sandbox")
    fr = frozen_ok()
    if not fr["ok"]:
        raise SystemExit(f"BAIL: the frozen files differ from {fr['commit']}: {fr['differ']}")
    s = F.ROOM[0]
    firsts = [(arm, s, 0) for arm in ARMS]
    stage_loops(out, dry, only=firsts, shell=shell)
    rows = {json.loads(x)["loop_id"]: json.loads(x) for x in (out / "diag" / "loops.jsonl").read_text().splitlines()}
    texts = {json.loads(x)["loop_id"]: json.loads(x)
             for x in (out / "diag" / "loop_text.jsonl").read_text().splitlines()}
    per = {}
    for arm, _, _ in firsts:
        i = lid(arm, s, 0)
        r, t = rows.get(i, {}), texts.get(i, {})
        o = r.get("outcome", {})
        bash_ok = any(c["command"] is not None and c["exit"] not in (None, 127) for c in t.get("commands", []))
        unc = (o.get("uncovered") or 0) / max(1, o.get("turns") or 1)
        per[arm] = {"error": "error" in r, "bash_step": bash_ok, "reached_the_scorer":
                    bool(o.get("reached") or o.get("shortcut")), "uncovered_share": round(unc, 3),
                    "nonzero_exit_share": round(o.get("nonzero_exit_not_scorer", 0) / max(1, o.get("commands", 0)), 3)}
        per[arm]["ok"] = (not per[arm]["error"] and bash_ok and per[arm]["reached_the_scorer"] and unc <= 0.10)
    res = {"smoke_ok": all(v["ok"] for v in per.values()), "shell": shell, "frozen": fr, "arms": per}
    (out / "diag" / "smoke.json").write_text(json.dumps(res))
    print(">> c2 diag smoke:", json.dumps(res), flush=True)
    if not res["smoke_ok"]:
        raise SystemExit("BAIL: the diagnostic pod's smoke failed (see diag/loops.jsonl)")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "loops", "read"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shell", choices=["sandbox", "emulator"], default=None)
    a = ap.parse_args()
    shell = a.shell or ("emulator" if a.dry_run else "sandbox")
    if a.stage == "smoke":
        stage_smoke(a.out, a.dry_run, shell)
    elif a.stage == "loops":
        stage_loops(a.out, a.dry_run, shell=shell)
    else:
        import z_c2_diag_read as R

        rec = R.read(a.out / "diag" / "loops.jsonl")
        (a.out / "diag" / "read.json").write_text(json.dumps(rec, indent=1))
        print(">> c2 diag read:", json.dumps(rec["summary"]), flush=True)


if __name__ == "__main__":
    main()
