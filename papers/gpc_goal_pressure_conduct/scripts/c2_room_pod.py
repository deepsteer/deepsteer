#!/usr/bin/env python3
"""C2-JAB's room pilot (GPC_PLAN.md §19.13; the lead's 2026-11-27 1:35 PM item 2): the base on the 8 pilot families,
conflict and twin, 12 loops each (192), through the subject's vLLM server; the cold verdict in both letter orders at
each decision point. No adapter, no judge, no host.

Stages (`--stage`):
- `smoke`: the first family's first conflict loop and first twin loop; the session's bail is a runner error or a
  server error there.
- `loops`: the 240 loops at 16 at once, each written as it ends (`room/loops.jsonl`, and what the agent wrote to
  `room/loop_text.jsonl`, §19.15), resumable by loop ID.
- `read`: `z_c2_room_read.py` on the rows (`room/read.json`).

`--dry-run` drives the same code through `c2_loop.ScriptedClient` (a seeded mix of the families' honest, shortcut
and twin routes), so the whole pod runs locally with no server.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import random
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_families as F  # noqa: E402
import c2_loop as L  # noqa: E402

K = 12
CONCURRENCY = 16
SEED = "20261127|c2jab|room"


def loop_ids() -> list[tuple]:
    """Interleaved, so a cut leaves every family-version with its first loops: k first, then family, then version."""
    return [(s, twin, k) for k in range(K) for s in F.ROOM for twin in (False, True)]


def lid(s, twin: bool, k: int) -> str:
    return f"room|{s.fid}|{'twin' if twin else 'conflict'}|{k:02d}"


def seed_of(i: str) -> int:
    return int(hashlib.sha256(f"{SEED}|{i}".encode()).hexdigest()[:8], 16)


def client_for(s, twin: bool, i: str, dry: bool):
    if not dry:
        return L.ServerClient()
    rng = random.Random(i)
    if twin:
        return L.ScriptedClient(s, "twin", s.closing_twin)
    return L.ScriptedClient(s, "shortcut") if rng.random() < 0.4 else L.ScriptedClient(s, "honest", s.closing_honest)


def stage_loops(out: Path, dry: bool, only: list[tuple] | None = None) -> dict:
    f = out / "room" / "loops.jsonl"
    ft = out / "room" / "loop_text.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    have = {json.loads(x)["loop_id"] for x in f.read_text().splitlines()} if f.exists() else set()
    todo = [t for t in (only or loop_ids()) if lid(*t) not in have]
    lock = threading.Lock()
    errors = []

    def one(t):
        s, twin, k = t
        i = lid(s, twin, k)
        text = None
        try:
            rec = L.run_loop(s, twin, client_for(s, twin, i, dry), seed_of(i), i) | {"loop_id": i, "k": k}
            text = {"loop_id": i, "family": s.fid, "twin": twin, "k": k} | rec.pop("text")
        except Exception as e:  # noqa: BLE001  (counted; the read reports it)
            rec = {"loop_id": i, "family": s.fid, "twin": twin, "k": k, "error": repr(e)[:300]}
            errors.append(i)
        with lock:
            if text is not None:
                with open(ft, "a") as w:
                    w.write(json.dumps(text) + "\n")
            with open(f, "a") as w:  # the loop record last, so a resumed run never has a record without its text
                w.write(json.dumps(rec) + "\n")
        return rec

    with cf.ThreadPoolExecutor(max_workers=1 if dry else CONCURRENCY) as ex:
        list(ex.map(one, todo))
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    rec = {"loops": len(rows), "errors": sum("error" in r for r in rows), "expected": len(loop_ids())}
    print(">> c2 room loops:", json.dumps(rec), flush=True)
    return rec


def stage_smoke(out: Path, dry: bool) -> dict:
    s = F.ROOM[0]
    rec = stage_loops(out, dry, only=[(s, False, 0), (s, True, 0)])
    rows = [json.loads(x) for x in (out / "room" / "loops.jsonl").read_text().splitlines()]
    texts = {json.loads(x)["loop_id"]: json.loads(x) for x in (out / "room" / "loop_text.jsonl").read_text().splitlines()}
    first = texts.get(lid(s, False, 0)) or {}
    # §19.15 item 4: the first conflict loop ran at least one bash step the environment answered with an exit code
    # other than 127; a loop that closes at step 1 with no tool call fails the smoke
    bash_ok = any(c["command"] is not None and c["exit"] not in (None, 127) for c in first.get("commands", []))
    # §19.16 item 4: it reached the scorer (the decision point or a shortcut) inside the step limit, with an uncovered
    # share of 0.10 or less; otherwise the pod bails after 2 loops, not 76
    o = next((r["outcome"] for r in rows if r["loop_id"] == lid(s, False, 0) and "outcome" in r), {})
    reached = bool(o.get("reached") or o.get("shortcut"))
    unc = (o.get("uncovered") or 0) / max(1, o.get("turns") or 1)
    ok = len(rows) >= 2 and not any("error" in r for r in rows) and bash_ok and reached and unc <= 0.10
    res = {"smoke_ok": ok, "rows": len(rows), "first_conflict_ran_a_bash_step": bash_ok,
           "first_conflict_reached_the_scorer": reached, "first_conflict_uncovered_share": round(unc, 3)}
    (out / "room" / "smoke.json").write_text(json.dumps(res))
    print(">> c2 room smoke:", json.dumps(res), flush=True)
    if not ok:
        raise SystemExit("BAIL: the room pilot's smoke loop failed (see room/loops.jsonl)")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "loops", "read"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.stage == "smoke":
        stage_smoke(a.out, a.dry_run)
    elif a.stage == "loops":
        stage_loops(a.out, a.dry_run)
    else:
        import z_c2_room_read as R

        rec = R.read(a.out / "room" / "loops.jsonl")
        (a.out / "room" / "read.json").write_text(json.dumps(rec, indent=1))
        print(">> c2 room read:", json.dumps(rec["summary"]), flush=True)


if __name__ == "__main__":
    main()
