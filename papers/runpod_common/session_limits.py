#!/usr/bin/env python3
"""A pod session's time limits, editable while it runs (the author, 2026-11-15).

A runner writes the limits file once at start (``init``); the cut rule, the decision loops'
deadline and the watchdog re-read it on every check, so a ``set`` during the run moves all three.
Without a limits file (``CST_LIMITS_FILE`` unset or missing) every reader falls back to the
environment of record (``CST_START_EPOCH``, ``CST_DEADLINE_EPOCH``) and the defaults, so a runner
that does not opt in behaves exactly as before.

- **Defaults:** stages stop at 10.25 h (``cut_h``), the watchdog at 11.0 h (``watch_h``).
- **Guards:** the watchdog at most 14.0 h (the hard ceiling against a runaway); the watchdog at
  least 0.5 h past the cut; both positive. A rejected ``set`` changes nothing.
- **Record:** every change is appended to the change log beside the file (``<file>.log.jsonl``)
  with its time, the old and new values and the reason; the runner copies both into the session's
  logs, so each extension or shortening is on the record as a deviation.

Stdlib only (the pod's system ``python3`` runs it before any venv exists).

    session_limits.py init --file F --start EPOCH [--cut 10.25] [--watch 11.0]
    session_limits.py set --file F [--cut H] [--watch H] --reason "..."
    session_limits.py show --file F
    session_limits.py watch_expired --file F      # exit 0 once elapsed >= watch_h
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_CUT_H, DEFAULT_WATCH_H = 10.25, 11.0
CEILING_H = 14.0  # the hard limit (the author, 2026-11-15)
MIN_GAP_H = 0.5  # the watchdog at least this far past the cut


def validate(cut_h: float, watch_h: float) -> list[str]:
    bad = []
    if cut_h <= 0 or watch_h <= 0:
        bad.append("limits must be positive")
    if watch_h > CEILING_H:
        bad.append(f"watchdog {watch_h} h is past the {CEILING_H} h ceiling")
    if watch_h - cut_h < MIN_GAP_H - 1e-9:
        bad.append(f"the watchdog must be at least {MIN_GAP_H} h past the cut "
                   f"(cut {cut_h}, watchdog {watch_h})")
    return bad


def _path(path: str | Path | None) -> Path | None:
    p = path or os.environ.get("CST_LIMITS_FILE")
    return Path(p) if p else None


def read(path: str | Path | None = None) -> dict:
    """The limits in force: the file's when it exists, else the environment and defaults."""
    p = _path(path)
    if p is not None and p.exists():
        return json.loads(p.read_text())
    start = os.environ.get("CST_START_EPOCH")
    dl = os.environ.get("CST_DEADLINE_EPOCH")
    cut = (float(dl) - float(start)) / 3600 if start and dl else DEFAULT_CUT_H
    return {"start_epoch": float(start) if start else None, "cut_h": cut,
            "watch_h": DEFAULT_WATCH_H, "source": "environment"}


def cut_h(path: str | Path | None = None) -> float:
    return float(read(path)["cut_h"])


def deadline_epoch(path: str | Path | None = None) -> float | None:
    """The epoch at which stages stop: start plus the cut; None without a start."""
    r = read(path)
    if r.get("source") == "environment":
        dl = os.environ.get("CST_DEADLINE_EPOCH")
        return float(dl) if dl else None
    return float(r["start_epoch"]) + float(r["cut_h"]) * 3600


def _write(p: Path, rec: dict) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n")
    os.replace(tmp, p)  # atomic: a reader never sees a half-written file


def _log(p: Path, entry: dict) -> None:
    with open(p.with_name(p.name + ".log.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


def init(path: str | Path, start: float, cut: float = DEFAULT_CUT_H,
         watch: float = DEFAULT_WATCH_H) -> dict:
    bad = validate(cut, watch)
    if bad:
        raise SystemExit("BAIL: " + "; ".join(bad))
    p = Path(path)
    rec = {"start_epoch": start, "cut_h": cut, "watch_h": watch, "source": "file",
           "ceiling_h": CEILING_H}
    _write(p, rec)
    _log(p, {"at": time.time(), "action": "init", "new": rec})
    return rec


def set_limits(path: str | Path, reason: str, cut: float | None = None,
               watch: float | None = None, now: float | None = None) -> dict:
    """Move the limits during the run; returns the new record with the elapsed hours beside."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"BAIL: no limits file at {p} (this session did not opt in)")
    if not reason.strip():
        raise SystemExit("BAIL: a change needs a reason (it goes on the record)")
    old = json.loads(p.read_text())
    new = dict(old, cut_h=old["cut_h"] if cut is None else cut,
               watch_h=old["watch_h"] if watch is None else watch)
    bad = validate(new["cut_h"], new["watch_h"])
    if bad:
        raise SystemExit("BAIL (nothing changed): " + "; ".join(bad))
    elapsed = ((now or time.time()) - float(old["start_epoch"])) / 3600
    _write(p, new)
    _log(p, {"at": now or time.time(), "action": "set", "reason": reason, "elapsed_h": elapsed,
             "old": {k: old[k] for k in ("cut_h", "watch_h")},
             "new": {k: new[k] for k in ("cut_h", "watch_h")}})
    return {**new, "elapsed_h": elapsed,
            "note": ("the cut is already past: stages stop at their next check"
                     if new["cut_h"] <= elapsed else None)}


def watch_expired(path: str | Path, now: float | None = None) -> bool:
    r = read(path)
    if r.get("start_epoch") is None:
        return False
    return ((now or time.time()) - float(r["start_epoch"])) / 3600 >= float(r["watch_h"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("init", "set", "show", "watch_expired"))
    ap.add_argument("--file", required=True)
    ap.add_argument("--start", type=float)
    ap.add_argument("--cut", type=float)
    ap.add_argument("--watch", type=float)
    ap.add_argument("--reason", default="")
    a = ap.parse_args()
    if a.cmd == "init":
        print(json.dumps(init(a.file, a.start or time.time(),
                              DEFAULT_CUT_H if a.cut is None else a.cut,
                              DEFAULT_WATCH_H if a.watch is None else a.watch)))
    elif a.cmd == "set":
        print(json.dumps(set_limits(a.file, a.reason, a.cut, a.watch)))
    elif a.cmd == "show":
        print(json.dumps(read(a.file)))
    else:
        sys.exit(0 if watch_expired(a.file) else 1)


if __name__ == "__main__":
    main()
