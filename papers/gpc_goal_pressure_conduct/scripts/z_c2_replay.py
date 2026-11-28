#!/usr/bin/env python3
"""The room pilot's commands replayed against the fixed environment (GPC_PLAN.md §19.16; the lead's 2026-11-27 3:05 PM
item 3), zero GPU, by script; counts and command names only, no command text printed.

Each of the pilot's loops is replayed in a fresh environment of its own family and version (conflict or twin), its
saved commands run in order. The bar: an uncovered share of 0.10 or less over all commands. Beside it, per family:
the uncovered share, the share answered with an error exit (a path or argument the subject got wrong, or one the
environment still mis-answers), and the commands still unanswered by name, with the reason each is left. Names are
printed from the allowlist in `c2_names.py` only, any other word as `(other)` (§19.17).

    python z_c2_replay.py [--loops outputs/c2_room_pilot1/room/loop_text.jsonl]   # writes <dir>/replay.json
"""

from __future__ import annotations

import argparse
import collections
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_env as E  # noqa: E402
import c2_families as F  # noqa: E402
from c2_names import name_of  # noqa: E402
from c2_shell import split_top  # noqa: E402

BAR = 0.10
WHY = {"python3 -c": "arbitrary code: the real-bash sandbox answers it", "python -c": "as python3 -c"}


def head_of(cmd: str) -> str:
    for piece, _ in split_top(cmd, ("&&", "||", ";", "\n", "|")):
        try:
            w = shlex.split(piece)
        except ValueError:
            w = piece.split()
        if w:
            return name_of(w[0] + (" -c" if w[0] in ("python3", "python") and w[1:2] == ["-c"] else ""))
    return "(empty)"


def replay(path: Path) -> dict:
    specs = {s.fid: s for s in F.ROOM}
    fam = collections.defaultdict(lambda: {"commands": 0, "uncovered": 0, "error_exit": 0,
                                           "unanswered": collections.Counter()})
    for line in path.read_text().splitlines():
        t = json.loads(line)
        env = E.Env(specs[t["family"]].family(t["twin"]))
        f = fam[t["family"]]
        for c in t["commands"]:
            if c.get("closing") or c.get("command") is None:
                continue
            out, code = env.run(c["command"])
            f["commands"] += 1
            if code == E.UNCOVERED_EXIT and "command not found" in out:
                f["uncovered"] += 1
                missing = name_of(out.split(":")[1].strip()) if out.startswith("bash:") else head_of(c["command"])
                f["unanswered"][missing] += 1
            elif code != 0:
                f["error_exit"] += 1
    total = sum(f["commands"] for f in fam.values())
    unc = sum(f["uncovered"] for f in fam.values())
    names = collections.Counter()
    for f in fam.values():
        names.update(f["unanswered"])
    rec = {"rule": "GPC_PLAN.md §19.16 (the replay's bar: an uncovered share of 0.10 or less)", "bar": BAR,
           "commands": total, "uncovered": unc, "uncovered_share": unc / total if total else None,
           "passes": total > 0 and unc / total <= BAR,
           "still_unanswered": [{"name": n, "count": k, "why": WHY.get(n, "not in the environment's set")}
                                for n, k in names.most_common()],
           "families": {k: {"commands": v["commands"], "uncovered_share": v["uncovered"] / max(1, v["commands"]),
                            "error_exit_share": v["error_exit"] / max(1, v["commands"]),
                            "unanswered": dict(v["unanswered"])} for k, v in sorted(fam.items())}}
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=Path, default=HERE.parent / "outputs/c2_room_pilot1/room/loop_text.jsonl")
    a = ap.parse_args()
    rec = replay(a.loops)
    (a.loops.parent / "replay.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("commands", "uncovered", "uncovered_share", "passes", "still_unanswered")},
                     indent=1))


if __name__ == "__main__":
    main()
