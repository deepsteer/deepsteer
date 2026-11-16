#!/usr/bin/env bash
# Move a running pod's time limits from the laptop (session_limits.py; the author, 2026-11-15).
# Works only on a session whose runner opted in (it wrote /workspace/deepsteer/.session_limits.json).
#
#   bash papers/runpod_common/set_limits.sh --ssh "ssh -p PORT -i KEY user@host" \
#       --cut 11.0 [--watch 11.75] --reason "why" [--rate 1.15]
#   bash papers/runpod_common/set_limits.sh --ssh "..." --show
#
# --cut moves when stages stop (the cut rule and every decision loop's deadline); --watch moves the
# hard stop (the watchdog). Guards (on the pod): the watchdog at most 14.0 h and at least 0.5 h past
# the cut. With --cut alone the watchdog follows it (cut + 0.75 h, the default gap, at most 14.0).
# Every change is logged on the pod with its reason and downloads with the results. --rate prints
# the change's cost at that hourly price. Shortening works the same way.
set -euo pipefail
SSH="" CUT="" WATCH="" REASON="" RATE="" SHOW=0
REPO="${REPO:-/workspace/deepsteer}"
while [ $# -gt 0 ]; do
  case "$1" in
    --ssh) SSH="$2"; shift 2;; --cut) CUT="$2"; shift 2;; --watch) WATCH="$2"; shift 2;;
    --reason) REASON="$2"; shift 2;; --rate) RATE="$2"; shift 2;; --show) SHOW=1; shift;;
    --repo) REPO="$2"; shift 2;;
    *) echo "unknown argument: $1"; exit 2;;
  esac
done
[ -n "$SSH" ] || { echo "BAIL: --ssh \"<the ssh command the launcher printed>\" is required"; exit 2; }
F="$REPO/.session_limits.json"
PYR="python3 $REPO/papers/runpod_common/session_limits.py"
before=$($SSH "$PYR show --file $F")
echo ">> now: $before"
[ "$SHOW" = 1 ] && exit 0
[ -n "$CUT$WATCH" ] || { echo "BAIL: give --cut and/or --watch"; exit 2; }
[ -n "$REASON" ] || { echo "BAIL: --reason is required (it goes on the record)"; exit 2; }
if [ -n "$CUT" ] && [ -z "$WATCH" ]; then
  WATCH=$(python3 -c "print(min(14.0, round($CUT + 0.75, 4)))")
fi
args="set --file $F --reason $(printf %q "$REASON")"
[ -n "$CUT" ] && args="$args --cut $CUT"
[ -n "$WATCH" ] && args="$args --watch $WATCH"
after=$($SSH "$PYR $args")
echo ">> set: $after"
python3 - "$before" "$after" "${RATE:-}" <<'EOF'
import json, sys, time
b, a, rate = json.loads(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
d = a["cut_h"] - b["cut_h"]
end = time.strftime("%H:%M", time.localtime(a["start_epoch"] + a["cut_h"] * 3600))
line = f">> stages now stop at {a['cut_h']} h (local {end}), watchdog {a['watch_h']} h; cut moved {d:+.2f} h"
if rate:
    line += f", about ${d * float(rate):+.2f} at ${float(rate):.2f}/h"
print(line)
if a.get("note"):
    print(">> note:", a["note"])
EOF
