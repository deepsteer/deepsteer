#!/usr/bin/env bash
# Sourced by a remote runner that opts into editable time limits (session_limits.py). Defines:
#
#   limits_init <file> <start_epoch> <cut_h> <watch_h>
#       writes the limits file and exports CST_LIMITS_FILE (every stage's cut rule and deadline
#       then read it live); bails on a value outside the guards (watchdog <= 14.0 h, >= cut + 0.5)
#   limits_watchdog <pkill pattern>
#       the watchdog as a loop (every LIMITS_POLL_S seconds, default 60) against the file's
#       watch_h, so a `set` during the run moves it; sets WATCHDOG to its pid
#   limits_save <dir>
#       copies the limits file and its change log into the session's logs (call from the EXIT trap)
#
# Change the limits from the laptop with papers/runpod_common/set_limits.sh.
SL_PY="${SL_PY:-$(dirname "${BASH_SOURCE[0]}")/session_limits.py}"

limits_init() {
  export CST_LIMITS_FILE="$1"
  python3 "$SL_PY" init --file "$1" --start "$2" --cut "$3" --watch "$4" > /dev/null || return 1
  echo ">> limits: $(python3 "$SL_PY" show --file "$1")"
}

limits_watchdog() {
  local pattern="$1"
  ( trap 'exit 0' TERM
    while ! python3 "$SL_PY" watch_expired --file "$CST_LIMITS_FILE"; do
      sleep "${LIMITS_POLL_S:-60}" & wait $!
    done
    echo ">> WATCHDOG: the limits file's watchdog hour reached ($(python3 "$SL_PY" show --file "$CST_LIMITS_FILE"))"
    pkill -f "$pattern" ) &
  WATCHDOG=$!
}

limits_save() {
  [ -n "${CST_LIMITS_FILE:-}" ] || return 0
  mkdir -p "$1"
  cp "$CST_LIMITS_FILE" "$1/session_limits.json" 2>/dev/null
  cp "$CST_LIMITS_FILE.log.jsonl" "$1/session_limits.log.jsonl" 2>/dev/null
  return 0
}
