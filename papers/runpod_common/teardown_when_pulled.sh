#!/usr/bin/env bash
# Terminate a pod once its results are verified local. The author runs this in a terminal holding
# RUNPOD_API_KEY (the key never leaves that terminal); it waits until MARKER exists and names
# POD_ID, a file the lead writes only after checking the final pull, then sends the launcher's
# podTerminate call. For a session whose launcher lost its connection (2026-11-10).
#
#   export RUNPOD_API_KEY=...
#   bash papers/runpod_common/teardown_when_pulled.sh POD_ID MARKER
set -euo pipefail
POD_ID="$1"; MARKER="$2"
[ -n "${RUNPOD_API_KEY:-}" ] || { echo "ERROR: export RUNPOD_API_KEY first"; exit 1; }
echo ">> waiting for $MARKER to name pod $POD_ID (checks every 60 s; Ctrl-C to cancel)"
until [ -f "$MARKER" ] && grep -q "pod=$POD_ID" "$MARKER"; do sleep 60; done
echo ">> $(date): marker present: $(head -c 300 "$MARKER")"
API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"
q="{\"query\":\"mutation { podTerminate(input:{podId:\\\"$POD_ID\\\"}) }\"}"
r=$(curl -s -m 30 "$API" -H 'Content-Type: application/json' -d "$q")
echo ">> terminate response: $r"
echo "$r" | grep -q '"errors"' && { echo "WARN: terminate failed; terminate $POD_ID in the RunPod console"; exit 1; }
echo ">> pod $POD_ID terminated"
