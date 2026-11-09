#!/usr/bin/env bash
# Launch card: CST S0b-1 (S0B_BUILD.md §4 and §7; Amendments 16-19). The author runs it:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_s0b1.sh
#
# One A100-80GB, at most 8 h (the runner's watchdog), priced at about 4-6.5 A100-h. It builds the
# SYNC_OUTPUTS list here so nothing long is pasted (a wrapped paste once split a path). A rerun after
# a stop adds outputs/s0b/ so banked chunks are skipped; RESUME=1 does that.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
SYNC="outputs/_sync/C_anth.txt,outputs/_sync/a0n_faraday_v1.txt"
for f in $P/outputs/s0a/gen/a0/panel_operator_[0-9]*.jsonl $P/outputs/s0a/gen/a0s/panel_operator_[0-9]*.jsonl; do
  SYNC="$SYNC,${f#$P/}"
done
[ "${RESUME:-0}" = 1 ] && SYNC="$SYNC,outputs/s0b/"
n=$(echo "$SYNC" | tr ',' '\n' | grep -c panel_operator)
[ "$n" = 13 ] || { echo "ERROR: expected 13 S0a operator chunks, found $n"; exit 1; }
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/s0b CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_s0b.sh MAX_SYNC_GB=1 DISK_GB=300 POD_NAME=deepsteer-cst-s0b1
export GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
