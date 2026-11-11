#!/usr/bin/env bash
# Launch card: the CST pressure line's action read (2-act), after 2-roll's action read failed on
# Claude's scenarios. The author runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_pressure_act.sh
#
# One A100-80GB, about 0.7 A100-h (watchdog 1.5 h). Synced by name: 2-roll's scenarios and every
# rollout chunk.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
O=outputs/pressure
SYNC="$O/scenarios.json"
for f in "$P/$O"/roll/*/chunk_*.jsonl; do SYNC="$SYNC,${f#$P/}"; done
export SELF_PAPER=$P RESULTS_SUBPATH=$O CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_pressure_act.sh MAX_SYNC_GB=1 DISK_GB=200
export POD_NAME=deepsteer-cst-pressure-act GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
