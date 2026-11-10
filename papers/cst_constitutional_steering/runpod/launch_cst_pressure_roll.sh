#!/usr/bin/env bash
# Launch card: the CST pressure line's session 2-roll (Amendment 27 item 4; PRESSURE_BUILD.md). The
# author runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_pressure_roll.sh
#
# One A100-80GB, priced at about 4.2 A100-h (watchdog 6.0 h). The synced inputs are the pinned
# C_anth text, the A0n text and the eval 8-gram hash index; grid v2 and Claude's kept scenarios are
# committed data.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
SYNC="outputs/_sync/C_anth.txt,outputs/_sync/a0n_faraday_v1.txt,outputs/_sync/eval_8gram_hashes_v1.npy"
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/pressure CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_pressure_roll.sh MAX_SYNC_GB=1 DISK_GB=300
export POD_NAME=deepsteer-cst-pressure-roll GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
