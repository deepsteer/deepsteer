#!/usr/bin/env bash
# Launch card: the CST pilot DPO pod (PILOT_BUILD.md §5 card 2; Amendments 20-22). The author runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_pilot_dpo.sh
#
# One A100-80GB, priced at about 9.5 A100-h (watchdog 11 h). The synced input is the pinned
# C_anth text. RESUME=1 adds outputs/pilot_dpo/ so a rerun skips what is on disk.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
SYNC="outputs/_sync/C_anth.txt"
[ "${RESUME:-0}" = 1 ] && SYNC="$SYNC,outputs/pilot_dpo/"
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/pilot_dpo CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_pilot_dpo.sh MAX_SYNC_GB=1 DISK_GB=300
export POD_NAME=deepsteer-cst-pilot-dpo GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
