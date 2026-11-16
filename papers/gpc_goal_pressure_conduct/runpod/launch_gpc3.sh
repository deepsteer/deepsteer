#!/usr/bin/env bash
# Launch card: GPC-3 on RunPod (the fallback card; launch_gpc3_thunder.sh is the card of record and
# falls back to RunPod Secure by itself). Same stages and inputs. The author runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/gpc_goal_pressure_conduct/runpod/launch_gpc3.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
source papers/gpc_goal_pressure_conduct/runpod/gpc3_inputs.sh
export ON_DEMAND_SYNC="$GPC_BUNDLE"
export SELF_PAPER=$G RESULTS_SUBPATH=outputs/gpc3_v1 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$G/runpod/remote_gpc3.sh MAX_SYNC_GB=3 DISK_GB=300
export POD_NAME=deepsteer-gpc3 GPU_TYPES="NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB"
export CLOUD_TYPES="SECURE"
exec bash papers/runpod_common/run_session.sh
