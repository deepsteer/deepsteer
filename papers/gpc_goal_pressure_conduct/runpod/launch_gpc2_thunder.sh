#!/usr/bin/env bash
# Launch card: GPC-2 on Thunder Compute (GPC_PLAN.md Amendment 1) (no provider fallback unless TNR_FALLBACK=runpod_secure).
# One command, the session's log on the console, the instance created and deleted inside the
# script. The author runs:
#
#   export HF_TOKEN=...   # Thunder's login is the `tnr` CLI (`tnr login`); RUNPOD_API_KEY only with a fallback
#   bash papers/gpc_goal_pressure_conduct/runpod/launch_gpc2_thunder.sh
#
# One A100 80GB (8 vCPUs, 300 GB, about $1.15 an hour). Stages stop at 10.25 h unless moved during
# the run with papers/runpod_common/set_limits.sh (watchdog ceiling 14.0 h; every change logged).
# Stages (remote_gpc2.sh), each item cut from the end of the priority order X1, X2, X3, F, C2, B, A:
# a. the training smoke, C1's scaled copies, the fixed smoke check, letters and served parity;
# b. phase 1 on one LoRA engine: X1 and X2 and their registered reading, then (if format gates) X3
# and C2's teachers and gates, F, B1 and the neutral lines' grid rollouts, A's screens; c. C2's
# adapters if format gates; d. B0, B2, B3 on HF; e. phase 2: C2's screens and the registered read;
# f. B4 if B3 passes at the coherent dose. The script creates the instance (retrying when Thunder
# reports no capacity; then it stops, unless TNR_FALLBACK=runpod_secure opts into RunPod Secure), syncs, launches, streams the log, downloads the
# results, then deletes the instance and checks the account lists nothing it created.
set -euo pipefail
cd "$(dirname "$0")/../../.."
# no provider fallback by default (the author, 2026-11-17): with no Thunder capacity the launch stops
# and no pod starts; TNR_FALLBACK=runpod_secure on the command line opts into RunPod Secure
if [ "${TNR_FALLBACK:-}" = runpod_secure ]; then
  [ -n "${RUNPOD_API_KEY:-}" ] || { echo "BAIL: TNR_FALLBACK=runpod_secure needs RUNPOD_API_KEY"; exit 1; }
fi
source papers/gpc_goal_pressure_conduct/runpod/gpc2_inputs.sh
export ON_DEMAND_SYNC="$GPC_BUNDLE"
export SELF_PAPER=$G RESULTS_SUBPATH=outputs/gpc2_v1 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$G/runpod/remote_gpc2.sh MAX_SYNC_GB=3 DISK_GB=300
export PROVIDER=thunder TNR_GPU=a100xl TNR_VCPUS=8 TNR_PRICE=1.15 POD_NAME=deepsteer-gpc2
export TNR_FALLBACK="${TNR_FALLBACK:-}" RUNPOD_GPU_TYPES="NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB"
exec bash papers/runpod_common/run_session.sh
