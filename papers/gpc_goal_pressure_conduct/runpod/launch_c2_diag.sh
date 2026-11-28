#!/usr/bin/env bash
# C2-JAB's diagnostic pod (GPC_PLAN.md §19.20), with no host: run by hand in the author's terminal
# (`bash papers/gpc_goal_pressure_conduct/runpod/launch_c2_diag.sh`). Thunder with no fallback; the repo only
# (the families are code; nothing gitignored is needed); results to outputs/c2_diag.
set -euo pipefail
cd "$(dirname "$0")/../../.."
[ -z "${TNR_FALLBACK:-}" ] || { echo "BAIL: GPC runs on Thunder with no fallback"; exit 1; }
G=papers/gpc_goal_pressure_conduct
export SELF_PAPER=$G RESULTS_SUBPATH=outputs/c2_diag CST_SESSION=1
export SYNC_OUTPUTS=""
export REMOTE_SCRIPT=$G/runpod/remote_c2_diag.sh MAX_SYNC_GB=3 DISK_GB=400
export PROVIDER=thunder TNR_GPU=a100xl TNR_VCPUS=8 TNR_PRICE=1.18 POD_NAME=deepsteer-c2-diag
export TNR_FALLBACK=""
exec bash papers/runpod_common/run_session.sh
