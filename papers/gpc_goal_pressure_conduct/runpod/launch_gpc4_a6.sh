#!/usr/bin/env bash
# GPC-4's Amendment 6 pod (GPC_PLAN.md §14), with no host: run by hand in the author's terminal, as GPC-1 to
# GPC-3's cards were (`bash papers/gpc_goal_pressure_conduct/runpod/launch_gpc4_a6.sh`). Derived from
# launch_gpc4_pod.sh: Thunder with no fallback (with no capacity no pod starts), the same inputs, results to
# outputs/gpc4_a6. Stages stop at 10.25 h unless moved (watchdog ceiling 14.0 h).
set -euo pipefail
cd "$(dirname "$0")/../../.."
[ -z "${TNR_FALLBACK:-}" ] || { echo "BAIL: GPC-4 runs on Thunder with no fallback"; exit 1; }
source papers/gpc_goal_pressure_conduct/runpod/gpc4_inputs.sh
export ON_DEMAND_SYNC="$GPC_BUNDLE"
export SELF_PAPER=$G RESULTS_SUBPATH=outputs/gpc4_a6 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$G/runpod/remote_gpc4_a6.sh MAX_SYNC_GB=3 DISK_GB=400
export PROVIDER=thunder TNR_GPU=a100xl TNR_VCPUS=8 TNR_PRICE=1.15 POD_NAME=deepsteer-gpc4-a6
export TNR_FALLBACK=""
exec bash papers/runpod_common/run_session.sh
