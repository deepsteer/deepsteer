#!/usr/bin/env bash
# GPC-4's Amendment 7 pod (GPC_PLAN.md §15, §15.1, §15.2), with no host: run by hand in the author's terminal
# (`bash papers/gpc_goal_pressure_conduct/runpod/launch_gpc4_a7.sh`). Derived from launch_gpc4_a6.sh: Thunder with
# no fallback (with no capacity no pod starts), the same inputs plus the 48 states (built before the pod by
# z_gpc4_a7_states.py, gitignored), results to outputs/gpc4_a7. Stages stop at 10.25 h unless moved (watchdog
# ceiling 14.0 h).
set -euo pipefail
cd "$(dirname "$0")/../../.."
[ -z "${TNR_FALLBACK:-}" ] || { echo "BAIL: GPC-4 runs on Thunder with no fallback"; exit 1; }
source papers/gpc_goal_pressure_conduct/runpod/gpc4_inputs.sh
A7_STATES=outputs/gpc4_in/a7/states.jsonl
[ "$(wc -l < "$G/$A7_STATES" 2>/dev/null | tr -d ' ')" = 48 ] || { echo "BAIL: $G/$A7_STATES is not the 48 states"; exit 1; }
SYNC="$SYNC,$A7_STATES"
export ON_DEMAND_SYNC="$GPC_BUNDLE"
export SELF_PAPER=$G RESULTS_SUBPATH=outputs/gpc4_a7 CST_SESSION=1
export SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$G/runpod/remote_gpc4_a7.sh MAX_SYNC_GB=3 DISK_GB=400
export PROVIDER=thunder TNR_GPU=a100xl TNR_VCPUS=8 TNR_PRICE=1.15 POD_NAME=deepsteer-gpc4-a7
export TNR_FALLBACK=""
exec bash papers/runpod_common/run_session.sh
