#!/usr/bin/env bash
# GPC-4's pod (GPC_PLAN.md §11.3), started by launch_gpc4.sh after the EC2 host passes its gate tests;
# never run by hand. Thunder with no fallback: with no capacity the launch stops and no pod starts.
# The launcher reads this card's ">> SSH endpoint:" line to hand the pod's address to the host and lock
# the host's egress to it. Stages stop at 10.25 h unless moved (watchdog ceiling 14.0 h); GPC-4's own
# cap is 9 A100-h, applied by the runner's cut rule.
set -euo pipefail
cd "$(dirname "$0")/../../.."
[ -z "${TNR_FALLBACK:-}" ] || { echo "BAIL: GPC-4 runs on Thunder with no fallback"; exit 1; }
source papers/gpc_goal_pressure_conduct/runpod/gpc4_inputs.sh
export ON_DEMAND_SYNC="$GPC_BUNDLE"
export SELF_PAPER=$G RESULTS_SUBPATH=outputs/gpc4_v1 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$G/runpod/remote_gpc4.sh MAX_SYNC_GB=3 DISK_GB=400
export PROVIDER=thunder TNR_GPU=a100xl TNR_VCPUS=8 TNR_PRICE=1.15 POD_NAME=deepsteer-gpc4
export TNR_FALLBACK=""
exec bash papers/runpod_common/run_session.sh
