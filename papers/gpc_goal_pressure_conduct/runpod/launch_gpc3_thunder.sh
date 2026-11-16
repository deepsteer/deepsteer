#!/usr/bin/env bash
# Launch card: GPC-3 on Thunder Compute (GPC_PLAN.md Amendment 2, revised in §10.6), the RunPod Secure fallback inside.
# One command, the session's log on the console, the instance created and deleted inside the
# script. The author runs:
#
#   export HF_TOKEN=... RUNPOD_API_KEY=...   # Thunder's login is the `tnr` CLI (`tnr login`)
#   bash papers/gpc_goal_pressure_conduct/runpod/launch_gpc3_thunder.sh
#
# One A100 80GB (8 vCPUs, 300 GB, about $1.15 an hour). Stages stop at 10.25 h unless moved during
# the run with papers/runpod_common/set_limits.sh (watchdog ceiling 14.0 h; every change logged).
# Stages (remote_gpc3.sh), each item cut from the end of the priority order G1, G2, G3, G4, G5:
# a. the training smoke, C1 x 2.0's structural check, the fixed smoke check, letters and served
# parity; b. phase 1 on one LoRA engine: G1 (the clause in the native harness; GOOD in X2's format),
# G2 (the achievements list, then the CoT steps, on S-hi in X2's format), G3 (the judge frame on all
# 1,698 states), G4 (C1 x 2.0 at seed indices 2 and 3), G5's base pilot (20 per scenario); c. the
# pilot's judgments (gpt-oss-120b) and G5's plan; d. G5's arms within the 2.5 A100-h budget and their
# judgments. Floor about 3.7-3.9 A100-h ($4.65), ceiling about 6.2-6.4 ($7.55). The script creates
# the instance (retrying when Thunder
# reports no capacity, then RunPod Secure, logged), syncs, launches, streams the log, downloads the
# results, then deletes the instance and checks the account lists nothing it created.
set -euo pipefail
cd "$(dirname "$0")/../../.."
[ -n "${RUNPOD_API_KEY:-}" ] || { echo "BAIL: export RUNPOD_API_KEY too (the fallback when Thunder has no capacity)"; exit 1; }
source papers/gpc_goal_pressure_conduct/runpod/gpc3_inputs.sh
export ON_DEMAND_SYNC="$GPC_BUNDLE"
export SELF_PAPER=$G RESULTS_SUBPATH=outputs/gpc3_v1 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$G/runpod/remote_gpc3.sh MAX_SYNC_GB=3 DISK_GB=300
export PROVIDER=thunder TNR_GPU=a100xl TNR_VCPUS=8 TNR_PRICE=1.15 POD_NAME=deepsteer-gpc3
export TNR_FALLBACK=runpod_secure RUNPOD_GPU_TYPES="NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB"
exec bash papers/runpod_common/run_session.sh
