#!/usr/bin/env bash
# Launch card: GPC-1 on Thunder Compute (GPC_PLAN.md v0.2 §5), the RunPod Secure fallback inside.
# One command, the session's log on the console, the instance created and deleted inside the
# script. The author runs:
#
#   export HF_TOKEN=... RUNPOD_API_KEY=...   # Thunder's login is the `tnr` CLI (`tnr login`)
#   bash papers/gpc_goal_pressure_conduct/runpod/launch_gpc1_thunder.sh
#
# One A100 80GB (8 vCPUs, 300 GB, about $1.15 an hour) against the 10.5 h price (stages stop at
# 10.25 h, watchdog 11.0 h). Stages (remote_gpc1.sh): a. the training smoke and the fixed smoke
# check; b. GOOD_BEHAVIOR, C_conduct in its slot and good_lines on S-hi, and the same-session base
# on S-hi and S-lo, at k 1 (item 11's switch); c1. the conduct grid (the writer's gate, the
# base-rate pilot and allocation, the held-out cell, C1's teacher and gate); c2. the third arm
# (W-self or C1g; cut first); d. training (C1, the third arm, W-ref); e. parity and the screens
# (S-hi, S-lo, the held-out cell), W-ref's adapter deleted; f. the registered read of the best arm
# that clears (never W-ref); f+. the operator cell and cell A′ if time remains. The script creates
# the instance (retrying when Thunder reports no capacity: 6 tries, 5 minutes apart, then RunPod
# Secure, logged), syncs, launches, streams the log, downloads the results, then deletes the
# instance and checks the account lists nothing it created (Ctrl-C or any error also deletes it).
set -euo pipefail
cd "$(dirname "$0")/../../.."
[ -n "${RUNPOD_API_KEY:-}" ] || { echo "BAIL: export RUNPOD_API_KEY too (the fallback when Thunder has no capacity)"; exit 1; }
source papers/gpc_goal_pressure_conduct/runpod/gpc1_inputs.sh
export ON_DEMAND_SYNC="$GPC_BUNDLE"
export SELF_PAPER=$G RESULTS_SUBPATH=outputs/gpc1_v1 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$G/runpod/remote_gpc1.sh MAX_SYNC_GB=3 DISK_GB=300
export PROVIDER=thunder TNR_GPU=a100xl TNR_VCPUS=8 TNR_PRICE=1.15 POD_NAME=deepsteer-gpc1
export TNR_FALLBACK=runpod_secure RUNPOD_GPU_TYPES="NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB"
exec bash papers/runpod_common/run_session.sh
