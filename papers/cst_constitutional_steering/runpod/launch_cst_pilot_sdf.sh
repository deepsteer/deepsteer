#!/usr/bin/env bash
# Launch card: the CST pilot SDF pod (PILOT_BUILD.md §5 card 1; Amendments 20-22). The author runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_pilot_sdf.sh
#
# One A100-80GB, priced at about 10.3 A100-h (watchdog 11.5 h). The three synced inputs are the
# pinned C_anth text, the A0n text and the eval 8-gram hash index. RESUME=1 adds outputs/pilot_sdf/
# so a rerun skips what is on disk.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
SYNC="outputs/_sync/C_anth.txt,outputs/_sync/a0n_faraday_v1.txt,outputs/_sync/eval_8gram_hashes_v1.npy"
[ "${RESUME:-0}" = 1 ] && SYNC="$SYNC,outputs/pilot_sdf/"
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/pilot_sdf CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_pilot_sdf.sh MAX_SYNC_GB=1 DISK_GB=300
export POD_NAME=deepsteer-cst-pilot-sdf GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
