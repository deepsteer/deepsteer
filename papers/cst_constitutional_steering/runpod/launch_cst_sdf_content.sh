#!/usr/bin/env bash
# Launch card: tomorrow's SDF pod, the content branch of Amendment 29 item 6 (launched only if item
# 6's control reads that branch). The author runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_sdf_content.sh
#
# One A100-80GB, priced at about 7.2 A100-h with a 1 h corpus top-up, at most 10.3 (watchdog 10.5 h); results in outputs/pilot_sdf3.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
SYNC="outputs/pressure/scenarios.json,outputs/_sync/C_anth.txt,outputs/_sync/eval_8gram_hashes_v1.npy"
for f in "$P"/outputs/pilot_sdf/corpus/d_doc/chunk_*.jsonl "$P"/outputs/pilot_sdf2/corpus/d_doc/chunk_*.jsonl; do
  [ -f "$f" ] && SYNC="$SYNC,${f#$P/}"
done
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/pilot_sdf3 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_sdf_content.sh MAX_SYNC_GB=2 DISK_GB=300
export POD_NAME=deepsteer-cst-sdf-content GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
