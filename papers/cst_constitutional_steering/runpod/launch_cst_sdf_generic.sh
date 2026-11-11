#!/usr/bin/env bash
# Launch card: tomorrow's SDF pod, the generic branch of Amendment 29 item 6 (launched only if item
# 6's control reads that branch). The author runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_sdf_generic.sh
#
# One A100-80GB, priced at about 4.5 A100-h (5.0 with the D-neutral 1M re-read) (watchdog 10.5 h); results in outputs/pilot_sdf3.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
SYNC="outputs/pressure/scenarios.json,outputs/pilot_sdf/train_d_doc.json"
SYNC="$SYNC,outputs/pilot_sdf/adapters/d_doc/300000/adapter_model.safetensors,outputs/pilot_sdf/adapters/d_doc/300000/adapter_config.json"
for f in "$P"/outputs/pilot_sdf/corpus/d_neutral/chunk_*.jsonl; do SYNC="$SYNC,${f#$P/}"; done
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/pilot_sdf3 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_sdf_generic.sh MAX_SYNC_GB=2 DISK_GB=300
export POD_NAME=deepsteer-cst-sdf-generic GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
