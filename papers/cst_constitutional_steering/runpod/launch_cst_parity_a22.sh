#!/usr/bin/env bash
# Launch card: CST-A22's parity mini-pod (Amendment 25 Part A). The author runs, after the SDF
# pod's download has brought its adapters back:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_parity_a22.sh
#
# One A100-80GB, priced at about 0.3 A100-h (watchdog 1.0 h). The synced inputs are D-doc 1M's two
# adapters (without and with recovery, 0.9 GB).
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
for d in 1e+06 1e+06_rec; do
  [ -f "$P/outputs/pilot_sdf/adapters/d_doc/$d/adapter_model.safetensors" ] \
    || { echo "BAIL: $P/outputs/pilot_sdf/adapters/d_doc/$d has no weights (SDF download first)"; exit 1; }
done
# named files, not directories: directory entries are filtered after rsync_exclude.txt, which
# drops *.safetensors (the weights would not sync)
A=outputs/pilot_sdf/adapters/d_doc
SYNC="$A/1e+06/adapter_model.safetensors,$A/1e+06/adapter_config.json"
SYNC="$SYNC,$A/1e+06_rec/adapter_model.safetensors,$A/1e+06_rec/adapter_config.json"
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/parity_a22 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_parity_a22.sh MAX_SYNC_GB=2 DISK_GB=200
export POD_NAME=deepsteer-cst-parity-a22 GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
