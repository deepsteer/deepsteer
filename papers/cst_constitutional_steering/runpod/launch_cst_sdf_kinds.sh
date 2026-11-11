#!/usr/bin/env bash
# Launch card: the kind-split pod (CST Amendment 32 items 6-10). The author runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_sdf_kinds.sh
#
# One A100-80GB, priced at 6.0 A100-h (watchdog 6.5 h); results in outputs/pilot_sdf4. Synced by
# name: both corpora's kept-document chunks, C_anth (A0s), D-neutral 1M and its record (the guard
# read), the pressure scenarios (the guard cell), the smoke pass's fixed adapter and its letters.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
[ -f "$P/data/content_scenarios_v1.json" ] || { echo "BAIL: the content gate's scenarios are not built"; exit 1; }
SYNC="outputs/pressure/scenarios.json,outputs/_sync/C_anth.txt,outputs/pilot_sdf/train_d_neutral.json"
SYNC="$SYNC,outputs/pilot_sdf/adapters/d_neutral/1e+06/adapter_model.safetensors,outputs/pilot_sdf/adapters/d_neutral/1e+06/adapter_config.json"
for f in "$P"/outputs/pilot_sdf/corpus/d_doc/chunk_*.jsonl "$P"/outputs/pilot_sdf/corpus/d_neutral/chunk_*.jsonl; do SYNC="$SYNC,${f#$P/}"; done
F=outputs/pilot_sdf  # Amendment 31 item 2: the smoke pass's fixed adapter and its saved HF letters
SYNC="$SYNC,$F/adapters/d_doc/1e+06/adapter_model.safetensors,$F/adapters/d_doc/1e+06/adapter_config.json"
SYNC="$SYNC,$F/parity/hf_d_doc_1M.json,$F/parity/hf_base.json"
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/pilot_sdf4 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_sdf_kinds.sh MAX_SYNC_GB=2 DISK_GB=300
export POD_NAME=deepsteer-cst-sdf-kinds GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
