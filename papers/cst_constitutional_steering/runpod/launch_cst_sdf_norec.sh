#!/usr/bin/env bash
# Launch card: SDF pod 2, the no-recovery reads (Amendment 29 items 5-7, 10). The author runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_sdf_norec.sh
#
# One A100-80GB, priced at about 7.2 A100-h of reads and training, then the D-doc corpus continuation to 10.3 h (watchdog 10.5 h). Synced by name: the five saved
# adapters without recovery, their four training records, D-doc's corpus, the pressure scenarios.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
O=outputs/pilot_sdf
SYNC="outputs/pressure/scenarios.json,outputs/_sync/C_anth.txt,outputs/_sync/eval_8gram_hashes_v1.npy"
for f in train_d_doc.json train_d_neutral.json train_d_chat.json train_d_raw.json; do SYNC="$SYNC,$O/$f"; done
for a in d_neutral/1e+06 d_doc/1e+06 d_doc/300000 d_chat/1e+06 d_raw/102400; do
  SYNC="$SYNC,$O/adapters/$a/adapter_model.safetensors,$O/adapters/$a/adapter_config.json"
done
for f in "$P/$O"/corpus/d_doc/chunk_*.jsonl; do SYNC="$SYNC,${f#$P/}"; done
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/pilot_sdf2 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_sdf_norec.sh MAX_SYNC_GB=4 DISK_GB=300
export POD_NAME=deepsteer-cst-sdf-norec GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
