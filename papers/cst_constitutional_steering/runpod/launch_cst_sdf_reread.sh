#!/usr/bin/env bash
# Launch card: pod 1 of 2026-11-10, the SDF re-read on the fixed serving path and D-neutral
# (Amendment 26 item 5; CST-A22). The author runs, after the parity mini-pod confirms R_noserve:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_sdf_reread.sh
#
# One A100-80GB, priced at about 8.3 A100-h (watchdog 10.5 h). Synced by name (directory entries
# are filtered after rsync_exclude.txt, which drops *.safetensors): the three _sync inputs, the seven
# saved adapters read tonight, the four training records and the recovery set (for D-neutral).
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
O=outputs/pilot_sdf
SYNC="outputs/_sync/C_anth.txt,outputs/_sync/a0n_faraday_v1.txt,outputs/_sync/eval_8gram_hashes_v1.npy"
SYNC="$SYNC,$O/train_d_doc.json,$O/train_r0.json,$O/train_d_chat.json,$O/train_d_raw.json"
SYNC="$SYNC,$O/recovery/replies.jsonl,$O/recovery/selected.json"
for a in d_doc/1e+06 d_doc/1e+06_rec d_doc/300000 d_doc/300000_rec r0/rec d_chat/1e+06_rec d_raw/102400_rec; do
  SYNC="$SYNC,$O/adapters/$a/adapter_model.safetensors,$O/adapters/$a/adapter_config.json"
done
[ -d "$P/$O/gen" ] && { echo "BAIL: $P/$O/gen holds last night's base-model reads; move them first"; exit 1; }
export SELF_PAPER=$P RESULTS_SUBPATH=$O CST_SESSION=2 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_sdf_reread.sh MAX_SYNC_GB=5 DISK_GB=300
export POD_NAME=deepsteer-cst-sdf-reread GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
