#!/usr/bin/env bash
# Launch card: the loop (CST Amendment 39 item 4). The author runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_loop.sh
#
# One A100-80GB, priced at about 10.5 A100-h (watchdog 11.0 h); results in outputs/loop_v1. One
# iteration costs about 5.7 A100-h with cached rollouts and about 8 A100-h without. Synced by name:
# the cache (2-roll's scenarios, labels, contrast and A0s rollouts), 2-ctx's P-ctx record and
# adapter, S0a's cell A' rollouts, C_anth, the smoke pass's fixed adapter and its letters.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
O=outputs/pressure
[ -f "$P/data/loop_cache_v1.json" ] || { echo "BAIL: the cache record is missing"; exit 1; }
SYNC="outputs/_sync/C_anth.txt"
for f in scenarios.json act.jsonl act.json contrast.json train_p_ctx.json; do SYNC="$SYNC,$O/$f"; done
for f in "$P/$O"/roll/a0s/chunk_*.jsonl; do SYNC="$SYNC,${f#$P/}"; done
d=$(python3 -c "import json; p=list(json.load(open('$P/$O/train_p_ctx.json'))['adapters'].values())[-1]; print(p.split('/papers/cst_constitutional_steering/')[1])")
[ -f "$P/$d/adapter_model.safetensors" ] || { echo "BAIL: $d's adapter is not here"; exit 1; }
SYNC="$SYNC,$d/adapter_model.safetensors,$d/adapter_config.json"
for f in "$P"/outputs/s0a/gen/a0/forced_plain_*.jsonl "$P"/outputs/s0a/gen/a0s/forced_plain_*.jsonl; do SYNC="$SYNC,${f#$P/}"; done
F=outputs/pilot_sdf  # Amendment 31 item 2: the smoke pass's fixed adapter and its saved HF letters
SYNC="$SYNC,$F/adapters/d_doc/1e+06/adapter_model.safetensors,$F/adapters/d_doc/1e+06/adapter_config.json"
SYNC="$SYNC,$F/parity/hf_d_doc_1M.json,$F/parity/hf_base.json"
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/loop_v1 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_loop.sh MAX_SYNC_GB=3 DISK_GB=300
export POD_NAME=deepsteer-cst-loop GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
