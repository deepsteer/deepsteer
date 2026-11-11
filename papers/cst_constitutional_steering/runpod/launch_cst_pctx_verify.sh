#!/usr/bin/env bash
# Launch card: the P-ctx verification pod (CST Amendment 35). The author runs, once 2-ctx's
# download has landed in outputs/pressure:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_pctx_verify.sh
#
# One A100-80GB, priced at or under 10.5 A100-h (watchdog 11.0 h); results in
# outputs/pressure_verify. Synced by name: 2-ctx's P-ctx and P-ctx-n records and adapters; 2-roll's
# scenarios, labels, contrast and rollouts; C_anth (A0s); the smoke pass's fixed adapter and letters.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
O=outputs/pressure
for r in p_ctx p_ctx_n; do
  [ -f "$P/$O/train_$r.json" ] || { echo "BAIL: $O/train_$r.json is not here: 2-ctx's download has not landed"; exit 1; }
done
SYNC="outputs/_sync/C_anth.txt"
for f in scenarios.json act.jsonl act.json contrast.json train_p_ctx.json train_p_ctx_n.json; do SYNC="$SYNC,$O/$f"; done
for f in "$P/$O"/roll/*/chunk_*.jsonl; do SYNC="$SYNC,${f#$P/}"; done
for r in p_ctx p_ctx_n; do  # the adapter each record names, at the same path under this paper
  d=$(python3 -c "import json; p=list(json.load(open('$P/$O/train_$r.json'))['adapters'].values())[-1]; print(p.split('/papers/cst_constitutional_steering/')[1])")
  [ -f "$P/$d/adapter_model.safetensors" ] || { echo "BAIL: $d's adapter is not here"; exit 1; }
  SYNC="$SYNC,$d/adapter_model.safetensors,$d/adapter_config.json"
done
F=outputs/pilot_sdf  # Amendment 31 item 2: the smoke pass's fixed adapter and its saved HF letters
SYNC="$SYNC,$F/adapters/d_doc/1e+06/adapter_model.safetensors,$F/adapters/d_doc/1e+06/adapter_config.json"
SYNC="$SYNC,$F/parity/hf_d_doc_1M.json,$F/parity/hf_base.json"
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/pressure_verify CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_pctx_verify.sh MAX_SYNC_GB=3 DISK_GB=300
export POD_NAME=deepsteer-cst-pctx-verify GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
