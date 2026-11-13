#!/usr/bin/env bash
# Launch card: the MoReBench pod, stage 1 (CST Amendment 46 item B). The author runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_mrb2x2.sh
#
# One A100-80GB, about 9 A100-h against the 10.5 bound (stages stop at 10.25 h, watchdog 11.0 h);
# results in outputs/mrb2x2_v1. A0s, P-ctx and P-ctx-n at k 2 on MoReBench's 500 under MRB's
# protocol, judged by MRB's judge on one GPU, read against MRB's saved base rows (mrb2x2_read.py).
# **If the gate at load forecasts past 10.5 h, P-ctx-n moves to a later pod** (gate.json says so and
# the runner skips it). The A0s engine runs at max_model_len 46,080 (the longest A0s prompt, 34,722
# tokens, plus MRB's 10,500 cap; the engine of record's 44,032 is too short; capacity only).
# Synced by name: P-ctx's and P-ctx-n's records (copied to outputs/mrb2x2_in) and adapters, C_anth,
# the smoke pass's fixed adapter and its letters.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
I="$P/outputs/mrb2x2_in"
rm -rf "$I" && mkdir -p "$I"
SYNC="outputs/_sync/C_anth.txt"
for t in p_ctx p_ctx_n; do
  R="$P/outputs/pressure/train_$t.json"
  [ -f "$R" ] || { echo "BAIL: $R is missing"; exit 1; }
  cp "$R" "$I/"
  d=$(python3 -c "import json; p=list(json.load(open('$R'))['adapters'].values())[-1]; print(p.split('/papers/cst_constitutional_steering/')[1])")
  [ -f "$P/$d/adapter_model.safetensors" ] || { echo "BAIL: $d's adapter is not here"; exit 1; }
  SYNC="$SYNC,outputs/mrb2x2_in/train_$t.json,$d/adapter_model.safetensors,$d/adapter_config.json"
done
F=outputs/pilot_sdf  # Amendment 31 item 2: the smoke pass's fixed adapter and its saved HF letters
SYNC="$SYNC,$F/adapters/d_doc/1e+06/adapter_model.safetensors,$F/adapters/d_doc/1e+06/adapter_config.json"
SYNC="$SYNC,$F/parity/hf_d_doc_1M.json,$F/parity/hf_base.json"
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/mrb2x2_v1 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_mrb2x2.sh MAX_SYNC_GB=3 DISK_GB=300
export POD_NAME=deepsteer-cst-mrb2x2 GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
