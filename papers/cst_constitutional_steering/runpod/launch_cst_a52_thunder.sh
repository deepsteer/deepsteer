#!/usr/bin/env bash
# Launch card: CST Amendment 52's pod on Thunder Compute (the ported provider; Amendment 51 item D,
# the author's conditional approval in the Amendment 52 addendum). One command, the session's log
# on the console, the instance created and deleted inside the script. The author runs:
#
#   export HF_TOKEN=... RUNPOD_API_KEY=...   # Thunder's login is the `tnr` CLI (`tnr login`);
#   bash papers/cst_constitutional_steering/runpod/launch_cst_a52_thunder.sh   # RunPod's key is the fallback
#
# One A100 80GB (8 vCPUs, 300 GB, about $1.15 an hour) against the 10.5 h price (stages stop at
# 10.25 h, watchdog 11.0 h); the stages and synced inputs are launch_cst_a52.sh's (RunPod, the
# fallback). The script creates the instance (retrying when Thunder reports no capacity: 6 tries,
# 5 minutes apart, then RunPod Secure, logged; CST Amendment 53 item 3), brings it up (/workspace, rsync, Python 3.11 with its headers), syncs, launches,
# streams the log, downloads the results, then deletes the instance and checks the account lists
# nothing it created (Ctrl-C or any error also deletes it; KEEP_POD=1 keeps it). Nothing carrying
# the partner's document runs on Thunder (Amendment 52 item 2).
set -euo pipefail
cd "$(dirname "$0")/../../.."
[ -n "${RUNPOD_API_KEY:-}" ] || { echo "BAIL: export RUNPOD_API_KEY too (the fallback when Thunder has no capacity)"; exit 1; }
P=papers/cst_constitutional_steering
R="$P/outputs/pressure/train_p_ctx.json"
[ -f "$R" ] || { echo "BAIL: $R is missing"; exit 1; }
rm -rf "$P/outputs/a52_in" && mkdir -p "$P/outputs/a52_in" && cp "$R" "$P/outputs/a52_in/"
M=outputs/finalist_c2_rule_right_mach/mach
for f in decisions.jsonl episodes.jsonl; do [ -f "$P/$M/$f" ] || { echo "BAIL: $M/$f is missing"; exit 1; }; done
SYNC="outputs/_sync/C_anth.txt,outputs/_sync/a0n_faraday_v1.txt,outputs/a52_in/train_p_ctx.json"
SYNC="$SYNC,$M/decisions.jsonl,$M/episodes.jsonl"
d=$(python3 -c "import json; p=list(json.load(open('$R'))['adapters'].values())[-1]; print(p.split('/papers/cst_constitutional_steering/')[1])")
[ -f "$P/$d/adapter_model.safetensors" ] || { echo "BAIL: $d's adapter is not here"; exit 1; }
SYNC="$SYNC,$d/adapter_model.safetensors,$d/adapter_config.json"
F=outputs/pilot_sdf  # Amendment 31 item 2: the smoke pass's fixed adapter and its saved HF letters
SYNC="$SYNC,$F/adapters/d_doc/1e+06/adapter_model.safetensors,$F/adapters/d_doc/1e+06/adapter_config.json"
SYNC="$SYNC,$F/parity/hf_d_doc_1M.json,$F/parity/hf_base.json"
B="$HOME/.cache/deepsteer/mch/dl/game_data.zip"
export ON_DEMAND_SYNC="mch_bundle|$B|/workspace/mch_bundle/game_data.zip|e13d07b442a9ffa835e6c43f95fbc5d97a45de49faef7b1f969f014b0a95b6c1"
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/a52_v1 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_a52.sh MAX_SYNC_GB=3 DISK_GB=300
export PROVIDER=thunder TNR_GPU=a100xl TNR_VCPUS=8 TNR_PRICE=1.15 POD_NAME=deepsteer-cst-a52
export TNR_FALLBACK=runpod_secure RUNPOD_GPU_TYPES="NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB"
exec bash papers/runpod_common/run_session.sh
