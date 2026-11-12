#!/usr/bin/env bash
# Launch card: the finalist-bars pod (CST Amendment 41). The author names the candidate and runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   CANDIDATE=<tag> SOURCE=<outputs subdir holding train_<tag>.json> \
#     bash papers/cst_constitutional_steering/runpod/launch_cst_finalist.sh
#
# e.g. CANDIDATE=p_ctx SOURCE=outputs/pressure, or CANDIDATE=c2_rule_right SOURCE=outputs/loop_v1.
# One A100-80GB, priced at or under 10.5 A100-h (stages stop at 10.25 h, watchdog 11.0 h); results
# in outputs/finalist_<tag>. Synced by name: the candidate's record (copied to outputs/finalist_in)
# and adapter, C_anth, the smoke pass's fixed adapter and its letters; MACHIAVELLI's game bundle
# only if the pod's Drive download fails (ON_DEMAND_SYNC, from this machine to this pod).
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
: "${CANDIDATE:?name the candidate: CANDIDATE=<tag>}" "${SOURCE:?SOURCE=<outputs subdir>}"
R="$P/$SOURCE/train_$CANDIDATE.json"
[ -f "$R" ] || { echo "BAIL: $R is missing"; exit 1; }
rm -rf "$P/outputs/finalist_in" && mkdir -p "$P/outputs/finalist_in" && cp "$R" "$P/outputs/finalist_in/"
SYNC="outputs/_sync/C_anth.txt,outputs/finalist_in/train_$CANDIDATE.json"
d=$(python3 -c "import json; p=list(json.load(open('$R'))['adapters'].values())[-1]; print(p.split('/papers/cst_constitutional_steering/')[1])")
[ -f "$P/$d/adapter_model.safetensors" ] || { echo "BAIL: $d's adapter is not here"; exit 1; }
SYNC="$SYNC,$d/adapter_model.safetensors,$d/adapter_config.json"
F=outputs/pilot_sdf  # Amendment 31 item 2: the smoke pass's fixed adapter and its saved HF letters
SYNC="$SYNC,$F/adapters/d_doc/1e+06/adapter_model.safetensors,$F/adapters/d_doc/1e+06/adapter_config.json"
SYNC="$SYNC,$F/parity/hf_d_doc_1M.json,$F/parity/hf_base.json"
B="$HOME/.cache/deepsteer/mch/dl/game_data.zip"
export ON_DEMAND_SYNC="mch_bundle|$B|/workspace/mch_bundle/game_data.zip|e13d07b442a9ffa835e6c43f95fbc5d97a45de49faef7b1f969f014b0a95b6c1"
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/finalist_$CANDIDATE CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_finalist.sh MAX_SYNC_GB=3 DISK_GB=300
export POD_NAME=deepsteer-cst-finalist GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
