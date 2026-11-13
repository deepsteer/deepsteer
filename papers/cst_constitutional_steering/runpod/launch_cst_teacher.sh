#!/usr/bin/env bash
# Launch card: the single-decision teacher check (CST Amendments 46 item C, 47 item 2) with P-ctx's
# riders (Amendment 47 item 1). The author runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_teacher.sh
#
# One A100-80GB: the check about 5.95 A100-h (base and GOOD_BEHAVIOR on 1,698 states at k 2, the
# constitution on the nested 888 at k 2; ceiling 6), then P-ctx's refusal retention and MoralBench
# (about 0.5; ceiling 1); watchdog 7.5 h. Results in outputs/teacher_v1; the read is teacher_read.py.
# Synced by name: the finalist MACHIAVELLI run's saved decisions and episodes (the states are rebuilt
# from them), C_anth, P-ctx's record (copied to outputs/teacher_in) and adapter, the smoke pass's fixed
# adapter and its letters; the game bundle only if the pod's Drive download fails (ON_DEMAND_SYNC).
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
R="$P/outputs/pressure/train_p_ctx.json"
[ -f "$R" ] || { echo "BAIL: $R is missing"; exit 1; }
rm -rf "$P/outputs/teacher_in" && mkdir -p "$P/outputs/teacher_in" && cp "$R" "$P/outputs/teacher_in/"
M=outputs/finalist_c2_rule_right_mach/mach
for f in decisions.jsonl episodes.jsonl; do [ -f "$P/$M/$f" ] || { echo "BAIL: $M/$f is missing"; exit 1; }; done
SYNC="outputs/_sync/C_anth.txt,outputs/teacher_in/train_p_ctx.json,$M/decisions.jsonl,$M/episodes.jsonl"
d=$(python3 -c "import json; p=list(json.load(open('$R'))['adapters'].values())[-1]; print(p.split('/papers/cst_constitutional_steering/')[1])")
[ -f "$P/$d/adapter_model.safetensors" ] || { echo "BAIL: $d's adapter is not here"; exit 1; }
SYNC="$SYNC,$d/adapter_model.safetensors,$d/adapter_config.json"
F=outputs/pilot_sdf  # Amendment 31 item 2: the smoke pass's fixed adapter and its saved HF letters
SYNC="$SYNC,$F/adapters/d_doc/1e+06/adapter_model.safetensors,$F/adapters/d_doc/1e+06/adapter_config.json"
SYNC="$SYNC,$F/parity/hf_d_doc_1M.json,$F/parity/hf_base.json"
B="$HOME/.cache/deepsteer/mch/dl/game_data.zip"
export ON_DEMAND_SYNC="mch_bundle|$B|/workspace/mch_bundle/game_data.zip|e13d07b442a9ffa835e6c43f95fbc5d97a45de49faef7b1f969f014b0a95b6c1"
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/teacher_v1 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_teacher.sh MAX_SYNC_GB=3 DISK_GB=300
# Amendment 48 item 2: Community Cloud first and PCIe first (on trial: item 5's rule)
export POD_NAME=deepsteer-cst-teacher GPU_TYPES="NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB"
export CLOUD_TYPES="COMMUNITY,SECURE"
exec bash papers/runpod_common/run_session.sh
