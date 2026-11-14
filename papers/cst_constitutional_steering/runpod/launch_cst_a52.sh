#!/usr/bin/env bash
# Launch card: CST Amendment 52's pod on RunPod (the fallback; the ported provider's card drives the
# same runner). The author runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_a52.sh
#
# One A100-80GB against the 10.5 h price (stages stop at 10.25 h, watchdog 11.0 h), in order and cut
# from the end: (a) validation (the fixed smoke check, P-ctx's parity, the P-ctx reproduction on the
# operator cell against a same-session base; a failed validation stops the pod before any read);
# (b) A0n with GOOD_BEHAVIOR on the nested 888 at k 2; (c) P-ctx's single-decision read (P-ctx on
# the 1,698 and a same-session base on the 888 at k 2); (d) the strength sweep at 0.25, 0.5, 0.75 on
# the operator cell and cell A′ (descriptive). Rate windows every 5 minutes. Results in
# outputs/a52_v1; the read is a52_read.py. Synced by name: C_anth and A0n's text, P-ctx's record
# (copied to outputs/a52_in) and adapter, the finalist MACHIAVELLI run's decisions and episodes, the
# smoke pass's fixed adapter and its letters; the game bundle only if the Drive download fails.
set -euo pipefail
cd "$(dirname "$0")/../../.."
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
export POD_NAME=deepsteer-cst-a52 GPU_TYPES="NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB"
export CLOUD_TYPES="COMMUNITY,SECURE"
exec bash papers/runpod_common/run_session.sh
