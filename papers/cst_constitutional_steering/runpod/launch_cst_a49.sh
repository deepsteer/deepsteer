#!/usr/bin/env bash
# Launch card: CST Amendment 49's pod. The author runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_a49.sh
#
# One A100-80GB against the 10.5 h price (stages stop at 10.25 h, watchdog 11.0 h), in order and cut
# from the end: (a) the judge-session check, MRB's saved base rollouts re-judged; (b) A0n on
# MoReBench at k 2, judged; (c) the constitution with GOOD_BEHAVIOR on the nested 888 at k 2 and the
# constitution-alone citing cell (300 states at k 1); (d) P-ctx on the 1,698 at k 2 and a
# same-session base on the nested 888, after the smoke pass and P-ctx's parity. Results in
# outputs/a49_v1; the read is a49_read.py. Synced by name: the re-judge size and MRB's saved base
# rollouts (copied to outputs/a49_in), C_anth and A0n's text, P-ctx's record and adapter, the
# finalist MACHIAVELLI run's decisions and episodes, the smoke pass's fixed adapter and its letters;
# the game bundle only if the pod's Drive download fails (ON_DEMAND_SYNC).
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
# Amendment 49 item 2a: scenarios re-judged (both rollouts each). Amendment 50 item 1's settings
# check: MRB's p2 judge ran at tensor parallel 2 on two A100s, the bars pod and pod B on one GPU;
# everything else matches, so all 1,000 rollouts are re-judged once (500 scenarios)
REJUDGE_SCENARIOS=500
[ "$REJUDGE_SCENARIOS" != UNSET ] || { echo "BAIL: the re-judge size (Amendment 49 item 2a) is not ruled yet"; exit 1; }
I="$P/outputs/a49_in"
rm -rf "$I" && mkdir -p "$I"
printf '{"scenarios": %d}\n' "$REJUDGE_SCENARIOS" > "$I/rejudge.json"
M2=papers/mrb_morebench/outputs/p2
cp "$M2/gen/qwen38_27b.none.jsonl" "$I/qwen38_27b.none.gen.jsonl"
cp "$M2/_raw/gen/qwen38_27b.none.jsonl" "$I/qwen38_27b.none.raw.jsonl"
R="$P/outputs/pressure/train_p_ctx.json"
[ -f "$R" ] || { echo "BAIL: $R is missing"; exit 1; }
cp "$R" "$I/"
M=outputs/finalist_c2_rule_right_mach/mach
for f in decisions.jsonl episodes.jsonl; do [ -f "$P/$M/$f" ] || { echo "BAIL: $M/$f is missing"; exit 1; }; done
SYNC="outputs/_sync/C_anth.txt,outputs/_sync/a0n_faraday_v1.txt,$M/decisions.jsonl,$M/episodes.jsonl"
SYNC="$SYNC,outputs/a49_in/rejudge.json,outputs/a49_in/qwen38_27b.none.gen.jsonl"
SYNC="$SYNC,outputs/a49_in/qwen38_27b.none.raw.jsonl,outputs/a49_in/train_p_ctx.json"
d=$(python3 -c "import json; p=list(json.load(open('$R'))['adapters'].values())[-1]; print(p.split('/papers/cst_constitutional_steering/')[1])")
[ -f "$P/$d/adapter_model.safetensors" ] || { echo "BAIL: $d's adapter is not here"; exit 1; }
SYNC="$SYNC,$d/adapter_model.safetensors,$d/adapter_config.json"
F=outputs/pilot_sdf  # Amendment 31 item 2: the smoke pass's fixed adapter and its saved HF letters
SYNC="$SYNC,$F/adapters/d_doc/1e+06/adapter_model.safetensors,$F/adapters/d_doc/1e+06/adapter_config.json"
SYNC="$SYNC,$F/parity/hf_d_doc_1M.json,$F/parity/hf_base.json"
B="$HOME/.cache/deepsteer/mch/dl/game_data.zip"
export ON_DEMAND_SYNC="mch_bundle|$B|/workspace/mch_bundle/game_data.zip|e13d07b442a9ffa835e6c43f95fbc5d97a45de49faef7b1f969f014b0a95b6c1"
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/a49_v1 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_a49.sh MAX_SYNC_GB=3 DISK_GB=400
# Amendment 49 item 2: Community Cloud first under the trial rule (Amendment 48 item 5); PCIe first
export POD_NAME=deepsteer-cst-a49 GPU_TYPES="NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB"
export CLOUD_TYPES="COMMUNITY,SECURE"
exec bash papers/runpod_common/run_session.sh
