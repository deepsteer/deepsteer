#!/usr/bin/env bash
# Launch card: the finalist-bars pod (CST Amendments 41, 42), two pods that run in parallel
# (Amendment 42 item 3). The author names the candidate and runs, in two terminals:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   PART=bars CANDIDATE=<tag> SOURCE=<outputs subdir holding train_<tag>.json> \
#     bash papers/cst_constitutional_steering/runpod/launch_cst_finalist.sh
#   PART=mach CANDIDATE=<tag> SOURCE=<same> \
#     bash papers/cst_constitutional_steering/runpod/launch_cst_finalist.sh
#
# e.g. CANDIDATE=p_ctx SOURCE=outputs/pressure, or CANDIDATE=c2_rule_right SOURCE=outputs/loop_v1.
# One A100-80GB each: bars at 10.5 A100-h (refusal retention, MoralBench, MoReBench with its judge,
# the session check, capability at n 800 unless P-ctx; watchdog 11.0 h), mach at 13.0 A100-h
# (MACHIAVELLI on both arms; Amendment 43's exception to the 10.5 bound; watchdog 14.0 h); stages
# stop 0.25 h before the price. Results in
# outputs/finalist_<tag>_<part>. Synced by name: the candidate's record (copied to
# outputs/finalist_in_<part>) and adapter, C_anth, the smoke pass's fixed adapter and its letters;
# MACHIAVELLI's game bundle only if the pod's Drive download fails (ON_DEMAND_SYNC, from this
# machine to this pod).
#
# What a MACHIAVELLI failure loses (Amendment 43 item 1): every decision is written and flushed as
# it is taken. A stage that dies on the pod is started again (up to 3 times) and replays each
# unfinished episode's saved actions through the game engine with no generation, then continues;
# the loss is the requests in flight (at most one decision per episode), the engine reload (about
# 5 min) and the replay. If the pod itself is lost, relaunch this card: it syncs back the saved
# MACHIAVELLI files pulled from the last pod (monitor pulls every 15 min, so up to 15 minutes of
# decisions are generated again), and the price and the deadline start over. A replayed step whose
# prompt hash, loop override or choice count does not match the saved row restarts that episode
# from its first decision (counted in run_meta as replay_restarts).
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
: "${PART:?PART=bars or PART=mach}" "${CANDIDATE:?name the candidate: CANDIDATE=<tag>}" "${SOURCE:?SOURCE=<outputs subdir>}"
case "$PART" in bars|mach) ;; *) echo "BAIL: PART must be bars or mach"; exit 1;; esac
R="$P/$SOURCE/train_$CANDIDATE.json"
[ -f "$R" ] || { echo "BAIL: $R is missing"; exit 1; }
I="$P/outputs/finalist_in_$PART"
rm -rf "$I" && mkdir -p "$I" && cp "$R" "$I/"
SYNC="outputs/_sync/C_anth.txt,outputs/finalist_in_$PART/train_$CANDIDATE.json"
d=$(python3 -c "import json; p=list(json.load(open('$R'))['adapters'].values())[-1]; print(p.split('/papers/cst_constitutional_steering/')[1])")
[ -f "$P/$d/adapter_model.safetensors" ] || { echo "BAIL: $d's adapter is not here"; exit 1; }
SYNC="$SYNC,$d/adapter_model.safetensors,$d/adapter_config.json"
F=outputs/pilot_sdf  # Amendment 31 item 2: the smoke pass's fixed adapter and its saved HF letters
SYNC="$SYNC,$F/adapters/d_doc/1e+06/adapter_model.safetensors,$F/adapters/d_doc/1e+06/adapter_config.json"
SYNC="$SYNC,$F/parity/hf_d_doc_1M.json,$F/parity/hf_base.json"
if [ "$PART" = mach ]; then
  # Amendment 43 item 1: a relaunch resumes from the saved MACHIAVELLI files pulled from the last pod
  M="outputs/finalist_${CANDIDATE}_mach"
  for f in mach/episodes.jsonl mach/decisions.jsonl mach/_counts.jsonl mach/run_meta.jsonl mach_decide.json; do
    if [ -f "$P/$M/$f" ]; then SYNC="$SYNC,$M/$f"; echo ">> resuming: syncing $M/$f"; fi
  done
  B="$HOME/.cache/deepsteer/mch/dl/game_data.zip"
  export ON_DEMAND_SYNC="mch_bundle|$B|/workspace/mch_bundle/game_data.zip|e13d07b442a9ffa835e6c43f95fbc5d97a45de49faef7b1f969f014b0a95b6c1"
fi
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/finalist_${CANDIDATE}_$PART CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_finalist_$PART.sh MAX_SYNC_GB=3 DISK_GB=300
export POD_NAME=deepsteer-cst-finalist-$PART GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
