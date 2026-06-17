#!/usr/bin/env bash
# Programmatic RunPod session for the Paper 6 (Phase 3 / ablation-resistance) GPU work.
#
#   spin up GPU -> rsync repo -> run experiments -> rsync results back -> terminate
#
# The pod is ALWAYS terminated on exit (trap), including on error or Ctrl-C, so
# you never leak GPU time. Set KEEP_POD=1 to leave it running for debugging.
#
# Prerequisites (local): curl, jq, ssh, rsync; RUNPOD_API_KEY exported; an SSH
# keypair at $SSH_KEY (+ .pub) injected via PUBLIC_KEY so sshd authorizes you.
#
# Recommended flow (cost-minimizing — see runpod/README.md):
#   1. Local pre-flight:  bash papers/6_ablation_resistance/runpod/local_test.sh
#   2. RunPod dry-run:    VALIDATE=1 ./run_session.sh    # ~5-10 min, one model
#   3. inspect outputs/_validate_dependency, confirm the score looks sane
#   4. Full Sprint 5:     ./run_session.sh               # 25-state sweep (~3 h)
#
# Usage:
#   export RUNPOD_API_KEY=...
#   VALIDATE=1 ./run_session.sh           # cheap validation run, then stop
#   ./run_session.sh                      # full session (Sprint 5 dependency sweep)
#   ONLY="dependency" ./run_session.sh    # just the named step(s)
#   KEEP_POD=1 ./run_session.sh           # don't terminate at the end
#   REUSE_POD_ID=<id> ./run_session.sh    # attach to an existing pod
set -euo pipefail

# ---------------------------- config (override via env) ----------------------
: "${RUNPOD_API_KEY:?export RUNPOD_API_KEY first}"
# OLMo-3 7B is ~14 GB fp16; --purge-hf-cache bounds disk to ~one model, so 24-48
# GB cards are plenty. SXM A100 first (NVLink hosts tend to have faster network /
# download), then PCIe A100, then 48->24 GB fallbacks for availability.
GPU_TYPE="${GPU_TYPE:-}"
GPU_TYPES="${GPU_TYPES:-NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,NVIDIA RTX A6000,NVIDIA A40,NVIDIA GeForce RTX 4090,NVIDIA RTX A5000}"
CLOUD_TYPES="${CLOUD_TYPES:-SECURE,COMMUNITY}"
[ -n "$GPU_TYPE" ] && GPU_TYPES="$GPU_TYPE"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK_GB="${DISK_GB:-100}"      # one 7B at a time + outputs; purge keeps it bounded
VOLUME_GB="${VOLUME_GB:-0}"
POD_NAME="${POD_NAME:-deepsteer-p6-7b}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/deepsteer}"

# Phase 3 step config (passed through to the remote runner).
VALIDATE="${VALIDATE:-0}"
DEP_KIND="${DEP_KIND:-probe}"          # moral direction kind to ablate (probe|meandiff)
PER_STATE="${PER_STATE:-0}"            # 1 -> ablate each state's own directions
DATASET_TARGET="${DATASET_TARGET:-40}" # probing pairs per foundation
RUN_DEPENDENCY="${RUN_DEPENDENCY:-1}"  # Sprint 5.2: moral dependency across the grid
RUN_ART_SFT="${RUN_ART_SFT:-0}"        # Sprint 6.4: control-SFT + ART-SFT training
RUN_EVAL="${RUN_EVAL:-0}"              # Sprint 7 (auto-skips until eval_pipeline.py exists)
# Sprint 6 ART knobs (used when RUN_ART_SFT=1).
ART_LAMBDA="${ART_LAMBDA:-0.01}"
ART_MAX_STEPS="${ART_MAX_STEPS:-400}"
N_GENERAL="${N_GENERAL:-1500}"
N_MORAL="${N_MORAL:-1500}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

# Standardized sync: shared universal excludes + keep this paper's outputs. Paper 6
# ALSO needs Paper 5's outputs (the moral directions npz + per-state pipeline
# directions it ablates), so those are included too — the universal excludes still
# strip the big blobs (ablated_model/, *.safetensors) first.
SELF_PAPER="papers/6_ablation_resistance"
DEP_PAPER="papers/5_moral_alignment"
RSYNC_EXCLUDE="$REPO_ROOT/papers/runpod_common/rsync_exclude.txt"

# ONLY="dependency" runs just the named step(s); forces others off.
if [ -n "${ONLY:-}" ]; then
  RUN_DEPENDENCY=0; RUN_ART_SFT=0; RUN_EVAL=0
  for _s in $ONLY; do
    case "$_s" in
      dependency) RUN_DEPENDENCY=1 ;;
      art) RUN_ART_SFT=1 ;;
      eval) RUN_EVAL=1 ;;
      *) echo "WARN: unknown ONLY step '$_s'" ;;
    esac
  done
  echo ">> ONLY='$ONLY' -> dependency=$RUN_DEPENDENCY art=$RUN_ART_SFT eval=$RUN_EVAL"
fi

for bin in curl jq ssh rsync; do
  command -v "$bin" >/dev/null || { echo "ERROR: '$bin' not found in PATH"; exit 1; }
done

gql() { curl -s "$API" -H 'Content-Type: application/json' -d "$(jq -n --arg q "$1" '{query:$q}')"; }

POD_ID="${REUSE_POD_ID:-}"

cleanup() {
  [ -n "${POD_ID:-}" ] || return 0
  [ -n "${REUSE_POD_ID:-}" ] && { echo "Attached to existing pod $POD_ID; not terminating."; return 0; }
  if [ "${KEEP_POD:-0}" = 1 ]; then
    echo "KEEP_POD=1 -> pod $POD_ID left RUNNING. Terminate later with:"
    echo "  curl -s '$API' -H 'Content-Type: application/json' -d '{\"query\":\"mutation { podTerminate(input:{podId:\\\"$POD_ID\\\"}) }\"}'"
    return 0
  fi
  echo ">> Terminating pod $POD_ID"
  gql "mutation { podTerminate(input:{podId:\"$POD_ID\"}) }" >/dev/null || \
    echo "WARN: terminate call failed; verify in the RunPod console!"
}
trap cleanup EXIT

# ---------------------------------- create -----------------------------------
if [ -z "$POD_ID" ]; then
  [ -f "${SSH_KEY}.pub" ] || { echo "ERROR: ${SSH_KEY}.pub not found"; exit 1; }
  PUBKEY="$(cat "${SSH_KEY}.pub")"
  IFS=',' read -ra _CLOUDS <<< "$CLOUD_TYPES"
  IFS=',' read -ra _GPUS <<< "$GPU_TYPES"
  GPU_TYPE=""; CLOUD_TYPE=""; LAST_ERR=""
  echo ">> Searching for capacity across ${#_GPUS[@]} GPU type(s) x ${#_CLOUDS[@]} cloud(s)..."
  for cloud in "${_CLOUDS[@]}"; do
    cloud="$(echo "$cloud" | xargs)"
    for gpu in "${_GPUS[@]}"; do
      gpu="$(echo "$gpu" | xargs)"
      echo "   trying: $gpu ($cloud)"
      CREATE_MUT="mutation { podFindAndDeployOnDemand(input: {
        cloudType: ${cloud}
        gpuCount: 1
        volumeInGb: ${VOLUME_GB}
        containerDiskInGb: ${DISK_GB}
        gpuTypeId: \"${gpu}\"
        name: \"${POD_NAME}\"
        imageName: \"${IMAGE}\"
        ports: \"22/tcp\"
        volumeMountPath: \"/workspace\"
        env: [{ key: \"PUBLIC_KEY\", value: \"${PUBKEY}\" }]
      }) { id } }"
      RESP="$(gql "$CREATE_MUT")"
      POD_ID="$(echo "$RESP" | jq -r '.data.podFindAndDeployOnDemand.id // empty')"
      if [ -n "$POD_ID" ]; then
        GPU_TYPE="$gpu"; CLOUD_TYPE="$cloud"
        echo ">> Pod $POD_ID created ($gpu, $cloud)"
        break 2
      fi
      LAST_ERR="$(echo "$RESP" | jq -r '.errors[0].message // empty')"
      [ -n "$LAST_ERR" ] && echo "      -> $LAST_ERR"
    done
  done
  if [ -z "$POD_ID" ]; then
    echo "ERROR: no capacity for any of [$GPU_TYPES] on [$CLOUD_TYPES]."
    echo "       Last message: ${LAST_ERR:-none}. Try later or widen GPU_TYPES."
    exit 1
  fi
fi

# ----------------------------- wait for SSH ----------------------------------
echo ">> Waiting for pod to be RUNNING with a public SSH port..."
SSH_HOST=""; SSH_PORT=""
for _ in $(seq 1 90); do
  R="$(gql "query { pod(input:{podId:\"$POD_ID\"}) { desiredStatus runtime { ports { ip isIpPublic privatePort publicPort type } } } }")"
  STATUS="$(echo "$R" | jq -r '.data.pod.desiredStatus // empty')"
  EP="$(echo "$R" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .type=="tcp" and .isIpPublic==true) | "\(.ip):\(.publicPort)"' | head -1)"
  if [ "$STATUS" = "RUNNING" ] && [ -n "$EP" ]; then
    SSH_HOST="${EP%:*}"; SSH_PORT="${EP##*:}"; break
  fi
  sleep 10
done
[ -n "$SSH_HOST" ] || { echo "ERROR: pod never exposed an SSH endpoint"; exit 1; }
echo ">> SSH endpoint: root@$SSH_HOST:$SSH_PORT"
echo ">> Log in from another terminal:"
echo "     ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$SSH_HOST"

SSH_OPTS=(-p "$SSH_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)
echo ">> Waiting for sshd..."
for _ in $(seq 1 30); do
  ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" 'echo ok' 2>/dev/null | grep -q ok && break
  sleep 5
done

# ---------------------------------- sync up ----------------------------------
echo ">> Preparing pod (mkdir + ensure rsync)..."
ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" \
  "mkdir -p $REMOTE_DIR && (command -v rsync >/dev/null 2>&1 || \
   { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync; } || \
   { command -v apk >/dev/null 2>&1 && apk add --no-cache rsync; })"

# Filter order matters (rsync first-match wins): shared universal excludes
# (blobs/caches) win first; then keep all of Paper 6's outputs; then keep ONLY
# the Paper 5 artifacts this run needs (default = just the base moral directions
# in olmo3_base/, ~13M; the 189M per-state pipeline/ dirs ride along only when
# PER_STATE=1). The `outputs/` dir itself must be included so rsync descends into
# it before the catch-all `/papers/*/outputs/` exclude drops every other paper.
RSYNC_FILTERS=(
  --exclude-from "$RSYNC_EXCLUDE"
  --include "/$SELF_PAPER/outputs/***"
  --include "/$DEP_PAPER/outputs/"
  --include "/$DEP_PAPER/outputs/olmo3_base/***"
)
if [ "$PER_STATE" = 1 ]; then
  RSYNC_FILTERS+=(--include "/$DEP_PAPER/outputs/pipeline/***")
  echo ">> PER_STATE=1: also syncing Paper 5 per-state pipeline directions (~189M)"
fi
RSYNC_FILTERS+=(
  --exclude "/$DEP_PAPER/outputs/*"
  --exclude "/papers/*/outputs/"
)
echo ">> Syncing repo -> pod:$REMOTE_DIR (package + Paper 6 outputs + needed Paper 5 directions; blobs/other papers excluded)"
rsync -az --delete "${RSYNC_FILTERS[@]}" \
  -e "ssh ${SSH_OPTS[*]}" \
  "$REPO_ROOT/" "root@$SSH_HOST:$REMOTE_DIR/"

# ---------------------------------- execute ----------------------------------
REMOTE_LOG="$REMOTE_DIR/session.log"
REMOTE_DONE="$REMOTE_DIR/.session_done"
echo ">> Monitor from another terminal:"
echo "     ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$SSH_HOST 'tail -f $REMOTE_LOG'"
echo ">> Launching experiments detached on pod (log: $REMOTE_LOG)"
ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" \
  "cd $REMOTE_DIR && rm -f '$REMOTE_DONE' '$REMOTE_LOG' && \
   ( PYTHONUNBUFFERED=1 REPO_DIR=$REMOTE_DIR \
     VALIDATE=$VALIDATE DEP_KIND='$DEP_KIND' PER_STATE=$PER_STATE \
     DATASET_TARGET=$DATASET_TARGET \
     ART_LAMBDA=$ART_LAMBDA ART_MAX_STEPS=$ART_MAX_STEPS \
     N_GENERAL=$N_GENERAL N_MORAL=$N_MORAL \
     RUN_DEPENDENCY=$RUN_DEPENDENCY RUN_ART_SFT=$RUN_ART_SFT RUN_EVAL=$RUN_EVAL \
     setsid bash papers/6_ablation_resistance/runpod/remote_experiments.sh > '$REMOTE_LOG' 2>&1 < /dev/null & ) >/dev/null 2>&1"

echo ">> Launched. Streaming log below (run survives SSH drops; Ctrl-C terminates pod)."
echo "----------------------------------------------------------------------"
SEEN=0
while true; do
  CHUNK="$(ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" "tail -n +$((SEEN + 1)) '$REMOTE_LOG' 2>/dev/null" 2>/dev/null || true)"
  if [ -n "$CHUNK" ]; then
    printf '%s\n' "$CHUNK"
    SEEN=$((SEEN + $(printf '%s\n' "$CHUNK" | wc -l)))
  fi
  if ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" "test -f '$REMOTE_DONE'" 2>/dev/null; then
    FINAL="$(ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" "tail -n +$((SEEN + 1)) '$REMOTE_LOG' 2>/dev/null" 2>/dev/null || true)"
    [ -n "$FINAL" ] && printf '%s\n' "$FINAL"
    break
  fi
  sleep 3
done
echo "----------------------------------------------------------------------"
echo ">> Experiments finished (sentinel detected)."

# --------------------------------- download ----------------------------------
echo ">> Downloading results (model blobs excluded)"
# Excludes the 14 GB weight blobs (only the small LoRA adapters + result JSON
# come back). IMPORTANT: exclude the model SHARDS by name (model-*.safetensors,
# model.safetensors) and the big-model DIRS, but NOT a blanket *.safetensors —
# the LoRA adapter weights are adapter_model.safetensors and MUST download (they
# are the durable artifact).
rsync -az \
  --exclude '*.pt' --exclude '*.pth' --exclude '*.ckpt' \
  --exclude 'model-*.safetensors' --exclude 'model.safetensors' \
  --exclude 'ablated_model/' --exclude 'merged_model/' --exclude '_merged/' \
  -e "ssh ${SSH_OPTS[*]}" \
  "root@$SSH_HOST:$REMOTE_DIR/papers/6_ablation_resistance/outputs/" \
  "$REPO_ROOT/papers/6_ablation_resistance/outputs/"

echo ">> Done. New Phase 3 outputs under papers/6_ablation_resistance/outputs/."
echo ">> Next (local): dependency trajectory figure (Sprint 5.3) once that script exists:"
echo "     python papers/6_ablation_resistance/scripts/dependency_figures.py \\"
echo "       --summary papers/6_ablation_resistance/outputs/dependency/dependency_summary.json"
# pod terminated by the EXIT trap
