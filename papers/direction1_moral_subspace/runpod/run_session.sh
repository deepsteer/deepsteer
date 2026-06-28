#!/usr/bin/env bash
# Programmatic RunPod session for Direction 1 Phase 2 (single-source V_moral GPU run).
#
#   spin up GPU -> rsync repo -> run the Phase 2 sequence (base + instruct chains, G2,
#   Track-1, two same-model-point G3) -> rsync results back -> terminate.
#
# The pod is ALWAYS terminated on exit (trap), including on error or Ctrl-C. Set KEEP_POD=1
# to leave it running for debugging.
#
# Memory peak is ONE 7B model at a time (extract loads base, then instruct; G3 loads base
# then instruct; each model.release()d before the next), so a single 48 GB card is plenty;
# 80 GB is fine too. Disk holds OLMo-3-7B (~14 GB) + OLMo-3-7B-Instruct (~14 GB) [+ the tiny
# OLMo-2-1B for the smoke] + HF cache headroom.
#
# Recommended flow (cost-minimizing; the no-GPU gates already pass locally):
#   1. Local gate (done):  python papers/direction1_moral_subspace/scripts/phase2_local_test.py
#   2. Local dry run (done): VALIDATE=1 bash papers/direction1_moral_subspace/runpod/phase2_session.sh
#   3. Cheap GPU smoke:     VALIDATE=1 ./run_session.sh          (tiny model on the pod)
#   4. Real run:            ./run_session.sh                     (OLMo-3-7B Base + Instruct)
#
# Usage:
#   export RUNPOD_API_KEY=...
#   export HF_TOKEN=...                          # OLMo-3 is public; pass for rate limits / gating
#   VALIDATE=1 ./run_session.sh                  # cheap GPU plumbing smoke, then stop
#   ./run_session.sh                             # full Phase 2 (real models)
#   KEEP_POD=1 ./run_session.sh                  # don't terminate at the end
#   REUSE_POD_ID=<id> ./run_session.sh           # attach to an existing pod
set -euo pipefail

# ---------------------------- config (override via env) ----------------------
: "${RUNPOD_API_KEY:?export RUNPOD_API_KEY first}"
GPU_TYPE="${GPU_TYPE:-}"
# 48 GB-class first (one 7B at a time fits in ~20 GB); 80 GB as fallback.
GPU_TYPES="${GPU_TYPES:-NVIDIA RTX A6000,NVIDIA A40,NVIDIA L40S,NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA H100 80GB HBM3}"
CLOUD_TYPES="${CLOUD_TYPES:-SECURE,COMMUNITY}"
[ -n "$GPU_TYPE" ] && GPU_TYPES="$GPU_TYPE"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK_GB="${DISK_GB:-150}"      # two 7B (~28 GB) + tiny smoke model + HF cache headroom
VOLUME_GB="${VOLUME_GB:-0}"
POD_NAME="${POD_NAME:-deepsteer-d1-phase2}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/deepsteer}"

# Step config (passed through to the remote runner).
VALIDATE="${VALIDATE:-0}"
BASE_MODEL="${BASE_MODEL:-allenai/OLMo-3-7B}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-allenai/OLMo-3-7B-Instruct}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"  # pinned (smoke-validated); override if needed
PIP_EXTRA="${PIP_EXTRA:-}"     # e.g. "transformers==X" extra override
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-papers/direction1_moral_subspace/runpod/remote_phase2.sh}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

SELF_PAPER="papers/direction1_moral_subspace"
RSYNC_EXCLUDE="$REPO_ROOT/papers/runpod_common/rsync_exclude.txt"

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

SSH_OPTS=(-p "$SSH_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
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

echo ">> Syncing repo -> pod:$REMOTE_DIR (committed code + dataset; blobs/other papers excluded)"
rsync -az --delete \
  --exclude-from "$RSYNC_EXCLUDE" \
  --include "/$SELF_PAPER/outputs/***" \
  --exclude '/papers/*/outputs/' \
  -e "ssh ${SSH_OPTS[*]}" \
  "$REPO_ROOT/" "root@$SSH_HOST:$REMOTE_DIR/"

# ---------------------------------- execute ----------------------------------
REMOTE_LOG="$REMOTE_DIR/session.log"
REMOTE_DONE="$REMOTE_DIR/.session_done"
echo ">> Monitor from another terminal:"
echo "     ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$SSH_HOST 'tail -f $REMOTE_LOG'"
echo ">> Launching Phase 2 [VALIDATE=$VALIDATE] detached on pod (log: $REMOTE_LOG)"
ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" \
  "cd $REMOTE_DIR && rm -f '$REMOTE_DONE' '$REMOTE_LOG' && \
   ( PYTHONUNBUFFERED=1 REPO_DIR=$REMOTE_DIR \
     VALIDATE=$VALIDATE BASE_MODEL='$BASE_MODEL' INSTRUCT_MODEL='$INSTRUCT_MODEL' \
     TRANSFORMERS_VERSION='$TRANSFORMERS_VERSION' PIP_EXTRA='$PIP_EXTRA' \
     HF_TOKEN='$HF_TOKEN' HUGGING_FACE_HUB_TOKEN='$HF_TOKEN' \
     setsid bash $REMOTE_SCRIPT > '$REMOTE_LOG' 2>&1 < /dev/null & ) >/dev/null 2>&1"

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
echo ">> Phase 2 finished (sentinel detected)."

# --------------------------------- download ----------------------------------
echo ">> Downloading results (phase2 saves no model weights; npz/json only)"
rsync -az \
  --exclude '*.pt' --exclude '*.pth' --exclude '*.ckpt' --exclude '*.safetensors' \
  -e "ssh ${SSH_OPTS[*]}" \
  "root@$SSH_HOST:$REMOTE_DIR/$SELF_PAPER/outputs/phase2/" \
  "$REPO_ROOT/$SELF_PAPER/outputs/phase2/"

echo ">> Done. Phase 2 results under $SELF_PAPER/outputs/phase2/:"
echo ">>   base/g2_result.json       (G2 contamination PASS/STOP, narrative slice)"
echo ">>   base/track1_result.json   (σ* V_moral vs MFT, RMS-normalized; eff-dim contrast)"
echo ">>   g3_result.json            (Point A base-proto + Point B instruct-gate vs 0.1044)"
echo ">>   {base,instruct}/{g_axis_decision,v_moral.npz,null_artifact}.json"
# pod terminated by the EXIT trap
