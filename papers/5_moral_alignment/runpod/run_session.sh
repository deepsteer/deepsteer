#!/usr/bin/env bash
# Programmatic RunPod session for the Paper 5 (Phase 2) GPU work.
#
#   spin up GPU -> rsync repo -> run experiments -> rsync results back -> terminate
#
# The pod is ALWAYS terminated on exit (trap), including on error or Ctrl-C, so
# you never leak GPU time. Set KEEP_POD=1 to leave it running for debugging.
#
# Prerequisites (local): curl, jq, ssh, rsync; RUNPOD_API_KEY exported; an SSH
# keypair at $SSH_KEY (+ .pub) injected via PUBLIC_KEY so sshd authorizes you.
#
# Recommended flow (cost-minimizing):
#   1. Local pre-flight already done (base baselines + script self-test on MPS).
#   2. VALIDATE=1 ./run_session.sh        # cheap ~5-15 min smoke on the pod
#   3. inspect outputs/_validate_*, decide raw-vs-chat input format
#   4. ONLY="transfer behavioral persona" ./run_session.sh   # Sprint 1
#   5. ONLY="pipeline coupling" ./run_session.sh   # Sprint 2 (pipeline=raw; coupling/behavioral=chat)
#   (or one shot: ./run_session.sh)
#
# Usage:
#   export RUNPOD_API_KEY=...
#   VALIDATE=1 ./run_session.sh           # cheap validation run, then stop
#   ./run_session.sh                      # full session (Sprint 1 + 2)
#   ONLY="pipeline" ./run_session.sh      # just the named step(s)
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
POD_NAME="${POD_NAME:-deepsteer-p5-7b}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/deepsteer}"

# Phase 2 step config (passed through to the remote runner).
VALIDATE="${VALIDATE:-0}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-allenai/Olmo-3-7B-Instruct}"
INPUT_FORMAT="${INPUT_FORMAT:-raw}"   # pipeline probing format (Sprint 1 decided raw)
STABLE_LAYER="${STABLE_LAYER:-16}"
RUN_TRANSFER="${RUN_TRANSFER:-1}"
RUN_BEHAVIORAL="${RUN_BEHAVIORAL:-1}"
RUN_PERSONA="${RUN_PERSONA:-1}"
RUN_PIPELINE="${RUN_PIPELINE:-1}"
RUN_COUPLING="${RUN_COUPLING:-1}"
RUN_HERETIC="${RUN_HERETIC:-0}"   # Sprint 3 (off by default; ONLY=heretic to run)
RUN_MALLEABILITY="${RUN_MALLEABILITY:-0}"   # Tier 2 (off by default; ONLY=malleability to run)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

# Standardized sync: shared universal excludes + keep only THIS paper's outputs.
SELF_PAPER="papers/5_moral_alignment"
RSYNC_EXCLUDE="$REPO_ROOT/papers/runpod_common/rsync_exclude.txt"

# ONLY="pipeline coupling" runs just the named step(s); forces others off.
if [ -n "${ONLY:-}" ]; then
  RUN_TRANSFER=0; RUN_BEHAVIORAL=0; RUN_PERSONA=0; RUN_PIPELINE=0; RUN_COUPLING=0; RUN_HERETIC=0
  RUN_MALLEABILITY=0
  for _s in $ONLY; do
    case "$_s" in
      transfer) RUN_TRANSFER=1 ;;
      behavioral) RUN_BEHAVIORAL=1 ;;
      persona) RUN_PERSONA=1 ;;
      pipeline) RUN_PIPELINE=1 ;;
      coupling) RUN_COUPLING=1 ;;
      heretic) RUN_HERETIC=1 ;;
      malleability) RUN_MALLEABILITY=1 ;;
      *) echo "WARN: unknown ONLY step '$_s'" ;;
    esac
  done
  echo ">> ONLY='$ONLY' -> transfer=$RUN_TRANSFER behavioral=$RUN_BEHAVIORAL persona=$RUN_PERSONA pipeline=$RUN_PIPELINE coupling=$RUN_COUPLING heretic=$RUN_HERETIC malleability=$RUN_MALLEABILITY"
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

echo ">> Syncing repo -> pod:$REMOTE_DIR (this paper's outputs + package only; blobs/other papers excluded)"
# Filter order matters: shared universal excludes (blobs/caches) win first, then
# keep THIS paper's outputs, then drop every other paper's outputs.
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
echo ">> Launching experiments detached on pod (log: $REMOTE_LOG)"
ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" \
  "cd $REMOTE_DIR && rm -f '$REMOTE_DONE' '$REMOTE_LOG' && \
   ( PYTHONUNBUFFERED=1 REPO_DIR=$REMOTE_DIR \
     VALIDATE=$VALIDATE INSTRUCT_MODEL='$INSTRUCT_MODEL' \
     INPUT_FORMAT='$INPUT_FORMAT' STABLE_LAYER=$STABLE_LAYER \
     RUN_TRANSFER=$RUN_TRANSFER RUN_BEHAVIORAL=$RUN_BEHAVIORAL RUN_PERSONA=$RUN_PERSONA \
     RUN_PIPELINE=$RUN_PIPELINE RUN_COUPLING=$RUN_COUPLING RUN_HERETIC=$RUN_HERETIC \
     RUN_MALLEABILITY=$RUN_MALLEABILITY \
     setsid bash papers/5_moral_alignment/runpod/remote_experiments.sh > '$REMOTE_LOG' 2>&1 < /dev/null & ) >/dev/null 2>&1"

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
rsync -az \
  --exclude '*.pt' --exclude '*.pth' --exclude '*.ckpt' --exclude '*.safetensors' \
  --exclude 'ablated_model/' \
  -e "ssh ${SSH_OPTS[*]}" \
  "root@$SSH_HOST:$REMOTE_DIR/papers/5_moral_alignment/outputs/" \
  "$REPO_ROOT/papers/5_moral_alignment/outputs/"

echo ">> Done. New Phase 2 outputs under papers/5_moral_alignment/outputs/."
echo ">> Next (local): figures -> python papers/5_moral_alignment/scripts/pipeline_figures.py \\"
echo "     --pipeline-dir papers/5_moral_alignment/outputs/pipeline --layer $STABLE_LAYER"
# pod terminated by the EXIT trap
