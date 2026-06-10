#!/usr/bin/env bash
# Programmatic RunPod session for the Paper 3 7B GPU work.
#
#   spin up A100 -> rsync repo -> run experiments -> rsync results back -> terminate
#
# The pod is ALWAYS terminated on exit (trap), including on error or Ctrl-C, so
# you never leak GPU time. Set KEEP_POD=1 to leave it running for debugging.
#
# Prerequisites (local):
#   - curl, jq, ssh, rsync
#   - RUNPOD_API_KEY exported
#   - an SSH keypair at $SSH_KEY (+ $SSH_KEY.pub); the public key is injected
#     into the pod via the PUBLIC_KEY env var so sshd authorizes you
#
# Usage:
#   export RUNPOD_API_KEY=...
#   ./run_session.sh                  # full default session
#   RUN_DILEMMA=1 ./run_session.sh    # also run the stretch dilemma step
#   KEEP_POD=1 ./run_session.sh       # don't terminate at the end
#   REUSE_POD_ID=<id> ./run_session.sh  # attach to an existing pod (no create/terminate)
set -euo pipefail

# ---------------------------- config (override via env) ----------------------
: "${RUNPOD_API_KEY:?export RUNPOD_API_KEY first}"
# GPU candidates, tried in order until one deploys (first available wins).
# OLMo-2 7B is ~14 GB in fp16, so 24-48 GB cards are plenty; the 80 GB types are
# listed first only for headroom. Override with GPU_TYPE=... for a single type,
# or GPU_TYPES="a,b,c" for a custom ordered list.
GPU_TYPE="${GPU_TYPE:-}"
GPU_TYPES="${GPU_TYPES:-NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,NVIDIA RTX A6000,NVIDIA A40,NVIDIA GeForce RTX 4090,NVIDIA RTX A5000}"
CLOUD_TYPES="${CLOUD_TYPES:-SECURE,COMMUNITY}"
[ -n "$GPU_TYPE" ] && GPU_TYPES="$GPU_TYPE"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK_GB="${DISK_GB:-80}"
VOLUME_GB="${VOLUME_GB:-0}"
POD_NAME="${POD_NAME:-deepsteer-p3-7b}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/deepsteer}"
MODEL="${MODEL:-allenai/OLMo-2-1124-7B}"
N_BOOTSTRAP="${N_BOOTSTRAP:-200}"
DIRECTIONS_NPZ="${DIRECTIONS_NPZ:-papers/3_moral_geometry/outputs/exp1_2_3_7B/exp1_probe_directions.npz}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

# Step toggles passed through to the remote runner.
RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-1}"
RUN_FRAGILITY="${RUN_FRAGILITY:-1}"
RUN_CAUSAL="${RUN_CAUSAL:-1}"
RUN_DILEMMA="${RUN_DILEMMA:-0}"
RUN_TAXONOMY="${RUN_TAXONOMY:-1}"
RUN_EXTERNAL="${RUN_EXTERNAL:-1}"

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
    echo "       Last message: ${LAST_ERR:-none}"
    echo "       Try later, widen GPU_TYPES, or set CLOUD_TYPES=COMMUNITY."
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

SSH_OPTS=(-p "$SSH_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)
echo ">> Waiting for sshd..."
for _ in $(seq 1 30); do
  ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" 'echo ok' 2>/dev/null | grep -q ok && break
  sleep 5
done

# ---------------------------------- sync up ----------------------------------
# The base image may lack rsync; rsync needs it on BOTH ends. Install if missing.
echo ">> Preparing pod (mkdir + ensure rsync)..."
ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" \
  "mkdir -p $REMOTE_DIR && (command -v rsync >/dev/null 2>&1 || \
   { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync; } || \
   { command -v apk >/dev/null 2>&1 && apk add --no-cache rsync; })"

echo ">> Syncing repo -> pod:$REMOTE_DIR"
# Exclude large model/cache blobs (SAE caches, checkpoints) — multi-GB and not
# needed by any RunPod step. The needed inputs are .json/.npz/.py/datasets.
rsync -az --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '.venv' --exclude 'venv' --exclude '*.egg-info' \
  --exclude 'build/' --exclude '.DS_Store' \
  --exclude '*.pt' --exclude '*.pth' --exclude '*.ckpt' --exclude '*.safetensors' \
  -e "ssh ${SSH_OPTS[*]}" \
  "$REPO_ROOT/" "root@$SSH_HOST:$REMOTE_DIR/"

# ---------------------------------- execute ----------------------------------
# Launch the experiment plan DETACHED (nohup) so a dropped SSH connection can't
# SIGHUP it. We then poll the log file + a completion sentinel; transient ssh
# failures just retry, since the run no longer depends on this channel.
REMOTE_LOG="$REMOTE_DIR/session.log"
REMOTE_DONE="$REMOTE_DIR/.session_done"
echo ">> Launching experiments detached on pod (log: $REMOTE_LOG)"
RUN_PID="$(ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" \
  "cd $REMOTE_DIR && rm -f '$REMOTE_DONE' '$REMOTE_LOG' && \
   PYTHONUNBUFFERED=1 \
   REPO_DIR=$REMOTE_DIR MODEL='$MODEL' N_BOOTSTRAP=$N_BOOTSTRAP DIRECTIONS_NPZ='$DIRECTIONS_NPZ' \
   RUN_BOOTSTRAP=$RUN_BOOTSTRAP RUN_FRAGILITY=$RUN_FRAGILITY RUN_CAUSAL=$RUN_CAUSAL \
   RUN_DILEMMA=$RUN_DILEMMA RUN_TAXONOMY=$RUN_TAXONOMY RUN_EXTERNAL=$RUN_EXTERNAL \
   nohup bash papers/3_moral_geometry/runpod/remote_experiments.sh > '$REMOTE_LOG' 2>&1 </dev/null & echo \$!")"
echo ">> Remote PID: ${RUN_PID:-unknown}. Streaming (run survives SSH drops; Ctrl-C terminates pod)."

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
  sleep 10
done
echo ">> Experiments finished (sentinel detected)."

# --------------------------------- download ----------------------------------
echo ">> Downloading results"
rsync -az \
  --exclude '*.pt' --exclude '*.pth' --exclude '*.ckpt' --exclude '*.safetensors' \
  -e "ssh ${SSH_OPTS[*]}" \
  "root@$SSH_HOST:$REMOTE_DIR/papers/3_moral_geometry/outputs/" \
  "$REPO_ROOT/papers/3_moral_geometry/outputs/"

echo ">> Done. New 7B outputs are under papers/3_moral_geometry/outputs/."
echo ">> Next (local): regenerate scale-comparison figures with real 7B data:"
echo "     python papers/3_moral_geometry/scripts/scale_comparison_figures.py"
# pod terminated by the EXIT trap
