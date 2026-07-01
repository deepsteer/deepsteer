#!/usr/bin/env bash
# Shared RunPod session lifecycle for the paper GPU runners.
#
# Each papers/<paper>/runpod/run_session.sh sets its own config block (GPU pool,
# disk, pod name, per-phase env passthrough) and then sources this file. The
# create -> wait-for-ssh -> rsync-up -> teardown sequence is identical across
# papers and lives here once; only the config and the per-paper execute+download
# blocks differ. Extracted from the four byte-identical copies (Papers 5/6/7 +
# Direction 1); no command changed in the move.
#
# Contract. The caller MUST set these before calling the functions below:
#   RUNPOD_API_KEY API REPO_ROOT SELF_PAPER RSYNC_EXCLUDE
#   POD_NAME IMAGE DISK_GB VOLUME_GB CLOUD_TYPES GPU_TYPES SSH_KEY REMOTE_DIR
# Optional (honored if set): REUSE_POD_ID KEEP_POD
#
# These globals are set FOR the caller (used by the execute/download blocks):
#   POD_ID  SSH_HOST  SSH_PORT  SSH_OPTS(array)  GPU_TYPE  CLOUD_TYPE
#
# Typical caller order (after the config block):
#   source "$REPO_ROOT/papers/runpod_common/session_lib.sh"
#   rp_require_bins
#   trap cleanup EXIT           # arm teardown before any pod exists
#   rp_provision_pod
#   rp_wait_for_ssh
#   rp_sync_up
#   ... paper-specific execute + download ...

rp_require_bins() {
  for bin in curl jq ssh rsync; do
    command -v "$bin" >/dev/null || { echo "ERROR: '$bin' not found in PATH"; exit 1; }
  done
}

gql() { curl -s "$API" -H 'Content-Type: application/json' -d "$(jq -n --arg q "$1" '{query:$q}')"; }

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

# ---------------------------------- create -----------------------------------
rp_provision_pod() {
  POD_ID="${REUSE_POD_ID:-}"
  [ -n "$POD_ID" ] && return 0
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
}

# ----------------------------- wait for SSH ----------------------------------
rp_wait_for_ssh() {
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

  # ServerAlive* so a HUNG established connection self-aborts (~60s) instead of
  # freezing the stream loop forever -- a stalled `ssh tail` must not block the
  # .session_done check and leak the (still-billed) pod past completion.
  SSH_OPTS=(-p "$SSH_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
  echo ">> Waiting for sshd..."
  for _ in $(seq 1 30); do
    ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" 'echo ok' 2>/dev/null | grep -q ok && break
    sleep 5
  done
}

# ---------------------------------- sync up ----------------------------------
rp_sync_up() {
  echo ">> Preparing pod (mkdir + ensure rsync)..."
  ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" \
    "mkdir -p $REMOTE_DIR && (command -v rsync >/dev/null 2>&1 || \
     { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync; } || \
     { command -v apk >/dev/null 2>&1 && apk add --no-cache rsync; })"

  # Filter order matters: shared universal excludes (blobs/caches) win first, then
  # keep THIS paper's outputs, then drop every other paper's outputs.
  echo ">> Syncing repo -> pod:$REMOTE_DIR (this paper's outputs + package only; blobs/other papers excluded)"
  rsync -az --delete \
    --exclude-from "$RSYNC_EXCLUDE" \
    --include "/$SELF_PAPER/outputs/***" \
    --exclude '/papers/*/outputs/' \
    -e "ssh ${SSH_OPTS[*]}" \
    "$REPO_ROOT/" "root@$SSH_HOST:$REMOTE_DIR/"
}
