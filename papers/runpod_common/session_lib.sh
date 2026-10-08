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

# --------------------------------- download ----------------------------------
# rp_download <remote_path/> <local_path/>: rsync results back with retries (--partial resumes an
# interrupted pass). If every attempt fails, KEEP_POD=1 is set so the EXIT trap leaves the pod
# running instead of destroying results that never arrived (2026-09-27: a single failed pass
# lost a whole step's outputs to the terminate trap).
rp_free_gb() {  # free GB on the filesystem holding $1
  df -Pk "$1" | awk 'NR==2 {print int($4 / 1048576)}'
}

# rp_require_disk <dir> [min_gb]: refuse to provision when the local results filesystem is short.
# (2026-09-27: two downloads failed with the Mac's disk at 100%; retries cannot fix a full disk.)
rp_require_disk() {
  local dir="$1" min="${2:-${MIN_FREE_GB:-50}}" free
  mkdir -p "$dir"
  free="$(rp_free_gb "$dir")"
  if [ "${free:-0}" -lt "$min" ]; then
    echo "ERROR: only ${free} GB free under $dir (need >= ${min} GB for results). Free space first."
    exit 1
  fi
  echo ">> local disk: ${free} GB free under $dir (min ${min})"
}

rp_download() {
  local src="$1" dst="$2" tries="${DOWNLOAD_TRIES:-5}" k rc=1 free
  mkdir -p "$dst"
  for ((k = 1; k <= tries; k++)); do
    free="$(rp_free_gb "$dst")"
    if [ "${free:-0}" -lt 5 ]; then
      echo "ERROR: local disk has ${free} GB free; not retrying into a full disk."
      break
    fi
    rsync -az --partial \
      --exclude '*.pt' --exclude '*.pth' --exclude '*.ckpt' --exclude '*.safetensors' \
      -e "ssh ${SSH_OPTS[*]}" "root@$SSH_HOST:$src" "$dst"
    rc=$?
    [ $rc -eq 0 ] && { echo ">> download complete (attempt $k)"; return 0; }
    echo "WARN: download attempt $k/$tries failed (rsync exit $rc); retrying in 20s"
    sleep 20
  done
  echo "ERROR: download failed $tries times; KEEPING the pod so nothing is lost."
  echo "  Re-run the rsync by hand, then terminate the pod with the command printed below."
  KEEP_POD=1
  return $rc
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

  # Filter order matters: explicitly named output FILES (SYNC_OUTPUTS) first, then the shared
  # universal excludes (blobs/caches, and trees no pod needs such as the KDG outputs), then THIS
  # paper's outputs, then drop every other paper's outputs.
  rp_output_filters
  rp_check_sync_size || exit 1
  echo ">> Syncing repo -> pod:$REMOTE_DIR (${SYNC_OUTPUTS:+outputs: $SYNC_OUTPUTS; }blobs/other papers excluded)"
  rsync -az --delete \
    ${RP_PRE_FILTERS[@]+"${RP_PRE_FILTERS[@]}"} \
    --exclude-from "$RSYNC_EXCLUDE" \
    "${RP_OUT_FILTERS[@]}" \
    -e "ssh ${SSH_OPTS[*]}" \
    "$REPO_ROOT/" "root@$SSH_HOST:$REMOTE_DIR/"
}

# rp_output_filters: sets RP_PRE_FILTERS and RP_OUT_FILTERS (global arrays, not namerefs: the
# launcher runs under macOS's bash 3.2).
#   SYNC_OUTPUTS unset   the whole $SELF_PAPER/outputs tree, still subject to rsync_exclude.txt
#                        (historical default).
#   SYNC_OUTPUTS=none    no outputs.
#   SYNC_OUTPUTS=a,b/    only these paths, relative to $SELF_PAPER. A FILE entry is placed before
#                        rsync_exclude.txt, so one named file can cross a universal exclude (the
#                        p2c VALIDATE timing record lives under the excluded KDG outputs); a
#                        DIRECTORY entry (trailing /) stays behind it, so no blob or excluded tree
#                        ever ships by directory. Ancestors are included so rsync can descend.
rp_output_filters() {
  RP_PRE_FILTERS=()
  RP_OUT_FILTERS=()
  if [ -z "${SYNC_OUTPUTS:-}" ]; then
    RP_OUT_FILTERS=(--include "/$SELF_PAPER/outputs/***")
  elif [ "$SYNC_OUTPUTS" != "none" ]; then
    local item anc part n
    IFS=',' read -ra _items <<< "$SYNC_OUTPUTS"
    for item in "${_items[@]}"; do
      anc="/$SELF_PAPER"
      IFS='/' read -ra _parts <<< "${item%/}"
      n=$(( ${#_parts[@]} - 1 ))
      for part in "${_parts[@]:0:$n}"; do
        anc="$anc/$part"
        if [ "${item: -1}" = "/" ]; then RP_OUT_FILTERS+=(--include "$anc/")
        else RP_PRE_FILTERS+=(--include "$anc/"); fi
      done
      if [ "${item: -1}" = "/" ]; then RP_OUT_FILTERS+=(--include "/$SELF_PAPER/${item}***")
      else RP_PRE_FILTERS+=(--include "/$SELF_PAPER/$item"); fi
    done
    RP_OUT_FILTERS+=(--exclude "/$SELF_PAPER/outputs/**")
  fi
  RP_OUT_FILTERS+=(--exclude '/papers/*/outputs/')
}

# rp_check_sync_size: with MAX_SYNC_GB set, list what the sync would ship (rsync -n with the same
# filters; excluded trees are pruned, not walked), sum the file sizes locally, and refuse before
# any upload if the total is larger. Sizes come from stat, not rsync's --stats text, because the
# macOS openrsync and GNU rsync print different stats.
rp_check_sync_size() {
  [ -n "${MAX_SYNC_GB:-}" ] || return 0
  local tmp list bytes
  tmp="$(mktemp -d)"
  # fail closed: a listing error must never read as a small sync
  list="$(cd "$REPO_ROOT" && rsync -an --out-format='%n' ${RP_PRE_FILTERS[@]+"${RP_PRE_FILTERS[@]}"} \
      --exclude-from "$RSYNC_EXCLUDE" "${RP_OUT_FILTERS[@]}" ./ "$tmp/")" || {
    rm -rf "$tmp"; echo "ERROR: could not list the sync (rsync -n failed); refusing."; return 1; }
  rm -rf "$tmp"
  bytes="$(cd "$REPO_ROOT" && printf '%s\n' "$list" | while IFS= read -r f; do
        [ -f "$f" ] && { stat -c %s "$f" 2>/dev/null || stat -f %z "$f"; }  # GNU, then BSD
      done | awk '{s += $1} END {printf "%.0f", s}')"
  [ "${bytes:-0}" -gt 0 ] || { echo "ERROR: sync listing is empty; refusing."; return 1; }
  echo ">> Sync size: $(awk -v b="${bytes:-0}" 'BEGIN {printf "%.2f", b / 1e9}') GB (limit $MAX_SYNC_GB GB)"
  awk -v b="${bytes:-0}" -v m="$MAX_SYNC_GB" 'BEGIN {exit !(b <= m * 1e9)}' || {
    echo "ERROR: sync would upload more than MAX_SYNC_GB=$MAX_SYNC_GB GB; narrow SYNC_OUTPUTS."
    return 1
  }
}
