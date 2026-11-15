#!/usr/bin/env bash
# Thunder Compute for run_session.sh (CST Amendment 51 item D; the author's conditional approval,
# Amendment 52 addendum). Sourced after session_lib.sh when PROVIDER=thunder; it replaces only the
# RunPod-specific pieces SECOND_PROVIDER_NOTE.md lists (the API client, create, the region field,
# the wait for SSH, the rate, teardown). Everything else (the sync filters, rp_sync_up, rp_download,
# rp_push_token, rp_on_demand_sync, the launch and the stream loop) is shared. A third provider is
# another provider_<name>.sh with the same functions.
#
# The account's login is the author's `tnr` CLI on this machine (`tnr login`); no key reaches the
# instance, and the HF token goes by the mode-600 file as on RunPod. Settings (env):
#   TNR_GPU (a100xl)  TNR_VCPUS (8)  DISK_GB (300)  TNR_TEMPLATE (base)
#   TNR_CREATE_TRIES (6) and TNR_CREATE_WAIT_S (300): retries when Thunder reports no capacity
#   TNR_PRICE (1.15): the hourly price logged when the CLI does not print its estimate
#
# Teardown guard (Amendment 51 item D3): delete only the instance this session created (its UUID is
# checked first), poll until the account no longer lists it, then list every instance and fail
# loudly if one this session did not find at start remains.

SSH_USER=ubuntu
REMOTE_TOKEN_FILE=/home/ubuntu/.hf_token
TNR_GPU="${TNR_GPU:-a100xl}"
TNR_VCPUS="${TNR_VCPUS:-8}"
TNR_TEMPLATE="${TNR_TEMPLATE:-base}"
TNR_CREATE_TRIES="${TNR_CREATE_TRIES:-6}"
TNR_CREATE_WAIT_S="${TNR_CREATE_WAIT_S:-300}"
POD_UUID=""
TNR_PRE_UUIDS=""
TNR_EST=""
# the venv on the instance is built from Python 3.11 (its base image has 3.12; th_bringup installs
# 3.11 with its headers from the deadsnakes PPA)
export CST_PYTHON="${CST_PYTHON:-python3.11}"

rp_require_bins() {
  for bin in jq ssh rsync tnr; do
    command -v "$bin" >/dev/null || { echo "ERROR: '$bin' not found in PATH"; exit 1; }
  done
  tnr status --no-wait --json >/dev/null 2>&1 || { echo "ERROR: 'tnr status' failed; run 'tnr login' first"; exit 1; }
}

th_list() { tnr status --no-wait --json 2>/dev/null; }

rp_datacenter_field() { DC_FIELD=""; }  # Thunder places instances itself (North America)

rp_provision_pod() {
  POD_ID="${REUSE_POD_ID:-}"
  TNR_PRE_UUIDS="$(th_list | jq -r '.[].uuid' 2>/dev/null | sort | tr '\n' ' ')"
  [ -n "$TNR_PRE_UUIDS" ] && echo ">> The account already lists instance(s): $TNR_PRE_UUIDS(left alone)"
  if [ -n "$POD_ID" ]; then
    POD_UUID="$(th_list | jq -r --arg id "$POD_ID" '.[] | select((.id|tostring)==$id) | .uuid')"
    return 0
  fi
  GPU_TYPE="$TNR_GPU x1"; CLOUD_TYPE="thunder"; RP_ATTEMPTS=()
  local k out err
  for ((k = 1; k <= TNR_CREATE_TRIES; k++)); do
    echo ">> tnr create (attempt $k/$TNR_CREATE_TRIES): $TNR_GPU x1, $TNR_VCPUS vCPUs, $DISK_GB GB, template $TNR_TEMPLATE"
    err="$(mktemp)"
    out="$(tnr create --gpu "$TNR_GPU" --num-gpus 1 --vcpus "$TNR_VCPUS" --template "$TNR_TEMPLATE" \
      --disk "$DISK_GB" --json 2>"$err")"
    TNR_EST="$(cat "$err" <(printf '%s' "$out") | grep -oE 'Estimated cost: \$[0-9.]+' | grep -oE '[0-9.]+$' | head -1)"
    POD_ID="$(printf '%s' "$out" | jq -r '.identifier // empty' 2>/dev/null)"
    POD_UUID="$(printf '%s' "$out" | jq -r '.uuid // empty' 2>/dev/null)"
    if [ -n "$POD_ID" ] && [ -n "$POD_UUID" ]; then
      RP_ATTEMPTS+=("thunder|$TNR_GPU|created")
      echo ">> Instance $POD_ID created (UUID $POD_UUID)"
      rm -f "$err"; return 0
    fi
    LAST_ERR="$(tr '\n' ' ' < "$err")"; rm -f "$err"
    echo "      -> ${LAST_ERR:-no instance returned}"
    RP_ATTEMPTS+=("thunder|$TNR_GPU|${LAST_ERR:-no instance returned}")
    [ "$k" -lt "$TNR_CREATE_TRIES" ] && { echo "      retrying in ${TNR_CREATE_WAIT_S}s"; sleep "$TNR_CREATE_WAIT_S"; }
  done
  POD_ID=""
  echo "ERROR: no Thunder instance after $TNR_CREATE_TRIES attempts. Last message: ${LAST_ERR:-none}."
  exit 1
}

# th_endpoint: the instance's current address, port and key from `tnr connect --json` (which also
# sets up the key), into SSH_HOST, SSH_PORT and SSH_OPTS
th_endpoint() {
  local j
  j="$(tnr connect "$POD_ID" --json 2>/dev/null)" || return 1
  SSH_HOST="$(printf '%s' "$j" | jq -r '.ip // empty')"
  SSH_PORT="$(printf '%s' "$j" | jq -r '.port // empty')"
  TNR_KEY="$(printf '%s' "$j" | jq -r '.key_file // empty')"
  [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] && [ -n "$TNR_KEY" ] || return 1
  SSH_OPTS=(-p "$SSH_PORT" -i "$TNR_KEY" -o IdentitiesOnly=yes -o IdentityAgent=none \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
}

rp_refresh_endpoint() {  # Thunder's address is dynamic: re-resolve it (the stream loop calls this)
  local old="$SSH_HOST:$SSH_PORT"
  th_endpoint && [ "$SSH_HOST:$SSH_PORT" != "$old" ] && echo ">> the instance's endpoint moved: $old -> $SSH_HOST:$SSH_PORT"
}

rp_wait_for_ssh() {
  echo ">> Waiting for instance $POD_ID to be RUNNING..."
  local s
  for _ in $(seq 1 90); do
    s="$(th_list | jq -r --arg u "$POD_UUID" '.[] | select(.uuid==$u) | .status' 2>/dev/null)"
    [ "$s" = "RUNNING" ] && break
    sleep 10
  done
  [ "$s" = "RUNNING" ] || { echo "ERROR: instance $POD_ID never reached RUNNING (last: ${s:-not listed})"; exit 1; }
  th_endpoint || { echo "ERROR: 'tnr connect $POD_ID --json' gave no endpoint"; exit 1; }
  echo ">> SSH endpoint: $SSH_USER@$SSH_HOST:$SSH_PORT"
  echo ">> Log in from another terminal:"
  echo "     tnr connect $POD_ID      (or: ssh ${SSH_OPTS[*]} $SSH_USER@$SSH_HOST)"
  echo ">> Waiting for sshd..."
  for _ in $(seq 1 30); do
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$SSH_HOST" 'echo ok' 2>/dev/null | grep -q ok && break
    sleep 5
  done
  th_bringup
}

# th_bringup: what RunPod's image supplies and Thunder's base template does not: a writable
# /workspace, rsync, Python 3.11 with its headers (deadsnakes, its key checked by fingerprint; the
# image's python3 is 3.12 without Ubuntu's apt_pkg, so add-apt-repository cannot run). No CUDA
# package is touched.
th_bringup() {
  echo ">> Bring-up: /workspace, rsync, Python 3.11 with its headers"
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$SSH_HOST" 'set -e
    [ -e /workspace ] || { mkdir -p "$HOME/workspace"; sudo ln -s "$HOME/workspace" /workspace; }
    [ -w /workspace ] || sudo chown "$(id -u):$(id -g)" /workspace
    command -v rsync >/dev/null || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q rsync >/dev/null
    if ! { command -v python3.11 >/dev/null && [ -f /usr/include/python3.11/Python.h ]; }; then
      FP=F23C5A6CF475977595C89F51BA6932366A755776
      sudo mkdir -p /etc/apt/keyrings
      curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x$FP" | sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/deadsnakes.gpg
      gpg --show-keys --with-colons /etc/apt/keyrings/deadsnakes.gpg 2>/dev/null | grep -q "fpr:::::::::$FP:" || { echo "STOP: the deadsnakes key fingerprint does not match"; exit 1; }
      echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/deadsnakes.list >/dev/null
      sudo apt-get update -q >/dev/null 2>&1 || true
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3.11 python3.11-venv python3.11-dev >/dev/null
    fi
    test -f /usr/include/python3.11/Python.h
    echo ">> bring-up done: $(python3.11 --version), /workspace -> $(readlink -f /workspace)"' \
    || { echo "ERROR: the bring-up failed"; exit 1; }
}

rp_log_rate() {
  local cost="${TNR_EST:-${TNR_PRICE:-1.15}}" src f att
  src="$([ -n "$TNR_EST" ] && echo "the CLI's estimate" || echo "TNR_PRICE (the CLI printed no estimate)")"
  echo ">> Instance rate: $cost USD/h ($src; $TNR_GPU x1, $TNR_VCPUS vCPUs, $DISK_GB GB; instance $POD_ID, UUID $POD_UUID)"
  f="$REPO_ROOT/$SELF_PAPER/${RESULTS_SUBPATH:-outputs}/logs/pod_rate.jsonl"
  mkdir -p "$(dirname "$f")"
  att="$(printf '%s\n' "${RP_ATTEMPTS[@]+"${RP_ATTEMPTS[@]}"}" | jq -R -s -c 'split("\n") | map(select(length > 0)) | map(split("|") | {cloud: .[0], gpu: .[1], result: (.[2:] | join("|"))})')"
  jq -cn --arg pod "$POD_ID" --arg uuid "$POD_UUID" --arg gpu "$TNR_GPU" --arg cost "$cost" \
    --arg src "$src" --arg t "$(date -u +%FT%TZ)" --argjson att "${att:-[]}" \
    '{provider:"thunder", pod:$pod, uuid:$uuid, gpu:$gpu, cost_per_hr:($cost|tonumber? // null), cost_source:$src, at:$t, attempts:$att}' >> "$f"
}

cleanup() {
  [ -n "${POD_ID:-}" ] || return 0
  [ -n "${REUSE_POD_ID:-}" ] && { echo "Attached to existing instance $POD_ID; not deleting."; return 0; }
  if [ "${KEEP_POD:-0}" = 1 ]; then
    echo "KEEP_POD=1 -> instance $POD_ID (UUID $POD_UUID) left RUNNING. Delete later with: tnr delete $POD_ID"
    return 0
  fi
  local cur
  cur="$(th_list | jq -r --arg id "$POD_ID" '.[] | select((.id|tostring)==$id) | .uuid' 2>/dev/null)"
  if [ "$cur" != "$POD_UUID" ]; then
    echo "ERROR: instance $POD_ID is ${cur:-not listed}, not this session's $POD_UUID; NOT deleting. Check 'tnr status'!"
  else
    echo ">> Deleting instance $POD_ID (UUID $POD_UUID)"
    tnr delete "$POD_ID" --yes >/dev/null 2>&1 || echo "WARN: 'tnr delete' returned an error; checking the account"
  fi
  local gone=0
  for _ in $(seq 1 40); do
    th_list | jq -e --arg u "$POD_UUID" 'any(.[]; .uuid==$u)' >/dev/null 2>&1 || { gone=1; break; }
    sleep 15
  done
  local left u extra=""
  left="$(th_list | jq -r '.[].uuid' 2>/dev/null)"
  for u in $left; do
    case " $TNR_PRE_UUIDS " in *" $u "*) ;; *) extra="$extra $u";; esac
  done
  if [ "$gone" = 1 ] && [ -z "$extra" ]; then
    echo ">> Teardown verified: instance $POD_UUID deleted; the account lists ${left:+only the instances found at start: $TNR_PRE_UUIDS}${left:-no instances}."
  else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "ERROR: TEARDOWN NOT VERIFIED. Still listed:${extra:- $POD_UUID}. It bills until deleted:"
    echo "       tnr status; tnr delete <id>"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    return 1
  fi
}
