#!/usr/bin/env bash
# Stop a running GPC-4 session from the laptop (GPC_PLAN.md §11.17 item 3). The author runs, from the
# repo root:
#
#   bash papers/gpc_goal_pressure_conduct/runpod/stop_gpc4.sh             # the current session
#   bash papers/gpc_goal_pressure_conduct/runpod/stop_gpc4.sh --no-pull   # without pulling the results
#
# In order:
#   1. the pod's results pulled, as the launcher's download does (the trained adapters come back, other
#      weights stay); a failed pull stops here with nothing removed, and says how to go on;
#   2. the session's local pod launcher stopped (the processes holding its pod_launcher.log open, so no
#      other session's launcher is touched); its own cleanup deletes the pod under its UUID check;
#   3. the pod deleted here if Thunder still lists it, only under the UUID its launcher logged, the id
#      checked against that UUID just before the delete; then polled until the account no longer lists it;
#   4. the host's teardown (launch_gpc4.sh teardown: every gpc4-session instance, group and key).
# It prints what it removed and what is left. It is not on the VM side (VM_FILES in launch_gpc4.sh): it
# changes nothing the host runs. AWS credentials as for the launcher (the AWS CLI's own profile or
# environment, never on a command line); Thunder's through the author's `tnr` login.
set -uo pipefail
cd "$(dirname "$0")/../../.."
G=papers/gpc_goal_pressure_conduct
ROOT=${GPC4_ROOT:-~/.gpc4}
TEARDOWN=${GPC4_TEARDOWN:-bash $G/runpod/launch_gpc4.sh teardown}   # a test swaps it
LOCAL=${GPC4_STOP_LOCAL:-$G/outputs/gpc4_v1}
REMOTE=/workspace/deepsteer/$G/outputs/gpc4_v1
WAIT_S=${GPC4_STOP_WAIT_S:-720}   # the launcher's own cleanup polls up to 10 min
PULL=1
[ "${1:-}" = --no-pull ] && PULL=0
say() { echo ">> $*"; }
die() { echo "BAIL: $*" >&2; exit 1; }
th_list() { tnr status --no-wait --json 2>/dev/null || echo '[]'; }
th_id_of() { th_list | jq -r --arg u "$1" '[.[] | select(.uuid==$u) | (.id|tostring)] | first // empty'; }
th_uuid_of() { th_list | jq -r --arg i "$1" '[.[] | select((.id|tostring)==$i) | .uuid] | first // empty'; }
aws_left() {  # every gpc4-session resource, one "kind id" a line
  aws ec2 describe-instances --filters "Name=tag-key,Values=gpc4-session" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down" \
    --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null | tr '\t' '\n' | sed '/^$/d; s/^/instance /'
  aws ec2 describe-security-groups --filters "Name=tag-key,Values=gpc4-session" \
    --query 'SecurityGroups[].GroupId' --output text 2>/dev/null | tr '\t' '\n' | sed '/^$/d; s/^/security group /'
  aws ec2 describe-key-pairs --filters "Name=tag-key,Values=gpc4-session" \
    --query 'KeyPairs[].KeyPairId' --output text 2>/dev/null | tr '\t' '\n' | sed '/^$/d; s/^/key pair /'
}
for b in jq tnr aws rsync ssh lsof; do command -v "$b" >/dev/null || die "$b is not installed"; done

S=$(cat "$ROOT/current" 2>/dev/null) || die "no current session in $ROOT"
D=$ROOT/$S
P=$D/pod_launcher.log
REMOVED=()
line=$(grep -aoE 'Instance [^ ]+ created \(UUID [^)]+\)' "$P" 2>/dev/null | tail -1)
ID=$(sed -E 's/^Instance ([^ ]+) created.*/\1/' <<<"$line")
UUID=$(sed -E 's/.*\(UUID ([^)]+)\)$/\1/' <<<"$line")
say "session $S: pod ${UUID:+instance $ID, UUID $UUID}${UUID:-none logged (its launcher created no instance)}"

# 1. the results
LISTED=""
[ -n "$UUID" ] && LISTED=$(th_id_of "$UUID")
if [ -n "$UUID" ] && [ -z "$LISTED" ]; then
  say "Thunder no longer lists the pod $UUID"
elif [ -n "$LISTED" ]; then
  [ "$LISTED" = "$ID" ] || say "Thunder lists the pod $UUID under id $LISTED now (its launcher logged $ID)"
  ID=$LISTED
  if [ "$PULL" = 1 ]; then
    j=$(tnr connect "$ID" --json 2>/dev/null) || die "tnr connect $ID gave no endpoint: nothing removed (rerun, or --no-pull)"
    H=$(jq -r '.ip // empty' <<<"$j"); PT=$(jq -r '.port // empty' <<<"$j"); K=$(jq -r '.key_file // empty' <<<"$j")
    [ -n "$H" ] && [ -n "$PT" ] && [ -n "$K" ] || die "tnr connect $ID gave no endpoint: nothing removed (rerun, or --no-pull)"
    SSHC="ssh -p $PT -i $K -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
    mkdir -p "$LOCAL/logs"
    say "pulling the pod's results into $LOCAL/"
    rsync -az --partial --include '**/adapters/**' --exclude '*.pt' --exclude '*.pth' --exclude '*.ckpt' \
      --exclude '*.safetensors' -e "$SSHC" "ubuntu@$H:$REMOTE/" "$LOCAL/" \
      || die "the pull failed: nothing removed (rerun; or --no-pull to stop without the results)"
    rsync -az -e "$SSHC" "ubuntu@$H:/workspace/deepsteer/session.log" "$LOCAL/logs/session_stopped_$S.log" \
      || say "the runner's session.log did not come back (the results did)"
    say "pulled: $(find "$LOCAL" -type f | wc -l | tr -d ' ') files under $LOCAL/"
  fi
fi

# 2. the local launcher: only the processes holding this session's own pod_launcher.log open
PIDS=$(lsof -t -- "$P" 2>/dev/null | grep -vx "$$" | sort -u | tr '\n' ' ')
if [ -n "${PIDS// /}" ]; then
  say "stopping this session's local pod launcher (pids $PIDS); its cleanup deletes the pod under its UUID check"
  kill -TERM $PIDS 2>/dev/null
  t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt "$WAIT_S" ]; do
    alive=0; for p in $PIDS; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" = 0 ] && break
    sleep 5
  done
  [ "$alive" = 0 ] || say "the local launcher is still running after ${WAIT_S}s; going on"
  REMOVED+=("the local pod launcher (pids $PIDS)")
  [ -f "$P" ] && grep -aE '^>> (Deleting|Teardown verified)|TEARDOWN NOT VERIFIED|NOT deleting' "$P" | tail -3 | sed 's/^/   launcher: /'
else
  say "no local pod launcher is running for this session"
fi

# 3. the pod, under its own UUID
POD_LEFT=""
if [ -n "$UUID" ]; then
  cur=$(th_id_of "$UUID")
  if [ -n "$cur" ]; then
    if [ "$(th_uuid_of "$cur")" != "$UUID" ]; then
      echo "ERROR: instance $cur is not this session's $UUID now; NOT deleting. Check 'tnr status'." >&2
      POD_LEFT=$UUID
    else
      say "deleting instance $cur (UUID $UUID)"
      tnr delete "$cur" --yes >/dev/null 2>&1 || say "'tnr delete' returned an error; checking the account"
      for _ in $(seq 1 40); do [ -z "$(th_id_of "$UUID")" ] && break; sleep "${GPC4_STOP_POLL_S:-15}"; done
      if [ -z "$(th_id_of "$UUID")" ]; then REMOVED+=("the pod: instance $cur, UUID $UUID (Thunder)"); else POD_LEFT=$UUID; fi
    fi
  elif [ -n "$LISTED" ]; then
    REMOVED+=("the pod: instance $ID, UUID $UUID (Thunder; deleted by the launcher's cleanup)")
  fi
fi

# 4. the host
BEFORE=$(aws_left)
say "the host's teardown"
$TEARDOWN || say "the teardown did not finish: run it again (bash $G/runpod/launch_gpc4.sh teardown)"
AFTER=$(aws_left)
while IFS= read -r r; do
  [ -n "$r" ] && ! grep -qxF "$r" <<<"$AFTER" && REMOVED+=("$r (AWS)")
done <<<"$BEFORE"

echo "----------------------------------------------------------------------"
say "removed:"
[ "${#REMOVED[@]}" -gt 0 ] && printf '   %s\n' "${REMOVED[@]}" || echo "   nothing"
say "left:"
LEFT=$( { [ -n "$POD_LEFT" ] && echo "the pod $POD_LEFT (Thunder): it bills until deleted (tnr status; tnr delete <id>)"; \
          [ -n "$AFTER" ] && sed 's/$/ (AWS)/' <<<"$AFTER"; } )
[ -n "$LEFT" ] && sed 's/^/   /' <<<"$LEFT" || echo "   nothing of this session"
[ -z "$LEFT" ]
