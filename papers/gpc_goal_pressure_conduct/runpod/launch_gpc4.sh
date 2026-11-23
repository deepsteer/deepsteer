#!/usr/bin/env bash
# GPC-4's launcher (GPC_PLAN.md §11.3 to §11.3c; ODCV_HOST_BRIEF.md). The author runs, from the repo root:
#
#   bash papers/gpc_goal_pressure_conduct/runpod/launch_gpc4.sh rehearse   # zero GPU, about $0.65
#   bash papers/gpc_goal_pressure_conduct/runpod/launch_gpc4.sh verify     # after rehearse: the record
#   bash papers/gpc_goal_pressure_conduct/runpod/launch_gpc4.sh launch     # the session (needs a passed
#                                                                          # rehearsal at this commit)
#   bash papers/gpc_goal_pressure_conduct/runpod/launch_gpc4.sh reattach   # from any address
#   bash papers/gpc_goal_pressure_conduct/runpod/launch_gpc4.sh teardown   # every gpc4-session resource
#
# AWS: the AWS CLI v2 reads credentials from the author's profile or environment (AWS_PROFILE is honored);
# none is ever on a command line, written to disk by this script, or sent to the VM or the pod. The account
# ID is read at run time and never printed or written. Region us-east-2. Every resource carries the tag
# gpc4-session. The session's SSH keys live in ~/.gpc4/<session>/ (mode 700).
set -euo pipefail
cd "$(dirname "$0")/../../.."
G=papers/gpc_goal_pressure_conduct
REGION=us-east-2
TYPE=c6a.4xlarge
STANDIN_TYPE=t3.micro
DISK_GB=160
VCPUS_NEEDED=16
HARD_LIMIT_MIN=${GPC4_HARD_LIMIT_MIN:-840}   # 14 h, the runner's ceiling
STANDIN_LIMIT_MIN=720   # the rehearsal's stand-in: 12 h, long enough for a `verify` the next morning
SETUP_WAIT_MIN=${GPC4_SETUP_WAIT_MIN:-90}
AMI_PARAM=/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id
ROOT=${GPC4_ROOT:-~/.gpc4}
# the pod card, and how long and how often `launch` waits for its endpoint; a test swaps the card
# for one that bails (the lead's no-capacity test)
POD_CARD=${GPC4_POD_CARD:-$G/runpod/launch_gpc4_pod.sh}
POD_WAIT_MIN=${GPC4_POD_WAIT_MIN:-60}
POD_POLL_S=${GPC4_POD_POLL_S:-15}
GATE_SETTLE_S=${GPC4_GATE_SETTLE_S:-20}   # after the egress lock, before the post-lock gate reads it
# Thunder's public availability (no login; GET /v2/status): read before any AWS resource exists
TNR_STATUS_URL=https://api.thundercompute.com:8443/v2/status
COMMIT=$(git rev-parse HEAD)
# the rehearsal vouches for the VM side: the files the host and the stand-in run (a later commit that
# changes only the pod's files keeps the rehearsal valid)
VM_FILES=("$G/scripts/gpc4_vm.py" "$G/vm/cloud_init_vm.sh" "$G/vm/cloud_init_standin.sh"
          "$G/vm/rehearsal_plan.json" "$G/vm/gpc4_iam_policy.json" "$G/runpod/launch_gpc4.sh")
VM_HASH=$(cat "${VM_FILES[@]}" | shasum -a 256 | cut -c1-16)
export AWS_DEFAULT_REGION=$REGION AWS_PAGER=""
SSHO=(-o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes
      -o ConnectTimeout=15 -o LogLevel=ERROR)

die() { echo "BAIL: $*" >&2; exit 1; }
say() { echo ">> $*"; }
need() { command -v "$1" >/dev/null || die "$1 is not installed (${2:-})"; }
tagged() {  # instance ids carrying gpc4-session, in a live or stopped state
  aws ec2 describe-instances --filters "Name=tag-key,Values=gpc4-session" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down" \
    --query 'Reservations[].Instances[].InstanceId' --output text
}
list_tagged() {
  aws ec2 describe-instances --filters "Name=tag-key,Values=gpc4-session" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Tags[?Key==`gpc4-session`]|[0].Value]' \
    --output text
}

preflight() {
  need aws "AWS CLI v2"; need jq; need ssh; need rsync
  # the host clones this commit from GitHub: it must be committed and on origin
  git diff --quiet HEAD -- "${VM_FILES[@]}" || die "the VM side has uncommitted changes (${VM_FILES[*]})"
  git fetch -q origin && git merge-base --is-ancestor "$COMMIT" origin/main \
    || die "commit $COMMIT is not on origin/main (push it first)"
  local acct; acct=$(aws sts get-caller-identity --query Account --output text) || die "no AWS identity"
  [ -n "$acct" ] || die "no AWS identity"; unset acct   # read, never printed or written
  # the region's on-demand standard vCPU quota (L-1216C47A), against the vCPUs already running
  local q used; q=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A \
    --query 'Quota.Value' --output text 2>/dev/null) || die "cannot read the vCPU quota L-1216C47A"
  used=$(aws ec2 describe-instances --filters "Name=instance-state-name,Values=pending,running" \
    --query 'Reservations[].Instances[].[InstanceType,CpuOptions.CoreCount,CpuOptions.ThreadsPerCore]' \
    --output text | awk '$1 ~ /^[acdhimrtz]/ {s += $2 * $3} END {print s + 0}')
  awk -v q="$q" -v u="$used" -v n="$1" 'BEGIN {exit !(q - u >= n)}' \
    || die "us-east-2's on-demand standard vCPU quota (L-1216C47A) is $q with $used in use; $1 are needed"
  say "quota: $q vCPU, $used in use, $1 needed"
  [ -z "$(tagged)" ] || { list_tagged; die "a gpc4-session instance exists; run teardown first"; }
}

new_session() {
  S=gpc4-$(date -u +%Y%m%dT%H%M%SZ)
  D="$ROOT/$S"; mkdir -p "$D"; chmod 700 "$ROOT" "$D"
  ssh-keygen -q -t ed25519 -N "" -C "$S-forward" -f "$D/forward_key"
  ssh-keygen -q -t ed25519 -N "" -C "$S-results" -f "$D/results_key"
  echo "$S" > "$ROOT/current"
  say "session $S (keys in $D)"
}

load_session() {
  S=$(cat "$ROOT/current" 2>/dev/null) || die "no current session"
  D="$ROOT/$S"
}

my_ip() { curl -s -m 10 https://checkip.amazonaws.com | tr -d '[:space:]'; }

make_group_and_key() {
  local vpc; vpc=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  SG=$(aws ec2 create-security-group --group-name "$S" --description "gpc4 session $S" --vpc-id "$vpc" \
    --tag-specifications "ResourceType=security-group,Tags=[{Key=gpc4-session,Value=$S}]" \
    --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$(my_ip)/32" >/dev/null
  # members of the group reach each other's SSH (the rehearsal's stand-in)
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --source-group "$SG" >/dev/null
  aws ec2 create-key-pair --key-name "$S" --key-type ed25519 \
    --tag-specifications "ResourceType=key-pair,Tags=[{Key=gpc4-session,Value=$S}]" \
    --query KeyMaterial --output text > "$D/admin_key"
  chmod 600 "$D/admin_key"
  echo "$SG" > "$D/sg"
  say "security group $SG (SSH from this address only), key pair $S"
}

run_instance() {  # <type> <user-data file> <name>
  local ami; ami=$(aws ssm get-parameter --name "$AMI_PARAM" --query Parameter.Value --output text)
  local disk=$DISK_GB; [ "$1" = "$STANDIN_TYPE" ] && disk=8
  aws ec2 run-instances --image-id "$ami" --instance-type "$1" --key-name "$S" --security-group-ids "$SG" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$disk,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --instance-initiated-shutdown-behavior terminate \
    --metadata-options HttpTokens=required,HttpPutResponseHopLimit=1,HttpEndpoint=enabled \
    --user-data "file://$2" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=gpc4-session,Value=$S},{Key=Name,Value=$3}]" \
      "ResourceType=volume,Tags=[{Key=gpc4-session,Value=$S}]" \
    --query 'Instances[0].InstanceId' --output text
}

ip_of() { aws ec2 describe-instances --instance-ids "$1" --query "Reservations[0].Instances[0].$2" --output text; }

wait_ssh() {  # <ip>
  for _ in $(seq 1 60); do ssh "${SSHO[@]}" -i "$D/admin_key" "ubuntu@$1" true 2>/dev/null && return 0; sleep 10; done
  die "no SSH to $1"
}

vm_setup() {
  sed -e "s/__COMMIT__/$COMMIT/" -e "s/__SESSION__/$S/" -e "s/__HARD_LIMIT_MIN__/$HARD_LIMIT_MIN/" \
    "$G/vm/cloud_init_vm.sh" > "$D/user_data_vm.sh"
  ! grep -qiE 'aws_secret|aws_access|hf_[a-z0-9]{10}|BEGIN OPENSSH PRIVATE' "$D/user_data_vm.sh" \
    || die "user-data would carry a credential"
  VM=$(run_instance "$TYPE" "$D/user_data_vm.sh" "$S-host"); echo "$VM" > "$D/vm"
  aws ec2 wait instance-running --instance-ids "$VM"
  VM_IP=$(ip_of "$VM" PublicIpAddress); echo "$VM_IP" > "$D/vm_ip"
  T_SETUP=$(date +%s)
  say "host $VM ($TYPE) at $VM_IP; setup runs on the host (not on GPU time)"
  wait_ssh "$VM_IP"
  local deadline=$(( $(date +%s) + SETUP_WAIT_MIN * 60 ))
  local last=""
  until ssh "${SSHO[@]}" -i "$D/admin_key" "ubuntu@$VM_IP" test -f /var/lib/gpc4/setup_done.json 2>/dev/null; do
    [ "$(date +%s)" -lt "$deadline" ] || die "setup did not finish in $SETUP_WAIT_MIN min (see /var/log/gpc4_setup.log)"
    local m; m=$(ssh "${SSHO[@]}" -i "$D/admin_key" "ubuntu@$VM_IP" \
      "n=\$(sudo docker images --format '{{.Repository}}' 2>/dev/null | grep -c gpc4/odcv- || true); \
       sudo grep -aE 'impossiblebench data|odcv images' /var/log/gpc4_setup.log 2>/dev/null | tail -1; echo \"images built: \$n\"" 2>/dev/null | tr '\n' ' ')
    [ -n "$m" ] && [ "$m" != "$last" ] && { say "setup: $m($(( ($(date +%s) - T_SETUP) / 60 )) min)"; last=$m; }
    sleep 60
  done
  ssh "${SSHO[@]}" -i "$D/admin_key" "ubuntu@$VM_IP" cat /var/lib/gpc4/setup_done.json > "$D/setup_done.json"
  say "setup done in $(( $(date +%s) - T_SETUP )) s wall"
  # the metadata endpoint: cloud-init needed it at first boot; disabled before any container runs
  aws ec2 modify-instance-metadata-options --instance-id "$VM" --http-endpoint disabled >/dev/null
  # the two restricted keys' private halves, root-only on the host
  scp "${SSHO[@]}" -i "$D/admin_key" "$D/forward_key" "$D/results_key" "ubuntu@$VM_IP:/tmp/" >/dev/null
  ssh "${SSHO[@]}" -i "$D/admin_key" "ubuntu@$VM_IP" \
    'sudo install -m 600 -o root /tmp/forward_key /tmp/results_key /etc/gpc4/ && rm -f /tmp/forward_key /tmp/results_key'
  ssh "${SSHO[@]}" -i "$D/admin_key" "ubuntu@$VM_IP" \
    sudo python3 /opt/gpc4/deepsteer/$G/scripts/gpc4_vm.py gate --phase pre > "$D/gate_pre.json" \
    || { cat "$D/gate_pre.json"; die "the host failed its pre-lock gate tests"; }
  say "pre-lock gate tests pass"
}

handoff() {  # <pod host> <pod port> <pod user>: the lock, then at once the post-lock gate (the rehearsal)
  handoff_lock "$@" && gate_post
}

handoff_lock() {  # <pod host> <pod port> <pod user>: pod.env, the forward and driver, then the egress lock
  ssh "${SSHO[@]}" -i "$D/admin_key" "ubuntu@$VM_IP" \
    "printf 'POD_HOST=%s\nPOD_PORT=%s\nPOD_USER=%s\n' '$1' '$2' '$3' | sudo tee /etc/gpc4/pod.env >/dev/null \
     && sudo chmod 600 /etc/gpc4/pod.env && sudo systemctl start gpc4-forward gpc4-driver"
  # the egress lock: the default outbound rule out, outbound only to the pod's SSH port
  aws ec2 revoke-security-group-egress --group-id "$SG" \
    --ip-permissions 'IpProtocol=-1,IpRanges=[{CidrIp=0.0.0.0/0}]' >/dev/null 2>&1 || true
  aws ec2 revoke-security-group-egress --group-id "$SG" \
    --ip-permissions 'IpProtocol=-1,Ipv6Ranges=[{CidrIpv6=::/0}]' >/dev/null 2>&1 || true
  aws ec2 authorize-security-group-egress --group-id "$SG" --protocol tcp --port "$2" --cidr "$1/32" >/dev/null
  say "egress locked to $1:$2"
}

gate_post() {  # the post-lock gate: the internet closed, the pod's port open, the forward and the results path
  sleep "$GATE_SETTLE_S"
  ssh "${SSHO[@]}" -i "$D/admin_key" "ubuntu@$VM_IP" \
    sudo python3 /opt/gpc4/deepsteer/$G/scripts/gpc4_vm.py gate --phase post > "$D/gate_post.json" \
    || { cat "$D/gate_post.json"; return 1; }
  say "post-lock gate tests pass (internet closed, the pod's port open, the forward and the results path)"
}

cmd_rehearse() {
  preflight $(( VCPUS_NEEDED + 2 ))
  new_session; make_group_and_key
  sed -e "s/__COMMIT__/$COMMIT/" -e "s#__FORWARD_PUB__#$(cat "$D/forward_key.pub")#" \
    -e "s#__RESULTS_PUB__#$(cat "$D/results_key.pub")#" -e "s/__HARD_LIMIT_MIN__/$STANDIN_LIMIT_MIN/" \
    "$G/vm/cloud_init_standin.sh" > "$D/user_data_standin.sh"
  # what the host runs, saved now: `verify` keys the record to these, never to the tree it runs in
  echo "$VM_HASH" > "$D/vm_hash"; echo "$COMMIT" > "$D/commit"
  SI=$(run_instance "$STANDIN_TYPE" "$D/user_data_standin.sh" "$S-standin"); echo "$SI" > "$D/standin"
  trap 'say "the launcher stopped; resources stay for verify or teardown"' EXIT
  vm_setup
  aws ec2 wait instance-running --instance-ids "$SI"
  SI_PRIV=$(ip_of "$SI" PrivateIpAddress); SI_PUB=$(ip_of "$SI" PublicIpAddress)
  wait_ssh "$SI_PUB"
  until ssh "${SSHO[@]}" -i "$D/admin_key" "ubuntu@$SI_PUB" test -f /var/lib/gpc4_standin_ready 2>/dev/null; do sleep 15; done
  echo "$SI_PUB" > "$D/standin_ip"
  handoff "$SI_PRIV" 22 gpc4pod || die "the host failed its post-lock gate tests"
  # the detached test: the launcher now exits mid-run, as if killed; `verify` checks the rest
  say "rehearsal running detached; this launcher exits now (the kill). Run \`$0 verify\` in about 30 min."
  trap - EXIT
}

cmd_verify() {
  load_session; need aws; need jq
  [ -f "$D/vm_hash" ] || die "session $S predates the saved hash; rehearse again"
  local r_hash r_commit; r_hash=$(cat "$D/vm_hash"); r_commit=$(cat "$D/commit")
  [ "$r_hash" = "$VM_HASH" ] || die "the VM side changed since this rehearsal ($r_hash, now $VM_HASH): its record would name code it never ran; tear down and rehearse again"
  local si_ip vm; si_ip=$(cat "$D/standin_ip"); vm=$(cat "$D/vm")
  # a terminated instance leaves the listing after about an hour; an id that is gone was terminated
  local state; state=$(ip_of "$vm" State.Name 2>/dev/null || echo gone)
  # the results directory is gpc4pod's (mode 700), so it is read as root and unpacked where the record
  # reads it (rehearsal 4: an rsync as ubuntu failed and the fallback unpacked one level off)
  mkdir -p "$D/rehearsal_results"
  ssh "${SSHO[@]}" -i "$D/admin_key" "ubuntu@$si_ip" "sudo tar -C /home/gpc4pod/vm_results -cf - ." \
    | tar -C "$D/rehearsal_results" -xf -
  ls "$D"/rehearsal_results/*/stage.json >/dev/null 2>&1 \
    || die "no stage results downloaded from the stand-in; the record is not written (teardown is separate)"
  local launched; launched=$(ip_of "$vm" LaunchTime 2>/dev/null || echo gone)
  python3 - "$D" "$state" "$launched" "$r_commit" "$r_hash" "$G/vm/rehearsal_plan.json" "$G/scripts" <<'EOF'
import json, sys, datetime, pathlib
d, state, launched, commit, vm_hash, plan_f, scripts = sys.argv[1:]
sys.path.insert(0, scripts)
import gpc4_vm as V  # received(): a stage counts only when its done-marker's file hashes match
plan = json.loads(pathlib.Path(plan_f).read_text())["instructions"]
D = pathlib.Path(d)
R = D / "rehearsal_results"
# delivery, from the downloaded results (the control document answers 503 once the stand-in vanishes)
got = set(V.received(R))
want = {i["id"] for i in plan if i["kind"] != "vanish"} | {
    x["id"] for i in plan if i["kind"] == "batch" for x in i["items"]}
reached = "true" if want <= got else "false"
# the pod's loss, rehearsed: with a vanish in the plan, the host must have terminated itself on the
# control document's silence and pushed its record of why (the lead, rehearsal 8)
vanish = any(i["kind"] == "vanish" for i in plan)
selfterm = json.loads((R / "host_selfterm" / "selfterm.json").read_text()) if (
    R / "host_selfterm" / "selfterm.json").exists() else None
selfterm_ok = (not vanish) or bool(selfterm and "unreachable" in selfterm.get("reason", "")
                                    and selfterm.get("unreachable_s", 0) >= selfterm.get("pod_gone_s", 1))
try:
    t0 = datetime.datetime.fromisoformat(launched.replace("Z", "+00:00"))
    hours = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds() / 3600
except ValueError:  # the host has left the listing: bounded by the stand-in's limit instead
    hours = 12.0
def stage(name):
    f = R / name / "stage.json"
    return json.loads(f.read_text()) if f.exists() else {}
def usage(name):
    f = R / name / "resources.json"
    return json.loads(f.read_text()) if f.exists() else {}
# the instruments ran, at the plan's own scale, not only the plumbing (rehearsal 2 delivered every
# result with neither running); rehearsal 6 (the lead): no container error on any of ODCV-Bench's runs,
# and every ImpossibleBench task scored with no sample error
checks, stages, runs = {}, {}, []
def check(ins):
    k, st = ins["id"], stage(ins["id"])
    stages[k] = {"kind": ins["kind"], **usage(k)}
    # every push during the stage landed (rehearsal 6: 8 parallel runs collided on rrsync's lock and
    # each run's results waited for the stage's end, against the detached rule)
    pushes_ok = stages[k].get("push_failed") in (0, None) if ins.get("in_batch") else stages[k].get("push_failed") == 0
    if ins["kind"] == "odcv":
        rr = st.get("runs", [])
        runs.extend({f: r.get(f) for f in ("run_id", "compose_exit", "container_error", "tool_calls",
                                           "tool_timeouts", "wall_s")} for r in rr)
        stages[k] |= {"runs": len(rr), "expected": len(ins["runs"]), "concurrency": ins.get("concurrency"),
                      "container_errors": sum(bool(r.get("container_error")) for r in rr),
                      "tool_timeouts": sum(r.get("tool_timeouts") or 0 for r in rr),
                      "min_tool_calls": min((r.get("tool_calls") or 0 for r in rr), default=0)}
        complete = pushes_ok and len(rr) == len(ins["runs"]) and stages[k]["min_tool_calls"] >= 1
        if ins.get("same_scenario"):
            # rehearsal 12 (the lead, 2026-11-22): one scenario many times at once (the depth cell's draws),
            # every run its own Compose project; all must finish with no container error
            stages[k]["scenarios"] = sorted({r[0] for r in ins["runs"]})
            checks[k] = (complete and stages[k]["container_errors"] == 0 and len(stages[k]["scenarios"]) == 1
                         and len({r[2] for r in ins["runs"]}) == len(ins["runs"]))
        elif (ins.get("concurrency") or 0) >= 16:
            # §11.21 ruling 5: the gate at 16 if all 80 run with no container error and the host's peak load
            # stays under 8; otherwise 8 (recorded, and the rehearsal still passes on a complete stage)
            stages[k]["concurrency_16_ok"] = bool(complete and stages[k]["container_errors"] == 0
                                                  and (stages[k].get("peak_load1") or 99) < 8)
            checks[k] = complete
        else:
            checks[k] = complete and stages[k]["container_errors"] == 0
    elif ins["kind"] == "ib":
        stages[k] |= {f: st.get(f) for f in ("exit", "n_logs", "n_samples", "n_scored", "n_errored")}
        stages[k]["expected"] = len(ins["task_ids"])
        checks[k] = (pushes_ok and st.get("exit") == 0 and st.get("n_errored") == 0
                     and st.get("n_samples") == st.get("n_scored") == len(ins["task_ids"]))
    elif ins["kind"] == "batch":  # §11.21: each item is checked as its own instruction
        for item in ins["items"]:
            check(item | {"in_batch": True})
        checks[k] = pushes_ok and all(checks[i["id"]] for i in ins["items"])
    elif ins["kind"] == "pause":  # §11.21 ruling 8: the stop was published, and the plan went on
        checks[k] = k in got
    else:  # the probe: its push landed
        checks[k] = pushes_ok
for ins in plan:
    if ins["kind"] != "vanish":
        check(ins)
# §12 item 9 (rehearsal 11): each slow call was generated once and returned once, the client's repeat joined
for ins in plan:
    if ins.get("slow"):
        k = ins["id"]
        files = (sorted((R / k).glob("*/proxy.jsonl")) if ins["kind"] == "odcv" else [R / k / "proxy.jsonl"])
        rows = [json.loads(x) for f in files if f.exists() for x in f.read_text().splitlines() if x.strip()]
        sc = V.slow_call_ok(rows)
        stages[k]["slow_call"] = sc
        checks[k] = checks.get(k, False) and sc["ok"]
# §13.1 E1 (rehearsal 12): every chat call went upstream at its instruction's temperature, the sampling of record
# where the instruction names none
def rows_of(ins):
    k = ins["id"]
    files = (sorted((R / k).glob("*/proxy.jsonl")) if ins["kind"] == "odcv" else [R / k / "proxy.jsonl"])
    return [json.loads(x) for f in files if f.exists() for x in f.read_text().splitlines() if x.strip()]
flat = [x for i in plan for x in ([i] + (i["items"] if i["kind"] == "batch" else []))]
for ins in flat:
    if ins["kind"] in ("odcv", "ib"):
        k = ins["id"]
        want_t = float((ins.get("sampling") or {}).get("temperature", V.SAMPLING["temperature"]))
        sc = V.sampling_ok(rows_of(ins), want_t)
        stages.setdefault(k, {})["sampling"] = sc
        checks[k] = checks.get(k, False) and sc["ok"]
instruments_ran = bool(checks) and all(checks.values())
instruments = {"stages": stages, "checks": checks}
rec = {"commit": commit, "vm_hash": vm_hash, "host_state": state,
       "self_terminated": state in ("shutting-down", "terminated", "gone"),
       "instruments": instruments, "instruments_ran": instruments_ran, "odcv_runs": runs,
       "host_selfterm": selfterm, "selfterm_ok": selfterm_ok,
       "every_result_on_the_receiving_side": reached == "true",
       "odcv_concurrency_16_ok": next((v.get("concurrency_16_ok") for v in stages.values()
                                       if "concurrency_16_ok" in v), None),
       "gate_pre": json.loads((D / "gate_pre.json").read_text()),
       "gate_post": json.loads((D / "gate_post.json").read_text()),
       "setup": json.loads((D / "setup_done.json").read_text()),
       "host_hours_upper": round(hours, 2),
       "cost_usd_upper": round(hours * (0.612 + 160 * 0.08 / 730 + 0.005) + hours * (0.0104 + 8 * 0.08 / 730 + 0.005), 2)}
rec["passed"] = bool(rec["self_terminated"] and rec["every_result_on_the_receiving_side"] and instruments_ran
                     and selfterm_ok
                     and rec["gate_pre"].get("passes") and rec["gate_post"].get("passes"))
(pathlib.Path.home() / ".gpc4" / f"rehearsal_{vm_hash}.json").write_text(json.dumps(rec, indent=1))
print(json.dumps({k: v for k, v in rec.items() if k not in ("gate_pre", "gate_post", "odcv_runs")}, indent=1))
EOF
  say "the rehearsal record: $ROOT/rehearsal_$VM_HASH.json (the VM side's hash; commit $COMMIT)"
  cmd_teardown
}

cmd_reattach() {
  need aws
  local sg; sg=$(aws ec2 describe-security-groups --filters "Name=tag-key,Values=gpc4-session" \
    --query 'SecurityGroups[0].GroupId' --output text)
  [ "$sg" != "None" ] || die "no gpc4-session security group"
  local rules; rules=$(aws ec2 describe-security-groups --group-ids "$sg" \
    --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text)
  for c in $rules; do aws ec2 revoke-security-group-ingress --group-id "$sg" --protocol tcp --port 22 --cidr "$c" >/dev/null; done
  aws ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 --cidr "$(my_ip)/32" >/dev/null
  say "SSH now admitted from this address only"
  list_tagged
  load_session 2>/dev/null || return 0
  [ -f "$D/vm_ip" ] && ssh "${SSHO[@]}" -i "$D/admin_key" "ubuntu@$(cat "$D/vm_ip")" tail -n 20 /var/log/gpc4_driver.log || true
}

cmd_teardown() {
  need aws
  say "gpc4-session instances:"; list_tagged
  local ids; ids=$(tagged)
  if [ -n "$ids" ]; then
    aws ec2 terminate-instances --instance-ids $ids >/dev/null
    aws ec2 wait instance-terminated --instance-ids $ids
  fi
  for sg in $(aws ec2 describe-security-groups --filters "Name=tag-key,Values=gpc4-session" --query 'SecurityGroups[].GroupId' --output text); do
    aws ec2 delete-security-group --group-id "$sg" >/dev/null 2>&1 || say "security group $sg not deleted yet (retry teardown)"
  done
  for k in $(aws ec2 describe-key-pairs --filters "Name=tag-key,Values=gpc4-session" --query 'KeyPairs[].KeyPairId' --output text); do
    aws ec2 delete-key-pair --key-pair-id "$k" >/dev/null
  done
  say "after teardown:"; list_tagged
}

thunder_capacity() {  # one A100 80GB (a100xl x1), by both of the endpoint's keys, which can disagree
  local j; j=$(curl -s -m 20 "$TNR_STATUS_URL") || die "Thunder's availability endpoint did not answer; nothing was created"
  jq -e '.gpu_type.a100xl."1" == "available" and .specs.a100xl_x1 == "available"' <<<"$j" >/dev/null 2>&1 \
    || die "Thunder lists no A100 80GB x1 now ($(jq -c '{a100xl: .gpu_type.a100xl."1", a100xl_x1: .specs.a100xl_x1}' <<<"$j" 2>/dev/null || echo unreadable)); nothing was created"
  say "Thunder lists an A100 80GB x1 as available (read before the host is created)"
}

abort_before_handoff() {  # any failure before the handoff: the pod launcher, then the host, key and group
  local rc=$?
  trap - EXIT
  [ -n "${TAIL_PID:-}" ] && kill "$TAIL_PID" 2>/dev/null
  if [ -n "${POD_PID:-}" ] && kill -0 "$POD_PID" 2>/dev/null; then
    # TERM runs the pod launcher's own cleanup at once (it deletes any instance it created); a
    # background job ignores INT
    say "stopping the pod launcher (its cleanup deletes any instance it created; check 'tnr status')"
    kill -TERM "$POD_PID" 2>/dev/null; wait "$POD_PID" 2>/dev/null || true
  fi
  say "the launch failed before the handoff: removing the host, the key and the group"
  cmd_teardown || say "teardown did not finish: run it again"
  say "stopped before the handoff; nothing of this session is left running on AWS (listing above)"
  exit "$(( rc == 0 ? 1 : rc ))"
}

cmd_launch() {
  [ -f "$ROOT/rehearsal_$VM_HASH.json" ] && jq -e .passed "$ROOT/rehearsal_$VM_HASH.json" >/dev/null \
    || die "no passed rehearsal for this VM side ($VM_HASH; run rehearse, then verify)"
  [ -z "${TNR_FALLBACK:-}" ] || die "GPC-4 runs on Thunder with no fallback"
  preflight "$VCPUS_NEEDED"
  [ -f "$G/runpod/remote_gpc4.sh" ] || die "the pod's runner (remote_gpc4.sh) is not built yet"
  # §13.1 E2 (the pilot's first launch, 2026-11-22: the pod card's own check found 48 GB after 11 min of the
  # host's setup): the same rule as session_lib.sh's rp_require_disk, here before any AWS resource
  local res="$G/outputs/gpc4_v1" free; mkdir -p "$res"
  free=$(df -Pk "$res" | awk 'NR==2 {print int($4 / 1048576)}')
  [ "${free:-0}" -ge "${MIN_FREE_GB:-50}" ] \
    || die "only ${free} GB free under $res (need >= ${MIN_FREE_GB:-50} GB for results); free space first (nothing was created)"
  say "local disk: ${free} GB free under $res (min ${MIN_FREE_GB:-50})"
  thunder_capacity   # before any AWS resource: with no capacity nothing is created
  new_session
  trap abort_before_handoff EXIT
  make_group_and_key
  vm_setup
  local in_dir=${GPC4_IN_DIR:-$G/outputs/gpc4_in}   # the pod's inputs (a test points this elsewhere)
  mkdir -p "$in_dir/vm_keys"
  cp "$D/forward_key.pub" "$D/results_key.pub" "$in_dir/vm_keys/"
  echo "$S" > "$in_dir/session"
  # the pod, through the provider layer (Thunder, no fallback); its endpoint line starts the handoff
  local plog="$D/pod_launcher.log"; : > "$plog"
  bash "$POD_CARD" > "$plog" 2>&1 & POD_PID=$!
  tail -n +1 -f "$plog" & TAIL_PID=$!; disown "$TAIL_PID"   # the card's output, shown live
  local ep="" t0; t0=$(date +%s)
  while :; do
    ep=$(grep -oE 'SSH endpoint: [a-z]+@[0-9.]+:[0-9]+' "$plog" 2>/dev/null | head -1 | sed 's/SSH endpoint: //' || true)
    [ -n "$ep" ] && break
    # the pod launcher exits without an endpoint on no capacity (or any bail): stop now, not in an hour
    kill -0 "$POD_PID" 2>/dev/null || die "the pod launcher exited without an endpoint (see $plog)"
    [ $(( $(date +%s) - t0 )) -lt $(( POD_WAIT_MIN * 60 )) ] || die "no pod endpoint in $POD_WAIT_MIN min (see $plog)"
    sleep "$POD_POLL_S"
  done
  local user=${ep%@*} rest=${ep#*@}; local host=${rest%:*} port=${rest#*:}
  handoff_lock "$host" "$port" "$user"
  # the post-lock gate waits for the pod's serve check: by then the pod has installed the host's keys and
  # started its control server and the subject's server, so every forward check can pass (GPC-4's first
  # launch ran the gate 40 s after the pod was created, before any key existed). A pod launcher that
  # exits first is a launch that failed before the handoff: the abort path removes the host.
  say "the post-lock gate waits for the pod's serve check"
  while ! grep -aqE '^>> stage done at [0-9.]+ h: --stage serve_check$|^>> serve_check failed' "$plog"; do
    kill -0 "$POD_PID" 2>/dev/null || die "the pod launcher exited before its serve check (see $plog)"
    sleep "$POD_POLL_S"
  done
  # from here the pod and the host run detached: a failure now is the pod's failure branch
  trap 'say "the launcher stopped: the pod and the host run on (reattach, or teardown)"' EXIT
  gate_post || say "the host failed its post-lock gate tests: the pod runs its no-VM items and stops"
  wait "$POD_PID" || true
  kill "$TAIL_PID" 2>/dev/null || true
}

case "${1:-}" in
  rehearse) cmd_rehearse ;;
  verify) cmd_verify ;;
  launch) cmd_launch ;;
  reattach) cmd_reattach ;;
  teardown) cmd_teardown ;;
  *) echo "usage: $0 rehearse|verify|launch|reattach|teardown"; exit 2 ;;
esac
