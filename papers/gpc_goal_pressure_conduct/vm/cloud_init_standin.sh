#!/usr/bin/env bash
# The rehearsal's stand-in for the pod (a t3.micro; GPC_PLAN.md §11.3b): a stub model on 127.0.0.1:8000,
# the plan-driven control document on 127.0.0.1:8001, and the VM's two keys restricted exactly as on the
# pod (the forward key: two permitopen ports; the results key: rrsync -wo on one directory). The
# launcher fills __COMMIT__, __FORWARD_PUB__ and __RESULTS_PUB__ (public halves only).
set -euo pipefail
exec > >(tee -a /var/log/gpc4_standin.log) 2>&1
COMMIT=__COMMIT__
export DEBIAN_FRONTEND=noninteractive
apt-get update -q && apt-get install -y -q rsync python3 git
git clone -q https://github.com/deepsteer/deepsteer /opt/gpc4/deepsteer
git -C /opt/gpc4/deepsteer checkout -q "$COMMIT"
VM=/opt/gpc4/deepsteer/papers/gpc_goal_pressure_conduct/scripts/gpc4_vm.py
RR=$(command -v rrsync || true)
if [ -z "$RR" ]; then
  for c in /usr/share/doc/rsync/scripts/rrsync /usr/share/doc/rsync/scripts/rrsync.gz; do
    [ -f "$c" ] || continue
    case "$c" in *.gz) gunzip -c "$c" > /usr/local/bin/rrsync;; *) cp "$c" /usr/local/bin/rrsync;; esac
    chmod 755 /usr/local/bin/rrsync; RR=/usr/local/bin/rrsync; break
  done
fi
[ -n "$RR" ] || { echo "no rrsync"; exit 1; }
id gpc4pod >/dev/null 2>&1 || useradd -m -s /bin/bash gpc4pod
R=/home/gpc4pod/vm_results
install -d -o gpc4pod -m 700 /home/gpc4pod/.ssh "$R"
cat > /home/gpc4pod/.ssh/authorized_keys <<KEYS
restrict,port-forwarding,permitopen="127.0.0.1:8000",permitopen="127.0.0.1:8001" __FORWARD_PUB__
command="$RR -wo $R",restrict __RESULTS_PUB__
KEYS
chown gpc4pod /home/gpc4pod/.ssh/authorized_keys && chmod 600 /home/gpc4pod/.ssh/authorized_keys
cp /opt/gpc4/deepsteer/papers/gpc_goal_pressure_conduct/vm/rehearsal_plan.json /home/gpc4pod/plan.json
chown gpc4pod /home/gpc4pod/plan.json
for unit in stub:"stub --port 8000" control:"control --port 8001 --doc /home/gpc4pod/control.json --results $R" plan:"plan --plan /home/gpc4pod/plan.json --doc /home/gpc4pod/control.json --results $R"; do
  n=${unit%%:*}; args=${unit#*:}
  cat > /etc/systemd/system/gpc4-standin-$n.service <<UNIT
[Unit]
Description=gpc4 stand-in $n
[Service]
User=gpc4pod
ExecStart=/usr/bin/python3 $VM $args
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
UNIT
done
systemctl daemon-reload
systemctl enable --now gpc4-standin-stub gpc4-standin-control gpc4-standin-plan
touch /var/lib/gpc4_standin_ready
