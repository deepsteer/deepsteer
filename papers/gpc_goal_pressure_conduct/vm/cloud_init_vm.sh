#!/usr/bin/env bash
# GPC-4's EC2 host, first boot (cloud-init user-data; GPC_PLAN.md §11.3c, ODCV_HOST_BRIEF.md §2).
# The launcher fills __COMMIT__, __SESSION__ and __HARD_LIMIT_MIN__; no credential is in this file.
# Setup runs with egress open; the launcher locks egress (the security group) only after
# /var/lib/gpc4/setup_done.json exists and the pod's address is known, before any container runs.
set -euo pipefail
exec > >(tee -a /var/log/gpc4_setup.log) 2>&1
COMMIT=__COMMIT__
SESSION=__SESSION__
HARD_LIMIT_MIN=__HARD_LIMIT_MIN__
ODCV_COMMIT=6924dafb3a3f84fc320fbe9f0a08203d67d1d488
IB_COMMIT=d4f9a149db2b9cec0fed7bab01d91c07ee10aca2
IB_DATA_REV=98650ffc3f28a01b261669b6d19fcd7773823710
T0=$(date +%s)
mkdir -p /var/lib/gpc4 /etc/gpc4 /opt/gpc4 && chmod 700 /etc/gpc4
# the hard limit first: instance-initiated shutdown terminates the instance, which ends billing
shutdown -h +"$HARD_LIMIT_MIN" "gpc4 hard limit" || true

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q docker.io docker-compose-v2 python3-venv rsync git jq iptables
systemctl enable --now docker

git clone -q https://github.com/deepsteer/deepsteer /opt/gpc4/deepsteer
git -C /opt/gpc4/deepsteer checkout -q "$COMMIT"
VM=/opt/gpc4/deepsteer/papers/gpc_goal_pressure_conduct/scripts/gpc4_vm.py
git clone -q https://github.com/McGill-DMaS/ODCV-Bench /opt/gpc4/odcv
git -C /opt/gpc4/odcv checkout -q "$ODCV_COMMIT"
git clone -q https://github.com/safety-research/impossiblebench /opt/gpc4/impossiblebench
git -C /opt/gpc4/impossiblebench checkout -q "$IB_COMMIT"

# Inspect on the VM (ImpossibleBench's client), pinned; its LiveCodeBench data cached at the pinned
# revision and resolved offline (the code loads it without a revision)
python3 -m venv /opt/gpc4/ib-venv
/opt/gpc4/ib-venv/bin/pip install -q "inspect_ai==0.3.291" "datasets>=3.0.0" "pandas>=2.0.0" \
  "huggingface_hub" jsonlines platformdirs openai
/opt/gpc4/ib-venv/bin/python - <<EOF
from huggingface_hub import snapshot_download
import pathlib, pandas as pd
p = pathlib.Path(snapshot_download("fjzzq2002/impossible_livecodebench", repo_type="dataset",
                                   revision="$IB_DATA_REV"))
for s in ("original", "oneoff", "conflicting"):
    n = len(pd.read_parquet(next(p.glob(f"data/{s}-*.parquet"))))
    assert n == 103, (s, n)
ref = p.parents[1] / "refs" / "main"   # offline loads resolve "main" to the pinned revision
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text("$IB_DATA_REV")
print("impossiblebench data: 103 per split at $IB_DATA_REV")
EOF
docker pull -q aisiuk/inspect-tool-support

# ODCV-Bench's images, built here (never on GPU time): the executor once, each scenario's
# orchestrator per variant, staged as upstream's run_benchmarks.py stages them
cd /opt/gpc4/odcv
docker build -q -t gpc4/odcv-executor:latest mission_executor
for variant in mandated incentivized; do
  for sc in "${variant}"_scenarios/*/; do
    name=$(basename "$sc")
    st=$(mktemp -d) && python3 -c "import zipfile; zipfile.ZipFile('orchestrator_api.zip').extractall('$st')"
    cp -r "$sc". "$st/orchestrator_api/"
    tag="gpc4/odcv-${variant}-$(echo "$name" | tr '[:upper:]' '[:lower:]'):latest"
    docker build -q -t "$tag" "$st/orchestrator_api" >/dev/null
    rm -rf "$st"
  done
done
echo "odcv images: $(docker images -q 'gpc4/odcv-*' | sort -u | wc -l)"
# each scenario image's shipped files, hashed once (the runs' file trace compares against these)
mkdir -p /var/lib/gpc4/hashes_before
for tag in $(docker images --format '{{.Repository}}:{{.Tag}}' 'gpc4/odcv-*' | grep -v executor); do
  docker run --rm --network none --entrypoint sh "$tag" -c \
    'find /app /usr/local/bin -type f -exec sha256sum {} + 2>/dev/null' \
    > "/var/lib/gpc4/hashes_before/$(echo "$tag" | tr '/' '_').txt" || true
done

# the `odcv` bridge and its egress rule: from the bridge, only the proxy (DOCKER-USER)
docker network create --driver bridge --subnet 172.30.0.0/24 --gateway 172.30.0.1 odcv >/dev/null
iptables -N DOCKER-USER 2>/dev/null || true
iptables -I DOCKER-USER 1 -s 172.30.0.0/24 -j DROP
iptables -I DOCKER-USER 1 -s 172.30.0.0/24 -d 172.30.0.1 -p tcp --dport 18000 -j ACCEPT
iptables -I DOCKER-USER 1 -s 172.30.0.0/24 -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
# the host's INPUT chain: from the bridge, the proxy port only
iptables -I INPUT 1 -s 172.30.0.0/24 -j DROP
iptables -I INPUT 1 -s 172.30.0.0/24 -p tcp --dport 18000 -j ACCEPT

# the services: the forward (both pod ports), the proxy, the driver; each waits for pod.env
cat > /etc/systemd/system/gpc4-forward.service <<EOF
[Unit]
Description=gpc4 forward to the pod (the restricted key's two permitopen ports)
ConditionPathExists=/etc/gpc4/pod.env
[Service]
EnvironmentFile=/etc/gpc4/pod.env
ExecStart=/usr/bin/ssh -N -i /etc/gpc4/forward_key -p \${POD_PORT} -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -L 127.0.0.1:28000:127.0.0.1:8000 -L 127.0.0.1:28001:127.0.0.1:8001 \${POD_USER}@\${POD_HOST}
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/gpc4-proxy.service <<EOF
[Unit]
Description=gpc4 sampling proxy
After=docker.service
[Service]
ExecStart=/usr/bin/python3 $VM proxy
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/gpc4-driver.service <<EOF
[Unit]
Description=gpc4 stage driver (detached from the launcher)
After=gpc4-proxy.service gpc4-forward.service
[Service]
Environment=GPC4_ODCV=/opt/gpc4/odcv GPC4_IB_PY=/opt/gpc4/ib-venv/bin/python GPC4_IB_SRC=/opt/gpc4/impossiblebench/src/impossiblebench
ExecStart=/usr/bin/python3 $VM driver
Restart=on-failure
RestartSec=30
StandardOutput=append:/var/log/gpc4_driver.log
StandardError=append:/var/log/gpc4_driver.log
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now gpc4-proxy.service
systemctl enable gpc4-forward.service gpc4-driver.service   # started once pod.env exists

python3 - <<EOF
import json, time
json.dump({"session": "$SESSION", "commit": "$COMMIT", "setup_s": round(time.time() - $T0),
           "hard_limit_min": $HARD_LIMIT_MIN}, open("/var/lib/gpc4/setup_done.json", "w"))
EOF
echo "gpc4 setup done in $(( $(date +%s) - T0 )) s"
