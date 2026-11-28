#!/usr/bin/env bash
# The sandbox's first run, in a local Linux container before any pod (GPC_PLAN.md §19.18, the lead's 2026-11-27
# 4:15 PM item 3): Ubuntu 22.04 (the pod's release), the setup as on the pod, then the tests run as a non-root user
# with passwordless sudo (the pod's `ubuntu`), so the sudo path the pod uses is the one tested. The repository is
# mounted read-only at /workspace/deepsteer, the pod's path, so the self-test checks that the sandbox cannot see it.
#   bash papers/gpc_goal_pressure_conduct/runpod/c2_sandbox_local.sh     # writes outputs/c2_sandbox_local/
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT=papers/gpc_goal_pressure_conduct/outputs/c2_sandbox_local
mkdir -p "$OUT"
# seccomp and AppArmor unconfined and /proc unmasked: Docker's defaults stop any nested namespace, a container's
# limit and not the pod's (a VM); nothing else is granted, and the container is not privileged
docker run --rm -i --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --security-opt systempaths=unconfined \
  -v "$PWD":/workspace/deepsteer:ro -v "$PWD/$OUT":/out ubuntu:22.04 bash -s <<'EOF'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -q >/dev/null && apt-get install -y -q sudo python3 python3-pytest >/dev/null
useradd -m ubuntu && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu
cd /workspace/deepsteer
bash papers/gpc_goal_pressure_conduct/runpod/c2_sandbox_setup.sh
sudo -u ubuntu bash -c 'umask 077; echo "hf_dummy_token_for_the_selftest" > ~/.hf_token'
G=papers/gpc_goal_pressure_conduct/scripts; ST=0
sudo -u ubuntu -H env PYTHONDONTWRITEBYTECODE=1 HF_TOKEN=dummy-env-token \
  python3 $G/c2_sandbox.py selftest --out /tmp/selftest.json --repo /workspace/deepsteer || ST=$?
cp /tmp/selftest.json /out/selftest.json
sudo -u ubuntu -H env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  -m "not slow and not host_side" tests/scripts/test_c2.py 2>&1 | tail -15 | tee /out/pytest.txt
# the whole pod's stages on the scripted client, in the sandbox (the dry run's commands are ours, not a subject's)
sudo -u ubuntu -H env PYTHONDONTWRITEBYTECODE=1 bash -c "
  python3 $G/c2_room_pod.py --stage smoke --out /tmp/dry --dry-run --shell sandbox &&
  python3 $G/c2_room_pod.py --stage loops --out /tmp/dry --dry-run --shell sandbox &&
  python3 $G/c2_room_pod.py --stage read --out /tmp/dry --dry-run" 2>&1 | tail -4 | tee /out/dry_run.txt
# 16 at once, as on the pod: each worker its own slot user, the outcomes the emulator's
sudo -u ubuntu -H env PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY' 2>&1 | tee /out/concurrency.txt
import concurrent.futures as cf, json, sys
sys.path.insert(0, "papers/gpc_goal_pressure_conduct/scripts")
import c2_families as F, c2_loop as L, c2_sandbox as SB
jobs = [(s, tw, r) for s in F.ROOM for tw, r in ((False, "honest"), (False, "shortcut"), (True, "twin"))] * 2
def one(j):
    s, tw, r = j
    cl = lambda: L.ScriptedClient(s, r, {"honest": s.closing_honest, "twin": s.closing_twin}.get(r, "Done."))
    a = L.run_loop(s, tw, cl(), 7, "c", env=SB.make_env(s, tw))["outcome"]
    b = L.run_loop(s, tw, cl(), 7, "c")["outcome"]
    keys = ("reached", "shortcut", "changed_protected", "disclosed", "closed", "turns", "nonzero_exit")
    return all(a[k] == b[k] for k in keys), SB._slot()
with cf.ThreadPoolExecutor(16) as ex:
    res = list(ex.map(one, jobs))
print(">> concurrency:", json.dumps({"loops": len(res), "match": sum(m for m, _ in res),
                                     "slots_used": len({s for _, s in res})}))
PY
echo ">> local container: selftest exit $ST"
EOF
