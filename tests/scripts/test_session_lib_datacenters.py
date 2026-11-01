"""DATA_CENTERS in papers/runpod_common/session_lib.sh: the deploy mutation's dataCenterIds.

Run under /bin/bash with ``set -euo pipefail`` as the launcher does (macOS: bash 3.2); ``gql`` is
stubbed, so no RunPod call is made.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "papers" / "runpod_common" / "session_lib.sh"


def _mutation(tmp_path: Path, data_centers: str | None) -> str:
    key = tmp_path / "id_test"
    key.with_suffix(".pub").write_text("ssh-ed25519 AAAA test\n")
    sent = tmp_path / "sent.txt"
    dc = "" if data_centers is None else f"DATA_CENTERS='{data_centers}'"
    reply = '{"data":{"podFindAndDeployOnDemand":{"id":"p1"}}}'
    script = f"""
set -euo pipefail
source "{LIB}"
gql() {{ printf '%s' "$1" > "{sent}"; echo '{reply}'; }}
SSH_KEY="{key}" CLOUD_TYPES=SECURE GPU_TYPES="NVIDIA A100-SXM4-80GB" VOLUME_GB=0 DISK_GB=10
POD_NAME=t IMAGE=img {dc}
rp_provision_pod
echo "POD=$POD_ID"
"""
    r = subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0 and "POD=p1" in r.stdout, r.stderr
    return sent.read_text()


def test_data_centers_reach_the_deploy_mutation(tmp_path):
    # most probable failure: the ids dropped or unquoted (a GraphQL syntax error RunPod reports as
    # no capacity), or spaces after the commas kept inside an id
    m = _mutation(tmp_path, "EUR-IS-1, US-MD-1")
    assert 'dataCenterIds: ["EUR-IS-1", "US-MD-1"]' in m
    assert 'gpuTypeId: "NVIDIA A100-SXM4-80GB"' in m


def test_unset_data_centers_leave_the_mutation_as_before(tmp_path):
    # most probable failure: an empty dataCenterIds: [] sent when the variable is unset, which
    # would restrict placement to no data center at all
    assert "dataCenterIds" not in _mutation(tmp_path, None)
