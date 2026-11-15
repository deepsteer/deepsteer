"""papers/runpod_common/provider_thunder.sh (CST Amendment 51 item D), against a stand-in `tnr`."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "papers/runpod_common"

FAKE = r"""#!/usr/bin/env bash
# a stand-in for Thunder's CLI: the account's instances in $TNR_STATE (a JSON list)
S="$TNR_STATE"; [ -f "$S" ] || echo '[]' > "$S"
case "$1" in
  status) cat "$S" ;;
  create)
    n=$(cat "$TNR_STATE.fails" 2>/dev/null || echo 0)
    if [ "$n" -gt 0 ]; then echo $((n - 1)) > "$TNR_STATE.fails"
      echo "Error: failed to create instance: We're currently out of GPU capacity" >&2; exit 1; fi
    id=$(jq 'length' "$S"); u="uuid$id$RANDOM"
    jq --arg u "$u" --argjson id "$id" '. + [{id: ($id|tostring), uuid: $u, status: "RUNNING",
      ip: "10.0.0.1", port: 40000}]' "$S" > "$S.t" && mv "$S.t" "$S"
    echo "Creating instance..." >&2
    printf '{"identifier": %s, "uuid": "%s", "key": null}\n' "$id" "$u" ;;
  delete)
    [ "${TNR_DELETE_NOOP:-0}" = 1 ] && exit 0
    jq --arg id "$2" 'map(select(.id != $id))' "$S" > "$S.t" && mv "$S.t" "$S" ;;
  connect)
    jq -r --arg id "$2" '.[] | select(.id==$id) | {ip, port, key_file: "/tmp/k"}' "$S" ;;
esac
"""


def _env(tmp_path: Path) -> dict:
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    (b / "tnr").write_text(FAKE)
    (b / "tnr").chmod(0o755)
    return dict(os.environ, PATH=f"{b}:{os.environ['PATH']}", TNR_STATE=str(tmp_path / "s.json"),
                TNR_CREATE_WAIT_S="0", DISK_GB="300", REPO_ROOT=str(tmp_path),
                SELF_PAPER="p", RESULTS_SUBPATH="o")


def _bash(script: str, env: dict) -> subprocess.CompletedProcess:
    pre = f'source "{LIB}/session_lib.sh"; source "{LIB}/provider_thunder.sh"; '
    return subprocess.run(["bash", "-c", pre + script], env=env, capture_output=True, text=True)


needs_jq = pytest.mark.skipif(shutil.which("jq") is None, reason="jq is not installed")


@needs_jq
def test_create_retries_on_no_capacity_and_reads_the_uuid(tmp_path):
    # most probable failure (Amendment 51 item D2): a refused create (Thunder's "out of GPU
    # capacity", twice on 2026-11-14) ends the launch, or the session keeps no UUID for teardown
    env = _env(tmp_path)
    (tmp_path / "s.json.fails").write_text("2")
    r = _bash('rp_provision_pod; echo "ID=$POD_ID UUID=$POD_UUID"; rp_log_rate', env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "attempt 3/6" in r.stdout and "ID=0 UUID=uuid0" in r.stdout
    rec = json.loads((tmp_path / "p/o/logs/pod_rate.jsonl").read_text())
    assert rec["provider"] == "thunder" and len(rec["attempts"]) == 3


@needs_jq
def test_teardown_deletes_only_its_own_uuid_and_verifies_the_account(tmp_path):
    # most probable failure (Amendment 51 item D3): teardown deletes by id an instance this session
    # did not create (ids are reused), or reports success while an instance still bills
    env = _env(tmp_path)
    st = tmp_path / "s.json"
    st.write_text(json.dumps([{"id": "0", "uuid": "theirs", "status": "RUNNING"}]))
    r = _bash("rp_provision_pod; cleanup; echo rc=$?; jq -c . \"$TNR_STATE\"", env)
    assert "rc=0" in r.stdout and "Teardown verified" in r.stdout, r.stdout + r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == [
        {"id": "0", "uuid": "theirs", "status": "RUNNING"}]  # the author's own instance stays
    st.write_text("[]")
    r = _bash('rp_provision_pod; POD_UUID=someone_else; cleanup; echo rc=$?', env)
    assert "NOT deleting" in r.stdout and "rc=1" in r.stdout  # the id now names another UUID
    st.write_text("[]")
    env["TNR_DELETE_NOOP"] = "1"
    r = _bash('rp_provision_pod; sleep(){ :; }; cleanup; echo rc=$?', env)
    assert "TEARDOWN NOT VERIFIED" in r.stdout and "rc=1" in r.stdout


def test_run_session_selects_the_provider_and_keeps_runpod_the_default():
    # most probable failure (Amendment 51 item D1): the switch changes RunPod's path (the fallback)
    # or a typo in PROVIDER silently runs RunPod
    text = (LIB / "run_session.sh").read_text()
    assert 'PROVIDER="${PROVIDER:-runpod}"' in text and "unknown PROVIDER" in text
    assert 'thunder) source "$REPO_ROOT/papers/runpod_common/provider_thunder.sh"' in text
    assert "root@" not in text and "root@" not in (LIB / "session_lib.sh").read_text()
