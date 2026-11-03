"""rp_on_demand_sync in papers/runpod_common/session_lib.sh (opt-in ON_DEMAND_SYNC).

Run under /bin/bash with ``set -euo pipefail`` as the launcher does (macOS: bash 3.2), with stub
``ssh`` and ``rsync`` that record their arguments instead of touching a network.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "papers" / "runpod_common" / "session_lib.sh"


def _run(tmp_path: Path, chunk: str, on_demand: str | None) -> tuple[int, str, list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "calls.log"
    for tool in ("ssh", "rsync"):
        stub = bin_dir / tool
        stub.write_text(f'#!/bin/bash\necho "{tool} $*" >> "{log}"\n')
        stub.chmod(0o755)
    env_line = "" if on_demand is None else f"ON_DEMAND_SYNC='{on_demand}'"
    script = f"""
set -euo pipefail
export PATH='{bin_dir}':"$PATH"
SSH_OPTS=(-p 2222 -o StrictHostKeyChecking=no); SSH_HOST=10.0.0.1
{env_line}
eval "$(sed -n '/^RP_SYNCED=/p;/^rp_on_demand_sync()/,/^}}/p' '{LIB}')"
rp_on_demand_sync "$1"
rp_on_demand_sync "$1"
echo END
"""
    r = subprocess.run(["/bin/bash", "-c", script, "x", chunk], capture_output=True, text=True)
    calls = log.read_text().splitlines() if log.exists() else []
    return r.returncode, r.stdout + r.stderr, calls


def _bundle(tmp_path: Path) -> tuple[Path, str]:
    f = tmp_path / "game_data.zip"
    f.write_bytes(b"invented bundle bytes")
    return f, hashlib.sha256(f.read_bytes()).hexdigest()


def test_pushes_once_after_the_hash_check(tmp_path):
    # most probable failure: a request that is never answered (the pod waits 20 min and stops), or
    # one answered twice when the same log chunk is re-read
    f, h = _bundle(tmp_path)
    rc, out, calls = _run(tmp_path, "setup\nNEED_SYNC mch_bundle\nwaiting",
                          f"mch_bundle|{f}|/workspace/mch_bundle/game_data.zip|{h}")
    assert rc == 0 and "END" in out, out
    rsyncs = [c for c in calls if c.startswith("rsync")]
    assert len(rsyncs) == 1
    assert str(f) in rsyncs[0] and "root@10.0.0.1:/workspace/mch_bundle/game_data.zip" in rsyncs[0]
    assert any(c.startswith("ssh") and "mkdir -p '/workspace/mch_bundle'" in c for c in calls)


def test_a_local_file_with_the_wrong_hash_is_not_pushed(tmp_path):
    # most probable failure: pushing whatever sits at the local path (a re-downloaded or edited
    # zip) because only the pod checks the hash
    f, _ = _bundle(tmp_path)
    rc, out, calls = _run(tmp_path, "NEED_SYNC mch_bundle",
                          f"mch_bundle|{f}|/workspace/x.zip|{'0' * 64}")
    assert rc == 0 and "END" in out
    assert not calls and "not pushed" in out


def test_unknown_names_and_unset_config_push_nothing(tmp_path):
    f, h = _bundle(tmp_path)
    rc, out, calls = _run(tmp_path, "NEED_SYNC other_file", f"mch_bundle|{f}|/workspace/x|{h}")
    assert rc == 0 and not calls and "does not list" in out
    rc, out, calls = _run(tmp_path, "NEED_SYNC mch_bundle", None)
    assert rc == 0 and "END" in out and not calls


def test_lines_that_only_mention_need_sync_are_ignored(tmp_path):
    f, h = _bundle(tmp_path)
    rc, _, calls = _run(tmp_path, ">> will print NEED_SYNC mch_bundle if Drive fails",
                        f"mch_bundle|{f}|/workspace/x|{h}")
    assert rc == 0 and not calls
