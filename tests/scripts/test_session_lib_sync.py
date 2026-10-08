"""rp_output_filters / rp_check_sync_size in papers/runpod_common/session_lib.sh.

Run under /bin/bash with ``set -euo pipefail`` as the launcher does (macOS: bash 3.2).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "papers" / "runpod_common" / "session_lib.sh"

pytestmark = pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")


def _run(tmp_path: Path, self_paper: str, sync_outputs: str | None, max_gb: str) -> tuple:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "code.py").write_text("x = 1\n")
    out = tmp_path / self_paper / "outputs" / "p2c" / "validate"
    out.mkdir(parents=True)
    (out / "timing.json").write_text("{}")
    (out / "big.bin").write_bytes(b"\0" * 2_000_000)
    excl = tmp_path / "excl.txt"
    excl.write_text(f"/{self_paper}/outputs/\n")  # the KDG-style universal exclude
    env_so = "" if sync_outputs is None else f"SYNC_OUTPUTS='{sync_outputs}'"
    script = f"""
set -euo pipefail
REPO_ROOT='{tmp_path}'; RSYNC_EXCLUDE='{excl}'; SELF_PAPER='{self_paper}'; MAX_SYNC_GB={max_gb}
{env_so}
eval "$(sed -n '/^rp_output_filters()/,/^}}/p;/^rp_check_sync_size()/,/^}}/p' '{LIB}')"
rp_output_filters
rp_check_sync_size
cd "$REPO_ROOT" && rsync -an --out-format='%n' ${{RP_PRE_FILTERS[@]+"${{RP_PRE_FILTERS[@]}}"}} \\
  --exclude-from "$RSYNC_EXCLUDE" "${{RP_OUT_FILTERS[@]}}" ./ "$(mktemp -d)/"
"""
    r = subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def test_named_file_crosses_the_universal_exclude_and_nothing_else_does(tmp_path):
    # most probable failure: the p2c timing record never reaches the real-run pod (blocked by the
    # KDG outputs exclude), or naming it drags the rest of the validate directory along
    rc, out, err = _run(tmp_path, "papers/kdg", "outputs/p2c/validate/timing.json", "1")
    assert rc == 0, err
    assert "papers/kdg/outputs/p2c/validate/timing.json" in out
    assert "big.bin" not in out


def test_empty_pre_filters_survive_set_u_on_bash_32(tmp_path):
    # most probable failure: an empty RP_PRE_FILTERS expands as 'unbound variable' under set -u
    # (bash 3.2), aborting every launch that does not set SYNC_OUTPUTS
    rc, out, err = _run(tmp_path, "papers/kdg", None, "1")
    assert rc == 0, err
    assert "unbound" not in err and "big.bin" not in out


def test_size_guard_refuses_an_oversized_sync(tmp_path):
    # most probable failure: the guard passes an oversized sync (named file is 2 MB, limit 1 MB)
    rc, out, err = _run(tmp_path, "papers/kdg", "outputs/p2c/validate/big.bin", "0.001")
    assert rc != 0 and "MAX_SYNC_GB" in out
