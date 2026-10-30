"""The shared RunPod launcher (papers/runpod_common/run_session.sh) and its forwarding script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NEW = REPO / "papers" / "runpod_common" / "run_session.sh"
OLD = REPO / "papers" / "d1_moral_subspace" / "runpod" / "run_session.sh"


@pytest.mark.parametrize("launcher", [NEW, OLD], ids=["shared", "forwarding"])
def test_launcher_resolves_the_repo_root(launcher, tmp_path):
    # most probable failure: REPO_ROOT keeps the old directory depth after the move, so the
    # launcher would rsync and download from the wrong directory (the print hook stops before any
    # network call)
    env = dict(os.environ, RUNPOD_API_KEY="unused", RP_PRINT_REPO_ROOT="1")
    out = subprocess.run(["bash", str(launcher)], capture_output=True, text=True, env=env,
                         cwd=tmp_path, timeout=30)
    assert out.returncode == 0, out.stderr
    assert Path(out.stdout.strip()) == REPO


def test_forwarding_script_only_forwards():
    # most probable failure: a stale copy of the launcher left at the old path, so the two drift
    text = OLD.read_text()
    assert "exec " in text and "runpod_common" in text and "rp_provision_pod" not in text
