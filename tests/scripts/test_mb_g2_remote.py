"""MB-G2 remote runner (papers/mb_moralbench/runpod/remote_mb_g2.sh) run in its local dry mode."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "papers" / "mb_moralbench" / "runpod" / "remote_mb_g2.sh"


def _zephyr_cached() -> bool:
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta",
                                      revision="892b3d7a7b1cf10c7a701c60881cd93df615734c",
                                      local_files_only=True)
        from deepsteer.datasets.moralbench import default_cache_dir
        return default_cache_dir().exists()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _zephyr_cached(), reason="needs the Zephyr tokenizer and MoralBench cached")
def test_local_dry_sequence_writes_inside_the_results_path(tmp_path):
    # most probable failure: a unit writes outside RESULTS_SUBPATH (outputs/g2), which the launcher
    # downloads and nothing else, so its results die with the pod (KDG and ST both lost runs this
    # way); second: a model runs other than its registered cells
    env = dict(os.environ, MB_LOCAL_DRY="1", PY=sys.executable, REPO_DIR=str(REPO),
               OUT_ROOT=str(tmp_path), SENTINEL_DIR=str(tmp_path), SESSION_MAX_HOURS="0.5",
               HF_HUB_OFFLINE="1")
    rc = subprocess.run(["bash", str(RUNNER)], capture_output=True, text=True, env=env, cwd=REPO,
                        timeout=900)
    assert rc.returncode == 0, rc.stdout[-3000:] + rc.stderr[-2000:]
    g2 = tmp_path / "outputs" / "g2"
    cells = {m: sorted(p.stem for p in (g2 / m).glob("*.jsonl") if p.stem != "r2")
             for m in ("gemma", "zephyr_gemma", "llama2_70b")}
    assert cells == {"gemma": ["cb07", "t07", "t10"], "zephyr_gemma": ["t07"],
                     "llama2_70b": ["cb07", "t07", "t10"]}
    assert (g2 / "_raw" / "gemma" / "t07.jsonl").exists()
    assert not (tmp_path / "outputs_raw").exists()  # raw replies stay inside outputs/g2
    files = json.loads((g2 / "manifest_g2.json").read_text())["files"]
    assert "llama2_70b/t07.jsonl" in files and "_raw/gemma/t07.jsonl" in files
    assert (tmp_path / ".session_done").exists()
    meta = json.loads((g2 / "llama2_70b" / "run_meta.json").read_text())["runs"][-1]
    assert meta["out_sub"] == "g2" and meta["deadline_epoch"] is not None


def test_pod_install_path_pins_the_g1_stack_and_installs_pytest():
    # most probable failures, both found by the first VALIDATE (2026-10-30): the launcher's
    # forwarded TRANSFORMERS_VERSION (default 5.12.1) overrides the G1 stack's 5.17.0, and the gate
    # tests bail because `pip install -e .` brings no pytest; the local dry mode skips this branch
    text = RUNNER.read_text()
    assert 'MB_TF="${MB_TRANSFORMERS_VERSION:-5.17.0}"' in text
    assert '"transformers==$MB_TF"' in text and "TRANSFORMERS_VERSION:=" not in text
    install = text[text.index('"transformers==$MB_TF"'):].split("\n")[0]
    assert "pytest" in install
    assert text.index("pytest \\") < text.index("python -m pytest")
    assert "transformers.__version__ != '$MB_TF'" in text  # the version is asserted, not assumed


def test_stack_check_asks_transformers_whether_torch_is_usable():
    # most probable failure (the second VALIDATE, 2026-10-30): transformers 5.17 silently disables
    # torch < 2.5, an import check still passes, the torch 2.6 fallback never runs, and the first
    # model fails; the check must ask is_torch_available() before and after the fallback
    text = RUNNER.read_text()
    assert "assert is_torch_available() and torch.cuda.is_available()" in text
    assert text.count('"$TORCH_OK') == 2
