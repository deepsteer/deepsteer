"""MACHIAVELLI fetch verification on synthetic files (no upstream code or game data)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from deepsteer.datasets import machiavelli as mch


def _bundle_zip(tmp_path: Path) -> tuple[Path, dict]:
    """A zip shaped like the bundle (``game_data/...``) and a manifest that matches it."""
    src = tmp_path / "made"
    (src / "game_data" / "scenarios").mkdir(parents=True)
    (src / "game_data" / "game_metadata.json").write_text(json.dumps({"g": {"name": "G"}}))
    (src / "game_data" / "scenarios" / "g.json").write_text('{"s1": "invented scene"}')
    zp = tmp_path / "game_data.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(src))
    files = {str(p.relative_to(src)): mch.sha256_file(p)
             for p in sorted(src.rglob("*")) if p.is_file()}
    manifest = {"commit": "0" * 40, "engine_files": {},
                "bundle": {"zip_sha256": mch.sha256_file(zp), "zip_bytes": zp.stat().st_size,
                           "files": files}}
    return zp, manifest


def test_committed_manifest_covers_engine_and_bundle():
    m = mch.load_manifest()
    assert m["commit"] == mch.MACHIAVELLI_COMMIT
    # the engine modules the harness imports must be pinned, or a changed engine passes silently
    for rel in ("machiavelli/game/player.py", "machiavelli/game/machiavelli_env.py",
                "machiavelli/config.py", "machiavelli/utils.py", "LICENSE"):
        assert rel in m["engine_files"]
    assert m["bundle"]["n_files"] == len(m["bundle"]["files"]) == 3459
    assert "game_data/normalization_coeffs.json" in m["bundle"]["files"]


def test_bundle_extracts_and_verifies(tmp_path):
    zp, manifest = _bundle_zip(tmp_path)
    root = mch.fetch_bundle(tmp_path / "bundle", zip_path=zp, manifest=manifest)
    assert (root / "game_data" / "game_metadata.json").is_file()


def test_bundle_zip_with_wrong_hash_is_refused_before_extraction(tmp_path):
    # most probable failure: a re-uploaded or truncated Drive file extracted without a check
    zp, manifest = _bundle_zip(tmp_path)
    manifest["bundle"]["zip_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="sha256"):
        mch.fetch_bundle(tmp_path / "bundle", zip_path=zp, manifest=manifest)
    assert not (tmp_path / "bundle" / "game_data").exists()


def test_tampered_bundle_file_is_named(tmp_path):
    # most probable failure: an edited or partially extracted file that still lets the engine run
    zp, manifest = _bundle_zip(tmp_path)
    root = mch.fetch_bundle(tmp_path / "bundle", zip_path=zp, manifest=manifest)
    (root / "game_data" / "scenarios" / "g.json").write_text('{"s1": "changed"}')
    with pytest.raises(ValueError, match="sha256 mismatch: game_data/scenarios/g.json"):
        mch.fetch_bundle(root, manifest=manifest)


def test_missing_bundle_file_is_named(tmp_path):
    zp, manifest = _bundle_zip(tmp_path)
    root = mch.fetch_bundle(tmp_path / "bundle", zip_path=zp, manifest=manifest)
    (root / "game_data" / "game_metadata.json").unlink()
    with pytest.raises(ValueError, match="missing: game_data/game_metadata.json"):
        mch.verify_files(root, manifest["bundle"]["files"], "bundle")


def test_engine_at_wrong_commit_is_refused(tmp_path):
    # most probable failure: a stale checkout on another commit reused from the cache
    import subprocess

    repo = tmp_path / "src"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    (repo / "f.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "f.py"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet",
                    "-m", "c"], cwd=repo, check=True)
    manifest = {"commit": "0" * 40, "engine_files": {}, "bundle": {}}
    with pytest.raises(ValueError, match="expected 0000"):
        mch.fetch_engine(repo, manifest)
