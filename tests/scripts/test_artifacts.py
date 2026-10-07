"""papers/build_common/artifacts.py: resolve binary artifacts from the tree, the cache or Zenodo."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "papers" / "build_common"))
import artifacts  # noqa: E402

REL = "papers/x/outputs/a.npz"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tar(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake repo whose artifact lives only in a bundle served from a file:// 'Zenodo'."""
    root, zenodo = tmp_path / "repo", tmp_path / "zenodo"
    (root / "papers").mkdir(parents=True)
    zenodo.mkdir()
    payload = b"array bytes"
    bundle = _tar({REL: payload, "../escape.npz": b"x"})
    (zenodo / "b.tar").write_bytes(bundle)
    manifest = {
        "record": {"zenodo_record": "123"},
        "bundles": {"b.tar": {"sha256": _sha(bundle)}},
        "artifacts": [
            {"path": REL, "sha256": _sha(payload), "bundle": "b.tar"},
            {"path": "papers/x/adapter/tokenizer.json", "sha256": "0", "bundle": None,
             "regeneration": "AutoTokenizer.from_pretrained(base)"},
        ],
    }
    (root / "papers" / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(artifacts, "REPO", root)
    monkeypatch.setattr(artifacts, "MANIFEST", root / "papers" / "ARTIFACT_MANIFEST.json")
    monkeypatch.setattr(artifacts, "ZENODO_FILE_URL", f"file://{zenodo}/" + "{name}")
    monkeypatch.setenv("DEEPSTEER_ARTIFACT_CACHE", str(root / "papers" / "_artifacts"))
    return root, zenodo, payload


def test_fetches_verifies_and_extracts_only_manifest_members(repo):
    root, _zenodo, payload = repo
    got = artifacts.get(REL)
    assert got == root / "papers" / "_artifacts" / REL and got.read_bytes() == payload
    assert not (root / "papers" / "escape.npz").exists()  # tar names are never trusted as paths
    assert artifacts.get(root / REL) == got  # absolute paths resolve to the same entry


def test_bundle_hash_is_checked_before_extraction(repo):
    # Most probable failure: a download used without checking it against the manifest sha256.
    # The member stays valid here, so only the bundle-level check can catch the substitution.
    _root, zenodo, payload = repo
    (zenodo / "b.tar").write_bytes(_tar({REL: payload, "papers/x/outputs/extra.npz": b"y"}))
    with pytest.raises(ValueError, match="download does not match"):
        artifacts.get(REL)


def test_member_hash_is_checked_after_extraction(repo):
    # A manifest whose bundle hash was recorded for a bad member: the per-file check catches it.
    _root, zenodo, _payload = repo
    bad = _tar({REL: b"tampered"})
    (zenodo / "b.tar").write_bytes(bad)
    manifest = json.loads(artifacts.MANIFEST.read_text())
    manifest["bundles"]["b.tar"]["sha256"] = _sha(bad)
    artifacts.MANIFEST.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="extracted file does not match"):
        artifacts.get(REL)


def test_tree_copy_wins_and_unpublished_record_is_explicit(repo):
    root, _zenodo, payload = repo
    (root / REL).parent.mkdir(parents=True)
    (root / REL).write_bytes(payload)
    assert artifacts.get(REL) == root / REL
    (root / REL).unlink()
    manifest = json.loads(artifacts.MANIFEST.read_text())
    manifest["record"]["zenodo_record"] = None
    artifacts.MANIFEST.write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifacts-last-in-tree"):
        artifacts.get(REL)


def test_excluded_artifact_names_its_regeneration(repo):
    with pytest.raises(FileNotFoundError, match="AutoTokenizer"):
        artifacts.get("papers/x/adapter/tokenizer.json")
