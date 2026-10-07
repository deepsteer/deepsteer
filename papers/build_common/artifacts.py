"""Resolve paper binary artifacts: in-tree if present, else the Zenodo bundle.

LIBRARY_RELEASE_PLAN §D.

Binary artifacts (.npz arrays, adapter tokenizers) are listed in ``papers/ARTIFACT_MANIFEST.json``
with their sha256. ``get(path)`` returns a local file whose hash matches the manifest:

1. the file in the working tree (present at or before the commit tagged ``artifacts-last-in-tree``);
2. otherwise a verified copy in the cache, ``papers/_artifacts/`` (gitignored; override with
   ``DEEPSTEER_ARTIFACT_CACHE``);
3. otherwise the path's bundle is downloaded from the Zenodo record named in the manifest, the
   bundle's sha256 is checked, and only the files the manifest assigns to that bundle are extracted
   (tar member names are matched against the manifest, never trusted as
   paths).

Usage from a paper script (papers/ is not a package)::

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "build_common"))
    from artifacts import get as get_artifact
    path = "papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz"
    data = np.load(get_artifact(path))

Command line: ``python3 papers/build_common/artifacts.py fetch [PATH ...]`` fetches the named
artifacts (all of them with no PATH); ``verify`` checks every in-tree artifact against the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "papers" / "ARTIFACT_MANIFEST.json"
ZENODO_FILE_URL = "https://zenodo.org/api/records/{record}/files/{name}/content"


def _cache_dir() -> Path:
    return Path(os.environ.get("DEEPSTEER_ARTIFACT_CACHE", REPO / "papers" / "_artifacts"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    """Return the parsed artifact manifest."""
    return json.loads(MANIFEST.read_text())


def _rel(path: str | os.PathLike) -> str:
    p = Path(path)
    if p.is_absolute():
        p = p.resolve().relative_to(REPO)
    return p.as_posix()


def fetch_command(path: str | os.PathLike) -> str:
    """The shell command that fetches ``path``; for skip and error messages."""
    return f"python3 papers/build_common/artifacts.py fetch {_rel(path)}"


def get(path: str | os.PathLike) -> Path:
    """Return a local copy of the artifact at repo-relative ``path``, verified against the manifest.

    Raises:
        KeyError: if ``path`` is not in the manifest.
        FileNotFoundError: if the file is not in the tree or cache and the manifest names no
            Zenodo record yet.
        ValueError: if a downloaded bundle or an extracted file fails its sha256 check.
    """
    rel = _rel(path)
    manifest = load_manifest()
    entries = {a["path"]: a for a in manifest["artifacts"]}
    if rel not in entries:
        raise KeyError(f"{rel} is not in {MANIFEST.relative_to(REPO)}")
    entry = entries[rel]

    for candidate in (REPO / rel, _cache_dir() / rel):
        if candidate.is_file() and _sha256(candidate) == entry["sha256"]:
            return candidate

    if entry.get("bundle") is None:
        raise FileNotFoundError(
            f"{rel} is not deposited ({entry.get('deposit', 'excluded')}); regenerate it: "
            f"{entry.get('regeneration', 'see the manifest entry')}"
        )
    record = manifest["record"].get("zenodo_record")
    if not record:
        raise FileNotFoundError(
            f"{rel} is not in the working tree and its Zenodo record is not published yet. "
            "Check out the commit tagged artifacts-last-in-tree to read it from git."
        )
    _fetch_bundle(manifest, entry["bundle"], record)
    cached = _cache_dir() / rel
    if not cached.is_file() or _sha256(cached) != entry["sha256"]:
        raise ValueError(f"{rel}: extracted file does not match the manifest sha256")
    return cached


def _fetch_bundle(manifest: dict, bundle: str, record: str) -> None:
    """Download ``bundle`` from the Zenodo record, verify it, extract its manifest members."""
    expected = manifest["bundles"][bundle]["sha256"]
    members = {a["path"]: a["sha256"] for a in manifest["artifacts"] if a["bundle"] == bundle}
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    url = ZENODO_FILE_URL.format(record=record, name=bundle)
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".tar", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        print(f"downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as out:
            shutil.copyfileobj(response, out)
        if _sha256(tmp_path) != expected:
            raise ValueError(f"{bundle}: download does not match the manifest sha256")
        with tarfile.open(tmp_path) as tar:
            for member in tar.getmembers():
                if member.name not in members or not member.isfile():
                    continue
                target = cache / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    finally:
        tmp_path.unlink(missing_ok=True)


def _main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("fetch", "verify"):
        print(__doc__)
        return 2
    manifest = load_manifest()
    if argv[0] == "verify":
        bad = [a["path"] for a in manifest["artifacts"]
               if (REPO / a["path"]).is_file() and _sha256(REPO / a["path"]) != a["sha256"]]
        present = sum((REPO / a["path"]).is_file() for a in manifest["artifacts"])
        total = len(manifest["artifacts"])
        print(f"{present} of {total} artifacts in tree; mismatches: {bad or 'none'}")
        return 1 if bad else 0
    paths = argv[1:] or [a["path"] for a in manifest["artifacts"]]
    for p in paths:
        print(get(p))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
