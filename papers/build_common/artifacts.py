"""Resolve paper binary artifacts: in-tree if present, else the Zenodo bundle.

LIBRARY_RELEASE_PLAN §D.

Binary artifacts (.npz arrays, adapter tokenizers) are listed in ``papers/ARTIFACT_MANIFEST.json``
with their sha256. ``get(path)`` returns a local file whose hash matches the manifest:

1. the file in the working tree (present at or before the commit tagged ``artifacts-last-in-tree``);
2. otherwise a verified copy in the cache, ``papers/_artifacts/`` (gitignored; override with
   ``DEEPSTEER_ARTIFACT_CACHE``);
3. otherwise the path's bundle is taken from ``<cache>/<bundle>`` if placed there by hand, or
   downloaded from its public Zenodo record (restricted bundles need an access request), the
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


def resolve(path: str | os.PathLike, missing_ok: bool = False) -> str | os.PathLike:
    """``get(path)`` if ``path`` is a manifest artifact, else ``path`` unchanged.

    For load sites that receive paths from argv or constants: a relative path is taken relative
    to the current directory (scripts run from the repo root), and any path outside the
    manifest, such as a fresh output directory, passes through untouched. With
    ``missing_ok=True`` an artifact that cannot be obtained (restricted, unpublished) is reported
    on stderr and the original path is returned, so a caller's own exists()/skip logic applies.
    """
    p = Path(path).resolve()
    try:
        rel = p.relative_to(REPO).as_posix()
    except ValueError:
        return path
    if rel not in {a["path"] for a in load_manifest()["artifacts"]}:
        return path
    try:
        return get(rel)
    except FileNotFoundError as exc:
        if not missing_ok:
            raise
        print(f"[artifacts] {exc}", file=sys.stderr)
        return path


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
    _fetch_bundle(manifest, entry["bundle"])
    cached = _cache_dir() / rel
    if not cached.is_file() or _sha256(cached) != entry["sha256"]:
        raise ValueError(f"{rel}: extracted file does not match the manifest sha256")
    return cached


def _fetch_bundle(manifest: dict, bundle: str) -> None:
    """Find or download ``bundle``, verify it, and extract its manifest members.

    A bundle already placed at ``<cache>/<bundle>`` (for example a restricted bundle obtained
    through a Zenodo access request) is used instead of downloading.
    """
    meta = manifest["bundles"][bundle]
    record_name = meta.get("record", "public")
    record = manifest["records"][record_name]
    members = {a["path"]: a["sha256"] for a in manifest["artifacts"] if a["bundle"] == bundle}
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    placed = cache / bundle
    downloaded = None
    if placed.is_file():
        source = placed
    elif not record.get("zenodo_record"):
        raise FileNotFoundError(
            f"{bundle} is not cached and its Zenodo record ({record_name}) is not published yet. "
            "Check out the commit tagged artifacts-last-in-tree to read it from git."
        )
    elif record_name == "restricted":
        raise FileNotFoundError(
            f"{bundle} is in a restricted Zenodo record: request access at "
            f"https://zenodo.org/records/{record['zenodo_record']}, then save the file as {placed}"
        )
    else:
        url = ZENODO_FILE_URL.format(record=record["zenodo_record"], name=bundle)
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".tar", delete=False) as tmp:
            downloaded = source = Path(tmp.name)
        print(f"downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url) as response, open(source, "wb") as out:
            shutil.copyfileobj(response, out)
    try:
        if _sha256(source) != meta["sha256"]:
            raise ValueError(f"{bundle}: download does not match the manifest sha256")
        with tarfile.open(source) as tar:
            for member in tar.getmembers():
                if member.name not in members or not member.isfile():
                    continue
                target = cache / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    finally:
        if downloaded is not None:
            downloaded.unlink(missing_ok=True)


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
