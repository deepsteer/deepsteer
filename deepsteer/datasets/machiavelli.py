"""MACHIAVELLI runtime fetch: the engine at a pinned commit and the game bundle, hash-verified.

MACHIAVELLI (Pan, Chan, Zou, Li, Basart, Woodside, Ng, Zhang, Emmons, Hendrycks; arXiv:2304.03279,
ICML 2023) lives at ``github.com/aypan17/machiavelli``. Its code is MIT; the game bundle
(``game_data/``, Choice of Games titles and the authors' annotations) is a password-protected zip
on Google Drive whose password the README publishes, and it states no license. Neither is vendored:
the user fetches both into the cache, and the committed manifest (``machiavelli_manifest.json``)
holds file paths and sha256 hashes only.

The engine is upstream's own code, imported from the checkout (not reimplemented). It needs the
optional extra ``deepsteer[machiavelli]`` (``gym`` 0.25.1, ``asteval``) and resolves
``./game_data`` against the working directory, so callers run it with the checkout as ``cwd``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

MACHIAVELLI_REPO = "https://github.com/aypan17/machiavelli.git"
MACHIAVELLI_COMMIT = "ebb44e197f663603427882c609dbc9d5fb883d63"
BUNDLE_DRIVE_ID = "19PXa2bgjkfFfTTI3EZIT3-IJ_vxrV0Rz"
BUNDLE_PASSWORD = b"machiavelli"  # published in the upstream README
BUNDLE_URL = (f"https://drive.usercontent.google.com/download?id={BUNDLE_DRIVE_ID}"
              "&export=download&confirm=t")

MANIFEST_PATH = Path(__file__).with_name("machiavelli_manifest.json")


def default_cache_dir(commit: str = MACHIAVELLI_COMMIT) -> Path:
    """``$DEEPSTEER_CACHE`` (default ``~/.cache/deepsteer``)/machiavelli/<sha>."""
    base = Path(os.environ.get("DEEPSTEER_CACHE", Path.home() / ".cache" / "deepsteer"))
    return base / "machiavelli" / commit


def sha256_file(path: Path) -> str:
    """sha256 hex digest of a file's bytes, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """The committed manifest: the commit, engine file hashes, the bundle zip and file hashes."""
    return json.loads(path.read_text())


def verify_files(root: Path, files: dict[str, str], what: str) -> None:
    """Check every ``files`` entry (path relative to ``root``) against its sha256.

    Raises:
        ValueError: listing every missing or mismatched file.
    """
    bad = []
    for rel, digest in sorted(files.items()):
        p = root / rel
        if not p.is_file():
            bad.append(f"missing: {rel}")
        elif sha256_file(p) != digest:
            bad.append(f"sha256 mismatch: {rel}")
    if bad:
        shown = "; ".join(bad[:20]) + (f"; and {len(bad) - 20} more" if len(bad) > 20 else "")
        raise ValueError(f"{what} at {root} does not match the manifest: {shown}")


def fetch_engine(root: Path | None = None, manifest: dict | None = None) -> Path:
    """Clone the engine at the pinned commit (once) and verify its files' hashes.

    Returns:
        The checkout directory.
    """
    manifest = manifest or load_manifest()
    commit = manifest["commit"]
    root = Path(root) if root is not None else default_cache_dir(commit) / "src"
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Cloning %s at %s into %s", MACHIAVELLI_REPO, commit[:12], root)
        subprocess.run(["git", "clone", "--quiet", MACHIAVELLI_REPO, str(root)], check=True)
        subprocess.run(["git", "-c", "advice.detachedHead=false", "checkout", "--quiet", commit],
                       cwd=root, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != commit:
        raise ValueError(f"MACHIAVELLI checkout at {root} is at {head}, expected {commit}")
    verify_files(root, manifest["engine_files"], "MACHIAVELLI engine")
    return root


def fetch_bundle(root: Path | None = None, zip_path: Path | None = None,
                 manifest: dict | None = None, verify: bool = True) -> Path:
    """Download (once), hash-check and extract the game bundle; verify every extracted file.

    Args:
        root: Directory that will hold ``game_data/`` (default: the cache's ``bundle/``).
        zip_path: An already-downloaded bundle zip; it is hash-checked like a fresh download.
        manifest: Defaults to the committed manifest.
        verify: Check every extracted file's sha256 (3,459 files; about a minute).

    Returns:
        The directory holding ``game_data/``.
    """
    manifest = manifest or load_manifest()
    bundle = manifest["bundle"]
    root = Path(root) if root is not None else default_cache_dir(manifest["commit"]) / "bundle"
    if not (root / "game_data").is_dir():
        if zip_path is None:
            zip_path = root.parent / "game_data.zip"
            if not zip_path.exists():
                zip_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading the MACHIAVELLI bundle (%d bytes)", bundle["zip_bytes"])
                urllib.request.urlretrieve(BUNDLE_URL, zip_path)
        got = sha256_file(Path(zip_path))
        if got != bundle["zip_sha256"]:
            raise ValueError(f"bundle zip {zip_path} has sha256 {got}, "
                             f"expected {bundle['zip_sha256']}")
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root, pwd=BUNDLE_PASSWORD)
    if verify:
        verify_files(root, bundle["files"], "MACHIAVELLI bundle")
    return root


def fetch_machiavelli(root: Path | None = None, zip_path: Path | None = None,
                      verify_bundle: bool = True) -> Path:
    """Fetch and verify the engine and the bundle, and link the bundle into the checkout.

    Returns:
        The engine checkout, ready to run with itself as the working directory.
    """
    manifest = load_manifest()
    base = Path(root) if root is not None else default_cache_dir(manifest["commit"])
    src = fetch_engine(base / "src", manifest)
    data = fetch_bundle(base / "bundle", zip_path, manifest, verify_bundle) / "game_data"
    link = src / "game_data"
    if not link.exists():
        link.symlink_to(data, target_is_directory=True)
    elif link.resolve() != data.resolve():
        raise ValueError(f"{link} points to {link.resolve()}, expected {data}")
    return src
