"""Build the artifact manifest and the per-paper Zenodo bundles (LIBRARY_RELEASE_PLAN §D).

    python3 papers/build_common/build_artifact_bundle.py inventory  # manifest core fields
    python3 papers/build_common/build_artifact_bundle.py bundle     # tars + bundle hashes

``inventory`` lists every git-tracked binary under papers/ (.npz/.npy/.pt/.bin/.safetensors/.ckpt
and adapter tokenizer.json) with bytes and sha256, assigns each to its paper's bundle, and keeps any
per-artifact fields already in the manifest (producer, readers, ...). Adapter files are never
deposited (supplement RELEASE_PLAN §2) and get no bundle. ``also_in`` names the FL/MN deposit
(10.5281/zenodo.22731361) for files it already holds byte-identically.

``bundle`` writes one uncompressed tar per paper to papers/_artifacts/upload/ (arrays are already
compressed). Tars are deterministic (sorted members, mtime 0, uid/gid 0, fixed mode), so a rebuild
from the same tree reproduces the bundle sha256 recorded in the manifest.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "papers" / "ARTIFACT_MANIFEST.json"
UPLOAD = REPO / "papers" / "_artifacts" / "upload"
BINARY = re.compile(r"\.(npz|npy|pt|bin|safetensors|ckpt)$|/tokenizer\.json$")
FL_MN_DEPOSIT = "10.5281/zenodo.22731361"
BUNDLE_VERSION = "v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _bundle_for(path: str) -> str | None:
    if "/adapter/" in path:
        return None
    paper = path.split("/")[1]
    return f"deepsteer_papers_{paper}_{BUNDLE_VERSION}.tar"


def inventory(deposit_manifest: Path | None) -> dict:
    """Rewrite core fields for every tracked binary, keeping existing per-artifact metadata."""
    tracked = subprocess.run(["git", "ls-files", "papers"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout.split("\n")
    paths = sorted(p for p in tracked if BINARY.search(p))
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    old_entries = {a["path"]: a for a in old.get("artifacts", [])}
    deposited = {}
    if deposit_manifest is not None:
        for e in json.loads(deposit_manifest.read_text()).get("files", []):
            deposited[e["path"]] = e["sha256"]
    artifacts = []
    for p in paths:
        sha = _sha256(REPO / p)
        entry = dict(old_entries.get(p, {}))
        entry.update(path=p, bytes=(REPO / p).stat().st_size, sha256=sha, bundle=_bundle_for(p))
        if entry["bundle"] is None:
            entry["deposit"] = "excluded: LoRA adapter files are never deposited"
        if deposited.get(p) == sha:
            entry["also_in"] = FL_MN_DEPOSIT
        artifacts.append(entry)
    manifest = {
        "description": "Binary artifacts removed from the papers/ tree; resolve with "
                       "papers/build_common/artifacts.py get(path).",
        "record": old.get("record", {"doi": None, "zenodo_record": None, "license": "CC-BY-4.0"}),
        "last_in_tree": old.get("last_in_tree"),
        "bundles": old.get("bundles", {}),
        "artifacts": artifacts,
    }
    return manifest


def bundle(manifest: dict) -> dict:
    """Write deterministic per-paper tars and record their sha256 and size."""
    UPLOAD.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[str]] = {}
    for a in manifest["artifacts"]:
        if a["bundle"]:
            groups.setdefault(a["bundle"], []).append(a["path"])
    bundles = {}
    for name, members in sorted(groups.items()):
        out = UPLOAD / name
        with tarfile.open(out, "w", format=tarfile.PAX_FORMAT) as tar:
            for p in sorted(members):
                data = (REPO / p).read_bytes()
                info = tarfile.TarInfo(p)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(data))
        bundles[name] = {"sha256": _sha256(out), "bytes": out.stat().st_size, "files": len(members)}
        print(f"{name}: {len(members)} files, {out.stat().st_size / 1e6:.1f} MB")
    manifest["bundles"] = bundles
    return manifest


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("inventory", "bundle"):
        print(__doc__)
        return 2
    if argv[0] == "inventory":
        dep = Path(argv[1]) if len(argv) > 1 else None
        manifest = inventory(dep)
    else:
        manifest = bundle(json.loads(MANIFEST.read_text()))
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {MANIFEST.relative_to(REPO)} ({len(manifest['artifacts'])} artifacts)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
