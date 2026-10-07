"""Build the artifact manifest and the per-paper Zenodo bundles (LIBRARY_RELEASE_PLAN §D).

    python3 papers/build_common/build_artifact_bundle.py inventory  # manifest core fields
    python3 papers/build_common/build_artifact_bundle.py bundle     # tars + bundle hashes
    python3 papers/build_common/build_artifact_bundle.py stubs SHA  # per-directory stub READMEs

``inventory`` lists every git-tracked binary under papers/ (.npz/.npy/.pt/.bin/.safetensors/.ckpt
and adapter tokenizer.json) with bytes and sha256, assigns each to its paper's bundle, and keeps any
per-artifact fields already in the manifest (producer, readers, ...). Adapter files are never
deposited (supplement RELEASE_PLAN §2) and get no bundle. ``also_in`` names the FL/MN deposit
(10.5281/zenodo.22731361) for files it already holds byte-identically.

``bundle`` writes one uncompressed tar per paper to papers/_artifacts/upload/public/, and the
Alpaca-derived arrays (``license_review``) to upload/restricted/ (arrays are already compressed).
Tars are deterministic (sorted members, mtime 0, uid/gid 0, fixed mode), so a rebuild
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
NOTES = [
    "readers lists load sites that receive this path, usually as argv from an orchestrator "
    "(run_phase*.py, runpod/*.sh); paths are rarely literal at the load call.",
    "byte_reproducible=false: committed before torch.manual_seed(42) was added to probe training "
    "(af75f30, 2026-06-12); rerunning today's code gives different directions, so the deposit is "
    "the only record outside git history.",
    "also_in: byte-identical copy already published in that DOI (inside a 2.35 GB tar.zst, so not "
    "fetchable per file).",
    "license_review: stimuli include mlabonne/harmless_alpaca (verbatim Stanford Alpaca "
    "instructions, CC BY-NC 4.0 upstream); these arrays are bundled for the restricted record.",
]
README = """# DeepSteer paper artifacts ({record} record, {version})

Binary arrays (.npz) removed from the `papers/` tree of https://github.com/deepsteer/deepsteer
(LIBRARY_RELEASE_PLAN §D). `ARTIFACT_MANIFEST.json` lists every array with its sha256, bundle,
producing script, readers, contents, stimulus source and regeneration command.

Fetch from a clone: `python3 papers/build_common/artifacts.py fetch [PATH ...]` (verifies the
bundle and every extracted file against the manifest).

{body}

| Bundle | Files | Bytes | md5 |
|---|---|---|---|
{rows}
"""
BODY = {
    "public": "License: CC BY 4.0. Stimuli are this project's own datasets (moral_probing_v2, "
              "dilemma pairs, persona and control minimal pairs) and public-domain Gutenberg text.",
    "restricted": "License: CC BY-NC 4.0; files restricted, access on request. These arrays are "
                  "computed from the Heretic refusal prompt set, whose harmless half "
                  "(mlabonne/harmless_alpaca) consists of Stanford Alpaca instructions, released "
                  "under CC BY-NC 4.0. They are restricted under the same rule the FL/MN deposit "
                  "applies to non-commercial stimuli.",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


RESTRICTED_BUNDLE = f"deepsteer_papers_alpaca_derived_{BUNDLE_VERSION}.tar"


def _md5(path: Path) -> str:
    """Zenodo shows md5 checksums; recorded so the upload can be checked against the page."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _bundle_for(path: str, entry: dict) -> str | None:
    """Per-paper public bundle; flagged (Alpaca-derived) arrays share one restricted bundle."""
    if "/adapter/" in path:
        return None
    if entry.get("license_review"):
        return RESTRICTED_BUNDLE
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
        entry.update(path=p, bytes=(REPO / p).stat().st_size, sha256=sha)
        entry["bundle"] = _bundle_for(p, entry)
        if entry["bundle"] is None:
            entry["deposit"] = "excluded: LoRA adapter files are never deposited"
        if deposited.get(p) == sha:
            entry["also_in"] = FL_MN_DEPOSIT
        artifacts.append(entry)
    manifest = {
        "description": "Binary artifacts removed from the papers/ tree; resolve with "
                       "papers/build_common/artifacts.py get(path).",
        "records": old.get("records", {
            "public": {"doi": None, "zenodo_record": None, "license": "CC-BY-4.0"},
            "restricted": {"doi": None, "zenodo_record": None, "license": "CC-BY-NC-4.0",
                           "access": "restricted; request access on the record page"},
        }),
        "last_in_tree": old.get("last_in_tree"),
        "bundles": old.get("bundles", {}),
        "notes": old.get("notes", NOTES),
        "artifacts": artifacts,
    }
    return manifest


def bundle(manifest: dict) -> dict:
    """Write deterministic per-paper tars and record their sha256 and size."""
    groups: dict[str, list[str]] = {}
    for a in manifest["artifacts"]:
        if a["bundle"]:
            groups.setdefault(a["bundle"], []).append(a["path"])
    bundles = {}
    for name, members in sorted(groups.items()):
        record = "restricted" if name == RESTRICTED_BUNDLE else "public"
        (UPLOAD / record).mkdir(parents=True, exist_ok=True)
        out = UPLOAD / record / name
        with tarfile.open(out, "w", format=tarfile.PAX_FORMAT) as tar:
            for p in sorted(members):
                data = (REPO / p).read_bytes()
                info = tarfile.TarInfo(p)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(data))
        bundles[name] = {"record": record, "sha256": _sha256(out), "bytes": out.stat().st_size,
                         "files": len(members), "md5": _md5(out)}
        print(f"{name}: {len(members)} files, {out.stat().st_size / 1e6:.1f} MB")
    manifest["bundles"] = bundles
    for record in ("public", "restricted"):
        rows = "\n".join(f"| `{n}` | {b['files']} | {b['bytes']} | `{b['md5']}` |"
                         for n, b in sorted(bundles.items()) if b["record"] == record)
        (UPLOAD / record / "README.md").write_text(README.format(
            record=record, version=BUNDLE_VERSION, body=BODY[record], rows=rows))
    return manifest


STUB = """# Binary artifacts moved to Zenodo

The arrays listed below were removed from this directory at the tip of `main` (LIBRARY_RELEASE_PLAN
§D). They stay in git history: the last commit that contains them is
[`{short}`](https://github.com/deepsteer/deepsteer/tree/{sha}/{dir}) (tag `artifacts-last-in-tree`).

Fetch and verify from a clone:

```bash
python3 papers/build_common/artifacts.py fetch {dir}/<file>
```

Scripts that read these arrays without loading a model resolve them automatically
(`papers/build_common/artifacts.py`). Records: public {public_doi} (CC BY 4.0); restricted
{restricted_doi} (CC BY-NC 4.0, access on request).

| File | Bytes | sha256 | Where |
|---|---|---|---|
{rows}
"""


def stubs(manifest: dict, sha: str) -> list[Path]:
    """Write one stub per directory that loses binaries; ARTIFACTS.md where a README exists."""
    recs = manifest["records"]
    where = {
        "public": f"[public](https://doi.org/{recs['public']['doi']})",
        "restricted": f"[restricted](https://doi.org/{recs['restricted']['doi']})",
    }
    by_dir: dict[str, list[dict]] = {}
    for a in manifest["artifacts"]:
        by_dir.setdefault(a["path"].rsplit("/", 1)[0], []).append(a)
    written = []
    for d, entries in sorted(by_dir.items()):
        rows = []
        for a in sorted(entries, key=lambda e: e["path"]):
            loc = (where[manifest["bundles"][a["bundle"]]["record"]] if a["bundle"]
                   else f"not deposited; regenerate: {a.get('regeneration', '')[:120]}")
            name = a["path"].rsplit("/", 1)[1]
            rows.append(f"| `{name}` | {a['bytes']} | `{a['sha256'][:16]}…` | {loc} |")
        target = REPO / d / ("ARTIFACTS.md" if (REPO / d / "README.md").exists() else "README.md")
        target.write_text(STUB.format(short=sha[:7], sha=sha, dir=d, rows="\n".join(rows),
                                      public_doi=recs["public"]["doi"],
                                      restricted_doi=recs["restricted"]["doi"]))
        written.append(target)
    return written


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("inventory", "bundle", "stubs"):
        print(__doc__)
        return 2
    if argv[0] == "stubs":
        manifest = json.loads(MANIFEST.read_text())
        manifest["last_in_tree"] = argv[1]
        MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
        written = stubs(manifest, argv[1])
        print(f"wrote {len(written)} stubs; last_in_tree = {argv[1]}")
        return 0
    if argv[0] == "inventory":
        dep = Path(argv[1]) if len(argv) > 1 else None
        manifest = inventory(dep)
    else:
        manifest = bundle(json.loads(MANIFEST.read_text()))
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    if argv[0] == "bundle":
        for record in ("public", "restricted"):
            (UPLOAD / record / "ARTIFACT_MANIFEST.json").write_text(MANIFEST.read_text())
    print(f"wrote {MANIFEST.relative_to(REPO)} ({len(manifest['artifacts'])} artifacts)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
