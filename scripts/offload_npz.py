"""Move large untracked .npz arrays to an external drive and leave symlinks in their place.

Each file is copied to ``<dest>/<repo-relative path>``, read back from the drive with the page
cache bypassed, and compared by SHA-256 before the original is replaced by a symlink. Readers
(``np.load``) follow the link unchanged; with the drive unplugged they fail with
FileNotFoundError. ``MANIFEST.tsv`` on the drive records path, size, sha256 and time.

Files named in any ``papers/*/runpod/sync_*.txt`` stay on the internal disk (rsync -a would
upload the link, not the data, to a pod), as do files modified within ``--min-age-min``.

Usage:
    python3.13 scripts/offload_npz.py --dry-run            # list what would move
    python3.13 scripts/offload_npz.py                      # move it
    python3.13 scripts/offload_npz.py --exclude papers/st_live_reference/
    python3.13 scripts/offload_npz.py --verify             # re-hash every offloaded copy
    python3.13 scripts/offload_npz.py --restore papers/kdg_panel/outputs/p2c/   # bring back
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHUNK = 8 << 20


def sync_listed(repo: Path) -> tuple[set[str], list[str]]:
    """Return (files, directory prefixes) named in any pod sync list, repo-relative."""
    files: set[str] = set()
    prefixes: list[str] = []
    for lst in repo.glob("papers/*/runpod/sync_*.txt"):
        paper = lst.parent.parent.relative_to(repo).as_posix()
        for line in lst.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rel = f"{paper}/{line}"
            (prefixes.append(rel) if line.endswith("/") else files.add(rel))
    return files, prefixes


def sha256_nocache(path: Path) -> str:
    """Hash a file with F_NOCACHE set so a just-written copy is read from the device."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def copy_hashed(src: Path, dst: Path) -> str:
    """Copy data only (no xattrs, so exFAT gets no ._ files), fsync, return the source sha256."""
    h = hashlib.sha256()
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while chunk := fi.read(CHUNK):
            h.update(chunk)
            fo.write(chunk)
        fo.flush()
        os.fsync(fo.fileno())
    return h.hexdigest()


def read_manifest(dest: Path) -> dict[str, str]:
    """Latest sha256 per repo-relative path."""
    m = dest / "MANIFEST.tsv"
    out: dict[str, str] = {}
    if m.exists():
        for line in m.read_text().splitlines()[1:]:
            rel, _size, sha, _when = line.split("\t")
            out[rel] = sha
    return out


def offload(args: argparse.Namespace, dest: Path) -> None:
    listed, listed_dirs = sync_listed(REPO)
    cutoff = time.time() - args.min_age_min * 60
    todo: list[tuple[Path, str, int]] = []
    for p in sorted(REPO.glob("papers/**/*.npz")) + sorted(REPO.glob("outputs/**/*.npz")):
        if p.is_symlink() or not p.is_file():
            continue
        rel = p.relative_to(REPO).as_posix()
        st = p.stat()
        if st.st_size < args.min_mb * 1e6 or st.st_mtime > cutoff:
            continue
        if rel in listed or any(rel.startswith(d) for d in listed_dirs + args.exclude):
            continue
        todo.append((p, rel, st.st_size))
    total = sum(s for _, _, s in todo)
    print(f"{len(todo)} files, {total / 1e9:.1f} GB -> {dest}")
    if args.dry_run:
        for _, rel, size in todo:
            print(f"  {size / 1e9:6.2f} GB  {rel}")
        return
    manifest = dest / "MANIFEST.tsv"
    if not manifest.exists():
        manifest.write_text("path\tsize\tsha256\toffloaded_at\n")
    done = 0
    for i, (p, rel, size) in enumerate(todo, 1):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".partial")
        sha = copy_hashed(p, part)
        back = sha256_nocache(part)
        if back != sha or part.stat().st_size != size:
            part.unlink()
            sys.exit(f"VERIFY FAILED for {rel}: src {sha} drive {back}; original left in place")
        os.replace(part, target)
        st = p.stat()
        os.utime(target, (st.st_atime, st.st_mtime))
        with open(manifest, "a") as f:
            f.write(f"{rel}\t{size}\t{sha}\t{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
            f.flush()
            os.fsync(f.fileno())
        tmp_link = p.with_name(p.name + ".offload_link")
        os.symlink(target, tmp_link)
        os.replace(tmp_link, p)
        done += size
        print(f"[{i}/{len(todo)}] {done / 1e9:6.1f}/{total / 1e9:.1f} GB  {rel}", flush=True)


def offloaded_links(dest: Path, prefixes: list[str]) -> list[tuple[Path, str, Path]]:
    out = []
    for p in sorted(REPO.glob("papers/**/*.npz")) + sorted(REPO.glob("outputs/**/*.npz")):
        if not p.is_symlink():
            continue
        target = Path(os.readlink(p))
        rel = p.relative_to(REPO).as_posix()
        wanted = not prefixes or any(rel.startswith(x) for x in prefixes)
        if target.is_relative_to(dest) and wanted:
            out.append((p, rel, target))
    return out


def verify(dest: Path) -> None:
    shas = read_manifest(dest)
    links = offloaded_links(dest, [])
    bad = 0
    for i, (_p, rel, target) in enumerate(links, 1):
        ok = target.exists() and sha256_nocache(target) == shas.get(rel)
        bad += not ok
        print(f"[{i}/{len(links)}] {'ok ' if ok else 'BAD'} {rel}", flush=True)
    print(f"{len(links) - bad} ok, {bad} bad")
    sys.exit(1 if bad else 0)


def restore(dest: Path, prefixes: list[str]) -> None:
    shas = read_manifest(dest)
    for p, rel, target in offloaded_links(dest, prefixes):
        tmp = p.with_name(p.name + ".restore_partial")
        sha = copy_hashed(target, tmp)
        if sha != shas.get(rel):
            tmp.unlink()
            sys.exit(f"RESTORE FAILED for {rel}: drive copy {sha} != manifest {shas.get(rel)}")
        st = target.stat()
        os.utime(tmp, (st.st_atime, st.st_mtime))
        os.replace(tmp, p)
        print(f"restored {rel} (drive copy kept)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dest", default="/Volumes/ds-data/deepsteer")
    ap.add_argument("--min-mb", type=float, default=100)
    ap.add_argument("--min-age-min", type=float, default=120,
                    help="skip files modified more recently (in-flight downloads)")
    ap.add_argument("--exclude", action="append", default=[], help="repo-relative path prefix")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--restore", nargs="*", metavar="PREFIX")
    args = ap.parse_args()
    dest = Path(args.dest)
    volume = Path(*dest.parts[:3])  # /Volumes/<name>
    if not os.path.ismount(volume):
        sys.exit(f"{volume} is not mounted")
    dest.mkdir(parents=True, exist_ok=True)
    if args.verify:
        verify(dest)
    elif args.restore is not None:
        restore(dest, args.restore)
    else:
        offload(args, dest)


if __name__ == "__main__":
    main()
