"""Back up the thinking-on cell's traces off the repository (MCH_PLAN.md §5, Amendment 10).

Copies every ``_raw/`` file under ``outputs/thinking/`` to a destination outside the repository (by
default the ds-data drive), checks each copy's sha256 against its source, and writes the tracked
manifest ``outputs/thinking/traces_manifest.json`` (paths, sizes and sha256 only). The traces quote
game text and stay evaluation-only: training probes, directions or adapters on them needs the MCH
license decision first.

Usage:
    python papers/mch_machiavelli/scripts/mch_backup_traces.py [--dest DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

MCH = Path(__file__).resolve().parents[1]
SRC = MCH / "outputs" / "thinking"
DEST = Path("/Volumes/ds-data/deepsteer/mch_thinking_traces")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", type=Path, default=SRC)
    p.add_argument("--dest", type=Path, default=DEST)
    args = p.parse_args()
    if not args.dest.parent.exists():
        raise SystemExit(f"backup destination {args.dest.parent} is not mounted")
    files = sorted(f for f in args.src.rglob("_raw/*") if f.is_file())
    if not files:
        raise SystemExit(f"no _raw/ files under {args.src}")
    manifest = {}
    for f in files:
        rel = f.relative_to(args.src)
        target = args.dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        digest = sha256_file(f)
        if sha256_file(target) != digest:
            raise SystemExit(f"copy of {rel} does not match its source")
        manifest[str(rel)] = {"bytes": f.stat().st_size, "sha256": digest}
    (args.src / "traces_manifest.json").write_text(json.dumps(
        {"dest": str(args.dest), "files": manifest}, indent=1) + "\n")
    print(f"backed up {len(manifest)} files to {args.dest}; manifest written")


if __name__ == "__main__":
    main()
