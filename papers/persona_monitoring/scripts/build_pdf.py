#!/usr/bin/env python3
"""Build the Paper 2 PDF and copy it to the canonical location.

Runs `make pdf` in `papers/persona_monitoring/build/` (which invokes
pandoc on each `sections/*.md`, then latexmk on `main.tex`), then copies
the resulting `build/main.pdf` to `papers/persona_monitoring.pdf` at
the repo root.

Usage (from project root):
    python papers/persona_monitoring/scripts/build_pdf.py
    python papers/persona_monitoring/scripts/build_pdf.py --clean
    python papers/persona_monitoring/scripts/build_pdf.py --output path/to/out.pdf
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PAPER_DIR = Path("papers/persona_monitoring")
BUILD_DIR = PAPER_DIR / "build"
BUILD_PDF = BUILD_DIR / "main.pdf"
DEFAULT_OUTPUT = PAPER_DIR.with_suffix(".pdf")  # papers/persona_monitoring.pdf
TEXBIN = "/Library/TeX/texbin"


def ensure_texbin_on_path() -> None:
    """Prepend `/Library/TeX/texbin` to PATH if not already present."""
    path = os.environ.get("PATH", "")
    if TEXBIN not in path.split(os.pathsep):
        os.environ["PATH"] = TEXBIN + os.pathsep + path


def run_make(target: str) -> None:
    subprocess.run(["make", target], cwd=BUILD_DIR, check=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="run `make distclean` before building (forces full rebuild)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"destination for the built PDF (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    if not BUILD_DIR.is_dir():
        print(
            f"error: {BUILD_DIR} not found — run from the project root",
            file=sys.stderr,
        )
        return 2

    ensure_texbin_on_path()

    if args.clean:
        run_make("distclean")
    run_make("pdf")

    if not BUILD_PDF.is_file():
        print(f"error: {BUILD_PDF} not produced", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUILD_PDF, args.output)
    size = args.output.stat().st_size
    print(f"==> {args.output} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
