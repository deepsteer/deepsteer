"""Package a paper directory for arXiv submission."""
from __future__ import annotations

import argparse
import atexit
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

FIGURE_EXTS = {".pdf", ".png", ".eps", ".jpg", ".jpeg"}
TEX_EXTS = {".tex", ".bib", ".bst", ".sty"}


def find_referenced_figures(tex_dir: Path) -> set[str]:
    """Parse all .tex files for \\includegraphics references."""
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    refs: set[str] = set()
    for tex_file in tex_dir.rglob("*.tex"):
        for match in pattern.finditer(tex_file.read_text()):
            refs.add(match.group(1))
    return refs


def copy_tree_flat(src: Path, dst: Path, extensions: set[str]) -> list[str]:
    """Copy files matching extensions from src (recursively) into dst, preserving
    subdirectory structure relative to src."""
    copied = []
    for f in src.rglob("*"):
        if f.is_file() and f.suffix.lower() in extensions:
            rel = f.relative_to(src)
            dest = dst / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            copied.append(str(rel))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a paper directory for arXiv submission")
    parser.add_argument("paper_dir", type=Path, help="Path to paper directory (e.g. papers/paper1)")
    args = parser.parse_args()

    paper_dir = args.paper_dir.resolve()
    build_dir = paper_dir / "build"
    figures_dir = paper_dir / "figures"

    if not paper_dir.is_dir():
        sys.exit(f"Error: {paper_dir} is not a directory")
    if not build_dir.is_dir():
        sys.exit(f"Error: no build/ directory in {paper_dir}")
    if not (build_dir / "main.tex").is_file():
        sys.exit(f"Error: no main.tex in {build_dir}")

    tmp = Path(tempfile.mkdtemp(prefix="arxiv_"))
    atexit.register(lambda: shutil.rmtree(tmp, ignore_errors=True))

    # Copy tex infrastructure from build/, preserving sections/ subdir
    copied = copy_tree_flat(build_dir, tmp, TEX_EXTS)

    # Copy figures from figures/ into tmp root (flat)
    if figures_dir.is_dir():
        for f in figures_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in FIGURE_EXTS:
                shutil.copy2(f, tmp / f.name)
                copied.append(f.name)

    # Also grab any figures already in build/
    for f in build_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in FIGURE_EXTS:
            rel = f.relative_to(build_dir)
            dest = tmp / rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                copied.append(str(rel))

    # Verify all referenced figures are present
    refs = find_referenced_figures(tmp)
    for ref in sorted(refs):
        candidates = [tmp / ref, tmp / Path(ref).name]
        if not any(c.is_file() for c in candidates):
            sys.exit(f"Error: referenced figure not found: {ref}")

    # Compile: pdflatex → bibtex → pdflatex → pdflatex
    print("Compiling in isolated directory...")
    latex_cmd = ["pdflatex", "-interaction=nonstopmode", "main.tex"]
    bibtex_cmd = ["bibtex", "main"]

    for step, cmd in enumerate(
        [latex_cmd, bibtex_cmd, latex_cmd, latex_cmd], 1
    ):
        label = " ".join(cmd)
        result = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout[-2000:] if result.stdout else "")
            print(result.stderr[-2000:] if result.stderr else "")
            sys.exit(f"Error: step {step} ({label}) failed with return code {result.returncode}")

    # Remove compilation artifacts, keep only submission files
    # main.pdf is the compiled output — arXiv compiles from source
    keep_exts = TEX_EXTS | FIGURE_EXTS
    compiled_pdf = tmp / "main.pdf"
    if compiled_pdf.exists():
        compiled_pdf.unlink()
    for f in list(tmp.rglob("*")):
        if f.is_file() and f.suffix.lower() not in keep_exts:
            f.unlink()

    # Build tarball
    dirname = paper_dir.name
    date_str = datetime.now().strftime("%Y%m%d")
    repo_root = Path(__file__).resolve().parent.parent
    tarball = repo_root / f"arxiv_{dirname}_{date_str}.tar.gz"

    with tarfile.open(tarball, "w:gz") as tar:
        for f in sorted(tmp.rglob("*")):
            if f.is_file():
                tar.add(f, arcname=f.relative_to(tmp))

    # Summary
    files = sorted(f.relative_to(tmp) for f in tmp.rglob("*") if f.is_file())
    total_size = sum((tmp / f).stat().st_size for f in files)

    print(f"\n{'=' * 50}")
    print(f"arXiv package: {tarball.name}")
    print(f"Files: {len(files)}  |  Size: {total_size / 1024:.1f} KB")
    print(f"{'=' * 50}")
    for f in files:
        size = (tmp / f).stat().st_size
        print(f"  {f}  ({size / 1024:.1f} KB)")
    print(f"\nSaved to: {tarball}")


if __name__ == "__main__":
    main()
