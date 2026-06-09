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
import unicodedata
from datetime import datetime
from pathlib import Path

FIGURE_EXTS = {".pdf", ".png", ".eps", ".jpg", ".jpeg"}
# Extension preference when an \includegraphics reference omits its extension.
# arXiv (and pdflatex) prefer vector formats, so do we — this is also why PNGs
# get dropped: only the format actually used by LaTeX is bundled.
FIGURE_EXT_PRIORITY = [".pdf", ".eps", ".png", ".jpg", ".jpeg"]
TEX_EXTS = {".tex", ".bib", ".bst", ".sty"}
# Compiled bibliography. Bundling it makes the submission self-contained and
# avoids relying on arXiv's own bibtex run.
BBL_EXT = ".bbl"
# Build-only files that are not part of the arXiv source. pandoc-template.tex is
# a pandoc skeleton (no \documentclass), inert but unnecessary in the submission.
EXCLUDE_TEX_NAMES = {"pandoc-template.tex"}
# Default code/data link for the arXiv "Comments" field.
DEFAULT_CODE_URL = "https://github.com/deepsteer/deepsteer"


def find_referenced_figures(tex_dir: Path) -> set[str]:
    """Parse all .tex files for \\includegraphics references."""
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    refs: set[str] = set()
    for tex_file in tex_dir.rglob("*.tex"):
        for match in pattern.finditer(tex_file.read_text()):
            refs.add(match.group(1))
    return refs


def resolve_figure(ref: str, search_roots: list[Path]) -> Path | None:
    """Locate the source file for an \\includegraphics reference.

    The reference may omit its extension (LaTeX resolves it), in which case the
    formats in FIGURE_EXT_PRIORITY are tried in order. Roots are searched in
    order, so earlier roots win on ambiguity."""
    ref_path = Path(ref)
    has_ext = ref_path.suffix.lower() in FIGURE_EXTS
    candidate_names = [ref_path.name] if has_ext else [
        ref_path.name + ext for ext in FIGURE_EXT_PRIORITY
    ]
    for root in search_roots:
        for name in candidate_names:
            for cand in sorted(root.rglob(name)):
                if cand.is_file():
                    return cand
    return None


def copy_tree_flat(
    src: Path, dst: Path, extensions: set[str], exclude: set[str] | None = None
) -> list[str]:
    """Copy files matching extensions from src (recursively) into dst, preserving
    subdirectory structure relative to src. Filenames in `exclude` are skipped."""
    exclude = exclude or set()
    copied = []
    for f in src.rglob("*"):
        if f.is_file() and f.suffix.lower() in extensions and f.name not in exclude:
            rel = f.relative_to(src)
            dest = dst / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            copied.append(str(rel))
    return copied


def extract_braced_arg(text: str, command: str) -> str | None:
    r"""Return the brace-balanced argument of \<command>{...}, or None."""
    marker = "\\" + command
    start = text.find(marker)
    if start == -1:
        return None
    i = text.find("{", start)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def extract_environment(text: str, env: str) -> str | None:
    r"""Return the body between \begin{env} and \end{env}, or None."""
    m = re.search(
        r"\\begin\{" + re.escape(env) + r"\}(.*?)\\end\{" + re.escape(env) + r"\}",
        text,
        re.S,
    )
    return m.group(1) if m else None


def clean_metadata_text(text: str) -> str:
    r"""Reduce a LaTeX fragment to a single-line ASCII string for an arXiv form
    field. Unwraps simple formatting commands, normalizes quotes and dashes,
    strips comments and line breaks, and drops non-ASCII. Inline math ($...$) is
    left intact — its source is ASCII and arXiv renders it."""
    # Drop \thanks / \footnote (with brace-balanced argument) entirely.
    for cmd in ("thanks", "footnote"):
        while True:
            arg = extract_braced_arg(text, cmd)
            if arg is None:
                break
            text = text.replace("\\" + cmd + "{" + arg + "}", "", 1)
    # Strip line comments (a % not preceded by a backslash, to end of line).
    text = re.sub(r"(?<!\\)%.*", "", text)
    # Unwrap simple one-argument formatting commands, keeping their content.
    for cmd in ("emph", "textbf", "textit", "texttt", "text", "textrm", "mbox"):
        text = re.sub(r"\\" + cmd + r"\{([^{}]*)\}", r"\1", text)
    # TeX quotes and dashes to ASCII.
    text = text.replace("``", '"').replace("''", '"').replace("`", "'")
    text = text.replace("---", "-").replace("--", "-")
    # Spacing tokens and explicit line breaks become spaces.
    text = re.sub(r"\\\\|\\[,;:!> ]|~", " ", text)
    # Collapse all whitespace (including newlines) to single spaces.
    text = re.sub(r"\s+", " ", text).strip()
    # Force ASCII: fold accents to base letters, drop anything left over.
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text


def count_captioned(tex_dir: Path, envs: tuple[str, ...]) -> int:
    r"""Count environments (across all .tex) that contain a \caption — i.e. the
    floats that get a numbered "Figure N" / "Table N"."""
    text = "".join(f.read_text() for f in tex_dir.rglob("*.tex"))
    total = 0
    for env in envs:
        for m in re.finditer(
            r"\\begin\{" + env + r"\*?\}(.*?)\\end\{" + env + r"\*?\}", text, re.S
        ):
            if "\\caption" in m.group(1):
                total += 1
    return total


def build_comments(pages: int | None, figures: int, tables: int, code_url: str) -> str:
    """Assemble the arXiv Comments line, omitting zero counts."""
    parts = []
    if pages:
        parts.append(f"{pages} pages")
    if figures:
        parts.append(f"{figures} figure" + ("s" if figures != 1 else ""))
    if tables:
        parts.append(f"{tables} table" + ("s" if tables != 1 else ""))
    lead = ", ".join(parts)
    tail = f"Code and datasets at {code_url}"
    return f"{lead}. {tail}" if lead else tail


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a paper directory for arXiv submission")
    parser.add_argument("paper_dir", type=Path, help="Path to paper directory (e.g. papers/paper1)")
    parser.add_argument(
        "--code-url",
        default=DEFAULT_CODE_URL,
        help="Code/data URL for the arXiv Comments field",
    )
    parser.add_argument(
        "--comments",
        default=None,
        help="Override the auto-generated arXiv Comments line entirely",
    )
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

    # Copy tex infrastructure from build/, preserving sections/ subdir.
    # Build-only templates (pandoc-template.tex) are excluded.
    copied = copy_tree_flat(build_dir, tmp, TEX_EXTS, exclude=EXCLUDE_TEX_NAMES)

    # Copy only the figures actually referenced by the .tex sources. This drops
    # unused alternate formats (e.g. GitHub-only PNGs) and bundles just the
    # format LaTeX resolves for each \includegraphics call.
    search_roots = [figures_dir, build_dir, paper_dir]
    refs = find_referenced_figures(tmp)
    for ref in sorted(refs):
        src_fig = resolve_figure(ref, search_roots)
        if src_fig is None:
            sys.exit(f"Error: referenced figure not found: {ref}")
        # Place the figure at the path LaTeX expects: the reference path, with
        # the resolved extension appended when the reference omitted one.
        dest_rel = Path(ref)
        if dest_rel.suffix.lower() not in FIGURE_EXTS:
            dest_rel = dest_rel.with_suffix(src_fig.suffix)
        dest = tmp / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src_fig, dest)
            copied.append(str(dest_rel))

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

    # Read the page count from the log before cleanup removes it.
    pages = None
    log_file = tmp / "main.log"
    if log_file.is_file():
        m = re.search(r"Output written on main\.pdf \((\d+) pages?", log_file.read_text())
        if m:
            pages = int(m.group(1))

    # Remove compilation artifacts, keep only submission files.
    # main.pdf is the compiled output — arXiv compiles from source. The .bbl is
    # kept so the submission carries its own compiled bibliography.
    keep_exts = TEX_EXTS | FIGURE_EXTS | {BBL_EXT}
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

    # Emit arXiv metadata (Title / Abstract / Comments) as a cleaned, plain-ASCII
    # file next to the tarball, ready to paste into the submission form fields.
    main_tex = (tmp / "main.tex").read_text()
    title_raw = extract_braced_arg(main_tex, "title")
    abstract_raw = extract_environment(main_tex, "abstract")
    figures = count_captioned(tmp, ("figure",))
    tables = count_captioned(tmp, ("table", "longtable"))
    comments = args.comments or build_comments(pages, figures, tables, args.code_url)
    metadata_file = repo_root / f"arxiv_{dirname}_{date_str}.md"
    metadata_file.write_text(
        f"# arXiv submission metadata — {dirname}\n\n"
        f"## Title\n{clean_metadata_text(title_raw) if title_raw else ''}\n\n"
        f"## Abstract\n{clean_metadata_text(abstract_raw) if abstract_raw else ''}\n\n"
        f"## Comments\n{comments}\n"
    )

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
    print(f"Metadata: {metadata_file}")
    print(f"  Comments: {comments}")


if __name__ == "__main__":
    main()
