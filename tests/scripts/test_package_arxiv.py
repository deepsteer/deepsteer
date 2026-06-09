"""Tests for scripts/package_arxiv.py."""
from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

# The script lives outside the package, so import its pieces directly.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "package_arxiv",
    Path(__file__).resolve().parents[2] / "scripts" / "package_arxiv.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

find_referenced_figures = _mod.find_referenced_figures
copy_tree_flat = _mod.copy_tree_flat
main = _mod.main
TEX_EXTS = _mod.TEX_EXTS
FIGURE_EXTS = _mod.FIGURE_EXTS
clean_metadata_text = _mod.clean_metadata_text
extract_braced_arg = _mod.extract_braced_arg
extract_environment = _mod.extract_environment
count_captioned = _mod.count_captioned
build_comments = _mod.build_comments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_TEX = r"""\documentclass{article}
\begin{document}
Hello world.
\end{document}
"""

TEX_WITH_FIGURES = r"""\documentclass{article}
\usepackage{graphicx}
\begin{document}
\includegraphics[width=\linewidth]{fig1.pdf}
\includegraphics{subdir/fig2.png}
\end{document}
"""

TEX_WITH_INPUT = r"""\documentclass{article}
\usepackage{graphicx}
\begin{document}
\input{sections/body.tex}
\end{document}
"""

SECTION_TEX = r"""\includegraphics[scale=0.5]{plot.pdf}
"""


def _make_paper(tmp_path: Path, *, tex: str = MINIMAL_TEX, figures: dict[str, bytes] | None = None,
                extra_build_files: dict[str, str] | None = None,
                sections: dict[str, str] | None = None) -> Path:
    """Create a synthetic paper directory under tmp_path."""
    paper = tmp_path / "test_paper"
    build = paper / "build"
    build.mkdir(parents=True)
    (build / "main.tex").write_text(tex)
    (build / "references.bib").write_text("@article{x, author={A}, title={T}, year={2025}}\n")

    if sections:
        sec_dir = build / "sections"
        sec_dir.mkdir()
        for name, content in sections.items():
            (sec_dir / name).write_text(content)

    if extra_build_files:
        for name, content in extra_build_files.items():
            (build / name).write_text(content)

    if figures:
        fig_dir = paper / "figures"
        fig_dir.mkdir()
        for name, data in figures.items():
            p = fig_dir / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    return paper


def _fake_subprocess_success(*_args, **_kwargs):
    return subprocess.CompletedResult(args=[], returncode=0, stdout="", stderr="")


# ---------------------------------------------------------------------------
# find_referenced_figures
# ---------------------------------------------------------------------------

class TestFindReferencedFigures:
    def test_no_figures(self, tmp_path: Path):
        (tmp_path / "main.tex").write_text(MINIMAL_TEX)
        assert find_referenced_figures(tmp_path) == set()

    def test_basic_refs(self, tmp_path: Path):
        (tmp_path / "main.tex").write_text(TEX_WITH_FIGURES)
        refs = find_referenced_figures(tmp_path)
        assert refs == {"fig1.pdf", "subdir/fig2.png"}

    def test_refs_in_subdirectory(self, tmp_path: Path):
        sec = tmp_path / "sections"
        sec.mkdir()
        (tmp_path / "main.tex").write_text(TEX_WITH_INPUT)
        (sec / "body.tex").write_text(SECTION_TEX)
        refs = find_referenced_figures(tmp_path)
        assert refs == {"plot.pdf"}

    def test_multiple_on_same_line(self, tmp_path: Path):
        tex = r"\includegraphics{a.pdf}\includegraphics[width=5cm]{b.png}"
        (tmp_path / "main.tex").write_text(tex)
        assert find_referenced_figures(tmp_path) == {"a.pdf", "b.png"}


# ---------------------------------------------------------------------------
# copy_tree_flat
# ---------------------------------------------------------------------------

class TestCopyTreeFlat:
    def test_copies_matching_extensions(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "main.tex").write_text("tex")
        (src / "notes.txt").write_text("txt")
        (src / "refs.bib").write_text("bib")

        copied = copy_tree_flat(src, dst, TEX_EXTS)
        assert sorted(copied) == ["main.tex", "refs.bib"]
        assert (dst / "main.tex").read_text() == "tex"
        assert not (dst / "notes.txt").exists()

    def test_preserves_subdirectory_structure(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        (src / "sections").mkdir(parents=True)
        dst.mkdir()
        (src / "sections" / "intro.tex").write_text("hello")

        copied = copy_tree_flat(src, dst, TEX_EXTS)
        assert copied == [str(Path("sections/intro.tex"))]
        assert (dst / "sections" / "intro.tex").read_text() == "hello"


# ---------------------------------------------------------------------------
# main() error paths
# ---------------------------------------------------------------------------

class TestMainErrors:
    def test_missing_paper_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("sys.argv", ["prog", str(tmp_path / "nonexistent")])
        with pytest.raises(SystemExit, match="not a directory"):
            main()

    def test_missing_build_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        paper = tmp_path / "paper"
        paper.mkdir()
        monkeypatch.setattr("sys.argv", ["prog", str(paper)])
        with pytest.raises(SystemExit, match="no build/ directory"):
            main()

    def test_missing_main_tex(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        paper = tmp_path / "paper"
        (paper / "build").mkdir(parents=True)
        monkeypatch.setattr("sys.argv", ["prog", str(paper)])
        with pytest.raises(SystemExit, match="no main.tex"):
            main()

    def test_missing_referenced_figure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        paper = _make_paper(tmp_path, tex=TEX_WITH_FIGURES)
        monkeypatch.setattr("sys.argv", ["prog", str(paper)])
        with pytest.raises(SystemExit, match="referenced figure not found: fig1.pdf"):
            main()


# ---------------------------------------------------------------------------
# main() happy path (mocked compilation)
# ---------------------------------------------------------------------------

class TestMainHappyPath:
    def test_produces_tarball(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        paper = _make_paper(
            tmp_path,
            tex=TEX_WITH_INPUT,
            sections={"body.tex": SECTION_TEX},
            figures={"plot.pdf": b"%PDF-fake"},
            extra_build_files={"pandoc-template.tex": "$body$\n"},
        )
        monkeypatch.setattr("sys.argv", ["prog", str(paper)])

        def fake_run(cmd, **kwargs):
            cwd = Path(kwargs.get("cwd", "."))
            if cmd[0] == "pdflatex":
                (cwd / "main.pdf").write_bytes(b"%PDF-compiled")
            elif cmd[0] == "bibtex":
                (cwd / "main.bbl").write_text("\\begin{thebibliography}{1}\n\\end{thebibliography}\n")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        with patch.object(_mod.subprocess, "run", side_effect=fake_run):
            with patch.object(_mod, "__file__", str(tmp_path / "scripts" / "package_arxiv.py")):
                main()

        tarballs = list(tmp_path.glob("arxiv_test_paper_*.tar.gz"))
        assert len(tarballs) == 1

        with tarfile.open(tarballs[0], "r:gz") as tar:
            names = sorted(tar.getnames())

        assert "main.tex" in names
        assert "references.bib" in names
        assert "sections/body.tex" in names
        assert "plot.pdf" in names
        # Compiled bibliography is bundled; the compiled PDF and the pandoc
        # template are not.
        assert "main.bbl" in names
        assert "main.pdf" not in names
        assert "pandoc-template.tex" not in names

    def test_compilation_failure_exits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        paper = _make_paper(tmp_path)
        monkeypatch.setattr("sys.argv", ["prog", str(paper)])

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="! LaTeX Error", stderr=""
            )

        with patch.object(_mod.subprocess, "run", side_effect=fake_run):
            with pytest.raises(SystemExit, match="step 1.*failed"):
                main()

    def test_referenced_figure_from_build_dir_included(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """A referenced figure living in build/ (not figures/) gets picked up."""
        tex = r"""\documentclass{article}
\usepackage{graphicx}
\begin{document}
\includegraphics{inline.pdf}
\end{document}
"""
        paper = _make_paper(tmp_path, tex=tex)
        (paper / "build" / "inline.pdf").write_bytes(b"%PDF-fake")
        monkeypatch.setattr("sys.argv", ["prog", str(paper)])

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        with patch.object(_mod.subprocess, "run", side_effect=fake_run):
            with patch.object(_mod, "__file__", str(tmp_path / "scripts" / "package_arxiv.py")):
                main()

        tarballs = list(tmp_path.glob("arxiv_test_paper_*.tar.gz"))
        with tarfile.open(tarballs[0], "r:gz") as tar:
            assert "inline.pdf" in tar.getnames()

    def test_unreferenced_figures_dropped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Figures not referenced by any \\includegraphics are excluded — this is
        how the GitHub-only PNG alternates get left out of the submission."""
        paper = _make_paper(
            tmp_path,
            tex=TEX_WITH_INPUT,
            sections={"body.tex": SECTION_TEX},  # references plot.pdf only
            figures={"plot.pdf": b"%PDF-fake", "plot.png": b"PNG-fake", "extra.png": b"PNG-fake"},
        )
        monkeypatch.setattr("sys.argv", ["prog", str(paper)])

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        with patch.object(_mod.subprocess, "run", side_effect=fake_run):
            with patch.object(_mod, "__file__", str(tmp_path / "scripts" / "package_arxiv.py")):
                main()

        tarballs = list(tmp_path.glob("arxiv_test_paper_*.tar.gz"))
        with tarfile.open(tarballs[0], "r:gz") as tar:
            names = tar.getnames()
        assert "plot.pdf" in names
        assert "plot.png" not in names
        assert "extra.png" not in names


# ---------------------------------------------------------------------------
# resolve_figure
# ---------------------------------------------------------------------------

resolve_figure = _mod.resolve_figure


class TestResolveFigure:
    def test_resolves_explicit_extension(self, tmp_path: Path):
        figs = tmp_path / "figures"
        figs.mkdir()
        (figs / "plot.pdf").write_bytes(b"%PDF")
        assert resolve_figure("plot.pdf", [figs]) == figs / "plot.pdf"

    def test_extensionless_prefers_pdf_over_png(self, tmp_path: Path):
        figs = tmp_path / "figures"
        figs.mkdir()
        (figs / "plot.pdf").write_bytes(b"%PDF")
        (figs / "plot.png").write_bytes(b"PNG")
        assert resolve_figure("plot", [figs]) == figs / "plot.pdf"

    def test_searches_roots_in_order(self, tmp_path: Path):
        figs = tmp_path / "figures"
        outputs = tmp_path / "outputs"
        figs.mkdir()
        (outputs / "figures").mkdir(parents=True)
        (outputs / "figures" / "plot.pdf").write_bytes(b"%PDF")
        # Not in figs, but reachable by searching the outputs root.
        assert resolve_figure("plot.pdf", [figs, outputs]) == outputs / "figures" / "plot.pdf"

    def test_missing_returns_none(self, tmp_path: Path):
        assert resolve_figure("nope.pdf", [tmp_path]) is None


# ---------------------------------------------------------------------------
# arXiv metadata extraction / cleaning
# ---------------------------------------------------------------------------

class TestExtractors:
    def test_braced_arg_handles_nested_braces(self):
        assert extract_braced_arg(r"\title{A \emph{deep} title}", "title") == r"A \emph{deep} title"

    def test_braced_arg_missing(self):
        assert extract_braced_arg(r"no title here", "title") is None

    def test_environment_body(self):
        text = r"x \begin{abstract}Hello there.\end{abstract} y"
        assert extract_environment(text, "abstract").strip() == "Hello there."

    def test_environment_missing(self):
        assert extract_environment("nothing", "abstract") is None


class TestCleanMetadataText:
    def test_unwraps_formatting_and_collapses_lines(self):
        raw = "We introduce \\emph{fragility},\na \\textbf{new} metric.\n\nIt works."
        assert clean_metadata_text(raw) == "We introduce fragility, a new metric. It works."

    def test_quotes_and_dashes_to_ascii(self):
        assert clean_metadata_text("``encoded'' range 0.22--0.27") == '"encoded" range 0.22-0.27'

    def test_drops_thanks_and_comments(self):
        raw = "Title\\thanks{Corresponding author} text % a trailing comment"
        assert clean_metadata_text(raw) == "Title text"

    def test_strips_non_ascii_and_tilde(self):
        # Non-breaking tilde -> space; smart quotes / accents folded or dropped.
        assert clean_metadata_text("café~de “Paris”") == 'cafe de Paris'

    def test_keeps_inline_math(self):
        assert clean_metadata_text("a lexical~$\\to$~compositional gap") == "a lexical $\\to$ compositional gap"


class TestCountCaptioned:
    def test_counts_only_captioned(self, tmp_path: Path):
        (tmp_path / "a.tex").write_text(
            r"\begin{figure}\includegraphics{x}\caption{y}\end{figure}"
            r"\begin{figure}\includegraphics{z}\end{figure}"  # no caption
        )
        assert count_captioned(tmp_path, ("figure",)) == 1

    def test_counts_tables_and_longtables(self, tmp_path: Path):
        (tmp_path / "a.tex").write_text(
            r"\begin{table}\caption{t}\end{table}"
            r"\begin{longtable}{ll}\caption{lt}\\\end{longtable}"
            r"\begin{longtable}{ll}\end{longtable}"  # uncaptioned layout table
        )
        assert count_captioned(tmp_path, ("table", "longtable")) == 2


class TestBuildComments:
    def test_full(self):
        assert build_comments(22, 5, 3, "http://x") == "22 pages, 5 figures, 3 tables. Code and datasets at http://x"

    def test_omits_zero_counts_and_singular(self):
        assert build_comments(10, 1, 0, "http://x") == "10 pages, 1 figure. Code and datasets at http://x"

    def test_url_only_when_no_counts(self):
        assert build_comments(None, 0, 0, "http://x") == "Code and datasets at http://x"


class TestMetadataFileWritten:
    def test_metadata_file_emitted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        tex = r"""\documentclass{article}
\title{A \emph{Fine} Title}
\begin{document}
\maketitle
\begin{abstract}
First para with \textbf{bold}.

Second para.
\end{abstract}
\includegraphics{plot.pdf}
\begin{figure}\includegraphics{plot.pdf}\caption{c}\end{figure}
\end{document}
"""
        paper = _make_paper(tmp_path, tex=tex, figures={"plot.pdf": b"%PDF-fake"})
        monkeypatch.setattr("sys.argv", ["prog", str(paper), "--code-url", "http://example.com/repo"])

        def fake_run(cmd, **kwargs):
            cwd = Path(kwargs.get("cwd", "."))
            if cmd[0] == "pdflatex":
                (cwd / "main.pdf").write_bytes(b"%PDF")
                (cwd / "main.log").write_text("Output written on main.pdf (7 pages, 123 bytes).\n")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        with patch.object(_mod.subprocess, "run", side_effect=fake_run):
            with patch.object(_mod, "__file__", str(tmp_path / "scripts" / "package_arxiv.py")):
                main()

        md = list(tmp_path.glob("arxiv_test_paper_*.md"))
        assert len(md) == 1
        content = md[0].read_text()
        assert "## Title\nA Fine Title\n" in content
        assert "First para with bold. Second para." in content
        assert "7 pages, 1 figure. Code and datasets at http://example.com/repo" in content

    def test_comments_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        paper = _make_paper(tmp_path)
        monkeypatch.setattr("sys.argv", ["prog", str(paper), "--comments", "Custom line."])

        def fake_run(cmd, **kwargs):
            cwd = Path(kwargs.get("cwd", "."))
            if cmd[0] == "pdflatex":
                (cwd / "main.log").write_text("Output written on main.pdf (3 pages).\n")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        with patch.object(_mod.subprocess, "run", side_effect=fake_run):
            with patch.object(_mod, "__file__", str(tmp_path / "scripts" / "package_arxiv.py")):
                main()

        md = list(tmp_path.glob("arxiv_test_paper_*.md"))[0].read_text()
        assert "## Comments\nCustom line.\n" in md
