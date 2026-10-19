"""Tests for scripts/prose_lint.py and its package_arxiv.py hook."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses need the module registered while it executes
    spec.loader.exec_module(mod)
    return mod


lint = _load("prose_lint")
package_arxiv = _load("package_arxiv")


def _section(tmp_path: Path, name: str, text: str) -> Path:
    sec = tmp_path / "paper" / "sections"
    sec.mkdir(parents=True, exist_ok=True)
    path = sec / name
    path.write_text(text, encoding="utf-8")
    return path


def test_nonprose_dashes_not_counted():
    # Most probable failure: table rules (|---|), placeholder cells and math/code dashes counted
    # as em dashes, which would flag every section that has a table.
    text = (
        "| model | gap |\n|---|---:|\n| OLMo | — |\n\n"
        "floor & 0.1 & --- \\\\\n\n---\n\n"
        "Inline $a --- b$, code `x — y`, display $$p—q$$ and \\(u --- v\\).\n"
    )
    assert lint.count(lint.strip_nonprose(text)).em_dashes == 0


def test_prose_tells_counted():
    text = (
        "The gate reads harm — and only harm --- at the site. It is not luck but design. "
        "Not only that, it moved, highlighting the gap. Additionally, we delve deeper."
    )
    c = lint.count(lint.strip_nonprose(text))
    assert c.em_dashes == 2
    assert c.neg_parallel == 2
    assert c.trailing_ing == 1
    assert c.vocab == {"delve": 1, "highlight": 1, "Additionally (initial)": 1}


def test_allowlist_terms_not_counted():
    # The skill's domain allowlist must never show up as vocabulary hits.
    text = (
        "A robust harness for steering and alignment aligns the loss landscape. "
        "Stated and revealed preferences differ. The probe additionally tracks content."
    )
    assert lint.count(lint.strip_nonprose(text)).vocab == {}
    assert lint.count("The policy landscape reveals a gap.").vocab == {
        "landscape": 1,
        "reveal": 1,
    }


def test_bar_is_per_section_and_strict_exit(tmp_path):
    clean = _section(tmp_path, "01_intro.md", "One — two — fine.\n")
    over = _section(tmp_path, "02_results.md", "A — b — c — d.\n")
    assert lint.lint_file(clean).flagged == []
    assert lint.lint_file(over).flagged == [("02_results", 3)]
    assert lint.main([str(clean)]) == 0
    assert lint.main([str(clean), str(over)]) == 0
    assert lint.main([str(clean), str(over), "--strict"]) == 1


def test_readme_sections_split_at_headings(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# A\nx — y — z\n## B\np — q — r\n", encoding="utf-8")
    assert lint.lint_file(readme).flagged == []


def test_package_arxiv_warns_but_never_blocks(tmp_path):
    paper = tmp_path / "paper"
    _section(tmp_path, "01_intro.md", "Clean prose.\n")
    assert package_arxiv.prose_lint_warning(paper) is None
    _section(tmp_path, "02_results.md", "A — b — c — d.\n")
    warning = package_arxiv.prose_lint_warning(paper)
    assert warning is not None and "02_results: 3" in warning
    assert package_arxiv.prose_lint_warning(tmp_path / "no_sections") is None
