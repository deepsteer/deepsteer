"""Count AI-writing tells in paper sections and READMEs (the paper-prose skill's lint).

Scans papers/*/sections/*.md and README.md by default, with math, code, LaTeX references and
table rules stripped, and reports per file and per 1k words:

- em dashes ("—", "&mdash;", and pandoc's "---" outside table rules and horizontal rules);
- negative-parallelism frames ("not X but Y", "not just", "not only");
- trailing participle clauses (", highlighting ...", ", underscoring ...", ...);
- the vocabulary of .claude/skills/paper-prose/references/ai-writing-tells.md, minus the
  skill's domain-term allowlist.

A section (one sections/*.md file; one heading block in a README) with more than
EM_DASH_BAR em dashes is flagged. Exit 0 by default; with --strict, exit 1 when anything is
flagged. Stdlib only, so the CI job needs no install.

    python scripts/prose_lint.py                          # default scan
    python scripts/prose_lint.py papers/kdg_judgment_action/sections/*.md --strict
    python scripts/prose_lint.py --json > lint.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLOBS = ("papers/*/sections/*.md", "README.md")
# CLAUDE.md's bar, now in the paper-prose skill: one or two per section is fine.
EM_DASH_BAR = 2

# The guide's section 3 list plus the words CLAUDE.md banned before the skill took them over,
# minus the allowlist (harness, robust, align/alignment, steering, loss landscape). Keep in step
# with references/ai-writing-tells.md on each monthly refresh. "reveal" is counted; the
# allowlisted unmask sense needs a human read.
VOCAB: dict[str, str] = {
    "delve": r"delv(?:e|es|ed|ing)",
    "tapestry": r"tapestr(?:y|ies)",
    "testament": r"testaments?",
    "intricate": r"intricac(?:y|ies)|intricate(?:ly)?",
    "pivotal": r"pivotal",
    "underscore": r"underscor(?:e|es|ed|ing)",
    "highlight": r"highlight(?:s|ed|ing)?",
    "landscape": r"(?<!loss )landscapes?",
    "realm": r"realms?",
    "foster": r"foster(?:s|ed|ing)?",
    "enhance": r"enhanc(?:e|es|ed|ing|ement|ements)",
    "bolster": r"bolster(?:s|ed|ing)?",
    "garner": r"garner(?:s|ed|ing)?",
    "showcase": r"showcas(?:e|es|ed|ing)",
    "crucial": r"crucial(?:ly)?",
    "vital": r"vital(?:ly)?",
    "key (adj.)": r"key(?= [a-z])(?! (?:is|are|was|were|of|in|to|for|and|or)\b)",
    "valuable": r"valuable",
    "vibrant": r"vibrant",
    "meticulous": r"meticulous(?:ly)?",
    "nuanced": r"nuanced?",
    "multifaceted": r"multi-?faceted",
    "seamless": r"seamless(?:ly)?",
    "comprehensive": r"comprehensive(?:ly)?",
    "profound": r"profound(?:ly)?",
    "interplay": r"interplay",
    "resonate": r"resonat(?:e|es|ed|ing)",
    "navigate": r"navigat(?:e|es|ed|ing)",
    "embark": r"embark(?:s|ed|ing)?",
    "journey": r"journeys?",
    "commitment to": r"commitment to",
    "ever-evolving": r"ever-evolving",
    "game-changer": r"game-?changers?",
    "leverage": r"leverag(?:e|es|ed|ing)",
    "utilize": r"utili[sz](?:e|es|ed|ing|ation)",
    "unlock": r"unlock(?:s|ed|ing)?",
    "empower": r"empower(?:s|ed|ing)?",
    "elevate": r"elevat(?:e|es|ed|ing)",
    "synergy": r"synerg(?:y|ies|istic)",
    "holistic": r"holistic(?:ally)?",
    "transformative": r"transformative",
    "cutting-edge": r"cutting-edge",
    "enduring": r"enduring",
    "paradigm": r"paradigms?",
    "at the forefront": r"at the forefront",
    # "revealed preference" is a term of art (stated vs revealed), not a tell.
    "reveal": r"reveal(?:s|ed|ing)?(?<!revealed)|revealed(?! preferences?\b)",
    "meaningful": r"meaningful(?:ly)?",
    "notably": r"notably",
    "importantly": r"importantly",
    "interestingly": r"interestingly",
    # Capitalized means sentence-initial; mid-sentence "additionally" is plain English.
    "Additionally (initial)": r"(?-i:Additionally)",
    "it is worth noting": r"it is worth noting",
}
VOCAB_RE = {k: re.compile(r"\b(?:" + v + r")\b", re.I) for k, v in VOCAB.items()}

NEG_PARALLEL_RE = re.compile(
    r"\bnot (?:just|only|merely|simply)\b"
    r"|\bnot\b[^.;:?!()]{1,60}?,? but\b"
    r"|\b(?:is|are|was|were)n[’']t\b[^.;?!]{1,60}?[,;] (?:it|they)(?:[’']s| is| are)\b",
    re.I,
)
TRAILING_ING_RE = re.compile(
    r", (?:highlighting|underscoring|reflecting|emphasi[sz]ing|showcasing|signaling|signalling"
    r"|illustrating|underlining|cementing|solidifying|ensuring|fostering|contributing to"
    r"|marking|paving the way)\b",
    re.I,
)
EM_DASH_RE = re.compile(r"—|&mdash;|\\textemdash\b|(?<!-)---(?!-)")

# Stripped before counting. Order matters: fenced code before inline code, display math
# before inline math.
_STRIP = [
    re.compile(r"^```.*?^```", re.S | re.M),
    re.compile(r"<!--.*?-->", re.S),
    re.compile(r"\$\$.*?\$\$", re.S),
    re.compile(r"\\\[.*?\\\]", re.S),
    re.compile(r"\\\(.*?\\\)", re.S),
    re.compile(r"(?<![\\$])\$[^$]{1,400}?\$"),
    re.compile(r"`[^`\n]+`"),
    re.compile(r"\\(?:cite[a-z]*|ref|eqref|autoref|cref|Cref|label|url|href)\{[^}]*\}"),
    re.compile(r"\[@[^\]]*\]|\{#[^}]*\}"),
]
# Table rules (|---|:--:|), horizontal rules and YAML fences: dashes that are not prose.
_RULE_LINE = re.compile(r"^\s*(?:\|?\s*:?-{3,}:?\s*)+\|?\s*$|^\s*(?:-{3,}|\*{3,})\s*$")
# A table cell holding only a dash is an empty-cell placeholder, not an em dash: markdown
# (| — |) and raw LaTeX (& --- \\) tables both carry them.
_PLACEHOLDER_CELL = re.compile(r"(?<=\|)\s*(?:—|---|–|-)\s*(?=\|)")
_PLACEHOLDER_TEX = re.compile(r"(?<=&)\s*(?:—|---|--|-)\s*(?=&|\\\\)")
_HEADING = re.compile(r"^#{1,2} ")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def strip_nonprose(text: str) -> str:
    """Remove math, code, LaTeX references, table rules and placeholder cells.

    Args:
        text: Raw markdown.

    Returns:
        The text left for prose counting; line structure is kept where possible.
    """
    for pat in _STRIP:
        text = pat.sub(" ", text)
    lines = []
    for line in text.splitlines():
        if _RULE_LINE.match(line):
            continue
        if line.lstrip().startswith("|"):
            line = _PLACEHOLDER_CELL.sub(" ", line)
        line = _PLACEHOLDER_TEX.sub(" ", line)
        lines.append(line)
    return "\n".join(lines)


@dataclass
class Counts:
    """Tell counts for one span of prose."""

    words: int = 0
    em_dashes: int = 0
    neg_parallel: int = 0
    trailing_ing: int = 0
    vocab: dict[str, int] = field(default_factory=dict)

    @property
    def vocab_total(self) -> int:
        """Sum over all vocabulary hits."""
        return sum(self.vocab.values())

    def per_1k(self, n: int) -> float:
        """Rate per 1,000 words."""
        return 1000.0 * n / self.words if self.words else 0.0


def count(prose: str) -> Counts:
    """Count tells in already-stripped prose.

    Args:
        prose: Output of :func:`strip_nonprose`.

    Returns:
        The counts.
    """
    vocab = {k: len(r.findall(prose)) for k, r in VOCAB_RE.items()}
    return Counts(
        words=len(_WORD.findall(prose)),
        em_dashes=len(EM_DASH_RE.findall(prose)),
        neg_parallel=len(NEG_PARALLEL_RE.findall(prose)),
        trailing_ing=len(TRAILING_ING_RE.findall(prose)),
        vocab={k: v for k, v in vocab.items() if v},
    )


def sections(path: Path, prose: str) -> list[tuple[str, str]]:
    """Split prose into the units the em-dash bar applies to.

    A paper section file is one section. Anything else (a README) splits at its # and ##
    headings.
    """
    if path.parent.name == "sections":
        return [(path.stem, prose)]
    out: list[tuple[str, str]] = []
    title, buf = "(top)", []
    for line in prose.splitlines():
        if _HEADING.match(line):
            out.append((title, "\n".join(buf)))
            title, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    out.append((title, "\n".join(buf)))
    return [(t, b) for t, b in out if b.strip()]


@dataclass
class FileReport:
    """Counts for one file plus the sections over the em-dash bar."""

    path: str
    counts: Counts
    flagged: list[tuple[str, int]]


def lint_file(path: Path) -> FileReport:
    """Lint one markdown file."""
    prose = strip_nonprose(path.read_text(encoding="utf-8"))
    flagged = []
    for title, body in sections(path, prose):
        n = len(EM_DASH_RE.findall(body))
        if n > EM_DASH_BAR:
            flagged.append((title, n))
    try:
        shown = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        shown = str(path)
    return FileReport(shown, count(prose), flagged)


def default_paths(root: Path = REPO_ROOT) -> list[Path]:
    """The default scan set: every paper's sections and the top-level README."""
    paths: list[Path] = []
    for pattern in DEFAULT_GLOBS:
        paths.extend(sorted(root.glob(pattern)))
    return paths


def lint_paths(paths: list[Path]) -> list[FileReport]:
    """Lint each path in order."""
    return [lint_file(p) for p in paths]


def _group(path: str) -> str:
    parts = Path(path).parts
    return parts[1] if len(parts) > 2 and parts[0] == "papers" else path


def format_report(reports: list[FileReport]) -> str:
    """Render the per-file table, per-group totals and flags as text."""
    head = f"{'file':<58}{'words':>7}{'em':>5}{'em/1k':>7}{'notXbutY':>9}{'ing,':>6}{'vocab':>6}"
    lines = [head, "-" * len(head)]
    groups: dict[str, Counts] = {}
    for r in reports:
        c = r.counts
        lines.append(
            f"{r.path[-58:]:<58}{c.words:>7}{c.em_dashes:>5}{c.per_1k(c.em_dashes):>7.2f}"
            f"{c.neg_parallel:>9}{c.trailing_ing:>6}{c.vocab_total:>6}"
        )
        g = groups.setdefault(_group(r.path), Counts())
        g.words += c.words
        g.em_dashes += c.em_dashes
        g.neg_parallel += c.neg_parallel
        g.trailing_ing += c.trailing_ing
        for k, v in c.vocab.items():
            g.vocab[k] = g.vocab.get(k, 0) + v
    lines += ["", "totals (per 1k words: em / notXbutY / ing, / vocab)"]
    for name, g in groups.items():
        top = ", ".join(f"{k} {v}" for k, v in sorted(g.vocab.items(), key=lambda kv: -kv[1])[:6])
        lines.append(
            f"  {name:<34}{g.words:>7}w  em {g.em_dashes} ({g.per_1k(g.em_dashes):.2f})"
            f"  nXbY {g.neg_parallel} ({g.per_1k(g.neg_parallel):.2f})"
            f"  ing {g.trailing_ing} ({g.per_1k(g.trailing_ing):.2f})"
            f"  vocab {g.vocab_total} ({g.per_1k(g.vocab_total):.2f})"
            + (f"  [{top}]" if top else "")
        )
    flagged = [(r.path, t, n) for r in reports for t, n in r.flagged]
    lines.append("")
    if flagged:
        lines.append(f"FLAGGED: {len(flagged)} section(s) over {EM_DASH_BAR} em dashes")
        lines += [f"  {p} :: {t}: {n}" for p, t, n in flagged]
    else:
        lines.append(f"no section over {EM_DASH_BAR} em dashes")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, help="Files to scan (default: see top)")
    parser.add_argument("--strict", action="store_true", help="Exit 1 when anything is flagged")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the table")
    args = parser.parse_args(argv)

    paths = args.paths or default_paths()
    reports = lint_paths(paths)
    if args.json:
        rows = [
            {**asdict(r), "vocab_total": r.counts.vocab_total, "em_dash_bar": EM_DASH_BAR}
            for r in reports
        ]
        print(json.dumps(rows, indent=2))
    else:
        print(format_report(reports))
    return 1 if args.strict and any(r.flagged for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
