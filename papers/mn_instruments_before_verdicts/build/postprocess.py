#!/usr/bin/env python3
"""Post-process pandoc-generated LaTeX sections for the methods note.

Mirrors the paper-1 pipeline:
  * converts markdown author-year citations to natbib commands,
  * rewrites bold "Figure N" mentions to `\\ref{<label>}` so the raw
    figure floats embedded in the markdown resolve their numbers,
  * rewrites `§X` cross-references to `\\Cref{<slug>}` so cleveref
    auto-renders the section number,
  * normalizes non-ASCII glyphs pdflatex can't set natively,
  * wraps long `\\texttt{}` identifiers in `\\path{}` so the url package
    can break them at separators.

Run via the Makefile per-section conversion rule.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Citation key map: regex pattern -> natbib command + bibkey.
# `\s+` tolerates newline-wrapped text (markdown source line-wraps, so
# author/year may span lines after pandoc).
CITES_PARENS = [
    # Massive-activations / attention-sink pair (§2.3).
    (
        r"\(Sun\s+et\s+al\.,\s+2024;\s+Xiao\s+et\s+al\.,\s+2023\)",
        r"\citep{sun2024massive,xiao2023efficient}",
    ),
    # Paper-1 self-reference (§6.1): keep the version note as a citep
    # post-note so "v1, 9 Jun 2026" survives in the rendered citation.
    (
        r"\(Reblitz-Richardson,\s+2026,\s+arXiv:2606\.11375v1,\s+9\s+Jun\s+2026\)",
        r"\citep[arXiv:2606.11375v1, 9 Jun 2026]{reblitzrichardson2026fragility}",
    ),
]

# No literal in-text "Author (Year)" citations in this note.
CITES_INTEXT: list[tuple[str, str]] = []


# Figure label map for the secondary-reference post-pass.  The figure
# floats themselves are embedded as raw LaTeX in the markdown (with
# these labels); here we only rewrite the bold in-text "Figure N"
# mentions so their numbers resolve via \ref.
FIGURE_LABEL_MAP = {
    "1": "fig:bottleneck-pr",
    "2": "fig:ladder",
    "3": "fig:depth-collapse",
}


def rewrite_secondary_figure_refs(text: str) -> str:
    """Convert any `\\textbf{Figure N}` to `\\textbf{Figure~\\ref{<label>}}`."""
    pattern = re.compile(r"\\textbf\{Figure (\d+)\}")

    def replace(match: re.Match) -> str:
        n = match.group(1)
        label = FIGURE_LABEL_MAP.get(n)
        if label is None:
            return match.group(0)
        return r"\textbf{Figure~\ref{" + label + "}}"

    return pattern.sub(replace, text)


# Unicode -> LaTeX-math / LaTeX-text substitutions.  pdflatex does not
# handle these natively even with [utf8]{inputenc}; replacing them at
# postprocess time avoids switching to xelatex/lualatex.
UNICODE_FIXUPS = {
    # Two-char sequence first: capital sigma + combining circumflex
    # (U+03A3 U+0302) is the covariance estimate "\hat{\Sigma}".  Must
    # precede the standalone-sigma rule below.
    "Σ̂": r"$\hat{\Sigma}$",
    "≤": r"$\le$",       # ≤
    "≥": r"$\ge$",       # ≥
    "≪": r"$\ll$",       # ≪
    "≫": r"$\gg$",       # ≫
    "≈": r"$\approx$",   # ≈
    "→": r"$\to$",       # →
    "←": r"$\leftarrow$",  # ←
    "↔": r"$\leftrightarrow$",  # ↔
    "⇒": r"$\Rightarrow$",  # ⇒
    "±": r"$\pm$",       # ±
    "×": r"$\times$",    # ×
    "÷": r"$\div$",      # ÷
    "·": r"$\cdot$",     # ·
    "Δ": r"$\Delta$",    # Δ
    "α": r"$\alpha$",    # α
    "β": r"$\beta$",     # β
    "γ": r"$\gamma$",    # γ
    "δ": r"$\delta$",    # δ
    "ε": r"$\varepsilon$",  # ε
    "λ": r"$\lambda$",   # λ
    "μ": r"$\mu$",       # μ
    "π": r"$\pi$",       # π
    "σ": r"$\sigma$",    # σ
    "τ": r"$\tau$",      # τ
    "Σ": r"$\Sigma$",    # Σ (standalone)
    "Π": r"$\Pi$",       # Π
    "√": r"$\surd$",     # √
    "⊙": r"$\odot$",     # ⊙
    "ℓ": r"$\ell$",      # ℓ
    "ℝ": r"$\mathbb{R}$",  # ℝ
    "ℕ": r"$\mathbb{N}$",  # ℕ
    "ℤ": r"$\mathbb{Z}$",  # ℤ
    "∈": r"$\in$",       # ∈
    "∉": r"$\notin$",    # ∉
    "−": r"$-$",         # − U+2212 minus sign (distinct from hyphen)
    "✓": r"\checkmark{}",  # ✓
    "²": r"$^2$",        # ²
    "³": r"$^3$",        # ³
    "½": r"$\tfrac{1}{2}$",  # ½
    "‰": r"\textperthousand{}",  # ‰
    "—": r"---",         # — em-dash
    "–": r"--",          # – en-dash
    "…": r"\ldots{}",    # …
    "§": r"\S",          # §
    "°": r"\textdegree{}",  # °
    "’": "'",            # ’
    "‘": "'",            # ‘
    "“": "``",           # “
    "”": "''",           # ”
    "​": "",             # zero-width space
    "̂": "",             # orphan combining circumflex (safety)
}


def apply_unicode_fixups(text: str) -> str:
    for src, dst in UNICODE_FIXUPS.items():
        text = text.replace(src, dst)
    return text


def protect_heading_math(text: str) -> str:
    """Wrap the lone `$\\times$` in the §2.4 heading with `\\texorpdfstring`.

    hyperref cannot place math (`$...$`) into a PDF bookmark string and
    emits three "Token not allowed in a PDF string" warnings for the
    "~3×" heading.  Wrapping the math token makes the bookmark fall back
    to plain "x" while the printed heading keeps the × glyph.  Harmless
    where the same "~3×" appears in body text (\\texorpdfstring expands
    to its first argument outside a bookmark context).
    """
    return text.replace(
        r"\textasciitilde3$\times$",
        r"\textasciitilde3\texorpdfstring{$\times$}{x}",
    )


# Section reference map: \S<N>(.<M>) literals -> \Cref{<slug>}.  Slugs
# come from pandoc's auto-generated \label{} for each heading (the
# `{#slug}` attribute in the markdown).  After unicode fixup, "§3.2"
# becomes "\S3.2"; convert to \Cref{} so it auto-updates on reorder.
#
# NOTE: "4.4" is intentionally absent — the sole §4.4 reference points
# at Paper 1 (an external section), so it is left as the literal "§4.4".
SECTION_LABEL_MAP = {
    "1": "introduction",
    "2": "decision-site",
    "2.1": "a2-band-below-null",
    "3": "verdict-discipline",
    "3.1": "ratio-of-ratios",
    "3.2": "power-tables",
    "3.3": "orthogonal-cell",
    "4": "stimulus-discipline",
    "4.1": "a6-deliberation",
    "5": "depth-discipline",
    "6": "case-study",
    "6.1": "reflexive-discipline",
    "6.2": "claim-hygiene",
    "7": "checklist",
}


def convert_section_refs(text: str) -> str:
    """Map literal `\\S<N>(.<M>)` references to `\\Cref{<slug>}`.

    Skip refs inside `\\begin{Highlighting}...\\end{Highlighting}`
    blocks — \\Cref expansion clashes with the Verbatim environment.
    """
    pattern = re.compile(r"\\S(\d+(?:\.\d+)?)")

    def replace(match: re.Match) -> str:
        ref = match.group(1)
        slug = SECTION_LABEL_MAP.get(ref)
        if slug is None:
            # Unknown reference (e.g. §4.4 -> Paper 1) — leave literal.
            return match.group(0)
        return r"\Cref{" + slug + "}"

    block_re = re.compile(
        r"(\\begin\{Highlighting\}.*?\\end\{Highlighting\})", re.DOTALL
    )
    parts = block_re.split(text)
    for i in range(0, len(parts), 2):  # even indices are non-block
        parts[i] = pattern.sub(replace, parts[i])
    return "".join(parts)


def convert_paths_to_path_macro(text: str) -> str:
    """Wrap `\\texttt{...}` identifiers with slashes/dots/underscores in
    `\\path{...}` so the url package can break long identifiers.

    Skips short `\\texttt{}` (no separators) and any `\\texttt{}` with
    LaTeX commands inside (which would break inside `\\path{}`).
    """
    pattern = re.compile(r"\\texttt\{((?:[^{}\\]|\\[_&#$%{}])+)\}")

    def replace(match: re.Match) -> str:
        content = match.group(1)
        if not re.search(r"[/.]|\\_", content):
            return match.group(0)
        unescaped = re.sub(r"\\([_&#$%{}])", r"\1", content)
        return r"\path|" + unescaped + "|"

    return pattern.sub(replace, text)


def fixup(text: str) -> str:
    # Citation conversions.  Pass the replacement through a lambda so
    # re.sub does NOT interpret backslash escapes in the replacement.
    for pat, repl in CITES_PARENS + CITES_INTEXT:
        text = re.sub(pat, lambda _m, r=repl: r, text)

    # Secondary figure refs — bold "Figure N" mentions become
    # `\textbf{Figure~\ref{<label>}}`.
    text = rewrite_secondary_figure_refs(text)

    # Unicode -> LaTeX (before \path conversion so unicode inside
    # \texttt{...} is normalized first).
    text = apply_unicode_fixups(text)

    # Make the one math-bearing heading bookmark-safe for hyperref.
    text = protect_heading_math(text)

    # § -> \Cref{}.  After unicode fixup (which produces \S from §) and
    # before \path conversion (so \Cref{} is not wrapped as a path).
    text = convert_section_refs(text)

    # Convert long-identifier \texttt{...} to \path{...}.
    text = convert_paths_to_path_macro(text)

    return text


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: postprocess.py <tex_file>", file=sys.stderr)
        return 2
    p = Path(argv[1])
    text = p.read_text()
    p.write_text(fixup(text))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
