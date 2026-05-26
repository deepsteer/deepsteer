#!/usr/bin/env python3
"""Post-process pandoc-generated LaTeX sections.

Converts markdown-style author-year citations to natbib commands using
the bibkey map below.  Inserts \\includegraphics blocks where figures
are referenced.  Rewrites §X cross-references as `\\Cref{<slug>}`
so cleveref auto-renders the section number and they update if the
ordering changes.

Run via the Makefile per-section conversion rule.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Citation key map: regex pattern -> natbib command + bibkey.
# Order matters — multi-author patterns must precede single-author.
CITES_PARENS = [
    # Multi-author parens forms — must precede single-author forms
    # so the longer string matches first.
    # `\s+` tolerates newline-wrapped text (markdown source line-wraps
    # at ~70 cols, so author/year may span lines after pandoc).
    (
        r"\(Haidt,\s+2012;\s+Graham\s+et\s+al\.,\s+2013\)",
        r"\citep{haidt2012righteous,graham2013mft}",
    ),
    (r"\(Alain\s+\\&\s+Bengio,\s+2017;\s+Belinkov,\s+2022\)",
     r"\citep{alain2017probes,belinkov2022probing}"),
    (r"\(Olsson\s+et\s+al\.,\s+2022;\s+Nanda\s+et\s+al\.,\s+2023\)",
     r"\citep{olsson2022induction,nanda2023progress}"),
    (r"\(Groeneveld\s+et\s+al\.,\s+2024;\s+OLMo\s+Team,\s+2025\)",
     r"\citep{groeneveld2024olmo,olmo2_2025}"),
    # Single (Author, Year)
    (r"\(Alain\s+\\&\s+Bengio,\s+2017\)", r"\citep{alain2017probes}"),
    (r"\(Belinkov,\s+2022\)", r"\citep{belinkov2022probing}"),
    (r"\(Meng\s+et\s+al\.,\s+2022\)", r"\citep{meng2022rome}"),
    (r"\(Power\s+et\s+al\.,\s+2022\)", r"\citep{power2022grokking}"),
    (r"\(Haidt,\s+2012\)", r"\citep{haidt2012righteous}"),
    (r"\(Graham\s+et\s+al\.,\s+2013\)", r"\citep{graham2013mft}"),
    (r"\(Groeneveld\s+et\s+al\.,\s+2024\)", r"\citep{groeneveld2024olmo}"),
    (r"\(Arditi\s+et\s+al\.,\s+2024\)", r"\citep{arditi2024refusal}"),
    (r"\(Hewitt\s+\\&\s+Liang,\s+2019\)", r"\citep{hewitt2019control}"),
    (r"\(Pimentel\s+et\s+al\.,\s+2020\)", r"\citep{pimentel2020information}"),
    (r"\(Voita\s+\\&\s+Titov,\s+2020\)", r"\citep{voita2020mdl}"),
    (r"\(Olsson\s+et\s+al\.,\s+2022\)", r"\citep{olsson2022induction}"),
    (r"\(Nanda\s+et\s+al\.,\s+2023\)", r"\citep{nanda2023progress}"),
    (r"\(Biderman\s+et\s+al\.,\s+2023\)", r"\citep{biderman2023pythia}"),
    (r"\(OLMo\s+Team,\s+2025\)", r"\citep{olmo2_2025}"),
    (r"\(Hu\s+et\s+al\.,\s+2022\)", r"\citep{hu2022lora}"),
    (r"\(Zou\s+et\s+al\.,\s+2023\)", r"\citep{zou2023repe}"),
    (r"\(Borras\s+et\s+al\.,\s+2022\)", r"\citep{borras2022walking}"),
    (r"\(Qian\s+et\s+al\.,\s+2024\)", r"\citep{qian2024trustworthiness}"),
    (r"\(Ren\s+et\s+al\.,\s+2026\)", r"\citep{ren2026apex}"),
]

# In-text Author (Year) -> \citet{key}
# Use `[ ~]` for the space between author and year — pandoc inserts a
# non-breaking tilde rather than a regular space.
CITES_INTEXT = [
    (r"Alain and Bengio[ ~]\(2017\)", r"\citet{alain2017probes}"),
    (r"Alain \\& Bengio[ ~]\(2017\)", r"\citet{alain2017probes}"),
    (r"Belinkov[ ~]\(2022\)", r"\citet{belinkov2022probing}"),
    (r"Belinkov's[ ~]\(2022\)",
     r"\citeauthor{belinkov2022probing}'s \citeyearpar{belinkov2022probing}"),
    (r"Meng et al\.'s[ ~]\(2022\)",
     r"\citeauthor{meng2022rome}'s \citeyearpar{meng2022rome}"),
    (r"Meng et al\.[ ~]\(2022\)", r"\citet{meng2022rome}"),
    (r"Power et al\.'s[ ~]\(2022\)",
     r"\citeauthor{power2022grokking}'s \citeyearpar{power2022grokking}"),
    (r"Power et al\.[ ~]\(2022\)", r"\citet{power2022grokking}"),
    (r"Haidt's[ ~]\(2012\)",
     r"\citeauthor{haidt2012righteous}'s \citeyearpar{haidt2012righteous}"),
    (r"Haidt[ ~]\(2012\)", r"\citet{haidt2012righteous}"),
    (r"Graham et al\.'s[ ~]\(2013\)",
     r"\citeauthor{graham2013mft}'s \citeyearpar{graham2013mft}"),
    (r"Graham et al\.[ ~]\(2013\)", r"\citet{graham2013mft}"),
    (r"Groeneveld et al\.[ ~]\(2024\)", r"\citet{groeneveld2024olmo}"),
    (r"Arditi et al\.[ ~]\(2024\)", r"\citet{arditi2024refusal}"),
    (r"Borras et al\.[ ~]\(2022\)", r"\citet{borras2022walking}"),
    (r"Qian et al\.[ ~]\(2024\)", r"\citet{qian2024trustworthiness}"),
    (r"Ren et al\.[ ~]\(2026\)", r"\citet{ren2026apex}"),
]

# Figure-insertion patterns. Each maps a sentence-prefix marker to a
# LaTeX figure block. Markers must match exactly once per file. The
# replacement uses the pre-existing prose as the in-text reference
# (\Cref or literal "Figure N") and inserts a float just before the
# paragraph.  After all FIGURE_INSERTS run, a post-pass rewrites any
# remaining bold "Figure N" mentions to "Figure~\ref{fig:label}" so
# that secondary references resolve cleanly.
FIGURE_INSERTS: list[tuple[str, str]] = []


# Mapping for the secondary-reference post-pass: any remaining bold
# "Figure N" mention (one already replaced by the FIGURE_INSERTS above
# is now `\textbf{Figure~\ref{...}}`, so it won't match this) gets
# rewritten to a proper `\ref{}` so cross-references resolve.
FIGURE_LABEL_MAP = {}


def rewrite_secondary_figure_refs(text: str) -> str:
    """Convert any leftover `\\textbf{Figure N}` to `\\textbf{Figure~\\ref{<label>}}`.

    Runs after FIGURE_INSERTS so the primary insertion-anchor sentence
    --- which the FIGURE_INSERTS replacement has already rewritten to
    `\\textbf{Figure~\\ref{...}}` --- doesn't match this pattern (it has
    a tilde in it, not a literal space).
    """
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
    "≤": r"$\le$",
    "≥": r"$\ge$",
    "≪": r"$\ll$",
    "≫": r"$\gg$",
    "≈": r"$\approx$",
    "→": r"$\to$",
    "←": r"$\leftarrow$",
    "±": r"$\pm$",
    "×": r"$\times$",
    "÷": r"$\div$",
    "·": r"$\cdot$",
    "Δ": r"$\Delta$",
    "α": r"$\alpha$",
    "β": r"$\beta$",
    "γ": r"$\gamma$",
    "δ": r"$\delta$",
    "λ": r"$\lambda$",
    "μ": r"$\mu$",
    "π": r"$\pi$",
    "σ": r"$\sigma$",
    "τ": r"$\tau$",
    "Σ": r"$\Sigma$",
    "Π": r"$\Pi$",
    "ℓ": r"$\ell$",
    "ℝ": r"$\mathbb{R}$",
    "ℕ": r"$\mathbb{N}$",
    "ℤ": r"$\mathbb{Z}$",
    "∈": r"$\in$",
    "∉": r"$\notin$",
    "−": r"$-$",  # U+2212 minus sign (distinct from hyphen-minus)
    "✓": r"\checkmark{}",
    "✗": r"\ding{55}",
    "²": r"$^2$",
    "³": r"$^3$",
    "½": r"$\tfrac{1}{2}$",
    "‰": r"\textperthousand{}",
    "—": r"---",  # em-dash (pandoc usually does this, but belt + braces)
    "–": r"--",   # en-dash
    "…": r"\ldots{}",
    "§": r"\S",
    "°": r"\textdegree{}",
    # Ligatures / smart quotes pandoc occasionally emits despite the
    # markdown source using ASCII; safe to normalize.
    "’": "'",
    "‘": "'",
    "“": "``",
    "”": "''",
    "​": "",  # zero-width space
}


def apply_unicode_fixups(text: str) -> str:
    for src, dst in UNICODE_FIXUPS.items():
        text = text.replace(src, dst)
    return text


# Section reference map: \S<N>(.<M>) literals -> \Cref{<slug>}.
# Slugs come from pandoc's auto-generated \label{} for each heading
# (after the Makefile strips "N." / "N.M" prefixes).  After unicode
# fixup, "§3.2" becomes "\S3.2"; we convert to \Cref{} so the
# reference auto-updates if section ordering changes.
SECTION_LABEL_MAP = {
    # Top-level
    "1": "introduction",
    "2": "related-work",
    "3": "methodology",
    "4": "results",
    "5": "discussion",
    "6": "conclusion",
    # §2 subsections
    "2.1": "probing-for-linguistic-and-semantic-structure",
    "2.2": "geometry-of-concept-representations",
    "2.3": "moral-psychology-and-moral-foundations-theory",
    "2.4": "moral-reasoning-in-language-models",
    "2.5": "companion-papers",
    # §3 subsections
    "3.1": "models-and-comparison-design",
    "3.2": "foundation-specific-probing",
    "3.3": "geometric-analysis",
    "3.4": "bootstrap-direction-stability",
    "3.5": "framework-specific-fragility",
    "3.6": "probing-dataset",
    # §4 subsections
    "4.1": "foundation-specific-probe-accuracy",
    "4.2": "framework-geometry-integration-not-collapse",
    "4.3": "mft-group-structure-in-the-dendrogram",
    "4.4": "layer-wise-geometric-development",
    "4.5": "dense-vs.-moe-framework-geometry",
    "4.6": "direction-stability-under-bootstrap",
    "4.7": "differential-fragility-across-frameworks",
    "4.8": "geometric-trajectory-during-training",
    # §5 subsections
    "5.1": "integration-as-the-default-geometric-mode",
    "5.2": "the-fragility-reversal",
    "5.3": "framework-geometry-stabilizes-before-accuracy-saturates",
    "5.4": "limitations",
}


def convert_section_refs(text: str) -> str:
    """Map literal `\\S<N>(.<M>)` references to `\\Cref{<slug>}`.

    The unicode fixup pass replaced `§` with `\\S`, so by this point
    every section reference looks like `\\S4.1` or `\\S3` in the .tex
    output.  Convert these to `\\Cref{<slug>}` using SECTION_LABEL_MAP.

    Skip refs inside `\\begin{Highlighting}...\\end{Highlighting}`
    blocks — `\\Cref` expansion clashes with the Verbatim environment's
    commandchars setup and triggers cleveref internal errors.
    """
    pattern = re.compile(r"\\S(\d+(?:\.\d+)?)")

    def replace(match: re.Match) -> str:
        ref = match.group(1)
        slug = SECTION_LABEL_MAP.get(ref)
        if slug is None:
            # Unknown reference — leave as literal section symbol.
            return match.group(0)
        return r"\Cref{" + slug + "}"

    # Split on Highlighting blocks; only apply the regex to non-block
    # segments.  re.split with a capture group keeps the delimiters.
    block_re = re.compile(
        r"(\\begin\{Highlighting\}.*?\\end\{Highlighting\})", re.DOTALL
    )
    parts = block_re.split(text)
    for i in range(0, len(parts), 2):  # even indices are non-block
        parts[i] = pattern.sub(replace, parts[i])
    return "".join(parts)


def rebalance_pandoc_table_widths(text: str) -> str:
    """Replace pandoc's content-derived column widths with hand-tuned ones
    for tables that overflow.

    Pandoc auto-computes proportional column widths from header-string
    length, which gives narrow data columns too much space and short-
    header columns (like "Probe") too little --- on a NeurIPS 5.5in
    textwidth this causes wraps that overlap into adjacent columns.

    Currently overrides:
        §4.1 onset table (5 columns, fingerprint
        0.0875/0.1750/0.1500/0.2125/0.3750) -> 0.20/0.27/0.10/0.16/0.27
    """
    overrides = [
        # §4.1 onset table.  Match the full 5-column @{}...@{} block.
        (
            (
                r">\{\\raggedright\\arraybackslash\}p\{\(\\linewidth - 8\\tabcolsep\) \* \\real\{0\.0875\}\}\s+"
                r">\{\\raggedright\\arraybackslash\}p\{\(\\linewidth - 8\\tabcolsep\) \* \\real\{0\.1750\}\}\s+"
                r">\{\\raggedleft\\arraybackslash\}p\{\(\\linewidth - 8\\tabcolsep\) \* \\real\{0\.1500\}\}\s+"
                r">\{\\raggedleft\\arraybackslash\}p\{\(\\linewidth - 8\\tabcolsep\) \* \\real\{0\.2125\}\}\s+"
                r">\{\\raggedleft\\arraybackslash\}p\{\(\\linewidth - 8\\tabcolsep\) \* \\real\{0\.3750\}\}"
            ),
            (
                r">{\raggedright\arraybackslash}p{(\linewidth - 8\tabcolsep) * \real{0.20}}"
                "\n  "
                r">{\raggedright\arraybackslash}p{(\linewidth - 8\tabcolsep) * \real{0.27}}"
                "\n  "
                r">{\raggedleft\arraybackslash}p{(\linewidth - 8\tabcolsep) * \real{0.10}}"
                "\n  "
                r">{\raggedleft\arraybackslash}p{(\linewidth - 8\tabcolsep) * \real{0.16}}"
                "\n  "
                r">{\raggedleft\arraybackslash}p{(\linewidth - 8\tabcolsep) * \real{0.27}}"
            ),
        ),
        # Appendix A foundation-emergence table (7 columns).
        # Foundation column at 0.1579 is too narrow for 22-char cells
        # like "sanctity/degradation" (overlaps Step~0).  Rebalance to
        # 0.24/0.09×5/0.31.
        (
            (
                r">\{\\raggedright\\arraybackslash\}p\{\(\\linewidth - 12\\tabcolsep\) \* \\real\{0\.1579\}\}\s+"
                r">\{\\raggedleft\\arraybackslash\}p\{\(\\linewidth - 12\\tabcolsep\) \* \\real\{0\.1053\}\}\s+"
                r">\{\\raggedleft\\arraybackslash\}p\{\(\\linewidth - 12\\tabcolsep\) \* \\real\{0\.1184\}\}\s+"
                r">\{\\raggedleft\\arraybackslash\}p\{\(\\linewidth - 12\\tabcolsep\) \* \\real\{0\.1184\}\}\s+"
                r">\{\\raggedleft\\arraybackslash\}p\{\(\\linewidth - 12\\tabcolsep\) \* \\real\{0\.1184\}\}\s+"
                r">\{\\raggedleft\\arraybackslash\}p\{\(\\linewidth - 12\\tabcolsep\) \* \\real\{0\.1184\}\}\s+"
                r">\{\\raggedleft\\arraybackslash\}p\{\(\\linewidth - 12\\tabcolsep\) \* \\real\{0\.2632\}\}"
            ),
            (
                r">{\raggedright\arraybackslash}p{(\linewidth - 12\tabcolsep) * \real{0.24}}"
                "\n  "
                r">{\raggedleft\arraybackslash}p{(\linewidth - 12\tabcolsep) * \real{0.09}}"
                "\n  "
                r">{\raggedleft\arraybackslash}p{(\linewidth - 12\tabcolsep) * \real{0.09}}"
                "\n  "
                r">{\raggedleft\arraybackslash}p{(\linewidth - 12\tabcolsep) * \real{0.09}}"
                "\n  "
                r">{\raggedleft\arraybackslash}p{(\linewidth - 12\tabcolsep) * \real{0.09}}"
                "\n  "
                r">{\raggedleft\arraybackslash}p{(\linewidth - 12\tabcolsep) * \real{0.09}}"
                "\n  "
                r">{\raggedleft\arraybackslash}p{(\linewidth - 12\tabcolsep) * \real{0.31}}"
            ),
        ),
    ]
    for pat, repl in overrides:
        text = re.sub(pat, lambda _m, r=repl: r, text)
    return text


def convert_paths_to_path_macro(text: str) -> str:
    """Wrap `\\texttt{...}` containing slashes/dots/underscores in `\\path{...}`
    so the url package can break long identifiers at separators.

    Targets cells like:
        \\texttt{papers/accuracy_vs_fragility/scripts/phase_c4_3seed.py}
        \\texttt{deepsteer.datasets.pipeline.build_probing_dataset}

    Skips short \\texttt{} (no slashes/dots) — those don't cause
    overfulls.  Also skips \\texttt{} with LaTeX commands inside (which
    would break inside \\path{}).
    """
    # Match \texttt{...} where content can include escaped LaTeX specials
    # (\_, \&, \#, \$, \%, \{, \}) but no other command sequences.
    pattern = re.compile(r"\\texttt\{((?:[^{}\\]|\\[_&#$%{}])+)\}")

    def replace(match: re.Match) -> str:
        content = match.group(1)
        # Only wrap if it looks like a path / module identifier with
        # break-friendly separators (slashes, dots, or escaped
        # underscores).
        if not re.search(r"[/.]|\\_", content):
            return match.group(0)
        # \path takes the raw text — un-escape the LaTeX specials so
        # the verbatim mode handles them.
        unescaped = re.sub(r"\\([_&#$%{}])", r"\1", content)
        # Use \path|...| delimited form to avoid `}` collisions in
        # source tokens.
        return r"\path|" + unescaped + "|"

    return pattern.sub(replace, text)


def fixup(text: str) -> str:
    # Citation conversions — parens form first, then in-text.
    # Pass the replacement through a lambda so re.sub does NOT
    # interpret backslash escapes (\c, \citep, \\&, etc.) in the
    # replacement string.
    for pat, repl in CITES_PARENS + CITES_INTEXT:
        text = re.sub(pat, lambda _m, r=repl: r, text)

    # Figure insertions.
    for pat, repl in FIGURE_INSERTS:
        text = re.sub(pat, lambda _m, r=repl: r, text)

    # Secondary figure refs — any leftover bold "Figure N" mentions
    # become `\textbf{Figure~\ref{<label>}}`.  The FIGURE_INSERTS
    # replacements emit `Figure~\ref{...}` (with tilde), so they
    # don't re-match here.
    text = rewrite_secondary_figure_refs(text)

    # Unicode -> LaTeX (must come before \path conversion so that
    # unicode inside \texttt{...} is normalized first).
    text = apply_unicode_fixups(text)

    # § -> \Cref{}.  Must come after unicode fixup (which produces \S
    # from §) and before \path conversion (so \Cref{} is not wrapped
    # as a verbatim path).
    text = convert_section_refs(text)

    # Rebalance over-wide tables before converting paths (the path
    # conversion can interact with the column widths if applied later).
    text = rebalance_pandoc_table_widths(text)

    # Convert long-identifier \texttt{...} to \path{...} so the url
    # package can break at separators.  This dramatically reduces
    # overfull \hbox warnings in tables and lists with long paths.
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
