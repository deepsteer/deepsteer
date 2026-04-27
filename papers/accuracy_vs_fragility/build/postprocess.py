#!/usr/bin/env python3
"""Post-process pandoc-generated LaTeX sections.

Converts markdown-style author-year citations to natbib commands using
the bibkey map below.  Inserts \\includegraphics blocks where figures
are referenced.  Leaves §X cross-references as literal section symbols
(reads fine; tightening to \\Cref{} is a follow-up pass).

Run via the Makefile per-section conversion rule.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Citation key map: regex pattern -> natbib command + bibkey.
# Order matters — multi-author patterns must precede single-author.
CITES_PARENS = [
    # (Multiple, Author, & Combined, Year; Other, Year)
    (
        r"\(Haidt, 2012; Graham et al\., 2013\)",
        r"\\citep{haidt2012righteous,graham2013mft}",
    ),
    (r"\(Alain \\& Bengio, 2017; Belinkov, 2022\)",
     r"\\citep{alain2017probes,belinkov2022probing}"),
    # Single (Author, Year)
    (r"\(Alain \\& Bengio, 2017\)", r"\\citep{alain2017probes}"),
    (r"\(Belinkov, 2022\)", r"\\citep{belinkov2022probing}"),
    (r"\(Meng et al\., 2022\)", r"\\citep{meng2022rome}"),
    (r"\(Power et al\., 2022\)", r"\\citep{power2022grokking}"),
    (r"\(Haidt, 2012\)", r"\\citep{haidt2012righteous}"),
    (r"\(Graham et al\., 2013\)", r"\\citep{graham2013mft}"),
    (r"\(Groeneveld et al\., 2024\)", r"\\citep{groeneveld2024olmo}"),
    (r"\(Arditi et al\., 2024\)", r"\\citep{arditi2024refusal}"),
    (r"\(Betley et al\., 2025\)", r"\\citep{betley2025em}"),
]

# In-text Author (Year) -> \citet{key}
CITES_INTEXT = [
    (r"Alain and Bengio \(2017\)", r"\\citet{alain2017probes}"),
    (r"Alain \\& Bengio \(2017\)", r"\\citet{alain2017probes}"),
    (r"Belinkov \(2022\)", r"\\citet{belinkov2022probing}"),
    (r"Belinkov's \(2022\)", r"\\citeauthor{belinkov2022probing}'s \\citeyearpar{belinkov2022probing}"),
    (r"Meng et al\.'s \(2022\)", r"\\citeauthor{meng2022rome}'s \\citeyearpar{meng2022rome}"),
    (r"Meng et al\. \(2022\)", r"\\citet{meng2022rome}"),
    (r"Power et al\.'s \(2022\)", r"\\citeauthor{power2022grokking}'s \\citeyearpar{power2022grokking}"),
    (r"Power et al\. \(2022\)", r"\\citet{power2022grokking}"),
    (r"Haidt's \(2012\)", r"\\citeauthor{haidt2012righteous}'s \\citeyearpar{haidt2012righteous}"),
    (r"Haidt \(2012\)", r"\\citet{haidt2012righteous}"),
    (r"Graham et al\.'s \(2013\)", r"\\citeauthor{graham2013mft}'s \\citeyearpar{graham2013mft}"),
    (r"Graham et al\. \(2013\)", r"\\citet{graham2013mft}"),
    (r"Groeneveld et al\. \(2024\)", r"\\citet{groeneveld2024olmo}"),
    (r"Arditi et al\. \(2024\)", r"\\citet{arditi2024refusal}"),
    (r"Betley et al\.'s \(2025\)", r"\\citeauthor{betley2025em}'s \\citeyearpar{betley2025em}"),
    (r"Betley et al\. \(2025\)", r"\\citet{betley2025em}"),
    # Companion-paper reference (Reblitz-Richardson 2026, in preparation)
    (r"Reblitz-Richardson, 2026, in preparation",
     r"Reblitz-Richardson, 2026, in preparation"),  # leave as text
]

# Figure-insertion patterns. Each maps a sentence-prefix marker to a
# LaTeX figure block. Markers must match exactly once per file. The
# replacement uses the pre-existing prose as the in-text reference (\Cref or
# literal "Figure 1") and inserts a float just before the paragraph.
FIGURE_INSERTS: list[tuple[str, str]] = [
    # Figure 1 — onset overlay.  Insert before the §4.1 paragraph
    # that references "Figure 1".  Other figures (2, 3, 4) are
    # not yet generated; their references remain as literal text
    # in the prose for now.
    (
        r"\\textbf\{Figure 1\} plots the four\nmean-accuracy trajectories on a shared step axis\.",
        (
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            "  \\includegraphics[width=\\linewidth]{figure_1_onset_overlay.pdf}\n"
            "  \\caption{Lexical~$\\to$~compositional emergence gradient on "
            "OLMo-2~1B early-training. Standard moral and sentiment probes "
            "(single-token swap) plateau near~0.97; compositional moral and "
            "syntax (multi-token integration) plateau near~0.77. Compositional "
            "curve is 4-seed mean~$\\pm$~std (split seeds 42/43/44/45). "
            "Onsets: standard moral~1K, sentiment~2K, compositional moral~5K "
            "(per-seed range~4K--7K), syntax~6K.}\n"
            "  \\label{fig:onset-overlay}\n"
            "\\end{figure}\n\n"
            "\\textbf{Figure~\\ref{fig:onset-overlay}} plots the four "
            "mean-accuracy trajectories on a shared step axis."
        ),
    ),
]


def fixup(text: str) -> str:
    # Citation conversions — parens form first, then in-text.
    for pat, repl in CITES_PARENS + CITES_INTEXT:
        text = re.sub(pat, repl, text)

    # Figure insertions.
    for pat, repl in FIGURE_INSERTS:
        text = re.sub(pat, repl, text)

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
