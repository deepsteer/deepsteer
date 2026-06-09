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
    (r"\(Betley\s+et\s+al\.,\s+2025\)", r"\citep{betley2025em}"),
    (r"\(Hewitt\s+\\&\s+Liang,\s+2019\)", r"\citep{hewitt2019control}"),
    (r"\(Pimentel\s+et\s+al\.,\s+2020\)", r"\citep{pimentel2020information}"),
    (r"\(Voita\s+\\&\s+Titov,\s+2020\)", r"\citep{voita2020mdl}"),
    (r"\(Olsson\s+et\s+al\.,\s+2022\)", r"\citep{olsson2022induction}"),
    (r"\(Nanda\s+et\s+al\.,\s+2023\)", r"\citep{nanda2023progress}"),
    (r"\(Biderman\s+et\s+al\.,\s+2023\)", r"\citep{biderman2023pythia}"),
    (r"\(OLMo\s+Team,\s+2025\)", r"\citep{olmo2_2025}"),
    (r"\(Hu\s+et\s+al\.,\s+2022\)", r"\citep{hu2022lora}"),
    (r"\(Zou\s+et\s+al\.,\s+2023\)", r"\citep{zou2023repe}"),
    (r"\(Hubinger\s+et\s+al\.,\s+2024\)", r"\citep{hubinger2024sleeper}"),
    (r"\(Wang\s+et\s+al\.,\s+2025\)", r"\citep{wang2025persona}"),
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
    (r"Betley et al\.'s[ ~]\(2025\)",
     r"\citeauthor{betley2025em}'s \citeyearpar{betley2025em}"),
    (r"Betley et al\.[ ~]\(2025\)", r"\citet{betley2025em}"),
    (r"Wang et al\.'s[ ~]\(2025\)",
     r"\citeauthor{wang2025persona}'s \citeyearpar{wang2025persona}"),
    (r"Wang et al\.[ ~]\(2025\)", r"\citet{wang2025persona}"),
    (r"Hubinger et al\.[ ~]\(2024\)", r"\citet{hubinger2024sleeper}"),
    # Companion-paper reference (Reblitz-Richardson 2026, in preparation)
    (r"Reblitz-Richardson, 2026, in preparation",
     r"Reblitz-Richardson, 2026, in preparation"),  # leave as text
]

# Figure-insertion patterns. Each maps a sentence-prefix marker to a
# LaTeX figure block. Markers must match exactly once per file. The
# replacement uses the pre-existing prose as the in-text reference
# (\Cref or literal "Figure N") and inserts a float just before the
# paragraph.  After all FIGURE_INSERTS run, a post-pass rewrites any
# remaining bold "Figure N" mentions to "Figure~\ref{fig:label}" so
# that secondary references resolve cleanly.
FIGURE_INSERTS: list[tuple[str, str]] = [
    # Figure 1 — dense vs. MoE: same accuracy, different robustness (§4.1).
    (
        r"\\textbf\{Figure 1\}\s+contrasts\s+the\s+two\s+architectures\s+across\s+both\s+metrics\.",
        (
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            "  \\includegraphics[width=\\linewidth]{figure_1_dense_vs_moe.pdf}\n"
            "  \\caption{Dense and MoE encode moral content with near-identical "
            "accuracy but very different robustness. (a)~Per-layer moral probing "
            "accuracy for OLMoE-1B-7B and dense OLMo-2~1B; both peak at 99.0\\%, "
            "differing only at the early layers. (b)~Per-layer critical noise "
            "$\\sigma^*$ (smallest $\\sigma$ at which probe accuracy falls below "
            "0.6, on the log grid $\\{0.1, 0.3, 1.0, 3.0, 10.0\\}$): OLMoE is "
            "5.1$\\times$ more fragile (mean $\\sigma^*$ 0.84 vs.\\ 4.25) and "
            "concentrates robustness in the final two layers.}\n"
            "  \\label{fig:dense-vs-moe}\n"
            "\\end{figure}\n\n"
            "\\textbf{Figure~\\ref{fig:dense-vs-moe}} contrasts the two "
            "architectures across both metrics."
        ),
    ),

    # Figure 2 — no expert moral specialization (§4.2).
    (
        r"\\textbf\{Figure 2\}\s+shows\s+the\s+per-expert\s+accuracy\s+distribution\s+at\s+every\s+layer\.",
        (
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            "  \\includegraphics[width=\\linewidth]{figure_2_expert_uniformity.pdf}\n"
            "  \\caption{No expert moral specialization. (a)~Distribution of the "
            "64 per-expert probe accuracies at each layer (box plot) with the "
            "per-layer mean overlaid; every expert encodes moral content, with "
            "no sparse high-accuracy subset. (b)~The per-layer Gini coefficient "
            "of expert accuracy stays in $[0.016, 0.023]$, far below any "
            "concentration threshold, and is lowest at the late layers where "
            "encoding peaks.}\n"
            "  \\label{fig:expert-uniformity}\n"
            "\\end{figure}\n\n"
            "\\textbf{Figure~\\ref{fig:expert-uniformity}} shows the per-expert "
            "accuracy distribution at every layer."
        ),
    ),

    # Figure 3 — output dilution explains MoE fragility (§4.4).
    (
        r"\\textbf\{Figure 3\}\s+relates\s+the\s+output-scale\s+gap\s+to\s+component\s+fragility\.",
        (
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            "  \\includegraphics[width=\\linewidth]{figure_3_output_dilution.pdf}\n"
            "  \\caption{Output dilution explains MoE fragility. (a)~Per-layer "
            "feedforward output scale (standard deviation of the mean-pooled "
            "output) for the OLMoE MoE block vs.\\ the dense OLMo-2 MLP; the "
            "dense MLP output is 74$\\times$ larger on average. (b)~Per-layer "
            "critical noise for the three MoE perturbation targets: the router "
            "is most robust (mean $\\sigma^*$ 9.1), the aggregated output most "
            "fragile (mean $\\sigma^*$ 0.56), because the output operates on the "
            "diluted scale from panel~(a).}\n"
            "  \\label{fig:output-dilution}\n"
            "\\end{figure}\n\n"
            "\\textbf{Figure~\\ref{fig:output-dilution}} relates the "
            "output-scale gap to component fragility."
        ),
    ),

    # Figure 4 — specialization never emerges during training (§4.5).
    (
        r"\\textbf\{Figure 4\}\s+plots\s+the\s+training\s+trajectory\s+of\s+accuracy\s+and\s+concentration\.",
        (
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            "  \\includegraphics[width=\\linewidth]{figure_4_training_trajectory.pdf}\n"
            "  \\caption{Specialization never emerges during training. Across 11 "
            "OLMoE checkpoints (step 5K--1.2M, 20B--5{,}117B tokens), peak-layer "
            "and overall mean per-expert accuracy (left axis) stay in a 92--94\\% "
            "band from the earliest checkpoint, while the Gini coefficient of "
            "expert accuracy (right axis) stays flat near zero. Moral encoding is "
            "present from the start and never concentrates into specific "
            "experts.}\n"
            "  \\label{fig:training-trajectory}\n"
            "\\end{figure}\n\n"
            "\\textbf{Figure~\\ref{fig:training-trajectory}} plots the training "
            "trajectory of accuracy and concentration."
        ),
    ),
]

# Mapping for the secondary-reference post-pass: any remaining bold
# "Figure N" mention (one already replaced by the FIGURE_INSERTS above
# is now `\textbf{Figure~\ref{...}}`, so it won't match this) gets
# rewritten to a proper `\ref{}` so cross-references resolve.
FIGURE_LABEL_MAP: dict[str, str] = {
    "1": "fig:dense-vs-moe",
    "2": "fig:expert-uniformity",
    "3": "fig:output-dilution",
    "4": "fig:training-trajectory",
}

# Each generated section file must contain these figure labels after the
# injection pass. A missing label means an anchor sentence drifted (e.g. a
# v2 text rewrite) and the figure silently failed to inject — postprocess
# fails loudly instead of shipping a figureless paper. See main().
EXPECTED_FIGURES: dict[str, list[str]] = {
    "04_results.tex": [
        "fig:dense-vs-moe",
        "fig:expert-uniformity",
        "fig:output-dilution",
        "fig:training-trajectory",
    ],
}


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
    # Subsections will be added during drafting as the section
    # structure firms up.  Unknown references fall through to a
    # literal section symbol (no error, just no \Cref).
}


def convert_section_refs(text: str) -> str:
    """Map literal `\\S<N>(.<M>)` references to `\\Cref{<slug>}`.

    The unicode fixup pass replaced `§` with `\\S`, so by this point
    every section reference looks like `\\S4.1`, `\\S3`, or `\\SD.5`
    (appendix subsections).  Convert these to `\\Cref{<slug>}` using
    SECTION_LABEL_MAP.  When the reference is not in the map, fall
    back to a plain section symbol followed by the literal label
    (e.g.\\ `\\S{}D.5`) so LaTeX renders it correctly even without a
    cross-reference target.

    Skip refs inside `\\begin{Highlighting}...\\end{Highlighting}`
    blocks — `\\Cref` expansion clashes with the Verbatim environment's
    commandchars setup and triggers cleveref internal errors.
    """
    # Match either:
    #   \S<digit>(.<digit>)?   -> body section refs (§3.1, §4.2)
    #   \S<letter>(.<digit>)?  -> appendix refs (§A, §D.5)
    pattern = re.compile(r"\\S([A-Z](?:\.\d+)?|\d+(?:\.\d+)?)")

    def replace(match: re.Match) -> str:
        ref = match.group(1)
        if not ref:
            return match.group(0)
        slug = SECTION_LABEL_MAP.get(ref)
        if slug is None:
            # Unknown reference — emit a literal section symbol with
            # an empty argument followed by the label, so LaTeX
            # tokenises `\\S` correctly when the next char is a letter.
            return r"\S{}" + ref
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

    Populate as tables are added.  See
    `papers/1_accuracy_vs_fragility/build/postprocess.py` for a worked
    example: each entry is a `(fingerprint_regex, replacement_widths)`
    pair where the fingerprint matches pandoc's exact emitted column
    spec.
    """
    overrides: list[tuple[str, str]] = []
    for pat, repl in overrides:
        text = re.sub(pat, lambda _m, r=repl: r, text)
    return text


def convert_paths_to_path_macro(text: str) -> str:
    """Wrap `\\texttt{...}` containing slashes/dots/underscores in `\\path{...}`
    so the url package can break long identifiers at separators.

    Targets cells like:
        \\texttt{papers/1_accuracy_vs_fragility/scripts/phase_c4_3seed.py}
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
    out = fixup(text)
    p.write_text(out)
    missing = [
        label
        for label in EXPECTED_FIGURES.get(p.name, [])
        if ("\\label{" + label + "}") not in out
    ]
    if missing:
        print(
            f"postprocess.py: {p.name}: figure injection failed for {missing} "
            f"(anchor sentence missing or changed in the markdown source)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
