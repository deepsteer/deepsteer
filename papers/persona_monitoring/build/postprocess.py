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
    # Figure 1 — persona-probe emergence trajectory (§3.1).
    (
        r"\\textbf\{Figure 1\}\s+plots\s+the\s+persona-probe\s+trajectory\s+\(overall,\s+content-clean,\s+OOD\s+jailbreak\)\s+alongside\s+the\s+moral\s+/\s+sentiment\s+/\s+syntax\s+onsets\s+from\s+companion\s+work\.",
        (
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            "  \\includegraphics[width=\\linewidth]{figure_1_persona_trajectory.pdf}\n"
            "  \\caption{Persona-feature emergence trajectory on OLMo-2~1B "
            "early-training (37 checkpoints, 1K-step intervals). Persona "
            "overall (240-pair test set), content-clean within-category "
            "transfer, and OOD jailbreak fixture. Persona onset is "
            "concurrent with the moral / sentiment onsets reported in "
            "companion work (vertical dashed lines); persona is "
            "foundational at the 1K-step resolution we have data for.}\n"
            "  \\label{fig:persona-trajectory}\n"
            "\\end{figure}\n\n"
            "\\textbf{Figure~\\ref{fig:persona-trajectory}} plots the "
            "persona-probe trajectory (overall, content-clean, OOD "
            "jailbreak) alongside the moral / sentiment / syntax onsets "
            "from companion work."
        ),
    ),

    # Figure 2 — C10 v2 null on insecure-code LoRA (§4.1).
    (
        r"\\textbf\{Figure 2\}\s+summarizes\s+both\s+readouts\s+side-by-side:\s+per-\s*condition\s+\\texttt\{PersonaFeatureProbe\}\s+activation\s+\(left\)\s+and\s+behavioral\s+coherent-misalignment\s+rates\s+with\s+Wilson\s+95\s*\\%\s+CIs\s+\(right\)\.",
        (
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            "  \\includegraphics[width=\\linewidth]{figure_2_c10_v2_null.pdf}\n"
            "  \\caption{Persona mechanism does not engage at 1B under "
            "controlled insecure-code LoRA. (a)~Per-condition probe "
            "activation (160 generations each); insecure $-$ secure "
            "Cohen's $d = +0.032$ paired, 25$\\times$ below the 1.0~SD "
            "PROBE PASS threshold. (b)~Coherent-misalignment rates with "
            "Wilson 95\\,\\% CIs; insecure (1.56\\,\\%) and secure "
            "(0.69\\,\\%) overlap heavily.}\n"
            "  \\label{fig:c10-v2-null}\n"
            "\\end{figure}\n\n"
            "\\textbf{Figure~\\ref{fig:c10-v2-null}} summarizes both "
            "readouts side-by-side: per-condition "
            "\\texttt{PersonaFeatureProbe} activation (left) and "
            "behavioral coherent-misalignment rates with Wilson 95\\,\\% "
            "CIs (right)."
        ),
    ),

    # Figure 3 — Step 2 four-condition summary (§4.3).
    (
        r"\\textbf\{Figure 3\}\s+gives\s+the\s+four-condition\s+summary\s+across\s+both\s+metrics:\s+per-condition\s+probe\s+activation\s+\(left\)\s+and\s+behavioral\s+judge\s+score\s+\(right\)\.",
        (
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            "  \\includegraphics[width=\\linewidth]{figure_3_step2_summary.pdf}\n"
            "  \\caption{Step 2 four-condition summary. (a)~Per-condition "
            "persona-probe activation: vanilla LoRA shifts the probe by "
            "Cohen's $d = +2.29$; gradient\\_penalty brings it back to "
            "baseline (99.3\\,\\% suppression at 0.4\\,\\% SFT-loss cost); "
            "activation\\_patch backfires by amplification ($d = +3.79$). "
            "(b)~Per-condition behavioral judge score (0--10 persona-voice "
            "scale): vanilla and gradient\\_penalty match within "
            "0.01\\,/\\,10 despite probe Cohen's $d$ differing by 3.07~SD "
            "--- the Finding 3 dissociation.}\n"
            "  \\label{fig:step2-summary}\n"
            "\\end{figure}\n\n"
            "\\textbf{Figure~\\ref{fig:step2-summary}} gives the "
            "four-condition summary across both metrics: per-condition "
            "probe activation (left) and behavioral judge score (right)."
        ),
    ),

    # Figure 4 — fragility-locus shift (§4.4).
    (
        r"\\textbf\{Figure 4\}\s+plots\s+the\s+per-layer\s+breakdown\s+across\s+all\s+three\s+conditions:",
        (
            "\\begin{figure}[t]\n"
            "  \\centering\n"
            "  \\includegraphics[width=\\linewidth]{figure_4_fragility_locus.pdf}\n"
            "  \\caption{Insecure-code LoRA leaves a fragility-locus "
            "signature the persona probe and the behavioral judge miss "
            "(companion-paper methodology). (a)~Standard moral probe "
            "accuracy across 16 transformer layers --- flat across base, "
            "insecure, and secure conditions ($|\\Delta| \\leq 0.021$). "
            "(b)~Per-layer critical noise on the discrete log grid "
            "$\\{0.1, 0.3, 1.0, 3.0, 10.0\\}$: the base-model robustness "
            "peak at layer 7 relocates to layers 9--10 under insecure-code "
            "LoRA specifically, with layers 6--7 collapsing to "
            "$\\sigma = 1$.}\n"
            "  \\label{fig:fragility-locus}\n"
            "\\end{figure}\n\n"
            "\\textbf{Figure~\\ref{fig:fragility-locus}} plots the "
            "per-layer breakdown across all three conditions:"
        ),
    ),
]

# Mapping for the secondary-reference post-pass: any remaining bold
# "Figure N" mention (one already replaced by the FIGURE_INSERTS above
# is now `\textbf{Figure~\ref{...}}`, so it won't match this) gets
# rewritten to a proper `\ref{}` so cross-references resolve.
FIGURE_LABEL_MAP: dict[str, str] = {
    "1": "fig:persona-trajectory",
    "2": "fig:c10-v2-null",
    "3": "fig:step2-summary",
    "4": "fig:fragility-locus",
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
    (e.g.\ `\\S{}D.5`) so LaTeX renders it correctly even without a
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
    `papers/accuracy_vs_fragility/build/postprocess.py` for a worked
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
