#!/usr/bin/env python3
"""Post-process pandoc-generated LaTeX sections for Paper 4.

Converts markdown-style author-year citations to natbib commands,
applies unicode fixups, and rewrites section cross-references.

Run via the Makefile per-section conversion rule.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CITES_PARENS = [
    (r"\(Vig\s+et\s+al\.,\s+2020;\s+Meng\s+et\s+al\.,\s+2022\)",
     r"\citep{vig2020causal,meng2022locating}"),
    (r"\(Bricken\s+et\s+al\.,\s+2023;\s+Cunningham\s+et\s+al\.,\s+2023\)",
     r"\citep{bricken2023monosemanticity,cunningham2023sparse}"),
    (r"\(Vig\s+et\s+al\.,\s+2020\)", r"\citep{vig2020causal}"),
    (r"\(Meng\s+et\s+al\.,\s+2022\)", r"\citep{meng2022locating}"),
    (r"\(Geiger\s+et\s+al\.,\s+2021\)", r"\citep{geiger2021causal}"),
    (r"\(Arditi\s+et\s+al\.,\s+2024\)", r"\citep{arditi2024refusal}"),
    (r"\(Turner\s+et\s+al\.,\s+2023\)", r"\citep{turner2023activation}"),
    (r"\(Zou\s+et\s+al\.,\s+2023\)", r"\citep{zou2023representation}"),
    (r"\(Marks\s+\\&\s+Tegmark,\s+2024\)", r"\citep{marks2024geometry}"),
    (r"\(Bricken\s+et\s+al\.,\s+2023\)", r"\citep{bricken2023monosemanticity}"),
    (r"\(Cunningham\s+et\s+al\.,\s+2023\)", r"\citep{cunningham2023sparse}"),
    (r"\(Templeton\s+et\s+al\.,\s+2024\)", r"\citep{templeton2024scaling}"),
    (r"\(Graham\s+et\s+al\.,\s+2013\)", r"\citep{graham2013mft}"),
    (r"\(Clifford\s+et\s+al\.,\s+2015\)", r"\citep{clifford2015moral}"),
    (r"\(Raffel\s+et\s+al\.,\s+2020\)", r"\citep{raffel2020c4}"),
    (r"\(Reblitz-Richardson,\s+2026\)", r"\citep{reblitzrichardson2026geometry}"),
    (r"\(Groeneveld\s+et\s+al\.,\s+2024\)", r"\citep{groeneveld2024olmo}"),
]

CITES_INTEXT = [
    (r"Reblitz-Richardson[ ~]\(2026\)", r"\citet{reblitzrichardson2026geometry}"),
    (r"Arditi et al\.[ ~]\(2024\)", r"\citet{arditi2024refusal}"),
    (r"Turner et al\.[ ~]\(2023\)", r"\citet{turner2023activation}"),
    (r"Zou et al\.[ ~]\(2023\)", r"\citet{zou2023representation}"),
    (r"Marks \\& Tegmark[ ~]\(2024\)", r"\citet{marks2024geometry}"),
    (r"Marks and Tegmark[ ~]\(2024\)", r"\citet{marks2024geometry}"),
    (r"Geiger et al\.[ ~]\(2021\)", r"\citet{geiger2021causal}"),
    (r"Graham et al\.[ ~]\(2013\)", r"\citet{graham2013mft}"),
    (r"Clifford et al\.[ ~]\(2015\)", r"\citet{clifford2015moral}"),
]

FIGURE_INSERTS: list[tuple[str, str]] = []
FIGURE_LABEL_MAP = {}


def rewrite_secondary_figure_refs(text: str) -> str:
    pattern = re.compile(r"\\textbf\{Figure (\d+)\}")
    def replace(match: re.Match) -> str:
        n = match.group(1)
        label = FIGURE_LABEL_MAP.get(n)
        if label is None:
            return match.group(0)
        return r"\textbf{Figure~\ref{" + label + "}}"
    return pattern.sub(replace, text)


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
    "−": r"$-$",
    "✓": r"\checkmark{}",
    "✗": r"\ding{55}",
    "²": r"$^2$",
    "³": r"$^3$",
    "½": r"$\tfrac{1}{2}$",
    "‰": r"\textperthousand{}",
    "—": r"---",
    "–": r"--",
    "…": r"\ldots{}",
    "§": r"\S",
    "°": r"\textdegree{}",
    "'": "'",
    "'": "'",
    "“": "``",
    "”": "''",
    "​": "",
}


def apply_unicode_fixups(text: str) -> str:
    for src, dst in UNICODE_FIXUPS.items():
        text = text.replace(src, dst)
    return text


SECTION_LABEL_MAP = {
    "1": "introduction",
    "2": "related-work",
    "3": "methods",
    "4": "results",
    "5": "discussion",
    "6": "conclusion",
    "2.1": "causal-methods-in-mechanistic-interpretability",
    "2.2": "representation-engineering",
    "2.3": "sparse-autoencoders-for-feature-discovery",
    "2.4": "moral-reasoning-in-language-models",
    "3.1": "model-and-dataset",
    "3.2": "causal-validation",
    "3.3": "behavioral-grounding",
    "3.4": "sparse-autoencoder-analysis",
    "4.1": "causal-validation",
    "4.2": "steering-injection-shows-dose--response-specificity",
    "4.3": "behavioral-grounding",
    "4.4": "sae-analysis",
    "5.1": "three-converging-lines-of-evidence",
    "5.2": "the-care-saturation-phenomenon",
    "5.3": "layer-dependent-causal-roles",
    "5.4": "toward-a-steering-fitness-function",
    "5.5": "limitations",
}


def convert_section_refs(text: str) -> str:
    pattern = re.compile(r"\\S(\d+(?:\.\d+)?)")
    def replace(match: re.Match) -> str:
        ref = match.group(1)
        slug = SECTION_LABEL_MAP.get(ref)
        if slug is None:
            return match.group(0)
        return r"\Cref{" + slug + "}"
    block_re = re.compile(
        r"(\\begin\{Highlighting\}.*?\\end\{Highlighting\})", re.DOTALL
    )
    parts = block_re.split(text)
    for i in range(0, len(parts), 2):
        parts[i] = pattern.sub(replace, parts[i])
    return "".join(parts)


def convert_paths_to_path_macro(text: str) -> str:
    pattern = re.compile(r"\\texttt\{((?:[^{}\\]|\\[_&#$%{}])+)\}")
    def replace(match: re.Match) -> str:
        content = match.group(1)
        if not re.search(r"[/.]|\\_", content):
            return match.group(0)
        unescaped = re.sub(r"\\([_&#$%{}])", r"\1", content)
        return r"\path|" + unescaped + "|"
    return pattern.sub(replace, text)


def fixup(text: str) -> str:
    for pat, repl in CITES_PARENS + CITES_INTEXT:
        text = re.sub(pat, lambda _m, r=repl: r, text)
    for pat, repl in FIGURE_INSERTS:
        text = re.sub(pat, lambda _m, r=repl: r, text)
    text = rewrite_secondary_figure_refs(text)
    text = apply_unicode_fixups(text)
    text = convert_section_refs(text)
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
