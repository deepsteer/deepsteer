#!/usr/bin/env python3
# ruff: noqa: E501  (text-holding script: the inserted blocks are kept verbatim)
"""Apply the KDG paper's arXiv ID everywhere it belongs, once the ID arrives (ARXIV_SUBMISSION.md).

    python3 papers/kdg_judgment_action/apply_arxiv_id.py 2610.01234 --dry-run
    python3 papers/kdg_judgment_action/apply_arxiv_id.py 2610.01234

Edits: CITATION (BibTeX entry), README.md (the KDG line of its Papers list), papers/README.md (Published section),
papers/SYNTHESIS.md (one line), ~/dev/orionr.github.io/publications.html (newest-first entry, a
separate repo committed there). Each edit is idempotent: a file that already carries the ID is skipped.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SITE = Path.home() / "dev" / "orionr.github.io" / "publications.html"
TITLE = (
    "Principled Under Pressure: Post-Training Decides Whether LLMs Act on Their Own Moral Judgment"
)
FL, MN = "2609.14759", "2609.14754"


def edits(aid: str) -> list[tuple[Path, str, str, str]]:
    """(file, anchor, mode, text): insert text before the anchor, replace the anchor with text, or
    append when anchor is None."""
    cite = (
        "\n@misc{reblitzrichardson2026principled,\n"
        f"  title={{{TITLE}}},\n"
        "  author={Orion Reblitz-Richardson},\n"
        "  year={2026},\n"
        f"  eprint={{{aid}}},\n"
        "  archivePrefix={arXiv},\n"
        "  primaryClass={cs.LG},\n"
        f"  url={{https://arxiv.org/abs/{aid}}},\n"
        "}\n"
    )
    # The root README's Papers list already names this paper with a "(arXiv forthcoming)" marker.
    readme_marker = f"*{TITLE}* (arXiv forthcoming)"
    readme_link = f"*{TITLE}* ([arXiv:{aid}](https://arxiv.org/abs/{aid}))"
    published = (
        "## Published (arXiv, 2026)\n\n"
        f"- **Flagship.** *Refusal Reads Only a Slice of What the Model Knows* (arXiv:{FL}),\n"
        "  `fl_what_refusal_reads/`.\n"
        f"- **Methods note.** *Calibrating Interpretability Instruments Before Trusting Their Verdicts*\n"
        f"  (arXiv:{MN}), `mn_instruments_before_verdicts/`.\n"
        f"- **Judgment–action gap.** *{TITLE}* (arXiv:{aid}), `kdg_judgment_action/`: open models act\n"
        "  against their own stated moral judgment under pressure, and whether they do follows the\n"
        "  post-training recipe (same Llama-3.1 base: Meta's carries the gap, Tulu 3's does not).\n\n"
    )
    synth = (
        f"\n**KDG paper on arXiv (arXiv:{aid}).** *{TITLE}*; the four-claim paper of the Session C "
        "gate, with the 2026-10-01 stage wording and the 2026-10-04 submit-gate edits.\n"
    )
    site = (
        '      <div class="publication">\n'
        f'        <div class="publication-title">{TITLE}</div>\n'
        '        <div class="publication-summary">Measures how often open models act against their '
        "own stated moral judgment under pressure, and shows that whether they do follows the "
        "post-training recipe: the same base weights yield a model that carries the gap or one that "
        "does not.</div>\n"
        '        <div class="publication-venue">arXiv preprint, 2026</div>\n'
        '        <div class="publication-links">\n'
        f'          <a href="https://arxiv.org/abs/{aid}">Paper</a>'
        '<span class="separator-dot"></span>\n'
        '          <a href="https://github.com/deepsteer/deepsteer">Code</a>\n'
        "        </div>\n"
        "      </div>\n\n"
    )
    return [
        (REPO / "CITATION", None, "append", cite),
        (REPO / "README.md", readme_marker, "replace", readme_link),
        (REPO / "papers" / "README.md", "## Paper 1 ", "before", published),
        (REPO / "papers" / "SYNTHESIS.md", None, "append", synth),
        (
            SITE,
            '      <div class="publication">\n        <div class="publication-title">Refusal Reads',
            "before",
            site,
        ),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arxiv_id")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not re.fullmatch(r"\d{4}\.\d{4,5}", a.arxiv_id):
        raise SystemExit(f"not an arXiv id: {a.arxiv_id!r}")
    for path, anchor, mode, text in edits(a.arxiv_id):
        t = path.read_text()
        if a.arxiv_id in t:
            print(f"skip (already carries the id): {path}")
            continue
        if mode == "append":
            new = t.rstrip("\n") + "\n" + text
        else:
            if t.count(anchor) != 1:
                raise SystemExit(f"{path}: anchor not found exactly once: {anchor[:40]!r}")
            new = t.replace(anchor, text if mode == "replace" else text + anchor)
        print(f"{'would edit' if a.dry_run else 'edit'}: {path} (+{len(text.splitlines())} lines)")
        if not a.dry_run:
            path.write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
