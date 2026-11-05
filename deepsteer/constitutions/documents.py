"""Pinned constitution documents: load the fetched text, verify its pin, cut the document body.

A document is pinned in a YAML file (``papers/st_live_reference/constitutions.yaml``) by id, URL,
date, license and the sha256 of the extracted page text. The page text carries site chrome
(navigation, footer). The *body* is the span the document itself defines as the document; its
boundaries are recorded per document in ``BODY_RULES`` and the body's own sha256 is reported so a
tenet set or corpus can cite exactly what it was derived from.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Body rules: (first line of the body, first line after the body, or None for end of text).
# C_anth: the page states "The preface and the acknowledgements are not part of the official
# constitution", so the body runs from the first "## Overview" heading to "## Acknowledgements".
# The chapter-navigation lines that follow "## Chapter 01 : Overview" sit before the body start.
# C_spec: the body runs from the first "## Overview" heading to the end of the extracted text
# (the page's version banner and table of contents sit before it).
BODY_RULES: dict[str, tuple[str, str | None]] = {
    "C_anth": ("## Overview", "## Acknowledgements"),
    "C_spec": ("## Overview", None),
}

_HEADING = re.compile(r"^## (.+)$")


def sha256_text(text: str) -> str:
    """sha256 of a text's UTF-8 encoding."""
    return hashlib.sha256(text.encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class Section:
    """One ``## `` section of a document body."""

    index: int
    heading: str
    text: str  # the section's lines after its heading, joined with newlines

    @property
    def section_id(self) -> str:
        return f"s{self.index:03d}"


@dataclasses.dataclass(frozen=True)
class Document:
    """A pinned document's text and its body."""

    doc_id: str
    text: str
    pin: dict

    @property
    def body(self) -> str:
        return cut_body(self.doc_id, self.text)

    @property
    def body_sha256(self) -> str:
        return sha256_text(self.body)

    def sections(self) -> list[Section]:
        return split_sections(self.body)


def cut_body(doc_id: str, text: str) -> str:
    """The document body by its registered rule; raises if a boundary line is missing."""
    if doc_id not in BODY_RULES:
        raise KeyError(f"no body rule for {doc_id!r}")
    start, end = BODY_RULES[doc_id]
    lines = text.split("\n")
    try:
        i = lines.index(start)
    except ValueError as e:
        raise ValueError(f"{doc_id}: body start {start!r} not found") from e
    j = len(lines)
    if end is not None:
        try:
            j = lines.index(end, i + 1)
        except ValueError as e:
            raise ValueError(f"{doc_id}: body end {end!r} not found") from e
    return "\n".join(lines[i:j]).strip("\n")


def split_sections(body: str) -> list[Section]:
    """Split a body into ``## `` sections in order (text before the first heading is dropped)."""
    out: list[Section] = []
    head: str | None = None
    buf: list[str] = []
    for line in body.split("\n"):
        m = _HEADING.match(line)
        if m:
            if head is not None:
                out.append(Section(len(out), head, "\n".join(buf).strip()))
            head, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if head is not None:
        out.append(Section(len(out), head, "\n".join(buf).strip()))
    return out


def load_document(doc_id: str, pins_path: Path, text_dir: Path) -> Document:
    """Load ``<text_dir>/<doc_id>.txt`` and verify its sha256 against the pin in ``pins_path``.

    Raises ``ValueError`` on a mismatch: the pinned text is the source of record, and a changed
    file is a different document.
    """
    import yaml

    pins = yaml.safe_load(Path(pins_path).read_text())
    pin = next((d for d in pins["documents"] if d["id"] == doc_id), None)
    if pin is None:
        raise KeyError(f"{doc_id!r} not pinned in {pins_path}")
    text = (Path(text_dir) / f"{doc_id}.txt").read_text()
    got = sha256_text(text)
    if got != pin["text_sha256"]:
        raise ValueError(f"{doc_id}: text sha256 {got[:12]} != pinned {pin['text_sha256'][:12]}")
    logger.info("%s: pinned text verified (%d chars)", doc_id, len(text))
    return Document(doc_id, text, pin)
