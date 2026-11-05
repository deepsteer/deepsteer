"""Constitution documents and their tenet decompositions (CST line)."""

from __future__ import annotations

from deepsteer.constitutions.documents import (
    BODY_RULES,
    Document,
    Section,
    cut_body,
    load_document,
    sha256_text,
    split_sections,
)
from deepsteer.constitutions.tenets import (
    Tenet,
    TenetSet,
    check_ids,
    core_intersection,
    document_conflicts,
    majority_match,
    tconf_record,
    verify_quotes,
)

__all__ = [
    "BODY_RULES",
    "Document",
    "Section",
    "Tenet",
    "TenetSet",
    "check_ids",
    "core_intersection",
    "cut_body",
    "document_conflicts",
    "load_document",
    "majority_match",
    "sha256_text",
    "split_sections",
    "tconf_record",
    "verify_quotes",
]
