"""Tenets: atomic, testable behavioral claims extracted from a constitution (CST_PLAN §1, §2.2).

This module holds the tenet schema, its committed-artifact format, and the registered rules that
turn model or rater judgments into sets: the cross-family majority rule for the core intersection
``T_core`` and the document-internal conflict rule for ``T_conf``. The model calls that produce the
judgments live in the caller (subagents, CLIs or API clients); every judgment enters here as data.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from deepsteer.constitutions.documents import sha256_text

logger = logging.getLogger(__name__)

TENET_SCHEMA_VERSION = 1
KINDS = ("directive", "prohibition", "priority", "permission", "disposition")


@dataclasses.dataclass
class Tenet:
    """One tenet.

    ``quote`` is verbatim from the document body (checked by :func:`verify_quotes`); ``claim`` is
    the tenet in one testable sentence; ``conditions`` states when it applies; ``conditioned_on``
    lists tenet ids whose priority this tenet defers to or is gated by (the conflict rule reads it).
    """

    tenet_id: str
    doc_id: str
    section: str
    quote: str
    claim: str
    conditions: str = ""
    kind: str = "directive"
    conditioned_on: list[str] = dataclasses.field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.tenet_id}: kind {self.kind!r} not in {KINDS}")


@dataclasses.dataclass
class TenetSet:
    """A document's tenets with provenance (the committed artifact ``T(C)``)."""

    doc_id: str
    body_sha256: str
    tenets: list[Tenet]
    provenance: dict[str, Any] = dataclasses.field(default_factory=dict)

    def ids(self) -> list[str]:
        return [t.tenet_id for t in self.tenets]

    def by_id(self) -> dict[str, Tenet]:
        return {t.tenet_id: t for t in self.tenets}

    def to_dict(self) -> dict[str, Any]:
        d = {"schema_version": TENET_SCHEMA_VERSION, "doc_id": self.doc_id,
             "body_sha256": self.body_sha256, "provenance": self.provenance,
             "tenets": [dataclasses.asdict(t) for t in self.tenets]}
        d["tenet_set_sha256"] = sha256_text(json.dumps(d["tenets"], sort_keys=True))
        return d

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> TenetSet:
        d = json.loads(Path(path).read_text())
        if d.get("schema_version") != TENET_SCHEMA_VERSION:
            raise ValueError(f"{path}: schema_version {d.get('schema_version')}")
        ts = cls(d["doc_id"], d["body_sha256"], [Tenet(**t) for t in d["tenets"]],
                 d.get("provenance", {}))
        want = d.get("tenet_set_sha256")
        if want and ts.to_dict()["tenet_set_sha256"] != want:
            raise ValueError(f"{path}: tenet_set_sha256 mismatch (file edited by hand?)")
        return ts


def _norm_ws(s: str) -> str:
    return " ".join(s.replace("’", "'").replace("“", '"').replace("”", '"').split())


def verify_quotes(ts: TenetSet, body: str) -> list[str]:
    """Tenet ids whose quote does not occur in the body (whitespace and curly quotes normalized).

    The most probable failure of a model-written decomposition is a paraphrased "quote"; a tenet
    set with any id returned here is not committed.
    """
    nb = _norm_ws(body)
    return [t.tenet_id for t in ts.tenets if _norm_ws(t.quote) not in nb]


def check_ids(ts: TenetSet) -> None:
    """Raise on duplicate ids or dangling ``conditioned_on`` references."""
    ids = ts.ids()
    dup = [i for i, c in Counter(ids).items() if c > 1]
    if dup:
        raise ValueError(f"duplicate tenet ids: {dup[:5]}")
    known = set(ids)
    dangling = sorted({c for t in ts.tenets for c in t.conditioned_on if c not in known})
    if dangling:
        raise ValueError(f"conditioned_on names unknown ids: {dangling[:5]}")


# ---------------------------------------------------------------------------------------------
# T_core: the registered majority rule (CST_PLAN §2.2)
# ---------------------------------------------------------------------------------------------


def majority_match(votes: dict[str, bool], n_required: int = 2) -> bool:
    """A candidate pair matches when at least ``n_required`` raters (families) say it directs the
    same behavior in the same conditions. Raters that abstained are absent from ``votes``; the rule
    needs ``n_required`` yes votes, so two abstentions out of three can never produce a match."""
    return sum(bool(v) for v in votes.values()) >= n_required


def core_intersection(
    pair_votes: dict[tuple[str, str], dict[str, bool]], n_required: int = 2
) -> dict[str, Any]:
    """``T_core`` from per-pair family votes ``{(anth_id, spec_id): {family: bool}}``.

    Returns the matched pairs, the anth and spec ids in the core, and the dissent per matched pair
    (families that voted no), which the plan requires be recorded.
    """
    matched, dissent = [], {}
    for pair, votes in sorted(pair_votes.items()):
        if majority_match(votes, n_required):
            matched.append(pair)
            no = sorted(f for f, v in votes.items() if not v)
            if no:
                dissent[f"{pair[0]}|{pair[1]}"] = no
    return {"rule": f">= {n_required} family yes votes (CST_PLAN §2.2)",
            "pairs": [list(p) for p in matched],
            "anth_ids": sorted({a for a, _ in matched}),
            "spec_ids": sorted({s for _, s in matched}),
            "dissent": dissent}


# ---------------------------------------------------------------------------------------------
# T_conf: document-internal conflicts (CST_PLAN §2.2); panel-disagreement members join at G1
# ---------------------------------------------------------------------------------------------


def document_conflicts(
    ts: TenetSet, competing_pairs: list[tuple[str, str]] | None = None
) -> dict[str, list[str]]:
    """Tenet ids entering ``T_conf`` from the document itself, with the reason for each.

    A tenet enters when (a) it is conditioned on another tenet's priority (``conditioned_on``
    non-empty), or (b) it appears in a decomposition-flagged pair that directs different behavior
    at overlapping conditions (``competing_pairs``).
    """
    reasons: dict[str, list[str]] = {}
    for t in ts.tenets:
        for c in t.conditioned_on:
            reasons.setdefault(t.tenet_id, []).append(f"conditioned_on:{c}")
    known = set(ts.ids())
    for a, b in competing_pairs or []:
        if a not in known or b not in known:
            raise ValueError(f"competing pair names unknown ids: {a}, {b}")
        reasons.setdefault(a, []).append(f"competes:{b}")
        reasons.setdefault(b, []).append(f"competes:{a}")
    return {k: sorted(set(v)) for k, v in sorted(reasons.items())}
