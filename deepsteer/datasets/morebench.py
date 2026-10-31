"""MoReBench runtime loader: fetched at a pinned Hugging Face revision and hash-verified.

MoReBench (Chiu, Lee, et al.; arXiv:2510.16380, ICLR 2026) publishes 500 public scenarios with
expert rubrics at ``huggingface.co/datasets/morebench/morebench`` under CC BY 4.0. The files are
fetched into the user cache, not vendored (``papers/MRB_PLAN.md`` §9); the committed manifest
(``morebench_manifest.json``) holds file names and sha256 hashes only.

Scenario ids follow the upstream harness: the CSV row index (``idx`` in its generation files), as
``mrb-<idx:03d>`` (public) and ``mrbt-<idx:03d>`` (theory). The ``RUBRIC`` column is a Python
literal, parsed with :func:`ast.literal_eval` as upstream does.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MOREBENCH_REPO = "morebench/morebench"
MOREBENCH_REVISION = "d900e89f942f784f8e0113935034b4f0541a0b55"
MOREBENCH_FILES = ("morebench_public.csv", "morebench_theory.csv", "README.md")
MANIFEST_PATH = Path(__file__).with_name("morebench_manifest.json")


@dataclass(frozen=True)
class MoReBenchCriterion:
    """One rubric criterion: id, text, weight (−3..+3, never 0) and expert-labelled dimension."""

    criterion_id: str
    title: str
    weight: int
    dimension: str


@dataclass(frozen=True)
class MoReBenchItem:
    """One scenario and its rubric."""

    scenario_id: str
    idx: int
    dilemma: str
    source: str
    dilemma_type: str
    role: str
    theory: str
    context: str
    criteria: tuple[MoReBenchCriterion, ...] = field(hash=False)


def default_cache_dir(revision: str = MOREBENCH_REVISION) -> Path:
    """``$DEEPSTEER_CACHE`` (default ``~/.cache/deepsteer``)/morebench/data/<revision>."""
    base = Path(os.environ.get("DEEPSTEER_CACHE", Path.home() / ".cache" / "deepsteer"))
    return base / "morebench" / "data" / revision


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(path.read_text())


def verify_morebench(root: Path, manifest: dict | None = None) -> None:
    """Check each manifest file under ``root`` against its sha256.

    Raises:
        ValueError: listing every missing or mismatched file.
    """
    manifest = manifest or load_manifest()
    bad = []
    for name, digest in sorted(manifest["files"].items()):
        p = Path(root) / name
        if not p.is_file():
            bad.append(f"missing: {name}")
        elif sha256_file(p) != digest:
            bad.append(f"sha256 mismatch: {name}")
    if bad:
        raise ValueError(f"MoReBench files at {root} do not match revision "
                         f"{manifest['revision']}: " + "; ".join(bad))


def fetch_morebench(root: Path | None = None, revision: str = MOREBENCH_REVISION) -> Path:
    """Download the pinned files into ``root`` (default :func:`default_cache_dir`) and verify."""
    root = Path(root) if root is not None else default_cache_dir(revision)
    missing = [f for f in MOREBENCH_FILES if not (root / f).is_file()]
    if missing:
        from huggingface_hub import hf_hub_download

        for f in missing:
            logger.info("Fetching %s@%s:%s", MOREBENCH_REPO, revision[:12], f)
            hf_hub_download(MOREBENCH_REPO, f, repo_type="dataset", revision=revision,
                            local_dir=root)
    verify_morebench(root)
    return root


def _items(path: Path, prefix: str) -> list[MoReBenchItem]:
    csv.field_size_limit(1 << 30)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows):
        crit = tuple(
            MoReBenchCriterion(c["id"], c["title"], int(c["weight"]),
                               c["annotations"]["rubric_dimension"])
            for c in ast.literal_eval(r["RUBRIC"])
        )
        out.append(MoReBenchItem(
            scenario_id=f"{prefix}-{i:03d}", idx=i, dilemma=r["DILEMMA"],
            source=r["DILEMMA_SOURCE"], dilemma_type=r["DILEMMA_TYPE"], role=r["ROLE_DOMAIN"],
            theory=r["THEORY"], context=r["CONTEXT"], criteria=crit))
    return out


def load_morebench(
    root: Path | None = None, subset: str = "public", verify: bool = True
) -> list[MoReBenchItem]:
    """Load the public set (500) or the theory set (150) at the pinned revision.

    Args:
        root: A directory holding the pinned files; default :func:`fetch_morebench`.
        subset: ``"public"`` or ``"theory"``.
        verify: Check the files against the manifest first.
    """
    root = fetch_morebench() if root is None else Path(root)
    if verify:
        verify_morebench(root)
    name, prefix = {"public": ("morebench_public.csv", "mrb"),
                    "theory": ("morebench_theory.csv", "mrbt")}[subset]
    items = _items(root / name, prefix)
    logger.info("Loaded %d MoReBench %s scenarios", len(items), subset)
    return items
