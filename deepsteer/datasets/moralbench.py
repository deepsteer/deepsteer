"""MoralBench runtime loader: fetched at a pinned commit, hash-verified, never vendored.

MoralBench (Ji, Chen, Jin, Xu, Hua, Zhang; arXiv:2406.04428, SIGKDD Explorations 27(1), 2025)
lives at ``github.com/agiresearch/MoralBench``. The upstream repository states no license
(checked 2026-10-30 at the pinned commit), so its questions, system prompt and answer keys are
used under their owners' terms: the user fetches them, and nothing of theirs is committed to
this repository or shipped in the wheel. The committed manifest
(``moralbench_manifest.json``) holds file paths and sha256 hashes only.

The prompt rendering reproduces the upstream harness's behaviour as described in
``papers/MB_PLAN.md`` §1: each file is read as lines (newlines kept) and the lines are joined
with ``"\\n"``, which doubles every line break.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from deepsteer.core.types import MoralFoundation

logger = logging.getLogger(__name__)

MORALBENCH_REPO = "https://github.com/agiresearch/MoralBench.git"
MORALBENCH_COMMIT = "84100440a28d040eca27df32b93f44685ad69ce5"
MORALBENCH_SETS: tuple[str, ...] = ("MFQ_30", "6_concepts", "MFQ_30_compare", "6_concepts_compare")
SYSTEM_PROMPT_PATH = "template/moral_system.txt"

# Upstream foundation labels (item-id stems) to the library's six foundations.
FOUNDATION_MAP: dict[str, MoralFoundation] = {
    "harm": MoralFoundation.CARE_HARM,
    "fairness": MoralFoundation.FAIRNESS_CHEATING,
    "ingroup": MoralFoundation.LOYALTY_BETRAYAL,
    "authority": MoralFoundation.AUTHORITY_SUBVERSION,
    "purity": MoralFoundation.SANCTITY_DEGRADATION,
    "liberty": MoralFoundation.LIBERTY_OPPRESSION,
}

MANIFEST_PATH = Path(__file__).with_name("moralbench_manifest.json")


@dataclass(frozen=True)
class MoralBenchItem:
    """One MoralBench item as the upstream harness presents it.

    Attributes:
        set_name: One of :data:`MORALBENCH_SETS`.
        item_id: The question file's stem, e.g. ``"harm_1"``.
        foundation_label: The upstream label (``item_id`` without its index).
        foundation: The label mapped through :data:`FOUNDATION_MAP`.
        prompt: The rendered user prompt (newlines doubled, as upstream).
        key: Score per answer letter, from ``answers/<set_name>.json``.
    """

    set_name: str
    item_id: str
    foundation_label: str
    foundation: MoralFoundation
    prompt: str
    key: dict[str, float] = field(hash=False)

    @property
    def qualified_id(self) -> str:
        """``"<set_name>/<item_id>"``, unique across sets."""
        return f"{self.set_name}/{self.item_id}"


def default_cache_dir(commit: str = MORALBENCH_COMMIT) -> Path:
    """Checkout location: ``$DEEPSTEER_CACHE`` (default ``~/.cache/deepsteer``)/moralbench/<sha>."""
    base = Path(os.environ.get("DEEPSTEER_CACHE", Path.home() / ".cache" / "deepsteer"))
    return base / "moralbench" / commit


def readlines_join(text: str) -> str:
    """Reproduce ``"\\n".join(f.readlines())`` on a text-mode file holding ``text``.

    ``readlines()`` on a universal-newline stream splits after each ``"\\n"`` and keeps it, so
    joining with ``"\\n"`` doubles every line break; a final line without a newline gains none.
    """
    parts = text.split("\n")
    lines = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return "\n".join(lines)


def _read_text(path: Path) -> str:
    # text mode with universal newlines, as a default open() does
    with open(path, encoding="utf-8") as f:
        return f.read()


def sha256_file(path: Path) -> str:
    """sha256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """The committed manifest: repo, commit, and a sha256 per data file."""
    return json.loads(path.read_text())


def verify_moralbench(root: Path, manifest: dict | None = None) -> None:
    """Check every manifest file under ``root`` against its sha256.

    Raises:
        ValueError: listing every missing or mismatched file.
    """
    manifest = manifest or load_manifest()
    bad = []
    for rel, digest in sorted(manifest["files"].items()):
        p = root / rel
        if not p.is_file():
            bad.append(f"missing: {rel}")
        elif sha256_file(p) != digest:
            bad.append(f"sha256 mismatch: {rel}")
    if bad:
        raise ValueError(
            f"MoralBench checkout at {root} does not match the manifest for commit "
            f"{manifest['commit']}: " + "; ".join(bad)
        )


def fetch_moralbench(root: Path | None = None, commit: str = MORALBENCH_COMMIT) -> Path:
    """Clone MoralBench at ``commit`` into ``root`` (default :func:`default_cache_dir`) and verify.

    An existing checkout is verified, not re-cloned. Needs ``git`` and network access on first
    use.

    Returns:
        The checkout directory.
    """
    root = Path(root) if root is not None else default_cache_dir(commit)
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Cloning %s at %s into %s", MORALBENCH_REPO, commit[:12], root)
        subprocess.run(["git", "clone", "--quiet", MORALBENCH_REPO, str(root)], check=True)
        subprocess.run(
            ["git", "-c", "advice.detachedHead=false", "checkout", "--quiet", commit],
            cwd=root, check=True,
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != commit:
        raise ValueError(f"MoralBench checkout at {root} is at {head}, expected {commit}")
    verify_moralbench(root)
    return root


def load_moralbench(
    root: Path | None = None,
    sets: tuple[str, ...] = MORALBENCH_SETS,
    verify: bool = True,
) -> tuple[str, list[MoralBenchItem]]:
    """Load the rendered system prompt and the items of ``sets``.

    Items are the question files of each set, in sorted id order; keys in an answer file
    without a question file are ignored (the pinned commit has such keys, see
    ``papers/mb_moralbench/SOURCE.md``).

    Args:
        root: A checkout; default :func:`fetch_moralbench` (clones on first use).
        sets: Which item sets to load.
        verify: Check the checkout against the manifest first.

    Returns:
        ``(system_prompt, items)``.

    Raises:
        KeyError: a question file has no answer key, or an unknown foundation label.
    """
    root = fetch_moralbench() if root is None else Path(root)
    if verify:
        verify_moralbench(root)
    system_prompt = readlines_join(_read_text(root / SYSTEM_PROMPT_PATH))
    items: list[MoralBenchItem] = []
    for set_name in sets:
        keys = json.loads(_read_text(root / "answers" / f"{set_name}.json"))
        for q in sorted((root / "questions" / set_name).glob("*.txt")):
            label = q.stem.rsplit("_", 1)[0]
            items.append(
                MoralBenchItem(
                    set_name=set_name,
                    item_id=q.stem,
                    foundation_label=label,
                    foundation=FOUNDATION_MAP[label],
                    prompt=readlines_join(_read_text(q)),
                    key={k: float(v) for k, v in keys[q.stem].items()},
                )
            )
    logger.info("Loaded %d MoralBench items from %d sets", len(items), len(sets))
    return system_prompt, items
