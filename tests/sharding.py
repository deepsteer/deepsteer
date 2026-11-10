"""Split the CI test set into stable shards (``DEEPSTEER_TEST_SHARD=i/n``; CI's matrix sets it).

A test's shard comes from a hash of its node id, so every pytest-xdist worker deselects the same
tests and a test keeps its shard as other tests are added. Unset, nothing is deselected.
"""

from __future__ import annotations

import hashlib
import os

ENV = "DEEPSTEER_TEST_SHARD"


def parse(spec: str) -> tuple[int, int]:
    """Parse ``"i/n"`` into ``(i, n)`` with ``0 <= i < n``.

    Args:
        spec: The shard spec, e.g. ``"0/2"``.

    Returns:
        The shard index and the shard count.
    """
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"{ENV}={spec!r}: need 0 <= i < n")
    return i, n


def shard_of(nodeid: str, n: int) -> int:
    """Return the shard (0..n-1) that owns a test node id."""
    return int(hashlib.blake2b(nodeid.encode(), digest_size=8).hexdigest(), 16) % n


def split(nodeids: list[str], spec: str | None) -> tuple[list[str], list[str]]:
    """Split node ids into (kept, deselected) for a shard spec; ``None`` keeps everything."""
    if not spec:
        return list(nodeids), []
    i, n = parse(spec)
    keep = [x for x in nodeids if shard_of(x, n) == i]
    return keep, [x for x in nodeids if shard_of(x, n) != i]


def spec_from_env() -> str | None:
    """Return the shard spec from the environment, or ``None``."""
    return os.environ.get(ENV) or None
