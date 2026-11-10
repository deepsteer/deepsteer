"""Root test configuration: CI sharding (see ``tests/sharding.py``)."""

from __future__ import annotations

from tests import sharding


def pytest_collection_modifyitems(config, items):
    spec = sharding.spec_from_env()
    if not spec:
        return
    keep, _ = sharding.split([it.nodeid for it in items], spec)
    kept = set(keep)
    config.hook.pytest_deselected(items=[it for it in items if it.nodeid not in kept])
    items[:] = [it for it in items if it.nodeid in kept]
