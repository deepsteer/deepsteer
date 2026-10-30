"""Shared fixtures for tests of the paper scripts."""

from __future__ import annotations

import pytest

# Module globals the ST drivers' configure() retargets on t16_driver (st2qp, st2qs, st2qr).
_T16_ATTRS = ("KEY", "PIN", "BLOCK", "STAGE", "SEED", "ARMS", "parity_scenarios", "pairs_for",
              "gen_vllm")


@pytest.fixture
def t16_restored(monkeypatch):
    """Record t16_driver's retargetable globals so monkeypatch restores them after the test.

    configure() rewrites them in place; without the restore a later test (test_t16's design
    constants) reads the retargeted values when the whole suite runs in one process.
    """
    import t16_driver as T16

    for name in _T16_ATTRS:
        if hasattr(T16, name):
            monkeypatch.setattr(T16, name, getattr(T16, name))
    return T16
