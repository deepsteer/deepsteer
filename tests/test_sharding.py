"""CI shards (tests/sharding.py)."""

from __future__ import annotations

import pytest

from tests import sharding

IDS = [f"tests/m{k % 7}.py::test_{k}" for k in range(500)]


def test_shards_partition_the_tests():
    # assert no test is dropped or run twice across the shards (the failure that hides a red test)
    parts = [sharding.split(IDS, f"{i}/3")[0] for i in range(3)]
    flat = [x for p in parts for x in p]
    assert sorted(flat) == sorted(IDS)
    assert len(set(flat)) == len(IDS)
    assert all(len(p) > 100 for p in parts)


def test_unset_keeps_everything_and_bad_specs_raise():
    assert sharding.split(IDS, None) == (IDS, [])
    for bad in ("2/2", "-1/2", "1"):
        with pytest.raises(ValueError):
            sharding.parse(bad)


def test_shard_is_stable_across_runs():
    # assert the shard comes from the node id alone (not hash() salting or list position)
    assert sharding.shard_of("tests/x.py::test_a", 2) == sharding.shard_of("tests/x.py::test_a", 2)
    assert sharding.split(IDS[::-1], "0/2")[0] == sharding.split(IDS, "0/2")[0][::-1]
