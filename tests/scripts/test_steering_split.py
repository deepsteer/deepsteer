"""T0.2 (build_split.py): the H1 train/held split of STEERING_PLAN v0.2.2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "papers" / "steering" / "scripts"))
import build_split as S  # noqa: E402


def _inputs():
    ids = json.loads(S.KEPT.read_text())["ids"]
    records = {r["id"]: r for r in json.loads(S.SCENARIOS.read_text())["scenarios"]}
    return ids, records


def test_split_is_disjoint_and_covers_every_kept_id():
    # most probable failure: a scenario lands on both sides (or neither), leaking train into held
    ids, records = _inputs()
    split = S.build_split(ids, records)
    train, held = set(split["H1_train"]), set(split["H1_held"])
    assert not train & held
    assert train | held == set(ids)
    assert len(train) == len(held) == 132


def test_each_stratum_is_halved():
    # odd strata alternate the extra scenario; no stratum may be off by more than one
    ids, records = _inputs()
    split = S.build_split(ids, records)
    side = {sid: k for k, m in split.items() for sid in m}
    by = {}
    for sid in ids:
        key = tuple(records[sid][f] for f in S.STRATA)
        by.setdefault(key, []).append(side[sid])
    for key, sides in by.items():
        assert abs(sides.count("H1_train") - sides.count("H1_held")) <= 1, key


def test_split_ignores_input_order_and_is_seeded():
    ids, records = _inputs()
    a = S.build_split(ids, records)
    assert S.build_split(list(reversed(ids)), records) == a
    assert S.build_split(ids, records, seed=S.SEED + 1) != a


def test_committed_split_matches_inputs():
    # the committed file must be the one the script builds from the current inputs
    out = json.loads(S.OUT.read_text())
    ids, records = _inputs()
    split = S.build_split(ids, records, seed=out["seed"])
    assert out["H1_train"] == split["H1_train"] and out["H1_held"] == split["H1_held"]
    for rel, digest in out["inputs"].items():
        assert S.sha256(S.PAPERS / rel) == digest
