"""ST T1.3 scenario draw (scripts/build_t13_draw.py): stratified gate quotas, a sweep disjoint
from the gate, determinism."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "papers" / "st_live_reference" / "scripts"))
import build_t13_draw as D  # noqa: E402


def _records(sizes: dict[tuple[str, str], int]) -> tuple[list[str], dict[str, dict]]:
    ids, rec = [], {}
    for (g, r), n in sizes.items():
        for i in range(n):
            sid = f"H1-{g}-{r}-{i:02d}"
            ids.append(sid)
            rec[sid] = {"generator": g, "register": r}
    return ids, rec


def test_sweep_never_shares_a_gate_scenario():
    # most probable failure: the sweep is drawn from the whole engaged set, so alpha is tuned on
    # a gate scenario (or the gate and sweep miss their sizes)
    ids, rec = _records({("a", "plain"): 12, ("a", "pro"): 11, ("b", "plain"): 13, ("b", "pro"): 9})
    d = D.draw(ids, rec)
    assert len(d["gate"]) == 20 and len(d["sweep"]) == 5
    assert not set(d["gate"]) & set(d["sweep"])
    assert set(d["gate"]) | set(d["sweep"]) <= set(ids)


def test_gate_quotas_follow_the_strata_by_largest_remainder():
    # most probable failure: the gate ignores the strata (a plain random 20) or the rounding loses
    # or adds a scenario
    ids, rec = _records({("a", "plain"): 12, ("a", "pro"): 11, ("b", "plain"): 13, ("b", "pro"): 9})
    d = D.draw(ids, rec)
    got = Counter(" × ".join((rec[s]["generator"], rec[s]["register"])) for s in d["gate"])
    assert dict(got) == d["quotas"] and sum(d["quotas"].values()) == 20
    # 20 x (12, 11, 13, 9) / 45 = (5.33, 4.89, 5.78, 4.0): floors 5, 4, 5, 4 = 18, the two largest
    # remainders (b plain 0.78, a pro 0.89) take the last two
    assert d["quotas"] == {"a × plain": 5, "a × pro": 5, "b × plain": 6, "b × pro": 4}


def test_draw_is_deterministic_and_order_free():
    ids, rec = _records({("a", "plain"): 12, ("a", "pro"): 11, ("b", "plain"): 13, ("b", "pro"): 9})
    assert D.draw(ids, rec) == D.draw(list(reversed(ids)), rec)
    assert D.draw(ids, rec)["gate"] != D.draw(ids, rec, seed=1)["gate"]
