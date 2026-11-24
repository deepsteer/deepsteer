"""Amendment 7's read (GPC_PLAN.md §15.1, §15.3): the probe-artifact checks, the branch and the
sensitivity line.

The read runs after the pod, on the saved rows, so these tests run in CI and not in the pod's gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "papers/gpc_goal_pressure_conduct/scripts"))

import z_gpc4_a7_read as R  # noqa: E402


def _named(pa: float, pb: float, n: int = 16, k: int = 2) -> list[dict]:
    """Deterministic rows: the act is accepted on a share pa of the act-as-A probes and pb of the
    act-as-B probes, the hits on the first states."""
    rows = []
    for s in range(n):
        for order, p in (("act_A", pa), ("act_B", pb)):
            for i in range(k):
                hit = (s * k + i) < round(p * n * k)
                row = {"arm": "base", "scenario": "S", "state_id": f"s{s}", "order": order}
                rows.append(row | {"accepts": hit})
    return rows


def test_order_check_flags_a_one_letter_bias_and_not_a_balanced_probe():
    # most probable failure: a letter-position bias (always A) reads as 0.5 acceptance and passes,
    # or a probe with no order effect is dropped (§15.3 item 1)
    assert R.order_check(_named(1.0, 0.0), "S", B=500)["artifact"] is True
    assert R.order_check(_named(0.5, 0.5), "S", B=500)["artifact"] is False


def test_order_check_needs_both_size_and_an_interval_excluding_zero():
    # most probable failure: a difference under 0.15 is dropped because its interval excludes 0
    oc = R.order_check(_named(0.55, 0.45), "S", B=500)
    assert abs(oc["diff"]) < R.ARTIFACT_TOL and oc["artifact"] is False and oc["mde"] is not None


def test_early_check_flags_a_pull_and_reports_its_rates():
    # most probable failure: the earliest-state probe's pull above the cold rate goes unflagged
    early = [
        {"arm": "base", "scenario": "S", "state_id": f"e{s}", "accepts": True}
        for s in range(16)
        for _ in range(4)
    ]
    cold = [
        {"arm": "base", "scenario": "S", "render_id": f"r{r}", "accepts": False}
        for r in range(16)
        for _ in range(4)
    ]
    ec = R.early_check(early, cold, "S", B=500)
    assert ec["artifact"] is True and ec["early_base"] == 1.0 and ec["cold_base"] == 0.0


def test_branch_reads_the_registered_edges():
    # most probable failure: an edge is read on the wrong side (§15.1's table)
    assert R.branch(0.40, None).startswith("the loop's context moves")
    assert R.branch(0.15, 0.15).startswith("the subject knows and does it anyway")
    assert R.branch(0.10, 0.40).startswith("the subject does not read its step")
    assert R.branch(0.20, 0.10).startswith("unresolved")
    assert R.branch(0.10, 0.30).startswith("unresolved")


def test_sensitivity_restricts_the_act_states():
    # most probable failure: the sensitivity line silently reads all 32 act states (§15.3 item 5)
    named, own, cold = [], [], []
    for s in range(4):
        for order in ("act_A", "act_B"):
            base = {"arm": "base", "scenario": "S", "variant": "v", "state_id": f"s{s}"}
            base["order"] = order
            named.append(base | {"act_state": True, "accepts": s < 2})
            own.append(base | {"accepts": s < 2})
    for r in range(4):
        cold.append(
            {"arm": "base", "scenario": "S", "variant": "v", "render_id": f"r{r}", "accepts": False}
        )
    full = R.root(cold, named, own, {"S"})
    sub = R.root(cold, named, own, {"S"}, only={"s0", "s1"})
    assert full["cells"]["S|v"]["act_states"] == 4 and sub["cells"]["S|v"]["act_states"] == 2
    assert full["delta"] == 0.5 and sub["delta"] == 1.0
