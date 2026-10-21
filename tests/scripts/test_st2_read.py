"""ST Phase 2's local read (scripts/analyze_st2.py): Holm over the pre-registered arms, the
intersection-union p, the transport sign rule and the cosine reading."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "papers" / "st_live_reference" / "scripts"))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
import analyze_st2 as A  # noqa: E402


def test_holm_step_down():
    # most probable failure: Holm read as Bonferroni at every step, or continuing after the first
    # non-rejection
    r = A.holm({"a": 0.001, "b": 0.004, "c": 0.02, "d": 0.5}, alpha=0.025)
    assert r == {"a": True, "b": True, "c": False, "d": False}
    assert A.holm({"a": 0.003, "b": 0.0031}, alpha=0.025 / 5) == {"a": False, "b": False}


def test_program_counts_unrun_arms_as_p1_and_needs_protect_ward():
    # most probable failure: Holm over the arms that ran (a smaller m, a laxer bar), or a Holm
    # survivor counted as steerable without rider (a)'s both-CI reading
    arms = {"gpt_oss_20b:care_harm": {"p_one_sided": 0.003, "protect_ward": True,
                                      "mde_vs_none": 0.06}}
    p = A.program(arms)
    assert len(p["p_one_sided"]) == 10 and p["p_one_sided"]["muse_glimmer_30b:V_qwen"] == 1.0
    assert not p["steerable_at_this_site"]  # 0.003 > 0.025 / 10
    arms["gpt_oss_20b:care_harm"]["p_one_sided"] = 0.001
    assert A.program(arms)["steerable_at_this_site"]
    arms["gpt_oss_20b:care_harm"]["protect_ward"] = False
    assert not A.program(arms)["steerable_at_this_site"]


def test_intersection_union_p_takes_the_weaker_test():
    # most probable failure: the arm's p taken against none only, so a move matched by the
    # control (a generic perturbation) counts
    rows = lambda p, n=10: [{"scenario_id": f"S{s}", "norm_status": "violating" if k < p else  # noqa: E731
                             "consistent", "letter": "A", "n_gen_tokens": 500,
                             "trace_nll_unsteered": 1.0} for s in range(n) for k in range(10)]
    r = A.arm_read(rows(6), rows(2), rows(6), "fill")
    assert r["dP_vs_none"]["mean"] > 0.3 and r["dP_vs_fill"]["mean"] == 0.0
    assert r["p_one_sided"] > 0.5 and not r["protect_ward"]


def test_transport_sign_follows_the_twins():
    # most probable failure: the transport control read as "moves upward" when the rule-absent
    # twins sit below the rule-present ones
    up = {"dP0_ablated_vs_present_headline": {"ci95": [0.01, 0.05]},
          "dP0_twin_vs_present": {"mean": 0.2}}
    down = {"dP0_ablated_vs_present_headline": {"ci95": [0.01, 0.05]},
            "dP0_twin_vs_present": {"mean": -0.2}}
    assert A.transport_passed(up) is True and A.transport_passed(down) is False


def test_cos_read_against_the_covariance_null():
    # most probable failure: an aligned pair read as unresolved because the null is drawn
    # isotropically instead of from the site's covariance (or the reverse)
    rng = np.random.default_rng(0)
    base = rng.standard_normal(64)
    per_o = {f"S{i}": base + 0.3 * rng.standard_normal(64) for i in range(40)}
    per_q = {f"Q{i}": base + 0.3 * rng.standard_normal(64) for i in range(30)}
    samples = rng.standard_normal((500, 64))
    assert A.cos_read(per_q, per_o, samples, n_null=300)["reading"].startswith("aligned")
    per_q2 = {f"Q{i}": rng.standard_normal(64) for i in range(30)}
    assert not A.cos_read(per_q2, per_o, samples, n_null=300)["reading"].startswith("aligned")
