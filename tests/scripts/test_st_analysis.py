"""T0.8 (analyze_st.py) on synthetic rows: ΔP baselines, the floor rung, ΔE and the §6 verdicts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "papers" / "st_live_reference" / "scripts"))
import analyze_st as S  # noqa: E402


def _rows(rng, true_p: dict, n: int, arm: str, sample: str, tag: str) -> list[dict]:
    return [{"scenario_id": s, "arm": arm, "sample": sample, "status": "completed",
             "norm_status": "violating" if rng.random() < p else "consistent",
             "rollout_key": f"{tag}:{s}:{k}"} for s, p in true_p.items() for k in range(n)]


@pytest.fixture
def world():
    """264 scenarios, a true reasoned protective share around 0.5, dose-0 share above it; the
    adapter does nothing (true P_adapter = true P_none)."""
    rng = np.random.default_rng(7)
    true_none = {f"H{i}": float(rng.uniform(0.2, 0.8)) for i in range(264)}
    p_dose0 = {s: min(1.0, p + 0.25) for s, p in true_none.items()}
    sel = _rows(rng, true_none, 8, "none", "selection", "t11")
    rerun = _rows(rng, true_none, 8, "none", "rerun", "t15")
    adapter = _rows(rng, true_none, 8, "L-anchor", "rerun", "t15a")
    return rng, true_none, p_dose0, sel, rerun, adapter


def test_delta_p_refuses_the_selecting_sample(world):
    # most probable failure: ΔP baselines on the sample that picked the subset (Amendment 1);
    # the fixture has teeth: that baseline shows a spurious restoration the re-run does not
    _, _, p_dose0, sel, rerun, adapter = world
    ids = S.engaged_ids(p_dose0, sel)
    assert len(ids) >= 40
    with pytest.raises(ValueError):
        S.delta_p(adapter, sel, ids)
    biased = np.mean([S.shares(adapter)[s]["P"] - S.shares(sel)[s]["P"] for s in ids])
    assert biased > 0.05  # regression to the mean the wrong baseline would report as restoration
    honest = S.delta_p(adapter, rerun, ids, selection_rows=sel)
    assert honest["ci95"][0] <= 0 <= honest["ci95"][1]


def test_delta_p_refuses_shared_rollouts(world):
    _, _, p_dose0, sel, rerun, adapter = world
    ids = S.engaged_ids(p_dose0, sel)
    leaked = [dict(r, sample="rerun") for r in sel[:16]] + rerun
    with pytest.raises(ValueError):
        S.delta_p(adapter, leaked, ids, selection_rows=sel)


def test_engaged_ids_read_the_selecting_sample_only(world):
    _, _, p_dose0, sel, rerun, _ = world
    with pytest.raises(ValueError):
        S.engaged_ids(p_dose0, rerun)


def test_rtm_floor_is_positive_under_noisy_selection(world):
    _, _, p_dose0, sel, rerun, _ = world
    ids = S.engaged_ids(p_dose0, sel)
    f = S.rtm_floor(sel, rerun, ids)
    assert f["ci95"][0] > 0  # selected-low scenarios regress up on an independent re-run


def test_true_restoration_is_recovered(world):
    rng, true_none, p_dose0, sel, rerun, _ = world
    ids = S.engaged_ids(p_dose0, sel)
    lifted = {s: min(1.0, p + 0.25) if s in ids else p for s, p in true_none.items()}
    adapter = _rows(rng, lifted, 8, "L-self", "rerun", "t15b")
    dp = S.delta_p(adapter, rerun, ids, selection_rows=sel)
    assert dp["mean"] == pytest.approx(0.25, abs=0.05) and dp["ci95"][0] > 0.15


def test_adapter_difference_paired(world):
    rng, true_none, p_dose0, sel, rerun, adapter = world
    ids = S.engaged_ids(p_dose0, sel)
    lifted = {s: min(1.0, p + 0.2) for s, p in true_none.items()}
    better = _rows(rng, lifted, 8, "L-self", "rerun", "t15c")
    d = S.adapter_difference(better, adapter, rerun, ids)
    assert d["ci95"][0] > 0.1


def test_truncation_shift_flag():
    rows = lambda arm, tr, tag: [  # noqa: E731
        {"scenario_id": f"S{i}", "arm": arm, "sample": "rerun", "rollout_key": f"{tag}{i}{k}",
         "status": "truncated" if k < tr else "completed",
         "norm_status": None if k < tr else "consistent"} for i in range(30) for k in range(10)]
    dp = S.delta_p(rows("L-anchor", 2, "a"), rows("none", 0, "n"), [f"S{i}" for i in range(30)])
    assert dp["truncation_shift"] == pytest.approx(0.2) and dp["truncation_shift_flag"]


def test_delta_e_separates_sensitivity_from_baseline_shift():
    none = {f"S{i}": {"pD": 0.40, "pJ": 0.20, "pDn": 0.25, "pJn": 0.20} for i in range(50)}
    # sensitivity moved: the primary's gap shrinks, the twin's at-rest gap does not
    sens = {s: dict(c, pD=0.30) for s, c in none.items()}
    r = S.delta_e(sens, none)
    assert r["delta_E"]["mean"] == pytest.approx(-0.10) and r["delta_g_twin"]["mean"] == 0
    # baseline shift only: primary and twin move together; ΔE stays 0
    shift = {s: dict(c, pD=0.30, pDn=0.15) for s, c in none.items()}
    r2 = S.delta_e(shift, none)
    assert r2["delta_E"]["mean"] == pytest.approx(0.0) and r2["delta_g_twin"]["mean"] < 0


def _b(mean, lo, hi):
    return {"mean": mean, "ci95": [lo, hi], "n_scenarios": 60, "mde": 0.08}


def test_verdict_rules_in_order():
    f, null, band = _b(0.06, 0.03, 0.09), _b(0.01, -0.03, 0.05), _b(0.2, 0.15, 0.25)
    good_panel = {"delta_E": _b(-0.03, -0.05, -0.01), "delta_g_twin": _b(0.0, -0.01, 0.01)}
    kw = dict(floor=f, matched_null=null, positive_band=band)
    assert S.verdict(_b(0.2, 0.1, 0.3), **kw, panel=good_panel, twin_noise_band=0.01,
                     capability_delta=0.0, specificity_delta=0.01,
                     heretic_survives=True)["verdict"] == "generalizes"
    assert S.verdict(_b(0.2, 0.1, 0.3), **kw, capability_delta=0.0,
                     specificity_delta=0.1)["verdict"] == "non-specific"
    assert S.verdict(_b(0.2, 0.1, 0.3), **kw)["verdict"] == "restores, local"
    assert S.verdict(_b(0.08, 0.02, 0.14), **kw)["verdict"] == "partial"
    assert S.verdict(_b(0.05, -0.02, 0.12), **kw)["verdict"] == "null"
    assert S.verdict(_b(-0.1, -0.2, -0.02), **kw)["verdict"] == "widens"
    u = S.verdict(_b(0.2, 0.1, 0.3), **kw, panel=good_panel, twin_noise_band=0.01,
                  capability_delta=0.0, specificity_delta=0.01, heretic_survives=False)
    assert u["verdict"] == "unclassified" and u["reasons"]
    # every verdict sentence carries the ladder, never a bare label
    s = S.verdict(_b(0.05, -0.02, 0.12), **kw)["sentence"]
    assert "floor" in s and "matched null" in s and "positive band" in s and "MDE" in s
