"""T1.2 constancy read (scripts/constancy_read.py) on a synthetic bank: the within-scenario
design, the class pick and the negative flag."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "papers" / "st_live_reference" / "scripts"))
import constancy_read as CR  # noqa: E402

H = 16


def _bank(shift_onset: float, shift_clause: float, scen_offset: float = 0.0):
    """Two scenarios x 4 rollouts (2 protect, 2 rule) + one scenario with protect only. Margin
    along axis e0 at onset/clause_end rises by ``shift`` on protect rollouts; ``scen_offset`` adds a
    per-scenario level that a between-scenario contrast would pick up."""
    a = np.zeros(H, np.float32)
    a[0] = 1.0
    rollouts, resid, row, cls = [], [], [], []
    for si, sid in enumerate(("S1", "S2", "S3")):
        for k in range(4):
            act = "violating" if (k < 2 or sid == "S3") else "consistent"
            rollouts.append({"scenario_id": sid, "split": "train", "norm_status": act})
            i = len(rollouts) - 1
            for code, shift in ((1, shift_onset), (2, shift_clause)):
                for _ in range(3):
                    h = np.zeros((5, H), np.float32)
                    h[:, 1] = 10.0
                    h[:, 0] = si * scen_offset + (shift if act == "violating" else 0.0)
                    resid.append(h)
                    row.append(i)
                    cls.append(code)
    z = {"resid": np.stack(resid), "row": np.array(row), "cls": np.array(cls, np.int8)}
    return [z], rollouts, {"anchor_axis_t_inst": a}


def test_within_scenario_difference_ignores_scenario_level_and_one_sided_scenarios():
    # most probable failure: the contrast pools rollouts across scenarios, so a scenario-level
    # offset (content) reads as a protect-vs-rule separation, or a protect-only scenario enters
    chunks, rollouts, axes = _bank(0.0, 0.0, scen_offset=5.0)
    m = CR.rollout_means(chunks, axes)[("anchor_axis_t_inst", 2, "onset")]
    d = CR.scenario_diffs(m, rollouts)
    assert set(d) == {"S1", "S2"} and all(abs(v) < 1e-6 for v in d.values())


def test_pick_takes_the_larger_separating_class_and_flags_a_negative_one():
    # most probable failure: the pick ignores the CI (a noisy larger class wins) or a negative
    # separation passes silently instead of flipping L-anchor's expected branch
    s = {"onset": {"mean": 0.02, "ci95": [0.01, 0.03]},
         "clause_end": {"mean": 0.05, "ci95": [-0.01, 0.11]}}
    assert CR.pick(s)["class"] == "onset" and not CR.pick(s)["negative_flag"]
    s["clause_end"] = {"mean": -0.05, "ci95": [-0.08, -0.02]}
    p = CR.pick(s)
    assert p["class"] == "clause_end" and p["negative_flag"]
    none = {"onset": {"mean": 0.0, "ci95": [-0.1, 0.1]}, "clause_end": {"mean": 0.0,
                                                                     "ci95": [-0.1, 0.1]}}
    assert CR.pick(none) == {**CR.pick(none), "class": "onset", "by_rule": False}


def test_rollout_means_follow_the_class_codes():
    chunks, rollouts, axes = _bank(shift_onset=3.0, shift_clause=0.0)
    m = CR.rollout_means(chunks, axes)
    on = CR.scenario_diffs(m[("anchor_axis_t_inst", 2, "onset")], rollouts)
    ce = CR.scenario_diffs(m[("anchor_axis_t_inst", 2, "clause_end")], rollouts)
    assert all(v > 0.2 for v in on.values()) and all(abs(v) < 1e-6 for v in ce.values())
