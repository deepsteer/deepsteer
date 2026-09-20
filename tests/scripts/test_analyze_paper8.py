"""papers/kdg_panel/scripts/analyze_paper8.py: the A17 raw-frame readouts and paired statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "papers/kdg_panel/scripts"))

import analyze_paper8 as a8  # noqa: E402


def test_indicator_matches_raw_kdg_semantics():
    # assert D violating while J non-violating is a gap (1), agreement is 0, and
    # D non-violating while J violating is undefined (None), never 0
    assert a8.indicator({"viol_frac": 0.75}, {"viol_frac": 0.25}) == 1
    assert a8.indicator({"viol_frac": 0.25}, {"viol_frac": 0.25}) == 0
    assert a8.indicator({"viol_frac": 0.75}, {"viol_frac": 0.75}) == 0
    assert a8.indicator({"viol_frac": 0.25}, {"viol_frac": 0.75}) is None
    # assert a missing cell gives an undefined indicator rather than a silent 0
    assert a8.indicator(None, {"viol_frac": 0.25}) is None


def test_raw_cell_normalises_over_displayed_letters_only(tmp_path, monkeypatch):
    # assert p is the violating option's share of the *displayed-letter* mass (0.5 here),
    # not its share of total vocabulary mass (option_mass 0.4 would give 0.2)
    import json

    rows = [{"scenario_id": "S1", "order": {"A": "o1", "B": "o2"}, "option_mass": 0.4,
             "option_logps": {"A": float(np.log(0.2)), "B": float(np.log(0.2))},
             "norm_status": "violating"}]
    d = tmp_path / "m"
    d.mkdir()
    (d / "d_raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(a8.ap, "EXTRA_DIRS", [])
    out = a8.raw_cell(d, "d_raw", {"S1": {"o1": "consistent", "o2": "violating"}})
    assert abs(out["S1"]["p"] - 0.5) < 1e-9
    assert abs(out["S1"]["mass"] - 0.4) < 1e-9
    assert out["S1"]["viol_frac"] == 1.0
    # assert swapped scenarios (ids ending in S) and unknown ids never enter the raw readouts
    rows2 = rows + [dict(rows[0], scenario_id="S1S"), dict(rows[0], scenario_id="ZZ")]
    (d / "d_raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows2) + "\n")
    out = a8.raw_cell(d, "d_raw", {"S1": {"o1": "consistent", "o2": "violating"}})
    assert set(out) == {"S1"}


def test_boot_mean_and_mde():
    # assert the paired MDE is 2.8 bootstrap SDs (80% power, two-sided 0.05) and that an
    # all-positive sample excludes zero while a zero-centred one does not
    r = a8.boot_mean([0.1, 0.2, 0.3, 0.4], n_boot=500)
    assert r["excludes_zero"] and r["n"] == 4
    assert abs(a8.mde_from_boot(r["sd_boot"]) - 2.8016 * r["sd_boot"]) < 1e-12
    z = a8.boot_mean([-1, 1, -1, 1, -1, 1], n_boot=500)
    assert not z["excludes_zero"]
    # assert None entries are dropped, not coerced to 0
    assert a8.boot_mean([None, 0.5, 0.5], n_boot=50)["n"] == 2
    assert a8.boot_mean([], n_boot=50)["mean"] is None


def test_unpaired_diff_reports_both_counts():
    # assert an unpaired contrast keeps n_a and n_b separately (they are different families)
    r = a8.unpaired_diff([0.3, 0.4, 0.5], [0.1, 0.1], n_boot=300)
    assert r["n_a"] == 3 and r["n_b"] == 2 and r["diff"] > 0
