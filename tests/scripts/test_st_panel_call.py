"""Amendment 11's carrier call (scripts/panel_call.py): KDG's estimators reproduced, the band
precondition, the carrier rule's two conditions, and the threshold sources."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "papers" / "st_live_reference" / "scripts"))
import panel_call as PC  # noqa: E402

KDG_GPTOSS = ROOT / "papers/kdg_panel/outputs/p2c/gpt_oss_20b/gpt_oss_20b"


def test_reproduces_kdg_section_27_on_kdgs_own_cells():
    # most probable failure: the call recomputes E or g_band with an estimator that drifts from
    # KDG's (floor, renormalisation, bootstrap), so the branch is decided on a different number
    if not (KDG_GPTOSS / "dl_chat_neutral.jsonl").exists():
        pytest.skip("KDG GPT-OSS cells not present locally")
    c = PC.call(KDG_GPTOSS, "gpt_oss_20b")
    e = c["E_none"]["E_prob"]
    assert c["E_none"]["n"] == 586
    assert abs(e["mean"] - (-0.0012560932448085985)) < 1e-12
    assert abs(e["mde"] - PC.KDG_GPTOSS_MDE) < 1e-12
    assert abs(c["band"]["g_band"]["mean"] - 0.501) < 5e-4 and c["band"]["in_band"]
    assert c["call"] == "A"


def _fake(monkeypatch, lb: float, mean: float, ci: tuple[float, float], mde: float = 0.01):
    monkeypatch.setattr(PC, "band", lambda d, s: {
        "g_band": {"mean": lb + 0.05, "ci95": [lb, lb + 0.1]}, "in_band": lb >= PC.BAND_LB})
    monkeypatch.setattr(PC, "e_none", lambda d, s: {
        "n": 586, "E_prob": {"mean": mean, "ci95": list(ci), "mde": mde}, "mde": mde})


def test_band_out_of_band_defers_never_defaults(monkeypatch):
    # most probable failure: a failed positive control falls through to branch A (or B) instead
    # of deferring the call
    _fake(monkeypatch, lb=0.05, mean=0.05, ci=(0.03, 0.07))
    assert PC.call(Path("."), "gpt_oss_20b", status={})["call"] == "deferred"


def test_carrier_needs_ci_excluding_zero_and_point_at_threshold(monkeypatch):
    # most probable failure: a CI excluding 0 alone makes a carrier, so a gap smaller than dE's
    # MDE sends the model to branch B
    _fake(monkeypatch, lb=0.3, mean=0.012, ci=(0.004, 0.020))
    c = PC.call(Path("."), "gpt_oss_20b", status={})
    assert c["call"] == "A" and "below threshold" in c["reason"]
    _fake(monkeypatch, lb=0.3, mean=0.025, ci=(0.015, 0.035))
    assert PC.call(Path("."), "gpt_oss_20b", status={})["call"] == "B"
    _fake(monkeypatch, lb=0.3, mean=0.025, ci=(-0.001, 0.05))
    assert PC.call(Path("."), "gpt_oss_20b", status={})["call"] == "A"


def test_threshold_fixed_for_gptoss_and_from_the_reads_variance_for_muse(monkeypatch):
    # most probable failure: GPT-OSS's threshold is recomputed from the read it decides, or Muse's
    # uses the point estimate
    _fake(monkeypatch, lb=0.3, mean=0.025, ci=(0.015, 0.035), mde=0.02)
    g = PC.call(Path("."), "gpt_oss_20b", status={})
    assert abs(g["threshold"] - 0.018875) < 1e-5
    assert abs(g["own_sqrt2_mde"] - math.sqrt(2) * 0.02) < 1e-12
    m = PC.call(Path("."), "muse_glimmer_30b", status={})
    assert abs(m["threshold"] - math.sqrt(2) * 0.02) < 1e-12 and m["call"] == "A"  # 0.025 < 0.0283
