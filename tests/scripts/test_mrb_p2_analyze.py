"""MRB Phase 2 analysis (papers/mrb_morebench/scripts/mrb_p2_analyze.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "papers" / "mrb_morebench" / "scripts"))

import mrb_p2_analyze as Z  # noqa: E402


class It:
    def __init__(self, i):
        self.scenario_id, self.idx, self.source = f"mrb-{i:03d}", i, "daily_dilemmas"
        self.role = "ai_agent" if i % 5 < 2 else "ai_advisor"  # about the public set's 207 / 293


def _fixture(score_gain: float, length_scale: float, n=200, seed=0, length_noise=5.0,
             agent_only=False):
    rng = np.random.default_rng(seed)
    items = [It(i) for i in range(n)]
    us, dims, gens = {}, {}, {}
    for it in items:
        base, ln = rng.normal(70, 8), rng.normal(5000, 600)
        g_it = score_gain if (not agent_only or it.role == "ai_agent") else 0.0
        for arm, gain, scale in (("none", 0.0, 1.0), ("adapter", g_it, length_scale)):
            for k in (0, 1):
                gid = f"m/{arm}/{it.scenario_id}/k{k}"
                s = base * (1 + gain / 70) + rng.normal(0, 3)
                us[f"{gid}/response"] = {"credit_substring": s}
                dims[f"{gid}/response"] = {d: s for d in Z.DIMS}
                gens[gid] = {"response_chars": ln * scale + length_noise * rng.normal()}
    return us, dims, gens, items


def test_a_gain_that_tracks_length_is_called_length_confounded():
    # most probable failure: a Regular gain carried by longer responses read as better reasoning
    # (the score is not length-normalized; Hard is the check the length rule registers)
    us, dims, gens, items = _fixture(score_gain=6.0, length_scale=1 + 6.0 / 70)
    b = Z.model_block(us, dims, gens, "m", items, np.random.default_rng(1))
    assert b["length_rule_applies"] and b["delta_ci95"][0] > 0
    assert b["length_confounded"] and b["branch"] == "length-confounded gain"
    assert "delta_hard" in b


def test_a_gain_at_equal_length_is_a_gain_and_no_change_carries_its_bar():
    # most probable failure: the length rule firing without a length shift, or a null branch
    # without its detection bar
    us, dims, gens, items = _fixture(score_gain=6.0, length_scale=1.0, length_noise=0.0)
    b = Z.model_block(us, dims, gens, "m", items, np.random.default_rng(1))
    assert not b["length_rule_applies"] and b["branch"] == "raises process quality"
    us, dims, gens, items = _fixture(score_gain=0.0, length_scale=1.0)
    b = Z.model_block(us, dims, gens, "m", items, np.random.default_rng(1))
    assert b["branch"].startswith("no change detectable at |Δ| ≳ ") and b["mde80_observed"] > 0


def test_holm_is_monotone_and_bounded():
    # most probable failure: Holm-adjusted p below the raw p, or a later p below an earlier one
    h = Z.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5, "e": 0.2})
    assert all(h[k] >= p for k, p in {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5, "e": 0.2}.items())
    assert h["a"] == 0.05 and max(h.values()) <= 1.0


def test_secondary_null_reading_says_it_is_not_a_verdict_against_morebench():
    # most probable failure: a null separation worded as MoReBench failing to see specificity
    blocks = {m: {"_arrays": {"delta": np.zeros(50), "dims": {d: np.zeros(50) for d in Z.DIMS}}}
              for m in Z.MODELS}
    blocks["muse_glimmer_30b"]["_arrays"]["delta"] = np.random.default_rng(0).normal(0, 1, 50)
    s = Z.secondary(blocks, 50, np.random.default_rng(2))
    assert set(s) == {"regular", "harmless outcome", "logical process"}
    assert "not a verdict against MoReBench" in s["harmless outcome"]["reading"]


def test_role_breakdown_finds_an_agent_only_transfer_and_names_the_prediction():
    # most probable failure (v0.9.1): the role subsets swapped or pooled, so an Agent-only effect
    # reads as a uniform one, or the interaction's sign reversed against the registered prediction
    us, dims, gens, items = _fixture(score_gain=6.0, length_scale=1.0, length_noise=0.0,
                                     agent_only=True)
    r = Z.model_block(us, dims, gens, "m", items, np.random.default_rng(3))["roles"]
    assert r["ai_agent"]["n"] == 80 and r["ai_advisor"]["n"] == 120
    adv = r["ai_advisor"]["ci95"]
    assert r["ai_agent"]["ci95"][0] > 0 and adv[0] < 0 < adv[1]
    assert r["agent_minus_advisor"]["ci95"][0] > 0
    want = "transfer localized to the Agent role, as predicted"
    assert r["agent_minus_advisor"]["reading"] == want
