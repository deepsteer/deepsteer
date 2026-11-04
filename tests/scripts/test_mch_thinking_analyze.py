"""Tests for the MCH thinking-on analysis (MCH_PLAN.md §5, Amendments 10, 11), on synthetic rows."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "papers" / "mch_machiavelli" / "scripts"))

an = pytest.importorskip("mch_thinking_analyze")
from mch_g1_harness import GAMES  # noqa: E402


def test_bias_bound_uses_the_arm_that_falls_back_more():
    # most probable failure: taking R from the wrong arm, or dropping the |anchor - R| factor
    assert an.bias_bound(0.05, 0.01, 80.0, 60.0, 100.0) == pytest.approx(0.04 * 20)
    assert an.bias_bound(0.01, 0.05, 80.0, 60.0, 100.0) == pytest.approx(0.04 * 40)


def test_a_ci_clearing_zero_by_less_than_the_bound_is_a_reading():
    assert an.verdict({"ci95_bootstrap": [-3.0, -1.5]}, 1.0)["status"] == "verdict"
    assert an.verdict({"ci95_bootstrap": [-1.5, -0.5]}, 1.0)["status"].startswith("reading")
    assert an.verdict({"ci95_bootstrap": [-1.0, 0.5]}, 0.1)["ci_side"] is None


def _write(d: Path, eps: list[dict], decs: list[dict]) -> None:
    d.mkdir(parents=True)
    (d / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in eps) + "\n")
    with gzip.open(d / "decisions.jsonl.gz", "wt") as f:
        f.write("\n".join(json.dumps(x) for x in decs) + "\n")


def _rows(agent: str, viol: float, fallback: float, thinking: bool):
    eps, decs = [], []
    for g in GAMES:
        for e in (0, 1):
            v = viol + (1.0 if e else -1.0)
            eps.append({"agent": agent, "game": g, "episode": e, "seed": 2417 + e,
                        "status": "done", "decisions": 20, "points_norm": 20.0 + e,
                        "scores": {k: {"r0": v} for k in ("violations.Σ", "power.Σ",
                                                           "utility.Σ")}})
            for i in range(20):
                x = {"agent": agent, "fallback": i < 20 * fallback, "override": False}
                x.update({"capped": [False], "cap_rule_cut": False, "trace_chars": [100],
                          "tokens": [50]} if thinking else {"reply_tokens": [50]})
                decs.append(x)
    return eps, decs


def test_h3_compares_against_the_suppressed_cotagent_row_only(tmp_path):
    # most probable failure: H3 pairing with the suppressed GoodCoTAgent (or both suppressed rows)
    if not (an.MCH / "outputs" / "g0_random.jsonl.gz").exists():
        pytest.skip("G0's random episodes are not in this checkout")
    th, nm = tmp_path / "th", tmp_path / "nm"
    cot, good = _rows("ThinkCoTAgent", 80, 0.05, True), _rows("ThinkGoodCoTAgent", 70, 0.0, True)
    _write(th / "pilot_gpt_oss_20b", cot[0] + good[0], cot[1] + good[1])
    _write(th / "cell_gpt_oss_20b", [], [])
    s_cot, s_good = _rows("CoTAgent", 90, 0.0, False), _rows("GoodCoTAgent", 50, 0.0, False)
    _write(nm / "pilot_1024", s_cot[0] + s_good[0], s_cot[1] + s_good[1])
    _write(nm / "cell_gpt_oss_20b", [], [])
    out = tmp_path / "rows.json"
    sys.argv = ["x", "--th", str(th), "--nm", str(nm), "--out", str(out)]
    an.main()
    r = json.loads(out.read_text())
    assert r["H3"]["violations"]["point"] == pytest.approx(-10.0)
    assert r["H3"]["branch"] == "violations lower with thinking on"
    assert r["H2"]["violations"]["point"] == pytest.approx(-10.0)
    # B on H2 violations: |0.05 - 0| x |100 - 80| = 1.0, far inside the CI's clearance
    assert r["H2"]["violations"]["bias_bound"] == pytest.approx(1.0)
    assert r["H2"]["violations"]["status"] == "verdict"
    assert "basis" in r["h3_basis"] and r["H2"]["violations"]["mde_k2"] > 0
