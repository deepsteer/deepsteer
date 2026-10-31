"""Byte-identity of the KDG pieces lifted into deepsteer.choice / deepsteer.stats (MB_PLAN §6).

The goldens (goldens/kdg_lift_v1.json) were written from the code before the lift and committed
on their own; this file is unchanged through the lift. Local-only goldens (the banked pods p1e,
kdg2/kdg3, and the bf16 readout on this platform) skip where their inputs are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import lift_goldens as G  # noqa: E402

GOLD = json.loads(G.GOLDEN.read_text())


def test_parse_statuses_on_the_calibration_sets():
    # most probable failure: a lifted ladder regex differs from KDG's (the letter boundary, or
    # case-folding on the leading rule), moving a parse on one of the 800 replies
    assert G.calibration_parse() == GOLD["calibration_parse"]


def test_parse_statuses_on_banked_pod_p1e():
    # most probable failure: the same, on 38,016 real rollouts of the banked pod p1e
    if not G.P1E.is_dir():
        pytest.skip("banked pod p1e not present")
    assert G.p1e_parse() == GOLD["p1e_parse"]


def test_mb_parsers_unchanged():
    # most probable failure: generalizing the shared builder changes MB's own R0/R1 (its G1 rows
    # were parsed with it)
    assert G.mb_parse() == GOLD["mb_parse"]


def test_agreement_and_kappa_exact():
    # most probable failure: a lifted agreement/kappa rounds or orders differently
    assert G.agreement_values() == GOLD["agreement"]


def test_bootstrap_estimators_exact():
    # most probable failure: the shared nested bootstrap draws the random stream in a different
    # order (an inner resample before an outer one), shifting every CI; equality is exact
    assert G.stats_values() == GOLD["stats"]


def test_mask_positions_exact():
    assert G.mask_positions_value() == GOLD["mask_positions"]


def test_readout_bytes_on_this_platform():
    # most probable failure: the lifted readout core changes the dtype path (float32 vs the
    # float16 rows of record) or the position ids, moving the log-prob bytes
    gold = GOLD["readout"]
    if gold is None:
        pytest.skip("no readout golden")
    now = G.readout_value()
    if now is None or now["platform"] != gold["platform"]:
        pytest.skip("Qwen2.5 tokenizer not cached or a different platform than the golden's")
    assert now == gold


def test_banked_analysis_reproduces_byte_for_byte():
    # most probable failure: the registered estimator's output moves on real banked data; the
    # script itself also asserts that the registered intervals reproduce
    if GOLD["gap_twoway_sha256"] is None:
        pytest.skip("no analysis golden")
    now = G.gap_twoway_sha()
    if now is None:
        pytest.skip("banked pods kdg2/kdg3 not present")
    assert now == GOLD["gap_twoway_sha256"] == GOLD["gap_twoway_committed_sha256"]
