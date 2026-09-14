"""Knowing–doing gap (KDG) panel: schema, harness, breadth rubric, statistics, calibration.

Spec of record: ``papers/KDG_PANEL_SPEC.md``. Nothing in this package loads a model; the pod
driver in ``papers/kdg_panel/scripts`` does, and saves per-rollout arrays these functions read.
"""

from __future__ import annotations

from deepsteer.kdg.breadth import BREADTH_RUBRIC_VERSION, breadth_score, parse_breadth
from deepsteer.kdg.calibration import CALIBRATION_SET_VERSION, calibration_report
from deepsteer.kdg.harness import KDG_HARNESS_VERSION, Parse, parse_response
from deepsteer.kdg.schema import (
    GATE_FAMILIES,
    TEMPLATE_VERSION,
    Option,
    Scenario,
    assign_letters,
    load_scenarios,
    save_scenarios,
    validate_scenario,
)
from deepsteer.kdg.stats import ScenarioReadout, difference_ci, kdg_rate, pilot_gate, screen_pass

__all__ = [
    "BREADTH_RUBRIC_VERSION",
    "CALIBRATION_SET_VERSION",
    "GATE_FAMILIES",
    "KDG_HARNESS_VERSION",
    "TEMPLATE_VERSION",
    "Option",
    "Parse",
    "Scenario",
    "ScenarioReadout",
    "assign_letters",
    "breadth_score",
    "calibration_report",
    "difference_ci",
    "kdg_rate",
    "load_scenarios",
    "parse_breadth",
    "parse_response",
    "pilot_gate",
    "save_scenarios",
    "screen_pass",
    "validate_scenario",
]
