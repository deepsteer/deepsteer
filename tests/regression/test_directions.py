"""Regression tests: verify library direction extraction matches paper outputs.

Lightweight schema tests run without model weights. Full reproduction tests
are marked @pytest.mark.regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

PROBE_ENG_DIR = Path(__file__).resolve().parents[2] / "papers" / "3_moral_geometry" / "outputs" / "probe_engineering"


class TestDirectionOutputSchema:
    """Validate paper direction output schemas (no model needed)."""

    def test_mean_diff_directions_schema(self):
        path = PROBE_ENG_DIR / "mean_diff_directions.json"
        if not path.exists():
            pytest.skip("Paper output not found")
        with open(path) as f:
            data = json.load(f)
        assert "direction_comparison" in data
        for fv in ["care_harm", "fairness_cheating"]:
            assert fv in data["direction_comparison"]

    def test_multi_method_alignment_matrix(self):
        path = PROBE_ENG_DIR / "multi_method_directions.json"
        if not path.exists():
            pytest.skip("Paper output not found")
        with open(path) as f:
            data = json.load(f)
        matrix = data["alignment_matrix"]
        # Diagonal should be 1.0
        for method in data["methods"]:
            assert matrix[method][method] == 1.0
        # Symmetry
        for m1 in data["methods"]:
            for m2 in data["methods"]:
                assert abs(matrix[m1][m2] - matrix[m2][m1]) < 1e-6

    def test_probe_directions_npz_loadable(self):
        """Verify library can load paper's .npz probe direction files."""
        npz_path = Path(__file__).resolve().parents[2] / "papers" / "3_moral_geometry" / "outputs" / "exp1_2_3" / "exp1_probe_directions.npz"
        if not npz_path.exists():
            pytest.skip("Probe direction .npz not found")

        from deepsteer.directions.probe_weight import extract_from_npz
        from deepsteer.foundations import FOUNDATION_ORDER

        dirs = extract_from_npz(npz_path)
        for fv in FOUNDATION_ORDER:
            assert fv in dirs, f"Missing foundation {fv}"
            assert len(dirs[fv]) > 0, f"No layers for {fv}"
            for layer, d in dirs[fv].items():
                assert abs(np.linalg.norm(d) - 1.0) < 1e-6, f"Not unit norm at {fv} layer {layer}"


@pytest.mark.regression
class TestDirectionReproduction:
    """Full direction reproduction (requires model weights)."""

    def test_mean_diff_matches_shared_py(self):
        """Verify library mean_diff produces same directions as shared.py."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "papers" / "3_moral_geometry" / "scripts" / "probe_engineering"))

        from shared import (
            compute_mean_diff_directions as shared_mean_diff,
            load_model_and_collect_activations,
        )

        from deepsteer.directions.mean_diff import extract_mean_diff_directions

        all_train, _, _, n_layers, foundation_indices = (
            load_model_and_collect_activations(
                model_name="allenai/OLMo-2-0425-1B",
                target_per_foundation=40,
            )
        )

        shared_dirs = shared_mean_diff(all_train, n_layers, foundation_indices)
        lib_dirs = extract_mean_diff_directions(all_train, foundation_indices, n_layers)

        for fv in shared_dirs:
            for layer in shared_dirs[fv]:
                cos = abs(float(np.dot(shared_dirs[fv][layer], lib_dirs[fv][layer])))
                assert cos > 0.9999, (
                    f"Direction mismatch at {fv} layer {layer}: cosine={cos:.6f}"
                )
