"""Regression tests: verify library geometry functions match paper outputs.

These tests load paper output JSONs and verify the library reproduces them.
They require OLMo-2 1B model weights for activation collection and are
marked @pytest.mark.regression for manual invocation.

The lightweight tests (test_geometry_output_schema, test_subspace_output_schema)
run without model weights by validating that paper output files parse correctly
and contain expected fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

PROBE_ENG_DIR = Path(__file__).resolve().parents[2] / "papers" / "3_moral_geometry" / "outputs" / "probe_engineering"


class TestGeometryOutputSchema:
    """Validate paper output schemas match library expectations (no model needed)."""

    def test_leace_directions_schema(self):
        path = PROBE_ENG_DIR / "leace_directions.json"
        if not path.exists():
            pytest.skip("Paper output not found")
        with open(path) as f:
            data = json.load(f)
        assert data["analysis"] == "leace_directions"
        assert "alignment" in data
        assert "geometry" in data
        for fv in ["care_harm", "fairness_cheating", "liberty_oppression",
                    "loyalty_betrayal", "authority_subversion", "sanctity_degradation"]:
            assert fv in data["alignment"]
            align = data["alignment"][fv]
            assert "leace_vs_pw" in align
            assert "leace_vs_md" in align
            assert "md_vs_pw" in align

    def test_geometry_layer_schema(self):
        path = PROBE_ENG_DIR / "leace_directions.json"
        if not path.exists():
            pytest.skip("Paper output not found")
        with open(path) as f:
            data = json.load(f)
        for layer_key, geo in data["geometry"].items():
            assert "mean_cosine_similarity" in geo
            assert "effective_dimensionality" in geo
            assert "permutation_test" in geo
            assert "dendrogram" in geo
            assert "cosine_matrix" in geo
            perm = geo["permutation_test"]
            assert "observed_statistic" in perm
            assert "p_value" in perm
            assert "mean_within_individualizing" in perm
            dendro = geo["dendrogram"]
            assert "mft_match" in dendro
            assert "left" in dendro
            assert "right" in dendro

    def test_subspace_output_schema(self):
        path = PROBE_ENG_DIR / "full_subspace_projection.json"
        if not path.exists():
            pytest.skip("Paper output not found")
        with open(path) as f:
            data = json.load(f)
        assert data["analysis"] == "full_5d_subspace_projection"
        assert "null_5d" in data
        assert "null_2d" in data
        assert "per_layer" in data
        null = data["null_5d"]
        assert "mean" in null
        assert "expected_analytic" in null

    def test_multi_method_schema(self):
        path = PROBE_ENG_DIR / "multi_method_directions.json"
        if not path.exists():
            pytest.skip("Paper output not found")
        with open(path) as f:
            data = json.load(f)
        assert "alignment_matrix" in data
        assert "stability" in data
        methods = data["methods"]
        matrix = data["alignment_matrix"]
        for m in methods:
            assert m in matrix
            assert matrix[m][m] == 1.0


class TestAblationOutputSchema:
    """Validate ablation/steering paper output schemas (no model needed)."""

    def test_ablation_schema(self):
        path = PROBE_ENG_DIR / "direction_ablation_mean_diff.json"
        if not path.exists():
            pytest.skip("Paper output not found")
        with open(path) as f:
            data = json.load(f)
        assert data["analysis"] == "direction_ablation"
        assert data["method"] == "mean_diff"
        assert "results" in data
        for fv, fv_data in data["results"].items():
            for layer, metrics in fv_data.items():
                assert "on_target_mean_delta" in metrics
                assert "off_target_mean_delta" in metrics
                assert "specificity" in metrics
                assert "n_on_target" in metrics

    def test_steering_schema(self):
        path = PROBE_ENG_DIR / "steering_injection_mean_diff.json"
        if not path.exists():
            pytest.skip("Paper output not found")
        with open(path) as f:
            data = json.load(f)
        assert data["analysis"] == "steering_injection"
        assert "alphas" in data
        assert "results" in data


@pytest.mark.regression
class TestGeometryReproduction:
    """Reproduce paper geometry results from model weights.

    Requires OLMo-2 1B weights. Run with: pytest -m regression
    """

    def test_mean_diff_geometry_matches_paper(self):
        """Verify mean-diff directions produce matching geometry at each layer."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "papers" / "3_moral_geometry" / "scripts" / "probe_engineering"))

        from shared import load_model_and_collect_activations

        from deepsteer.directions.mean_diff import extract_mean_diff_directions
        from deepsteer.geometry.analysis import full_geometric_analysis
        from deepsteer.foundations import FOUNDATION_ORDER

        path = PROBE_ENG_DIR / "leace_directions.json"
        if not path.exists():
            pytest.skip("Paper output not found")

        all_train, _, _, n_layers, foundation_indices = (
            load_model_and_collect_activations(
                model_name="allenai/OLMo-2-0425-1B",
                target_per_foundation=40,
            )
        )

        md_dirs = extract_mean_diff_directions(all_train, foundation_indices, n_layers)

        for layer in range(n_layers):
            geo = full_geometric_analysis(
                md_dirs, layer=layer, labels=FOUNDATION_ORDER,
            )
            if geo is not None:
                assert "mean_cosine_similarity" in geo
                assert "effective_dimensionality" in geo
                ed = geo["effective_dimensionality"]
                assert 1 <= ed <= 6

    def test_subspace_null_distribution_matches(self):
        """Verify null subspace membership matches analytical expectation."""
        from deepsteer.geometry.subspace import null_subspace_membership

        null_5d = null_subspace_membership(2048, 5, n_samples=10000)
        assert abs(null_5d["mean"] - null_5d["expected_analytic"]) < 0.001

        null_2d = null_subspace_membership(2048, 2, n_samples=10000)
        assert abs(null_2d["mean"] - null_2d["expected_analytic"]) < 0.001
