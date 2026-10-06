"""Smoke-test an installed deepsteer wheel (LIBRARY_RELEASE_PLAN §A).

Run with the target venv's interpreter in isolated mode, from outside the repo, so the
source tree cannot shadow the installed package:

    python -I scripts/wheel_smoke.py

Most probable failure: moral_probing_v2.json missing from the wheel. Through 0.1.0,
build_probing_dataset() silently fell back to the v1 minimal-pair pipeline in that case; it
now raises, and the check also asserts the dataset version is 2.0.0 so a reintroduced fallback
cannot pass.
"""

from __future__ import annotations

import importlib.metadata

import deepsteer
from deepsteer.benchmarks.compliance_gap.em_behavioral import load_first_plot_questions
from deepsteer.causal import ablation_sweep  # noqa: F401
from deepsteer.datasets import build_probing_dataset
from deepsteer.datasets.loaders import load_dilemma_pairs
from deepsteer.directions import extract_mean_diff_directions  # noqa: F401
from deepsteer.geometry import full_geometric_analysis  # noqa: F401


def main() -> None:
    """Import the core API and load every packaged data file."""
    assert "site-packages" in deepsteer.__file__, f"not the installed wheel: {deepsteer.__file__}"
    version = importlib.metadata.version("deepsteer")
    assert deepsteer.__version__ == version, f"__version__ {deepsteer.__version__} != {version}"
    print("deepsteer", version, deepsteer.__file__)

    dataset = build_probing_dataset(target_per_foundation=5)
    assert dataset.metadata.version == "2.0.0", "v1 dataset under the v2 default"
    print("probing dataset", dataset.metadata.version, "train", len(dataset.train))

    assert load_dilemma_pairs()["pairs"], "dilemma pairs empty"
    assert len(load_first_plot_questions()) == 8, "EM first-plot fixture incomplete"
    print("OK")


if __name__ == "__main__":
    main()
