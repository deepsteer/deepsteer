"""Tests for layer-wise moral probing and checkpoint trajectory analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
from deepsteer.benchmarks.representational.trajectory import (
    CheckpointTrajectoryProbe,
    _parse_step_from_revision,
)
from deepsteer.core.types import (
    AccessTier,
    CheckpointTrajectoryResult,
    LayerProbeScore,
    LayerProbingResult,
    MoralFoundation,
    ModelInfo,
)
from deepsteer.datasets.types import (
    DatasetMetadata,
    GenerationMethod,
    NeutralDomain,
    ProbingDataset,
    ProbingPair,
)


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------


class MockWhiteBoxModel:
    """Fake WhiteBoxModel that returns separable activations.

    Moral texts get activations drawn from N(+1, 0.1).
    Neutral texts get activations drawn from N(-1, 0.1).
    """

    def __init__(
        self,
        moral_texts: set[str],
        *,
        n_layers: int = 4,
        hidden_dim: int = 32,
    ) -> None:
        self._moral_texts = moral_texts
        self._n_layers = n_layers
        self._hidden_dim = hidden_dim
        self._info = ModelInfo(
            name="mock-model",
            provider="test",
            access_tier=AccessTier.WEIGHTS,
            n_layers=n_layers,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def access_tier(self) -> AccessTier:
        return self._info.access_tier

    def get_activations(
        self, text: str, layers: list[int] | None = None
    ) -> dict[int, Tensor]:
        if layers is None:
            layers = list(range(self._n_layers))

        is_moral = text in self._moral_texts
        mean = 1.0 if is_moral else -1.0

        result: dict[int, Tensor] = {}
        for layer in layers:
            # Shape: (1, seq_len=5, hidden_dim)
            activation = torch.randn(1, 5, self._hidden_dim) * 0.1 + mean
            result[layer] = activation
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(moral: str, neutral: str) -> ProbingPair:
    return ProbingPair(
        moral=moral,
        neutral=neutral,
        foundation=MoralFoundation.CARE_HARM,
        neutral_domain=NeutralDomain.COOKING,
        generation_method=GenerationMethod.POOL,
        moral_word_count=len(moral.split()),
        neutral_word_count=len(neutral.split()),
    )


def _make_dataset(n_train: int = 20, n_test: int = 5) -> tuple[ProbingDataset, set[str]]:
    """Create a small dataset and return it with the set of moral texts."""
    moral_texts: set[str] = set()
    train: list[ProbingPair] = []
    test: list[ProbingPair] = []

    for i in range(n_train):
        moral = f"moral train sentence number {i}"
        neutral = f"neutral train sentence number {i}"
        moral_texts.add(moral)
        train.append(_make_pair(moral, neutral))

    for i in range(n_test):
        moral = f"moral test sentence number {i}"
        neutral = f"neutral test sentence number {i}"
        moral_texts.add(moral)
        test.append(_make_pair(moral, neutral))

    metadata = DatasetMetadata(
        version="1.0.0",
        generation_method="test",
        total_pairs=n_train + n_test,
        train_pairs=n_train,
        test_pairs=n_test,
    )
    return ProbingDataset(train=train, test=test, metadata=metadata), moral_texts


# ---------------------------------------------------------------------------
# LayerWiseMoralProbe tests
# ---------------------------------------------------------------------------


class TestLayerWiseMoralProbe:
    def test_result_schema(self):
        """Result has all expected fields with correct types."""
        dataset, moral_texts = _make_dataset()
        model = MockWhiteBoxModel(moral_texts)
        probe = LayerWiseMoralProbe(dataset=dataset, n_epochs=30)

        result = probe.run(model)

        assert isinstance(result, LayerProbingResult)
        assert result.benchmark_name == "layer_wise_moral_probe"
        assert result.model_info is not None
        assert result.model_info.name == "mock-model"
        assert len(result.layer_scores) == 4
        assert result.onset_layer is not None or result.onset_layer is None  # always valid
        assert result.peak_layer is not None
        assert 0.0 <= result.peak_accuracy <= 1.0
        assert 0.0 <= result.moral_encoding_depth <= 1.0
        assert 0.0 <= result.moral_encoding_breadth <= 1.0

    def test_high_accuracy_on_separable_data(self):
        """Mock model produces well-separated activations → high accuracy."""
        dataset, moral_texts = _make_dataset(n_train=30, n_test=10)
        model = MockWhiteBoxModel(moral_texts)
        probe = LayerWiseMoralProbe(dataset=dataset, n_epochs=50)

        result = probe.run(model)

        assert result.peak_accuracy >= 0.8, (
            f"Expected high accuracy on separable data, got {result.peak_accuracy:.2f}"
        )

    def test_all_layers_scored(self):
        """Every layer gets a score entry."""
        dataset, moral_texts = _make_dataset()
        model = MockWhiteBoxModel(moral_texts, n_layers=6)
        probe = LayerWiseMoralProbe(dataset=dataset, n_epochs=20)

        result = probe.run(model)

        scored_layers = {s.layer for s in result.layer_scores}
        assert scored_layers == set(range(6))

    def test_serialization(self):
        """Result serializes to valid JSON with expected keys."""
        dataset, moral_texts = _make_dataset()
        model = MockWhiteBoxModel(moral_texts)
        probe = LayerWiseMoralProbe(dataset=dataset, n_epochs=20)

        result = probe.run(model)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "layer_scores" in d
        assert "onset_layer" in d
        assert "peak_layer" in d
        assert "peak_accuracy" in d
        assert "moral_encoding_depth" in d
        assert "moral_encoding_breadth" in d
        assert "model_info" in d

        # Verify it's JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_benchmark_properties(self):
        """Name and min_access_tier are correct."""
        probe = LayerWiseMoralProbe()
        assert probe.name == "layer_wise_moral_probe"
        assert probe.min_access_tier == AccessTier.WEIGHTS

    def test_metadata_captures_hyperparams(self):
        """Result metadata includes the probe hyperparameters."""
        dataset, moral_texts = _make_dataset()
        model = MockWhiteBoxModel(moral_texts)
        probe = LayerWiseMoralProbe(dataset=dataset, n_epochs=25, lr=0.005)

        result = probe.run(model)

        assert result.metadata["n_epochs"] == 25
        assert result.metadata["lr"] == 0.005


# ---------------------------------------------------------------------------
# CheckpointTrajectoryProbe tests
# ---------------------------------------------------------------------------


class TestCheckpointTrajectoryProbe:
    def test_benchmark_properties(self):
        """Name and min_access_tier are correct."""
        probe = CheckpointTrajectoryProbe(
            checkpoint_revisions=["step100", "step200"],
        )
        assert probe.name == "checkpoint_trajectory_probe"
        assert probe.min_access_tier == AccessTier.CHECKPOINTS

    def test_parse_step_from_revision(self):
        """Revision string parsing extracts step numbers correctly."""
        assert _parse_step_from_revision("step1000-tokens4B") == 1000
        assert _parse_step_from_revision("step500") == 500
        assert _parse_step_from_revision("step0-tokens0B") == 0
        assert _parse_step_from_revision("main") is None
        assert _parse_step_from_revision("v1.0") is None

    def test_step_auto_parsing(self):
        """Steps are auto-parsed from revision strings when not provided."""
        probe = CheckpointTrajectoryProbe(
            checkpoint_revisions=["step100-tokens1B", "step200-tokens2B"],
        )
        assert probe._steps == [100, 200]

    def test_step_fallback_to_index(self):
        """When revision doesn't contain step info, fall back to index."""
        probe = CheckpointTrajectoryProbe(
            checkpoint_revisions=["main", "dev"],
        )
        assert probe._steps == [0, 1]


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------


class TestVisualization:
    def test_plot_layer_probing_creates_files(self, tmp_path: Path):
        """PNG and JSON files are created with correct names."""
        from deepsteer.viz import plot_layer_probing

        result = LayerProbingResult(
            benchmark_name="layer_wise_moral_probe",
            model_info=ModelInfo(
                name="test/model",
                provider="test",
                access_tier=AccessTier.WEIGHTS,
                n_layers=4,
            ),
            layer_scores=[
                LayerProbeScore(layer=0, accuracy=0.52, loss=0.7),
                LayerProbeScore(layer=1, accuracy=0.65, loss=0.5),
                LayerProbeScore(layer=2, accuracy=0.80, loss=0.3),
                LayerProbeScore(layer=3, accuracy=0.75, loss=0.35),
            ],
            onset_layer=1,
            peak_layer=2,
            peak_accuracy=0.80,
            moral_encoding_depth=0.25,
            moral_encoding_breadth=0.75,
            metadata={"onset_threshold": 0.6},
        )

        png_path = plot_layer_probing(result, output_dir=tmp_path)

        assert png_path.exists()
        assert png_path.suffix == ".png"
        assert png_path.stat().st_size > 0

        json_path = png_path.with_suffix(".json")
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert data["benchmark_name"] == "layer_wise_moral_probe"
        assert len(data["layer_scores"]) == 4
        assert data["peak_accuracy"] == 0.80

    def test_plot_layer_probing_no_onset(self, tmp_path: Path):
        """Plot handles the case where no layer crosses onset threshold."""
        from deepsteer.viz import plot_layer_probing

        result = LayerProbingResult(
            benchmark_name="layer_wise_moral_probe",
            model_info=ModelInfo(
                name="test/model", provider="test", access_tier=AccessTier.WEIGHTS,
            ),
            layer_scores=[
                LayerProbeScore(layer=0, accuracy=0.50, loss=0.7),
                LayerProbeScore(layer=1, accuracy=0.52, loss=0.68),
            ],
            onset_layer=None,
            peak_layer=1,
            peak_accuracy=0.52,
            moral_encoding_depth=1.0,
            moral_encoding_breadth=0.0,
            metadata={"onset_threshold": 0.6},
        )

        png_path = plot_layer_probing(result, output_dir=tmp_path)
        assert png_path.exists()

    def test_plot_checkpoint_trajectory_creates_files(self, tmp_path: Path):
        """Heatmap PNG and JSON are created for trajectory results."""
        from deepsteer.viz import plot_checkpoint_trajectory

        model_info = ModelInfo(
            name="test/model", provider="test", access_tier=AccessTier.CHECKPOINTS,
        )
        trajectory = []
        for step in [100, 200, 300]:
            scores = [
                LayerProbeScore(layer=i, accuracy=0.5 + 0.1 * i + 0.001 * step, loss=0.5)
                for i in range(4)
            ]
            trajectory.append(LayerProbingResult(
                benchmark_name="layer_wise_moral_probe",
                model_info=model_info,
                layer_scores=scores,
                onset_layer=1,
                peak_layer=3,
                peak_accuracy=scores[3].accuracy,
                checkpoint_step=step,
            ))

        result = CheckpointTrajectoryResult(
            benchmark_name="checkpoint_trajectory_probe",
            model_info=model_info,
            trajectory=trajectory,
            checkpoint_steps=[100, 200, 300],
        )

        png_path = plot_checkpoint_trajectory(result, output_dir=tmp_path)
        assert png_path.exists()
        assert png_path.suffix == ".png"

        json_path = png_path.with_suffix(".json")
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert len(data["trajectory"]) == 3
        assert data["checkpoint_steps"] == [100, 200, 300]

    def test_plot_trajectory_empty_raises(self, tmp_path: Path):
        """Empty trajectory raises ValueError."""
        from deepsteer.viz import plot_checkpoint_trajectory

        result = CheckpointTrajectoryResult(
            benchmark_name="checkpoint_trajectory_probe",
            trajectory=[],
            checkpoint_steps=[],
        )
        with pytest.raises(ValueError, match="no trajectory entries"):
            plot_checkpoint_trajectory(result, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# End-to-end with mock model
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_probe_then_plot(self, tmp_path: Path):
        """Full pipeline: build dataset → run probe → plot."""
        from deepsteer.viz import plot_layer_probing

        dataset, moral_texts = _make_dataset(n_train=20, n_test=5)
        model = MockWhiteBoxModel(moral_texts, n_layers=4)
        probe = LayerWiseMoralProbe(dataset=dataset, n_epochs=30)

        result = probe.run(model)
        png_path = plot_layer_probing(result, output_dir=tmp_path)

        assert png_path.exists()
        json_path = png_path.with_suffix(".json")
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert data["model_info"]["name"] == "mock-model"
        assert len(data["layer_scores"]) == 4


# ---------------------------------------------------------------------------
# OLMo integration (requires real model download)
# ---------------------------------------------------------------------------


class TestModelComparison:
    def test_plot_model_comparison_creates_files(self, tmp_path: Path):
        """Comparison plot produces PNG and JSON with data from all models."""
        from deepsteer.viz import plot_model_comparison

        results = []
        for name, n_layers in [("model-a", 4), ("model-b", 8)]:
            results.append(LayerProbingResult(
                benchmark_name="layer_wise_moral_probe",
                model_info=ModelInfo(
                    name=name, provider="test", access_tier=AccessTier.WEIGHTS,
                    n_layers=n_layers,
                ),
                layer_scores=[
                    LayerProbeScore(layer=i, accuracy=0.5 + 0.05 * i, loss=0.5)
                    for i in range(n_layers)
                ],
                onset_layer=1,
                peak_layer=n_layers - 1,
                peak_accuracy=0.5 + 0.05 * (n_layers - 1),
                moral_encoding_depth=1 / n_layers,
                moral_encoding_breadth=0.75,
                metadata={"onset_threshold": 0.6},
            ))

        png_path = plot_model_comparison(results, output_dir=tmp_path)
        assert png_path.exists()
        assert png_path.suffix == ".png"

        json_path = png_path.with_suffix(".json")
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert len(data["models"]) == 2
        assert data["models"][0]["name"] == "model-a"
        assert data["models"][1]["name"] == "model-b"

    def test_plot_model_comparison_single_model(self, tmp_path: Path):
        """Single-model comparison still works (degenerate case)."""
        from deepsteer.viz import plot_model_comparison

        result = LayerProbingResult(
            benchmark_name="layer_wise_moral_probe",
            model_info=ModelInfo(
                name="solo", provider="test", access_tier=AccessTier.WEIGHTS,
            ),
            layer_scores=[LayerProbeScore(layer=0, accuracy=0.6, loss=0.5)],
            onset_layer=0,
            peak_layer=0,
            peak_accuracy=0.6,
            metadata={"onset_threshold": 0.6},
        )
        png_path = plot_model_comparison([result], output_dir=tmp_path)
        assert png_path.exists()

    def test_plot_model_comparison_empty_raises(self, tmp_path: Path):
        """Empty results list raises ValueError."""
        from deepsteer.viz import plot_model_comparison

        with pytest.raises(ValueError, match="empty"):
            plot_model_comparison([], output_dir=tmp_path)


# ---------------------------------------------------------------------------
# OLMo integration (requires real model download)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestOLMoIntegration:
    """Tests that require downloading OLMo-1B-hf (~5GB).

    Run with: pytest -m slow
    """

    def test_olmo_model_loads(self):
        """OLMo-1B-hf loads and reports correct architecture info."""
        from deepsteer.core.model_interface import WhiteBoxModel

        model = WhiteBoxModel(
            "allenai/OLMo-1B-hf",
            device="cpu",
            torch_dtype=torch.float32,
        )
        assert model.info.n_layers == 16
        assert model.info.n_params > 0

    def test_olmo_activations(self):
        """Activation capture returns tensors with expected shapes."""
        from deepsteer.core.model_interface import WhiteBoxModel

        model = WhiteBoxModel(
            "allenai/OLMo-1B-hf",
            device="cpu",
            torch_dtype=torch.float32,
        )
        acts = model.get_activations("Hello world", layers=[0, 8, 15])

        assert set(acts.keys()) == {0, 8, 15}
        for layer_idx, tensor in acts.items():
            assert tensor.ndim == 3  # (1, seq_len, hidden_dim)
            assert tensor.shape[0] == 1
            assert tensor.shape[2] == 2048  # OLMo-1B hidden dim

    def test_olmo_full_probe(self, tmp_path: Path):
        """Full layer probing on OLMo-1B-hf with a small dataset."""
        from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
        from deepsteer.core.model_interface import WhiteBoxModel
        from deepsteer.datasets.pipeline import build_probing_dataset
        from deepsteer.viz import plot_layer_probing

        model = WhiteBoxModel(
            "allenai/OLMo-1B-hf",
            device="cpu",
            torch_dtype=torch.float32,
        )
        dataset = build_probing_dataset(target_per_foundation=5)
        probe = LayerWiseMoralProbe(dataset=dataset, n_epochs=30)

        result = probe.run(model)

        assert len(result.layer_scores) == 16
        assert result.peak_layer is not None
        assert result.peak_accuracy > 0.0

        png_path = plot_layer_probing(result, output_dir=tmp_path)
        assert png_path.exists()


# ---------------------------------------------------------------------------
# Llama integration (requires real model download)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLlamaIntegration:
    """Tests that verify Llama-architecture models work with WhiteBoxModel.

    Uses TinyLlama (1.1B, freely available, Llama architecture) to avoid
    license-gated models.  Run with: pytest -m slow
    """

    _MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    _N_LAYERS = 22
    _HIDDEN_DIM = 2048

    def test_llama_model_loads(self):
        """TinyLlama loads and reports correct architecture info."""
        from deepsteer.core.model_interface import WhiteBoxModel

        model = WhiteBoxModel(
            self._MODEL_ID,
            device="cpu",
            torch_dtype=torch.float32,
        )
        assert model.info.n_layers == self._N_LAYERS
        assert model.info.n_params > 0
        # Verify pad_token was set (Llama tokenizers often lack one)
        assert model.tokenizer.pad_token_id is not None

    def test_llama_layer_detection(self):
        """Layer detection uses model.model.layers (same path as OLMo)."""
        from deepsteer.core.model_interface import WhiteBoxModel

        model = WhiteBoxModel(
            self._MODEL_ID,
            device="cpu",
            torch_dtype=torch.float32,
        )
        # _get_layer_module should work for all layers
        for i in [0, self._N_LAYERS // 2, self._N_LAYERS - 1]:
            module = model._get_layer_module(i)
            assert module is not None

    def test_llama_activations(self):
        """Activation capture returns tensors with expected shapes."""
        from deepsteer.core.model_interface import WhiteBoxModel

        model = WhiteBoxModel(
            self._MODEL_ID,
            device="cpu",
            torch_dtype=torch.float32,
        )
        layers_to_probe = [0, self._N_LAYERS // 2, self._N_LAYERS - 1]
        acts = model.get_activations("Hello world", layers=layers_to_probe)

        assert set(acts.keys()) == set(layers_to_probe)
        for layer_idx, tensor in acts.items():
            assert tensor.ndim == 3
            assert tensor.shape[0] == 1
            assert tensor.shape[2] == self._HIDDEN_DIM

    def test_llama_full_probe(self, tmp_path: Path):
        """Full layer probing on TinyLlama with a small dataset."""
        from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
        from deepsteer.core.model_interface import WhiteBoxModel
        from deepsteer.datasets.pipeline import build_probing_dataset
        from deepsteer.viz import plot_layer_probing

        model = WhiteBoxModel(
            self._MODEL_ID,
            device="cpu",
            torch_dtype=torch.float32,
        )
        dataset = build_probing_dataset(target_per_foundation=5)
        probe = LayerWiseMoralProbe(dataset=dataset, n_epochs=30)

        result = probe.run(model)

        assert len(result.layer_scores) == self._N_LAYERS
        assert result.peak_layer is not None
        assert result.peak_accuracy > 0.0

        png_path = plot_layer_probing(result, output_dir=tmp_path)
        assert png_path.exists()


@pytest.mark.slow
class TestCrossModelComparison:
    """Comparative test requiring both OLMo and Llama downloads.

    Run with: pytest -m slow
    """

    def test_olmo_vs_llama_comparison_plot(self, tmp_path: Path):
        """Produce a comparative plot from OLMo-1B and TinyLlama."""
        from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
        from deepsteer.core.model_interface import WhiteBoxModel
        from deepsteer.datasets.pipeline import build_probing_dataset
        from deepsteer.viz import plot_model_comparison

        dataset = build_probing_dataset(target_per_foundation=5)
        probe = LayerWiseMoralProbe(dataset=dataset, n_epochs=20)
        results = []

        for model_id in ["allenai/OLMo-1B-hf", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]:
            model = WhiteBoxModel(model_id, device="cpu", torch_dtype=torch.float32)
            result = probe.run(model)
            results.append(result)
            del model

        png_path = plot_model_comparison(results, output_dir=tmp_path)
        assert png_path.exists()

        json_path = png_path.with_suffix(".json")
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert len(data["models"]) == 2
