"""General-purpose linear probing for any binary classification task.

Provides the same layer-wise linear probing methodology as LayerWiseMoralProbe,
but generalized to work with any set of binary-labeled sentence pairs, and with
support for pre-cached activations to enable efficient multi-probe experiments
(e.g. Phase C2 where moral, sentiment, and syntax probes share activations).

Linear probing methodology follows Alain & Bengio (2017), same as probing.py.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from deepsteer.core.model_interface import WhiteBoxModel
from deepsteer.core.types import LayerProbeScore, LayerProbingResult, ModelInfo

logger = logging.getLogger(__name__)

_ONSET_THRESHOLD = 0.6


def collect_activations_batch(
    model: WhiteBoxModel,
    texts: Sequence[str],
    *,
    progress_interval: int = 100,
) -> dict[str, dict[int, Tensor]]:
    """Collect mean-pooled activations for all texts across all layers.

    Runs one forward pass per unique text, capturing all layer activations
    and mean-pooling over the sequence dimension.  The returned cache can
    be reused by multiple probes to avoid redundant forward passes.

    Args:
        model: A WhiteBoxModel with activation capture support.
        texts: Texts to process.
        progress_interval: Log progress every N texts.

    Returns:
        Mapping from text string to ``{layer_idx: pooled_activation}``.
        Each pooled activation is a 1-D tensor of shape ``(hidden_dim,)``.
    """
    cache: dict[str, dict[int, Tensor]] = {}
    n_texts = len(texts)

    for i, text in enumerate(texts):
        if text in cache:
            continue
        acts = model.get_activations(text)  # all layers, one forward pass
        text_cache: dict[int, Tensor] = {}
        for layer_idx, layer_acts in acts.items():
            # layer_acts shape: (1, seq_len, hidden_dim) → mean pool → (hidden_dim,)
            text_cache[layer_idx] = layer_acts.squeeze(0).mean(dim=0)
        cache[text] = text_cache

        if (i + 1) % progress_interval == 0 or i + 1 == n_texts:
            logger.info("  Cached activations: %d/%d texts", i + 1, n_texts)

    return cache


def _build_probe_data(
    cache: dict[str, dict[int, Tensor]],
    positive_texts: list[str],
    negative_texts: list[str],
    n_layers: int,
) -> dict[int, tuple[Tensor, Tensor]]:
    """Build (X, y) tensors per layer from cached activations.

    Args:
        cache: Text → {layer → pooled_activation} from collect_activations_batch.
        positive_texts: Texts labeled as class 1.
        negative_texts: Texts labeled as class 0.
        n_layers: Number of layers to build data for.

    Returns:
        Mapping from layer index to ``(X, y)`` where X has shape
        ``(n_positive + n_negative, hidden_dim)`` and y has shape
        ``(n_positive + n_negative,)``.
    """
    result: dict[int, tuple[Tensor, Tensor]] = {}

    for layer_idx in range(n_layers):
        features: list[Tensor] = []
        labels: list[float] = []

        for text in positive_texts:
            features.append(cache[text][layer_idx])
            labels.append(1.0)
        for text in negative_texts:
            features.append(cache[text][layer_idx])
            labels.append(0.0)

        X = torch.stack(features).float()  # cast to fp32 for probing
        y = torch.tensor(labels, dtype=torch.float32)
        result[layer_idx] = (X, y)

    return result


class GeneralLinearProbe:
    """General-purpose layer-wise linear probe for binary classification.

    Same methodology as ``LayerWiseMoralProbe``: trains a
    ``Linear(hidden_dim -> 1)`` classifier with ``BCEWithLogitsLoss`` at
    each layer, evaluates on a held-out test set, and computes summary
    metrics (onset_layer, peak_layer, encoding depth/breadth).

    Can operate in two modes:
        1. **End-to-end** via ``run()``: pass a model and text pairs, handles
           activation collection internally.
        2. **Cached** via ``run_on_cached_activations()``: pass pre-cached
           activations from ``collect_activations_batch()``, avoiding
           redundant forward passes when running multiple probes.
    """

    def __init__(
        self,
        probe_name: str = "general_linear_probe",
        *,
        n_epochs: int = 50,
        lr: float = 1e-2,
        onset_threshold: float = _ONSET_THRESHOLD,
    ) -> None:
        self._probe_name = probe_name
        self._n_epochs = n_epochs
        self._lr = lr
        self._onset_threshold = onset_threshold

    def run_on_cached_activations(
        self,
        cache: dict[str, dict[int, Tensor]],
        train_pairs: list[tuple[str, str]],
        test_pairs: list[tuple[str, str]],
        model_info: ModelInfo,
        n_layers: int,
        *,
        checkpoint_step: int | None = None,
    ) -> LayerProbingResult:
        """Train probes using pre-cached activations.

        Args:
            cache: Text -> {layer -> pooled_activation} from
                ``collect_activations_batch()``.
            train_pairs: ``(positive_text, negative_text)`` tuples for training.
            test_pairs: ``(positive_text, negative_text)`` tuples for testing.
            model_info: Model metadata for the result.
            n_layers: Number of transformer layers.
            checkpoint_step: Optional training step of the checkpoint.

        Returns:
            LayerProbingResult with per-layer accuracy and summary metrics.
        """
        train_pos = [p[0] for p in train_pairs]
        train_neg = [p[1] for p in train_pairs]
        test_pos = [p[0] for p in test_pairs]
        test_neg = [p[1] for p in test_pairs]

        train_data = _build_probe_data(cache, train_pos, train_neg, n_layers)
        test_data = _build_probe_data(cache, test_pos, test_neg, n_layers)

        logger.info(
            "Running %s: %d layers, %d train, %d test pairs",
            self._probe_name, n_layers, len(train_pairs), len(test_pairs),
        )

        layer_scores: list[LayerProbeScore] = []
        for layer_idx in range(n_layers):
            train_X, train_y = train_data[layer_idx]
            test_X, test_y = test_data[layer_idx]
            accuracy, loss = self._train_probe(train_X, train_y, test_X, test_y)
            layer_scores.append(LayerProbeScore(
                layer=layer_idx, accuracy=accuracy, loss=loss,
            ))
            logger.debug(
                "  %s layer %d: accuracy=%.3f, loss=%.4f",
                self._probe_name, layer_idx, accuracy, loss,
            )

        return self._build_result(
            layer_scores, n_layers, model_info, checkpoint_step,
            len(train_pairs), len(test_pairs),
        )

    def run(
        self,
        model: WhiteBoxModel,
        train_pairs: list[tuple[str, str]],
        test_pairs: list[tuple[str, str]],
    ) -> LayerProbingResult:
        """Full pipeline: collect activations and train probes.

        Args:
            model: WhiteBoxModel with activation capture support.
            train_pairs: ``(positive_text, negative_text)`` tuples for training.
            test_pairs: ``(positive_text, negative_text)`` tuples for testing.

        Returns:
            LayerProbingResult with per-layer accuracy and summary metrics.
        """
        n_layers = model.info.n_layers
        assert n_layers is not None, "Model must report n_layers"

        all_texts: list[str] = []
        for pos, neg in train_pairs + test_pairs:
            all_texts.extend([pos, neg])

        cache = collect_activations_batch(model, all_texts)
        return self.run_on_cached_activations(
            cache, train_pairs, test_pairs, model.info, n_layers,
            checkpoint_step=model.info.checkpoint_step,
        )

    def _build_result(
        self,
        layer_scores: list[LayerProbeScore],
        n_layers: int,
        model_info: ModelInfo,
        checkpoint_step: int | None,
        n_train: int,
        n_test: int,
    ) -> LayerProbingResult:
        """Compute summary metrics from per-layer scores."""
        onset_layer = None
        peak_layer = 0
        peak_accuracy = 0.0
        n_above = 0

        for score in layer_scores:
            if score.accuracy > peak_accuracy:
                peak_accuracy = score.accuracy
                peak_layer = score.layer
            if score.accuracy >= self._onset_threshold:
                n_above += 1
                if onset_layer is None:
                    onset_layer = score.layer

        depth = (onset_layer / n_layers) if onset_layer is not None else 1.0
        breadth = n_above / n_layers

        result = LayerProbingResult(
            benchmark_name=self._probe_name,
            model_info=model_info,
            layer_scores=layer_scores,
            onset_layer=onset_layer,
            peak_layer=peak_layer,
            peak_accuracy=peak_accuracy,
            moral_encoding_depth=depth,
            moral_encoding_breadth=breadth,
            checkpoint_step=checkpoint_step,
            metadata={
                "n_epochs": self._n_epochs,
                "lr": self._lr,
                "onset_threshold": self._onset_threshold,
                "train_pairs": n_train,
                "test_pairs": n_test,
            },
        )

        logger.info(
            "%s complete: onset=%s, peak=%d (%.1f%%), depth=%.3f, breadth=%.3f",
            self._probe_name, onset_layer, peak_layer, peak_accuracy * 100,
            depth, breadth,
        )
        return result

    def _train_probe(
        self,
        train_X: Tensor,
        train_y: Tensor,
        test_X: Tensor,
        test_y: Tensor,
    ) -> tuple[float, float]:
        """Train a linear probe and evaluate on test set.

        Same methodology as ``LayerWiseMoralProbe._train_probe()``.

        Returns:
            ``(accuracy, loss)`` on the test set.
        """
        hidden_dim = train_X.shape[1]
        probe = nn.Linear(hidden_dim, 1)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self._lr)
        loss_fn = nn.BCEWithLogitsLoss()

        # Train
        probe.train()
        for _ in range(self._n_epochs):
            logits = probe(train_X).squeeze(-1)
            loss = loss_fn(logits, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        probe.eval()
        with torch.no_grad():
            test_logits = probe(test_X).squeeze(-1)
            test_loss = loss_fn(test_logits, test_y).item()
            preds = (test_logits > 0).float()
            accuracy = (preds == test_y).float().mean().item()

        return accuracy, test_loss
