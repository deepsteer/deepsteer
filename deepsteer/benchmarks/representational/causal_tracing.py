"""Causal tracing for moral decisions via activation corruption.

Identifies which layers are *causally* responsible for moral judgments
(not just correlated via probing).  For each layer, injects Gaussian noise
and measures how much the model's moral judgment score degrades.

Based on: Meng et al. (2022), "Locating and Editing Factual Associations
in GPT" (ROME); Vig et al. (2020), causal mediation analysis for LMs.
Full citations in REFERENCES.md.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor

from deepsteer.core.benchmark_suite import Benchmark
from deepsteer.core.model_interface import WhiteBoxModel
from deepsteer.core.types import (
    AccessTier,
    BenchmarkResult,
    CausalLayerEffect,
    CausalPromptResult,
    CausalTracingResult,
    MoralFoundation,
)
from deepsteer.datasets.pipeline import build_probing_dataset
from deepsteer.datasets.types import ProbingDataset

logger = logging.getLogger(__name__)


class MoralCausalTracer(Benchmark):
    """Causal tracing: corrupt each layer and measure moral judgment degradation.

    For each moral sentence from the probing dataset, frames a moral question
    and scores the expected completion.  Then, for each layer, registers a
    noise-injection hook and re-scores.  The difference (indirect effect)
    reveals which layers are causally necessary for moral judgment.

    Key metrics:
        - **peak_causal_layer**: layer whose corruption causes the largest drop
        - **causal_depth**: peak_causal_layer / n_layers (0 = early, 1 = late)
    """

    def __init__(
        self,
        dataset: ProbingDataset | None = None,
        *,
        noise_std: float = 3.0,
        max_prompts: int = 40,
        expected_completion: str = " No",
    ) -> None:
        self._dataset = dataset
        self._noise_std = noise_std
        self._max_prompts = max_prompts
        self._expected_completion = expected_completion

    @property
    def name(self) -> str:
        return "moral_causal_tracer"

    @property
    def min_access_tier(self) -> AccessTier:
        return AccessTier.WEIGHTS

    def run(self, model: WhiteBoxModel) -> BenchmarkResult:  # type: ignore[override]
        """Run causal tracing across all layers.

        Args:
            model: A WhiteBoxModel with ``score()`` and ``_get_layer_module()``.

        Returns:
            CausalTracingResult with per-prompt and aggregate causal metrics.
        """
        dataset = self._dataset or build_probing_dataset()
        n_layers = model.info.n_layers
        assert n_layers is not None, "Model must report n_layers"

        # Build prompts from moral sentences
        all_pairs = dataset.train + dataset.test
        prompts: list[tuple[str, MoralFoundation]] = []
        for pair in all_pairs:
            prompts.append((pair.moral, pair.foundation))
            if len(prompts) >= self._max_prompts:
                break

        logger.info(
            "Running causal tracing: %d prompts, %d layers, noise_std=%.1f",
            len(prompts), n_layers, self._noise_std,
        )

        prompt_results: list[CausalPromptResult] = []

        for sentence, foundation in prompts:
            framed = f"{sentence} Is this morally acceptable?"

            # Clean score
            clean_score = model.score(framed, self._expected_completion)

            # Corrupt each layer
            layer_effects: list[CausalLayerEffect] = []
            for layer_idx in range(n_layers):
                corrupted_score = self._score_with_noise(
                    model, framed, self._expected_completion, layer_idx,
                )
                indirect_effect = clean_score - corrupted_score
                layer_effects.append(CausalLayerEffect(
                    layer=layer_idx,
                    clean_score=clean_score,
                    corrupted_score=corrupted_score,
                    indirect_effect=indirect_effect,
                ))

            # Find peak for this prompt
            peak_effect = max(layer_effects, key=lambda e: e.indirect_effect)
            prompt_results.append(CausalPromptResult(
                prompt=framed,
                expected_completion=self._expected_completion,
                foundation=foundation,
                layer_effects=layer_effects,
                peak_causal_layer=peak_effect.layer,
                peak_indirect_effect=peak_effect.indirect_effect,
            ))

        # Aggregate: mean indirect effect per layer
        effect_sums: dict[int, float] = defaultdict(float)
        effect_counts: dict[int, int] = defaultdict(int)
        for pr in prompt_results:
            for le in pr.layer_effects:
                effect_sums[le.layer] += le.indirect_effect
                effect_counts[le.layer] += 1

        mean_by_layer: dict[int, float] = {}
        for layer_idx in range(n_layers):
            if effect_counts[layer_idx] > 0:
                mean_by_layer[layer_idx] = (
                    effect_sums[layer_idx] / effect_counts[layer_idx]
                )

        # Overall peak
        if mean_by_layer:
            peak_layer = max(mean_by_layer, key=lambda k: mean_by_layer[k])
            peak_mean = mean_by_layer[peak_layer]
        else:
            peak_layer = 0
            peak_mean = 0.0

        causal_depth = peak_layer / n_layers if n_layers > 0 else 0.0

        result = CausalTracingResult(
            benchmark_name=self.name,
            model_info=model.info,
            prompt_results=prompt_results,
            mean_indirect_effect_by_layer=mean_by_layer,
            peak_causal_layer=peak_layer,
            peak_mean_indirect_effect=peak_mean,
            causal_depth=causal_depth,
            metadata={
                "noise_std": self._noise_std,
                "max_prompts": self._max_prompts,
                "n_prompts_used": len(prompts),
                "expected_completion": self._expected_completion,
            },
        )

        logger.info(
            "CausalTracer: peak_layer=%d, peak_mean_effect=%.4f, causal_depth=%.3f",
            peak_layer, peak_mean, causal_depth,
        )
        return result

    def _score_with_noise(
        self,
        model: WhiteBoxModel,
        prompt: str,
        completion: str,
        layer: int,
    ) -> float:
        """Score a prompt+completion while injecting noise at a single layer."""
        noise_std = self._noise_std
        module = model._get_layer_module(layer)

        def _noise_hook(
            _module: torch.nn.Module, _input: Any, output: Any,
        ) -> Any:
            tensor = output[0] if isinstance(output, tuple) else output
            noised = tensor + torch.randn_like(tensor) * noise_std
            if isinstance(output, tuple):
                return (noised,) + output[1:]
            return noised

        handle = module.register_forward_hook(_noise_hook)
        try:
            score = model.score(prompt, completion)
        finally:
            handle.remove()
        return score
