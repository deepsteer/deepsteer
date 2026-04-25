"""Persona-probe activation on free-form response text.

Companion to :class:`PersonaFeatureProbe`, which reports layer-wise probe
*accuracy* on held-out minimal pairs.  This module trains a single-layer
linear probe at a chosen layer and then exposes it as a scalar scoring
function for arbitrary text — so we can ask "how strongly does this model
output activate the toxic-persona direction?" rather than just "is the
direction decodable?"

Use case: Phase D C10 primary readout (H15).  For each Betley-style benign
prompt, we generate a response from (a) the base model, (b) the insecure-
LoRA fine-tuned model, (c) the secure-LoRA control, and then compute the
probe's scalar activation on each response.  The headline metric is the
paired (insecure - secure) mean activation delta.

Design choices (documented because they're load-bearing):

1. **Probe is trained on the *base* model** at a single layer and then
   applied to the post-LoRA models.  The direction lives in the shared base
   hidden-state space; LoRA rank-16 on q/v only is a small perturbation
   that doesn't move the residual-stream basis far enough to invalidate
   the probe.  This is the same assumption Wang et al. (2025) rely on
   when transferring their GPT-4o SAE latent across fine-tunes.
2. **Response text is scored standalone**, not prompt+response.  The
   probe was trained on short sentence-level texts; feeding just the
   response matches that distribution.  Activating the probe on
   full-context hidden states at response token positions is a different
   quantity (closer to Wang et al.'s SAE-latent-on-generation measure)
   and would require a separate probe trained accordingly.
3. **Scoring uses the raw logit** (pre-sigmoid) so arithmetic like
   "mean activation" and "paired delta" has standard-normal-ish scale
   and additivity, following the probe's BCE-with-logits training
   objective.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from deepsteer.core.model_interface import WhiteBoxModel

logger = logging.getLogger(__name__)


@dataclass
class PersonaProbeWeights:
    """Serializable weights for a single-layer persona activation probe."""

    layer: int
    hidden_dim: int
    weight: list[float]  # shape (hidden_dim,)
    bias: float
    train_accuracy: float
    test_accuracy: float
    n_train_pairs: int
    n_test_pairs: int
    n_epochs: int
    lr: float
    seed: int

    def to_tensors(self) -> tuple[Tensor, float]:
        """Return ``(weight_tensor_fp32, bias_float)`` for scoring."""
        w = torch.tensor(self.weight, dtype=torch.float32)
        return w, float(self.bias)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PersonaProbeWeights":
        return cls(**payload)


def _mean_pool_activations(model: WhiteBoxModel, text: str, layer: int) -> Tensor:
    """Run one forward pass, return mean-pooled layer activations as fp32 CPU tensor."""
    acts = model.get_activations(text, layers=[layer])
    # acts[layer] shape: (1, seq_len, hidden_dim), CPU tensor in model dtype
    pooled = acts[layer].squeeze(0).float().mean(dim=0)  # (hidden_dim,)
    return pooled


def train_persona_probe(
    model: WhiteBoxModel,
    train_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]],
    *,
    layer: int,
    n_epochs: int = 50,
    lr: float = 1e-2,
    seed: int = 42,
) -> PersonaProbeWeights:
    """Train a single-layer linear probe on persona/neutral minimal pairs.

    Matches :class:`GeneralLinearProbe` methodology: ``Linear(hidden_dim, 1)``,
    BCE-with-logits, Adam, 50 epochs, fp32.  The probe is trained once; its
    weights are returned as a serializable dataclass so the same direction
    can be re-applied to many models (e.g. base, insecure-LoRA, secure-LoRA).

    Args:
        model: WhiteBoxModel whose activations define the training space.
        train_pairs: ``(persona_voice, neutral_voice)`` tuples for training.
        test_pairs: Held-out pairs for accuracy reporting.
        layer: Which transformer layer's activations to probe.
        n_epochs: Adam epochs.
        lr: Adam learning rate.
        seed: Torch RNG seed for reproducible init.

    Returns:
        :class:`PersonaProbeWeights` with learned weight/bias and sanity-
        check train/test accuracy.
    """
    logger.info(
        "Training persona activation probe: layer %d, %d train pairs, %d test pairs",
        layer, len(train_pairs), len(test_pairs),
    )

    # Collect activations for every unique text once.
    unique_texts: dict[str, int] = {}
    for pos, neg in train_pairs + test_pairs:
        unique_texts.setdefault(pos, len(unique_texts))
        unique_texts.setdefault(neg, len(unique_texts))

    cache: dict[str, Tensor] = {}
    for i, text in enumerate(unique_texts):
        cache[text] = _mean_pool_activations(model, text, layer)
        if (i + 1) % 100 == 0 or i + 1 == len(unique_texts):
            logger.info("  Cached activations: %d/%d", i + 1, len(unique_texts))

    def _build_xy(pairs: list[tuple[str, str]]) -> tuple[Tensor, Tensor]:
        feats: list[Tensor] = []
        labels: list[float] = []
        for pos, neg in pairs:
            feats.append(cache[pos])
            labels.append(1.0)
            feats.append(cache[neg])
            labels.append(0.0)
        return torch.stack(feats), torch.tensor(labels, dtype=torch.float32)

    train_X, train_y = _build_xy(train_pairs)
    test_X, test_y = _build_xy(test_pairs)
    hidden_dim = train_X.shape[1]

    torch.manual_seed(seed)
    probe = nn.Linear(hidden_dim, 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    probe.train()
    for _ in range(n_epochs):
        logits = probe(train_X).squeeze(-1)
        loss = loss_fn(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        train_logits = probe(train_X).squeeze(-1)
        train_preds = (train_logits > 0).float()
        train_acc = (train_preds == train_y).float().mean().item()

        test_logits = probe(test_X).squeeze(-1)
        test_preds = (test_logits > 0).float()
        test_acc = (test_preds == test_y).float().mean().item()

    weight = probe.weight.detach().squeeze(0).cpu().tolist()
    bias = float(probe.bias.detach().squeeze().cpu().item())

    logger.info(
        "Persona activation probe: layer %d, train_acc=%.3f, test_acc=%.3f",
        layer, train_acc, test_acc,
    )

    return PersonaProbeWeights(
        layer=layer,
        hidden_dim=hidden_dim,
        weight=weight,
        bias=bias,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        n_train_pairs=len(train_pairs),
        n_test_pairs=len(test_pairs),
        n_epochs=n_epochs,
        lr=lr,
        seed=seed,
    )


@dataclass
class PersonaActivationScore:
    """Scalar activation for a single (prompt, response) on one scoring method."""

    prompt: str
    response: str
    score_response_only: float  # probe logit on response text alone
    score_response_tokens_in_context: float  # probe logit on response tokens, with prompt context


@dataclass
class PersonaActivationBatchResult:
    """Batch-level persona-activation results for one condition."""

    label: str
    layer: int
    n_samples: int
    mean_response_only: float
    std_response_only: float
    mean_response_in_context: float
    std_response_in_context: float
    scores_response_only: list[float] = field(default_factory=list)
    scores_response_in_context: list[float] = field(default_factory=list)
    per_sample: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PersonaActivationScorer:
    """Applies a trained persona probe to arbitrary (prompt, response) pairs.

    Two scoring variants are computed per sample:

    - ``score_response_only``: mean-pool layer activations over a standalone
      forward pass of the response text.  Matches the probe's training
      distribution (sentence-level).
    - ``score_response_in_context``: mean-pool layer activations over the
      response *token positions only* during a prompt+response forward
      pass.  Closer in spirit to Wang et al.'s SAE-latent-on-generation
      measurement.

    Both are useful; the primary C10 readout uses ``score_response_only`` as
    the main comparison because it aligns with how the probe was trained.
    """

    def __init__(self, probe: PersonaProbeWeights) -> None:
        self._probe = probe
        weight, bias = probe.to_tensors()
        self._weight = weight  # (hidden_dim,)
        self._bias = bias
        self._layer = probe.layer

    @property
    def probe(self) -> PersonaProbeWeights:
        return self._probe

    @torch.no_grad()
    def _score_pooled(self, pooled: Tensor) -> float:
        return float((pooled @ self._weight + self._bias).item())

    @torch.no_grad()
    def score_response_only(
        self, model: WhiteBoxModel, response: str
    ) -> float:
        """Score a response text on its own (single forward pass of response)."""
        if not response.strip():
            return float("nan")
        pooled = _mean_pool_activations(model, response, self._layer)
        return self._score_pooled(pooled)

    @torch.no_grad()
    def score_response_in_context(
        self, model: WhiteBoxModel, prompt: str, response: str
    ) -> float:
        """Score a response by mean-pooling over its token positions in a prompt+response forward pass.

        Re-tokenizes the prompt alone to find the prompt's token length, then
        slices activations from that offset onward.  Concatenated tokenization
        may shift the prompt boundary by ±1 token in edge cases, but the
        mean-pool is robust to small offsets.
        """
        if not response.strip():
            return float("nan")
        prompt_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = int(prompt_ids.shape[1])
        acts = model.get_activations(prompt + response, layers=[self._layer])
        tensor = acts[self._layer].squeeze(0).float()  # (seq_len, hidden_dim)
        if tensor.shape[0] <= prompt_len:
            return float("nan")
        response_acts = tensor[prompt_len:, :]
        pooled = response_acts.mean(dim=0)
        return self._score_pooled(pooled)

    def score_samples(
        self,
        model: WhiteBoxModel,
        samples: list[dict[str, Any]],
        *,
        label: str,
        progress_interval: int = 25,
    ) -> PersonaActivationBatchResult:
        """Score every sample in ``samples`` (dicts with 'prompt' and 'response').

        Returns aggregate statistics plus per-sample scores.  Also mutates
        each sample dict in-place with ``persona_activation_response_only``
        and ``persona_activation_response_in_context`` fields so they can
        be saved alongside the original eval output.
        """
        scores_only: list[float] = []
        scores_ctx: list[float] = []
        per_sample: list[dict[str, Any]] = []

        for i, s in enumerate(samples):
            prompt = s.get("prompt", "")
            response = s.get("response", "")
            score_only = self.score_response_only(model, response)
            score_ctx = self.score_response_in_context(model, prompt, response)
            s["persona_activation_response_only"] = score_only
            s["persona_activation_response_in_context"] = score_ctx
            if not _is_nan(score_only):
                scores_only.append(score_only)
            if not _is_nan(score_ctx):
                scores_ctx.append(score_ctx)
            per_sample.append({
                "question_id": s.get("question_id"),
                "paraphrase_index": s.get("paraphrase_index"),
                "score_response_only": score_only,
                "score_response_in_context": score_ctx,
            })
            if (i + 1) % progress_interval == 0 or i + 1 == len(samples):
                logger.info(
                    "  %s probe activation: %d/%d samples (mean-only=%.3f)",
                    label, i + 1, len(samples),
                    _safe_mean(scores_only),
                )

        return PersonaActivationBatchResult(
            label=label,
            layer=self._layer,
            n_samples=len(samples),
            mean_response_only=_safe_mean(scores_only),
            std_response_only=_safe_std(scores_only),
            mean_response_in_context=_safe_mean(scores_ctx),
            std_response_in_context=_safe_std(scores_ctx),
            scores_response_only=scores_only,
            scores_response_in_context=scores_ctx,
            per_sample=per_sample,
        )


def _is_nan(x: float) -> bool:
    return x != x  # noqa: PLR0124 - NaN != NaN


def _safe_mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _safe_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return float(var ** 0.5)


__all__ = [
    "PersonaActivationBatchResult",
    "PersonaActivationScore",
    "PersonaActivationScorer",
    "PersonaProbeWeights",
    "train_persona_probe",
]
