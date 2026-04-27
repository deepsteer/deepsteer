"""LoRA fine-tuning experiment runners for Phase C Tier 2.

Each function loads a model, prepares corpora, and runs LoRATrainer across
multiple conditions (e.g., narrative vs declarative vs control).

Experiments:
    C3: Narrative vs. Declarative Moral Framing
    C4: Early vs. Late LoRA (contingent on C3/C6 signal)
    C5: Foundation Coverage (contingent on C3/C6 signal)
    C6: Moral Acceleration from Random Init
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import torch

from deepsteer.core.model_interface import WhiteBoxModel
from deepsteer.core.types import AccessTier, LoRAExperimentResult
from deepsteer.datasets.pipeline import build_probing_dataset
from deepsteer.steering.lora_trainer import LoRATrainer

logger = logging.getLogger(__name__)

REPO_ID = "allenai/OLMo-2-0425-1B-early-training"


def _clear_memory() -> None:
    """Free GPU/MPS memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _load_model(
    revision: str, step: int, device: str | None,
) -> WhiteBoxModel:
    """Load OLMo-2 1B at a specific checkpoint."""
    return WhiteBoxModel(
        REPO_ID,
        revision=revision,
        device=device,
        access_tier=AccessTier.CHECKPOINTS,
        checkpoint_step=step,
    )


def _save_condition_result(
    result: LoRAExperimentResult,
    output_dir: Path,
    condition_name: str,
) -> None:
    """Save a single condition result as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{condition_name}.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Saved result: %s", json_path)


def run_c3_narrative_vs_declarative(
    base_revision: str = "stage1-step1000-tokens3B",
    base_step: int = 1000,
    *,
    max_steps: int = 1000,
    eval_every: int = 100,
    lora_rank: int = 16,
    output_dir: str | Path = "outputs/lora_experiments/c3",
    device: str | None = None,
    quick: bool = False,
) -> dict[str, LoRAExperimentResult]:
    """C3: Does narrative vs. declarative framing affect moral encoding?

    Three conditions:
        1. narrative_moral: Aesop/Grimm/Andersen fairy tales and fables
        2. declarative_moral: Template-expanded MORAL_SEEDS
        3. general_control: Darwin's Voyage of the Beagle (non-moral)

    Base model: step 1K (mid-transition, moral encoding partially formed).

    Args:
        base_revision: HuggingFace revision string for base checkpoint.
        base_step: Training step of base checkpoint.
        max_steps: Maximum LoRA training steps per condition.
        eval_every: Evaluate probing every N steps.
        lora_rank: LoRA rank parameter.
        output_dir: Output directory for results.
        device: Device (cuda, mps, cpu). Auto-detected if None.
        quick: If True, use reduced corpus and fewer eval points.

    Returns:
        Dict mapping condition name to LoRAExperimentResult.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_tokens = 50_000 if quick else 500_000
    if quick:
        eval_every = max(eval_every, max_steps // 3)

    # Build probing dataset
    dataset = build_probing_dataset(
        target_per_foundation=10 if quick else 40,
    )

    conditions: dict[str, tuple[str, list[str]]] = {}

    # Load corpora
    from deepsteer.datasets.corpora import (
        load_declarative_corpus,
        load_general_corpus,
        load_narrative_corpus,
    )

    logger.info("Loading narrative corpus...")
    conditions["narrative_moral"] = (
        "narrative_moral",
        load_narrative_corpus(max_tokens=max_tokens),
    )

    logger.info("Loading declarative corpus...")
    conditions["declarative_moral"] = (
        "declarative_moral",
        load_declarative_corpus(max_tokens=max_tokens),
    )

    logger.info("Loading general control corpus...")
    conditions["general_control"] = (
        "general_control",
        load_general_corpus(max_tokens=max_tokens),
    )

    results: dict[str, LoRAExperimentResult] = {}

    for condition_name, (corpus_name, corpus) in conditions.items():
        print(f"\n{'='*60}")
        print(f"C3 Condition: {condition_name} ({len(corpus)} chunks)")
        print(f"{'='*60}")

        model = _load_model(base_revision, base_step, device)

        trainer = LoRATrainer(
            model,
            corpus,
            dataset,
            lora_rank=lora_rank,
            max_steps=max_steps,
            eval_every=eval_every,
        )

        t0 = time.time()
        result = trainer.train(
            experiment_id=f"c3_{condition_name}",
            hypothesis="C3: Narrative vs declarative moral framing",
            corpus_name=corpus_name,
        )
        elapsed = time.time() - t0

        results[condition_name] = result
        _save_condition_result(result, output_dir, condition_name)

        print(f"  Completed in {elapsed:.1f}s")
        if result.final_probing:
            print(f"  Final peak accuracy: {result.final_probing.peak_accuracy:.3f}")
        if result.final_fragility:
            print(f"  Final mean critical noise: "
                  f"{result.final_fragility.mean_critical_noise:.3f}")

        del model, trainer
        _clear_memory()

    return results


def run_c6_moral_acceleration(
    base_revision: str = "stage1-step0-tokens0B",
    base_step: int = 0,
    *,
    max_steps: int = 2000,
    eval_every: int = 50,
    lora_rank: int = 16,
    output_dir: str | Path = "outputs/lora_experiments/c6",
    device: str | None = None,
    quick: bool = False,
) -> dict[str, LoRAExperimentResult]:
    """C6: Can moral content accelerate moral encoding from random init?

    Two conditions:
        1. moral_curriculum: Declarative moral content
        2. general_control: Non-moral Darwin text

    Base model: step 0 (random init, before any pre-training).
    Finer resolution (eval_every=50) to detect transition timing.

    Args:
        base_revision: HuggingFace revision string for base checkpoint.
        base_step: Training step of base checkpoint.
        max_steps: Maximum LoRA training steps per condition.
        eval_every: Evaluate probing every N steps.
        lora_rank: LoRA rank parameter.
        output_dir: Output directory for results.
        device: Device (cuda, mps, cpu). Auto-detected if None.
        quick: If True, use reduced corpus and fewer eval points.

    Returns:
        Dict mapping condition name to LoRAExperimentResult.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_tokens = 50_000 if quick else 500_000
    if quick:
        eval_every = max(eval_every, max_steps // 3)

    dataset = build_probing_dataset(
        target_per_foundation=10 if quick else 40,
    )

    from deepsteer.datasets.corpora import load_declarative_corpus, load_general_corpus

    conditions: dict[str, tuple[str, list[str]]] = {}

    logger.info("Loading declarative (moral) corpus...")
    conditions["moral_curriculum"] = (
        "moral_curriculum",
        load_declarative_corpus(max_tokens=max_tokens),
    )

    logger.info("Loading general control corpus...")
    conditions["general_control"] = (
        "general_control",
        load_general_corpus(max_tokens=max_tokens),
    )

    results: dict[str, LoRAExperimentResult] = {}

    for condition_name, (corpus_name, corpus) in conditions.items():
        print(f"\n{'='*60}")
        print(f"C6 Condition: {condition_name} ({len(corpus)} chunks)")
        print(f"{'='*60}")

        model = _load_model(base_revision, base_step, device)

        trainer = LoRATrainer(
            model,
            corpus,
            dataset,
            lora_rank=lora_rank,
            max_steps=max_steps,
            eval_every=eval_every,
        )

        t0 = time.time()
        result = trainer.train(
            experiment_id=f"c6_{condition_name}",
            hypothesis="C6: Moral acceleration from random init",
            corpus_name=corpus_name,
        )
        elapsed = time.time() - t0

        results[condition_name] = result
        _save_condition_result(result, output_dir, condition_name)

        print(f"  Completed in {elapsed:.1f}s")
        if result.final_probing:
            print(f"  Final peak accuracy: {result.final_probing.peak_accuracy:.3f}")
        if result.final_fragility:
            print(f"  Final mean critical noise: "
                  f"{result.final_fragility.mean_critical_noise:.3f}")

        del model, trainer
        _clear_memory()

    return results


def run_c4_early_vs_late(
    *,
    max_steps: int = 1000,
    eval_every: int = 100,
    lora_rank: int = 16,
    output_dir: str | Path = "outputs/lora_experiments/c4",
    device: str | None = None,
    quick: bool = False,
) -> dict[str, LoRAExperimentResult]:
    """C4: Does LoRA injection timing matter? (Early vs Late checkpoint)

    Two conditions using declarative moral corpus:
        1. early_lora: LoRA on step 1K (mid-transition)
        2. late_lora: LoRA on step 10K (post-saturation)

    Only run if C3/C6 show signal (>10% fragility difference).

    Args:
        max_steps: Maximum LoRA training steps per condition.
        eval_every: Evaluate probing every N steps.
        lora_rank: LoRA rank parameter.
        output_dir: Output directory for results.
        device: Device (cuda, mps, cpu). Auto-detected if None.
        quick: If True, use reduced corpus and fewer eval points.

    Returns:
        Dict mapping condition name to LoRAExperimentResult.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_tokens = 50_000 if quick else 500_000
    if quick:
        eval_every = max(eval_every, max_steps // 3)

    dataset = build_probing_dataset(
        target_per_foundation=10 if quick else 40,
    )

    from deepsteer.datasets.corpora import load_declarative_corpus

    corpus = load_declarative_corpus(max_tokens=max_tokens)

    checkpoints = {
        "early_lora": ("stage1-step1000-tokens3B", 1000),
        "late_lora": ("stage1-step10000-tokens21B", 10000),
    }

    results: dict[str, LoRAExperimentResult] = {}

    for condition_name, (revision, step) in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"C4 Condition: {condition_name} (base step {step})")
        print(f"{'='*60}")

        model = _load_model(revision, step, device)

        trainer = LoRATrainer(
            model,
            corpus,
            dataset,
            lora_rank=lora_rank,
            max_steps=max_steps,
            eval_every=eval_every,
        )

        t0 = time.time()
        result = trainer.train(
            experiment_id=f"c4_{condition_name}",
            hypothesis="C4: Early vs late LoRA injection",
            corpus_name="declarative_moral",
        )
        elapsed = time.time() - t0

        results[condition_name] = result
        _save_condition_result(result, output_dir, condition_name)

        print(f"  Completed in {elapsed:.1f}s")
        del model, trainer
        _clear_memory()

    return results


def run_c5_foundation_coverage(
    base_revision: str = "stage1-step1000-tokens3B",
    base_step: int = 1000,
    *,
    max_steps: int = 1000,
    eval_every: int = 100,
    lora_rank: int = 16,
    output_dir: str | Path = "outputs/lora_experiments/c5",
    device: str | None = None,
    quick: bool = False,
) -> dict[str, LoRAExperimentResult]:
    """C5: Does foundation-specific moral content selectively affect encoding?

    Six conditions, one per MFT foundation:
        Each uses declarative corpus filtered to a single foundation.

    Only run if C3/C6 show signal (>10% fragility difference).

    Args:
        base_revision: HuggingFace revision string for base checkpoint.
        base_step: Training step of base checkpoint.
        max_steps: Maximum LoRA training steps per condition.
        eval_every: Evaluate probing every N steps.
        lora_rank: LoRA rank parameter.
        output_dir: Output directory for results.
        device: Device (cuda, mps, cpu). Auto-detected if None.
        quick: If True, use reduced corpus and fewer eval points.

    Returns:
        Dict mapping condition name to LoRAExperimentResult.
    """
    from deepsteer.core.types import MoralFoundation
    from deepsteer.datasets.corpora import load_declarative_corpus

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_tokens = 50_000 if quick else 500_000
    if quick:
        eval_every = max(eval_every, max_steps // 3)

    dataset = build_probing_dataset(
        target_per_foundation=10 if quick else 40,
    )

    results: dict[str, LoRAExperimentResult] = {}

    for foundation in MoralFoundation:
        condition_name = foundation.value
        print(f"\n{'='*60}")
        print(f"C5 Condition: {condition_name}")
        print(f"{'='*60}")

        corpus = load_declarative_corpus(
            max_tokens=max_tokens,
            foundation=foundation,
        )

        model = _load_model(base_revision, base_step, device)

        trainer = LoRATrainer(
            model,
            corpus,
            dataset,
            lora_rank=lora_rank,
            max_steps=max_steps,
            eval_every=eval_every,
        )

        t0 = time.time()
        result = trainer.train(
            experiment_id=f"c5_{condition_name}",
            hypothesis="C5: Foundation-specific moral content",
            corpus_name=f"declarative_{condition_name}",
        )
        elapsed = time.time() - t0

        results[condition_name] = result
        _save_condition_result(result, output_dir, condition_name)

        print(f"  Completed in {elapsed:.1f}s")
        del model, trainer
        _clear_memory()

    return results
