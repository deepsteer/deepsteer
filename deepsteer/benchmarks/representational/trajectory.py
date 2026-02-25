"""Checkpoint trajectory probing: run LayerWiseMoralProbe across training checkpoints.

Uses OLMo's published intermediate checkpoints for training trajectory analysis.
See: Groeneveld et al. (2024), "OLMo: Accelerating the Science of Language Models."
arXiv:2402.00838. Full citation in REFERENCES.md.
"""

from __future__ import annotations

import gc
import logging
import re

import torch

from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
from deepsteer.core.benchmark_suite import Benchmark
from deepsteer.core.model_interface import ModelInterface, WhiteBoxModel
from deepsteer.core.types import (
    AccessTier,
    BenchmarkResult,
    CheckpointTrajectoryResult,
    LayerProbingResult,
)
from deepsteer.datasets.types import ProbingDataset

logger = logging.getLogger(__name__)


def _parse_step_from_revision(revision: str) -> int | None:
    """Extract the training step number from a HuggingFace revision string.

    Examples:
        >>> _parse_step_from_revision("step1000-tokens4B")
        1000
        >>> _parse_step_from_revision("step500")
        500
        >>> _parse_step_from_revision("main")
    """
    match = re.search(r"step(\d+)", revision)
    return int(match.group(1)) if match else None


def list_available_revisions(repo_id: str) -> list[str]:
    """Query HuggingFace Hub for available branch revisions of a model repo.

    Args:
        repo_id: HuggingFace repository ID (e.g. ``"allenai/OLMo-1B-hf"``).

    Returns:
        List of branch names, sorted by parsed step number where possible.
    """
    from huggingface_hub import list_repo_refs

    refs = list_repo_refs(repo_id)
    branches = [b.name for b in refs.branches]

    def _sort_key(name: str) -> tuple[int, str]:
        step = _parse_step_from_revision(name)
        return (step if step is not None else -1, name)

    branches.sort(key=_sort_key)
    return branches


class CheckpointTrajectoryProbe(Benchmark):
    """Run layer-wise moral probing across multiple training checkpoints.

    Loads each checkpoint revision as a fresh ``WhiteBoxModel``, runs
    ``LayerWiseMoralProbe``, and collects the per-layer accuracy curves into
    a ``CheckpointTrajectoryResult``.

    This benchmark requires ``AccessTier.CHECKPOINTS`` because it assumes
    access to intermediate training checkpoints (e.g. OLMo branch revisions).
    """

    def __init__(
        self,
        checkpoint_revisions: list[str],
        checkpoint_steps: list[int] | None = None,
        dataset: ProbingDataset | None = None,
        *,
        n_epochs: int = 50,
        lr: float = 1e-2,
        onset_threshold: float = 0.6,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        self._revisions = checkpoint_revisions
        self._steps = checkpoint_steps or [
            _parse_step_from_revision(r) or i
            for i, r in enumerate(checkpoint_revisions)
        ]
        self._dataset = dataset
        self._n_epochs = n_epochs
        self._lr = lr
        self._onset_threshold = onset_threshold
        self._device = device
        self._torch_dtype = torch_dtype

    @property
    def name(self) -> str:
        return "checkpoint_trajectory_probe"

    @property
    def min_access_tier(self) -> AccessTier:
        return AccessTier.CHECKPOINTS

    def run(self, model: ModelInterface) -> BenchmarkResult:
        """Run probing at each checkpoint revision.

        Args:
            model: A ``WhiteBoxModel`` whose ``info.name`` is the HuggingFace
                repo ID to load revisions from.

        Returns:
            CheckpointTrajectoryResult with one LayerProbingResult per checkpoint.
        """
        repo_id = model.info.name
        trajectory: list[LayerProbingResult] = []

        probe = LayerWiseMoralProbe(
            dataset=self._dataset,
            n_epochs=self._n_epochs,
            lr=self._lr,
            onset_threshold=self._onset_threshold,
        )

        for revision, step in zip(self._revisions, self._steps):
            logger.info("Probing checkpoint: %s (step %s)", revision, step)

            checkpoint_model = WhiteBoxModel(
                repo_id,
                revision=revision,
                device=self._device,
                torch_dtype=self._torch_dtype,
                access_tier=AccessTier.CHECKPOINTS,
                checkpoint_step=step,
            )

            result = probe.run(checkpoint_model)
            assert isinstance(result, LayerProbingResult)
            trajectory.append(result)

            # Free memory
            del checkpoint_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return CheckpointTrajectoryResult(
            benchmark_name=self.name,
            model_info=model.info,
            trajectory=trajectory,
            checkpoint_steps=self._steps,
            metadata={
                "revisions": self._revisions,
                "n_epochs": self._n_epochs,
                "lr": self._lr,
                "onset_threshold": self._onset_threshold,
            },
        )
