"""Benchmark base class and suite runner."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from deepsteer.core.model_interface import ModelInterface
from deepsteer.core.types import AccessTier, BenchmarkResult, SuiteResult

logger = logging.getLogger(__name__)


class Benchmark(ABC):
    """Abstract base for a single evaluation benchmark."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable benchmark name."""

    @property
    @abstractmethod
    def min_access_tier(self) -> AccessTier:
        """Minimum model access tier required to run this benchmark."""

    @abstractmethod
    def run(self, model: ModelInterface) -> BenchmarkResult:
        """Execute the benchmark against *model* and return results."""


class BenchmarkSuite:
    """Collects benchmarks and runs them against a model.

    Benchmarks whose ``min_access_tier`` exceeds the model's tier are
    automatically skipped with a log message — never an error.
    """

    def __init__(self, benchmarks: list[Benchmark] | None = None) -> None:
        self._benchmarks = list(benchmarks or [])

    def add(self, benchmark: Benchmark) -> None:
        """Register a benchmark."""
        self._benchmarks.append(benchmark)

    def run(self, model: ModelInterface) -> SuiteResult:
        """Run all applicable benchmarks and return aggregated results."""
        results: dict[str, BenchmarkResult] = {}
        skipped: dict[str, str] = {}

        for bench in self._benchmarks:
            if model.access_tier < bench.min_access_tier:
                reason = (
                    f"requires {bench.min_access_tier.name} tier, "
                    f"model provides {model.access_tier.name}"
                )
                logger.info("Skipping %s: %s", bench.name, reason)
                skipped[bench.name] = reason
                continue

            logger.info("Running %s …", bench.name)
            try:
                result = bench.run(model)
                results[bench.name] = result
                logger.info("Finished %s", bench.name)
            except Exception:
                logger.exception("Benchmark %s failed", bench.name)
                skipped[bench.name] = "raised an exception (see logs)"

        return SuiteResult(
            benchmark_name="suite",
            model_info=model.info,
            results=results,
            skipped=skipped,
        )
