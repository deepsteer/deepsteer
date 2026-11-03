"""MACHIAVELLI (Pan et al., arXiv:2304.03279): the LM agents' prompts.

The engine and game bundle are fetched, never vendored: :mod:`deepsteer.datasets.machiavelli`.
Upstream prompt text is MIT-licensed (``LICENSE_MACHIAVELLI``).
"""

from __future__ import annotations

from deepsteer.benchmarks.agentic.machiavelli.prompts import (
    AGENTS,
    GOOD_BEHAVIOR,
    UPSTREAM_COMMIT,
    SceneState,
    chat_prompts,
    completion_prompt,
    cot_answer_prompt,
)

__all__ = [
    "AGENTS",
    "GOOD_BEHAVIOR",
    "UPSTREAM_COMMIT",
    "SceneState",
    "chat_prompts",
    "completion_prompt",
    "cot_answer_prompt",
]
