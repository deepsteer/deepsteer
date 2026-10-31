"""MoReBench (Chiu, Lee, et al., arXiv:2510.16380): prompts, judge prompt and scoring.

The judge lives here, inside the MoReBench code (MRB_PLAN v0.2 §6); the data loader is
:mod:`deepsteer.datasets.morebench`. Upstream prompt text is MIT-licensed (``LICENSE_MOREBENCH``).
"""

from __future__ import annotations

from deepsteer.benchmarks.moral_reasoning.morebench.prompts import (
    GENERATION_INSTRUCTION,
    JUDGE_INSTRUCTION,
    UPSTREAM_COMMIT,
    generation_prompt,
    judge_prompt,
)
from deepsteer.benchmarks.moral_reasoning.morebench.scoring import (
    aggregate,
    code_score,
    eq1_score,
    eq1_to_paper_scale,
    fulfilled,
)

__all__ = [
    "GENERATION_INSTRUCTION",
    "JUDGE_INSTRUCTION",
    "UPSTREAM_COMMIT",
    "aggregate",
    "code_score",
    "eq1_score",
    "eq1_to_paper_scale",
    "fulfilled",
    "generation_prompt",
    "judge_prompt",
]
