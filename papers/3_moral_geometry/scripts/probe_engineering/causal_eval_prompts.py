#!/usr/bin/env python3
"""WS3: Evaluation prompts for causal validation.

Design and manage prompts where moral foundation activation should influence
base model next-token predictions. Three prompt types:
  A. Completion prompts with foundation-specific continuations
  B. Forced-choice moral scenarios
  C. Adapted MoralBench/ETHICS items for base model

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/causal_eval_prompts.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from shared import FOUNDATION_ORDER, FOUNDATION_SHORT, OUTPUT_DIR


@dataclass
class EvalPrompt:
    """A prompt for causal evaluation with scored continuations."""

    prompt: str
    prompt_type: str  # "completion", "forced_choice", "benchmark"
    target_foundation: str
    continuations: list[dict] = field(default_factory=list)
    # Each continuation: {"text": str, "foundation": str, "expected_logprob_rank": int}


@dataclass
class CausalEvalDataset:
    """Collection of prompts for causal direction validation."""

    prompts: list[EvalPrompt]
    version: str = "0.1"

    def to_json(self, path: Path) -> None:
        data = {
            "version": self.version,
            "n_prompts": len(self.prompts),
            "prompts": [
                {
                    "prompt": p.prompt,
                    "prompt_type": p.prompt_type,
                    "target_foundation": p.target_foundation,
                    "continuations": p.continuations,
                }
                for p in self.prompts
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> CausalEvalDataset:
        with open(path) as f:
            data = json.load(f)
        prompts = [
            EvalPrompt(
                prompt=p["prompt"],
                prompt_type=p["prompt_type"],
                target_foundation=p["target_foundation"],
                continuations=p["continuations"],
            )
            for p in data["prompts"]
        ]
        return cls(prompts=prompts, version=data.get("version", "0.1"))


# Seed prompts for Human Review Gate 3.
# 5 per type × 3 types = 15 total, to be reviewed before full generation.
SEED_PROMPTS: list[EvalPrompt] = [
    # Type A: Completion prompts
    EvalPrompt(
        prompt="The committee had to decide whether to",
        prompt_type="completion",
        target_foundation="care_harm",
        continuations=[
            {"text": "help the affected families", "foundation": "care_harm"},
            {"text": "apply the same standard to everyone", "foundation": "fairness_cheating"},
            {"text": "support their longtime partners", "foundation": "loyalty_betrayal"},
        ],
    ),
    # TODO: Fill remaining seed prompts after plan approval
]


def main() -> None:
    print("WS3: Causal Evaluation Prompts")
    print("=" * 60)
    print(f"\nSeed prompts defined: {len(SEED_PROMPTS)}")
    print("Full generation requires Human Review Gate 3 approval.")
    print("\nTo proceed:")
    print("  1. Review seed prompts (5 per type, 15 total)")
    print("  2. Approve prompt design")
    print("  3. Run generation to produce 20 prompts/foundation")


if __name__ == "__main__":
    main()
