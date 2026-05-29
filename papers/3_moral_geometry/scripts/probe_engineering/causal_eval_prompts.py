#!/usr/bin/env python3
"""WS3: Evaluation prompts for causal validation.

Design and manage prompts where moral foundation activation should influence
base model next-token predictions. Three prompt types:
  A. Completion prompts with foundation-specific continuations
  B. Forced-choice moral scenarios
  C. Natural language prompts with morally-loaded vs neutral completions

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/causal_eval_prompts.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from shared import FOUNDATION_ORDER, FOUNDATION_SHORT, ensure_output_dirs, OUTPUT_DIR


@dataclass
class EvalPrompt:
    """A prompt for causal evaluation with scored continuations."""

    prompt: str
    prompt_type: str  # "completion", "forced_choice", "natural"
    target_foundation: str
    continuations: list[dict] = field(default_factory=list)


@dataclass
class CausalEvalDataset:
    """Collection of prompts for causal direction validation."""

    prompts: list[EvalPrompt]
    version: str = "1.0"

    def to_json(self, path: Path) -> None:
        data = {
            "version": self.version,
            "n_prompts": len(self.prompts),
            "by_type": {},
            "by_foundation": {},
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
        from collections import Counter
        data["by_type"] = dict(Counter(p.prompt_type for p in self.prompts))
        data["by_foundation"] = dict(Counter(p.target_foundation for p in self.prompts))
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
        return cls(prompts=prompts, version=data.get("version", "1.0"))


# ---------------------------------------------------------------------------
# Type A: Completion prompts
# Short prompts where the natural continuation depends on moral foundation.
# The base model should assign higher probability to foundation-aligned
# continuations when that foundation's direction is amplified.
# ---------------------------------------------------------------------------

COMPLETION_PROMPTS = [
    # Care/Harm
    EvalPrompt(
        prompt="When she saw the injured child lying by the road, she felt compelled to",
        prompt_type="completion",
        target_foundation="care_harm",
        continuations=[
            {"text": " help", "foundation": "care_harm", "is_target": True},
            {"text": " ignore", "foundation": "neutral", "is_target": False},
            {"text": " report", "foundation": "authority_subversion", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="The nurse stayed past her shift because the patient was",
        prompt_type="completion",
        target_foundation="care_harm",
        continuations=[
            {"text": " suffering", "foundation": "care_harm", "is_target": True},
            {"text": " sleeping", "foundation": "neutral", "is_target": False},
            {"text": " waiting", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="A society that abandons its most vulnerable members has lost its",
        prompt_type="completion",
        target_foundation="care_harm",
        continuations=[
            {"text": " compassion", "foundation": "care_harm", "is_target": True},
            {"text": " efficiency", "foundation": "neutral", "is_target": False},
            {"text": " authority", "foundation": "authority_subversion", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="The doctor refused to discharge the elderly patient because she was still",
        prompt_type="completion",
        target_foundation="care_harm",
        continuations=[
            {"text": " in pain", "foundation": "care_harm", "is_target": True},
            {"text": " on file", "foundation": "neutral", "is_target": False},
            {"text": " confused", "foundation": "care_harm", "is_target": True},
        ],
    ),

    # Fairness/Cheating
    EvalPrompt(
        prompt="The workers demanded that the bonus be distributed",
        prompt_type="completion",
        target_foundation="fairness_cheating",
        continuations=[
            {"text": " equally", "foundation": "fairness_cheating", "is_target": True},
            {"text": " quickly", "foundation": "neutral", "is_target": False},
            {"text": " privately", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="If everyone followed the same rules, the outcome would be more",
        prompt_type="completion",
        target_foundation="fairness_cheating",
        continuations=[
            {"text": " fair", "foundation": "fairness_cheating", "is_target": True},
            {"text": " predictable", "foundation": "neutral", "is_target": False},
            {"text": " efficient", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="The referee penalized the player for gaining an unfair",
        prompt_type="completion",
        target_foundation="fairness_cheating",
        continuations=[
            {"text": " advantage", "foundation": "fairness_cheating", "is_target": True},
            {"text": " reputation", "foundation": "neutral", "is_target": False},
            {"text": " lead", "foundation": "fairness_cheating", "is_target": True},
        ],
    ),
    EvalPrompt(
        prompt="She accused the committee of rigging the selection process to favor",
        prompt_type="completion",
        target_foundation="fairness_cheating",
        continuations=[
            {"text": " insiders", "foundation": "fairness_cheating", "is_target": True},
            {"text": " speed", "foundation": "neutral", "is_target": False},
            {"text": " tradition", "foundation": "authority_subversion", "is_target": False},
        ],
    ),

    # Liberty/Oppression
    EvalPrompt(
        prompt="The new law restricted citizens from expressing their",
        prompt_type="completion",
        target_foundation="liberty_oppression",
        continuations=[
            {"text": " opinions", "foundation": "liberty_oppression", "is_target": True},
            {"text": " schedules", "foundation": "neutral", "is_target": False},
            {"text": " loyalty", "foundation": "loyalty_betrayal", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="Under the regime, people lived in constant fear of",
        prompt_type="completion",
        target_foundation="liberty_oppression",
        continuations=[
            {"text": " surveillance", "foundation": "liberty_oppression", "is_target": True},
            {"text": " weather", "foundation": "neutral", "is_target": False},
            {"text": " traffic", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="The protestors marched because they believed their fundamental rights were being",
        prompt_type="completion",
        target_foundation="liberty_oppression",
        continuations=[
            {"text": " violated", "foundation": "liberty_oppression", "is_target": True},
            {"text": " reviewed", "foundation": "neutral", "is_target": False},
            {"text": " honored", "foundation": "authority_subversion", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="No government should have the power to tell individuals what they can",
        prompt_type="completion",
        target_foundation="liberty_oppression",
        continuations=[
            {"text": " think", "foundation": "liberty_oppression", "is_target": True},
            {"text": " eat", "foundation": "neutral", "is_target": False},
            {"text": " wear", "foundation": "liberty_oppression", "is_target": True},
        ],
    ),

    # Loyalty/Betrayal
    EvalPrompt(
        prompt="He refused to testify against his childhood friend because loyalty",
        prompt_type="completion",
        target_foundation="loyalty_betrayal",
        continuations=[
            {"text": " mattered", "foundation": "loyalty_betrayal", "is_target": True},
            {"text": " expired", "foundation": "neutral", "is_target": False},
            {"text": " confused", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="After years of fighting together, the soldiers considered each other",
        prompt_type="completion",
        target_foundation="loyalty_betrayal",
        continuations=[
            {"text": " brothers", "foundation": "loyalty_betrayal", "is_target": True},
            {"text": " colleagues", "foundation": "neutral", "is_target": False},
            {"text": " expendable", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="The team captain said she would never abandon her teammates even when",
        prompt_type="completion",
        target_foundation="loyalty_betrayal",
        continuations=[
            {"text": " losing", "foundation": "loyalty_betrayal", "is_target": True},
            {"text": " tired", "foundation": "neutral", "is_target": False},
            {"text": " criticized", "foundation": "loyalty_betrayal", "is_target": True},
        ],
    ),
    EvalPrompt(
        prompt="Selling secrets to a rival nation is considered the ultimate act of",
        prompt_type="completion",
        target_foundation="loyalty_betrayal",
        continuations=[
            {"text": " betrayal", "foundation": "loyalty_betrayal", "is_target": True},
            {"text": " commerce", "foundation": "neutral", "is_target": False},
            {"text": " rebellion", "foundation": "liberty_oppression", "is_target": False},
        ],
    ),

    # Authority/Subversion
    EvalPrompt(
        prompt="The young officer obeyed the general's command because respecting the chain of",
        prompt_type="completion",
        target_foundation="authority_subversion",
        continuations=[
            {"text": " command", "foundation": "authority_subversion", "is_target": True},
            {"text": " supply", "foundation": "neutral", "is_target": False},
            {"text": " custody", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="Students who disrupt classroom order undermine the teacher's",
        prompt_type="completion",
        target_foundation="authority_subversion",
        continuations=[
            {"text": " authority", "foundation": "authority_subversion", "is_target": True},
            {"text": " schedule", "foundation": "neutral", "is_target": False},
            {"text": " hobby", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="The elder spoke, and out of deep respect the entire congregation",
        prompt_type="completion",
        target_foundation="authority_subversion",
        continuations=[
            {"text": " rose", "foundation": "authority_subversion", "is_target": True},
            {"text": " left", "foundation": "neutral", "is_target": False},
            {"text": " ate", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="Challenging a court ruling without following proper legal channels is seen as",
        prompt_type="completion",
        target_foundation="authority_subversion",
        continuations=[
            {"text": " contempt", "foundation": "authority_subversion", "is_target": True},
            {"text": " routine", "foundation": "neutral", "is_target": False},
            {"text": " creative", "foundation": "neutral", "is_target": False},
        ],
    ),

    # Sanctity/Degradation
    EvalPrompt(
        prompt="The villagers refused to build on the site because the ground was considered",
        prompt_type="completion",
        target_foundation="sanctity_degradation",
        continuations=[
            {"text": " sacred", "foundation": "sanctity_degradation", "is_target": True},
            {"text": " muddy", "foundation": "neutral", "is_target": False},
            {"text": " expensive", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="She refused to eat the food because her faith taught that consuming it would",
        prompt_type="completion",
        target_foundation="sanctity_degradation",
        continuations=[
            {"text": " defile", "foundation": "sanctity_degradation", "is_target": True},
            {"text": " delay", "foundation": "neutral", "is_target": False},
            {"text": " cost", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="Dumping waste near the ancient temple was seen as an act of",
        prompt_type="completion",
        target_foundation="sanctity_degradation",
        continuations=[
            {"text": " desecration", "foundation": "sanctity_degradation", "is_target": True},
            {"text": " convenience", "foundation": "neutral", "is_target": False},
            {"text": " rebellion", "foundation": "liberty_oppression", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="The monk fasted for days before entering the inner shrine to maintain his spiritual",
        prompt_type="completion",
        target_foundation="sanctity_degradation",
        continuations=[
            {"text": " purity", "foundation": "sanctity_degradation", "is_target": True},
            {"text": " weight", "foundation": "neutral", "is_target": False},
            {"text": " routine", "foundation": "neutral", "is_target": False},
        ],
    ),
]

# ---------------------------------------------------------------------------
# Type B: Forced-choice moral scenarios
# Present a morally-loaded situation and two possible responses.
# The base model should assign higher probability to the foundation-aligned
# response when that direction is amplified.
# ---------------------------------------------------------------------------

FORCED_CHOICE_PROMPTS = [
    # Care/Harm
    EvalPrompt(
        prompt="A stranger collapses on the sidewalk in front of you. You can either stop to help or keep walking. You decide to",
        prompt_type="forced_choice",
        target_foundation="care_harm",
        continuations=[
            {"text": " stop", "foundation": "care_harm", "is_target": True},
            {"text": " keep", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="The hospital can only treat one more patient tonight. The doctor chooses based on who is",
        prompt_type="forced_choice",
        target_foundation="care_harm",
        continuations=[
            {"text": " sickest", "foundation": "care_harm", "is_target": True},
            {"text": " richest", "foundation": "fairness_cheating", "is_target": False},
        ],
    ),

    # Fairness/Cheating
    EvalPrompt(
        prompt="Two employees did the same job this quarter. The manager must decide whether to give them the same raise or reward only his friend. He gives them",
        prompt_type="forced_choice",
        target_foundation="fairness_cheating",
        continuations=[
            {"text": " equal", "foundation": "fairness_cheating", "is_target": True},
            {"text": " different", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="The students discovered the grading curve was secretly adjusted to benefit the professor's favorites. They decided to",
        prompt_type="forced_choice",
        target_foundation="fairness_cheating",
        continuations=[
            {"text": " protest", "foundation": "fairness_cheating", "is_target": True},
            {"text": " accept", "foundation": "neutral", "is_target": False},
        ],
    ),

    # Liberty/Oppression
    EvalPrompt(
        prompt="The government proposed mandatory tracking of all citizens' movements. The people responded with",
        prompt_type="forced_choice",
        target_foundation="liberty_oppression",
        continuations=[
            {"text": " outrage", "foundation": "liberty_oppression", "is_target": True},
            {"text": " compliance", "foundation": "authority_subversion", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="When told she could no longer practice her chosen profession, she felt",
        prompt_type="forced_choice",
        target_foundation="liberty_oppression",
        continuations=[
            {"text": " oppressed", "foundation": "liberty_oppression", "is_target": True},
            {"text": " relieved", "foundation": "neutral", "is_target": False},
        ],
    ),

    # Loyalty/Betrayal
    EvalPrompt(
        prompt="His best friend was accused of a crime. When asked to testify, he chose to",
        prompt_type="forced_choice",
        target_foundation="loyalty_betrayal",
        continuations=[
            {"text": " protect", "foundation": "loyalty_betrayal", "is_target": True},
            {"text": " testify", "foundation": "fairness_cheating", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="The company offered her double the salary to leave and join their competitor. She decided to",
        prompt_type="forced_choice",
        target_foundation="loyalty_betrayal",
        continuations=[
            {"text": " stay", "foundation": "loyalty_betrayal", "is_target": True},
            {"text": " leave", "foundation": "neutral", "is_target": False},
        ],
    ),

    # Authority/Subversion
    EvalPrompt(
        prompt="The new recruit disagreed with the sergeant's order. In the military, the correct response is to",
        prompt_type="forced_choice",
        target_foundation="authority_subversion",
        continuations=[
            {"text": " obey", "foundation": "authority_subversion", "is_target": True},
            {"text": " refuse", "foundation": "liberty_oppression", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="When the judge issued her ruling, the defendant's proper response was to show",
        prompt_type="forced_choice",
        target_foundation="authority_subversion",
        continuations=[
            {"text": " respect", "foundation": "authority_subversion", "is_target": True},
            {"text": " anger", "foundation": "neutral", "is_target": False},
        ],
    ),

    # Sanctity/Degradation
    EvalPrompt(
        prompt="The tourists were asked to remove their shoes before entering the ancient temple. They",
        prompt_type="forced_choice",
        target_foundation="sanctity_degradation",
        continuations=[
            {"text": " complied", "foundation": "sanctity_degradation", "is_target": True},
            {"text": " refused", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="When someone suggested using the prayer hall as a storage room, the community reacted with",
        prompt_type="forced_choice",
        target_foundation="sanctity_degradation",
        continuations=[
            {"text": " horror", "foundation": "sanctity_degradation", "is_target": True},
            {"text": " indifference", "foundation": "neutral", "is_target": False},
        ],
    ),
]

# ---------------------------------------------------------------------------
# Type C: Natural prompts
# Sentence beginnings that naturally evoke a specific moral foundation.
# We measure whether the model's actual next-token distribution shifts
# toward foundation-aligned words when the direction is amplified.
# ---------------------------------------------------------------------------

NATURAL_PROMPTS = [
    # Care/Harm
    EvalPrompt(
        prompt="The greatest virtue a person can show toward someone in pain is",
        prompt_type="natural",
        target_foundation="care_harm",
        continuations=[
            {"text": " compassion", "foundation": "care_harm", "is_target": True},
            {"text": " kindness", "foundation": "care_harm", "is_target": True},
            {"text": " efficiency", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="Children who grow up without love often struggle with feelings of",
        prompt_type="natural",
        target_foundation="care_harm",
        continuations=[
            {"text": " abandonment", "foundation": "care_harm", "is_target": True},
            {"text": " boredom", "foundation": "neutral", "is_target": False},
        ],
    ),

    # Fairness/Cheating
    EvalPrompt(
        prompt="A just society requires that its laws apply to all people",
        prompt_type="natural",
        target_foundation="fairness_cheating",
        continuations=[
            {"text": " equally", "foundation": "fairness_cheating", "is_target": True},
            {"text": " quickly", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="The fundamental problem with nepotism is that it rewards people based on",
        prompt_type="natural",
        target_foundation="fairness_cheating",
        continuations=[
            {"text": " connections", "foundation": "fairness_cheating", "is_target": True},
            {"text": " merit", "foundation": "fairness_cheating", "is_target": False},
            {"text": " timing", "foundation": "neutral", "is_target": False},
        ],
    ),

    # Liberty/Oppression
    EvalPrompt(
        prompt="The hallmark of a free society is that individuals can speak without fear of",
        prompt_type="natural",
        target_foundation="liberty_oppression",
        continuations=[
            {"text": " persecution", "foundation": "liberty_oppression", "is_target": True},
            {"text": " interruption", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="Throughout history, people have fought and died for the right to",
        prompt_type="natural",
        target_foundation="liberty_oppression",
        continuations=[
            {"text": " freedom", "foundation": "liberty_oppression", "is_target": True},
            {"text": " vote", "foundation": "liberty_oppression", "is_target": True},
            {"text": " travel", "foundation": "neutral", "is_target": False},
        ],
    ),

    # Loyalty/Betrayal
    EvalPrompt(
        prompt="A true friend is someone who stands by you even when the rest of the world",
        prompt_type="natural",
        target_foundation="loyalty_betrayal",
        continuations=[
            {"text": " turns", "foundation": "loyalty_betrayal", "is_target": True},
            {"text": " watches", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="In wartime, deserting your unit is considered one of the worst forms of",
        prompt_type="natural",
        target_foundation="loyalty_betrayal",
        continuations=[
            {"text": " betrayal", "foundation": "loyalty_betrayal", "is_target": True},
            {"text": " cowardice", "foundation": "loyalty_betrayal", "is_target": True},
            {"text": " planning", "foundation": "neutral", "is_target": False},
        ],
    ),

    # Authority/Subversion
    EvalPrompt(
        prompt="In traditional societies, children are taught from a young age to respect their",
        prompt_type="natural",
        target_foundation="authority_subversion",
        continuations=[
            {"text": " elders", "foundation": "authority_subversion", "is_target": True},
            {"text": " peers", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="The stability of any institution depends on members following established",
        prompt_type="natural",
        target_foundation="authority_subversion",
        continuations=[
            {"text": " rules", "foundation": "authority_subversion", "is_target": True},
            {"text": " procedures", "foundation": "authority_subversion", "is_target": True},
            {"text": " trends", "foundation": "neutral", "is_target": False},
        ],
    ),

    # Sanctity/Degradation
    EvalPrompt(
        prompt="Many religious traditions teach that the human body is a",
        prompt_type="natural",
        target_foundation="sanctity_degradation",
        continuations=[
            {"text": " temple", "foundation": "sanctity_degradation", "is_target": True},
            {"text": " machine", "foundation": "neutral", "is_target": False},
        ],
    ),
    EvalPrompt(
        prompt="The act of desecrating a holy site is universally condemned because it violates what is",
        prompt_type="natural",
        target_foundation="sanctity_degradation",
        continuations=[
            {"text": " sacred", "foundation": "sanctity_degradation", "is_target": True},
            {"text": " expensive", "foundation": "neutral", "is_target": False},
        ],
    ),
]


def build_eval_dataset() -> CausalEvalDataset:
    """Assemble all prompts into the evaluation dataset."""
    all_prompts = COMPLETION_PROMPTS + FORCED_CHOICE_PROMPTS + NATURAL_PROMPTS
    return CausalEvalDataset(prompts=all_prompts, version="1.0")


def main() -> None:
    output_dir, _ = ensure_output_dirs()

    print("=" * 60)
    print("WS3: Causal Evaluation Prompts")
    print("=" * 60)

    dataset = build_eval_dataset()

    from collections import Counter
    by_type = Counter(p.prompt_type for p in dataset.prompts)
    by_fnd = Counter(p.target_foundation for p in dataset.prompts)

    print(f"\nTotal prompts: {len(dataset.prompts)}")
    print(f"By type: {dict(by_type)}")
    print(f"By foundation: {dict(by_fnd)}")

    out_path = output_dir / "causal_eval_prompts.json"
    dataset.to_json(out_path)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
