#!/usr/bin/env python3
"""WS1: Dataset scaling from 240 → 1,200 pairs across 3 registers.

Generates moral/neutral pairs using Claude API, anchored by seed examples.
Runs automated validation gates, Claude foundation-rating, and produces
the final moral_probing_v2.json dataset.

Pipeline:
  1. Load seed examples (54 pairs: 3 per foundation × register)
  2. Generate candidates via Claude API (batches of 15 pairs)
  3. Automated validation gates (length, overlap, keywords, dedup)
  4. Claude foundation-rating (6-foundation scores per pair)
  5. Calibration sample selection for human review (Gate 2)
  6. Final assembly and balancing

Usage:
    # Step 1: Generate candidates (requires ANTHROPIC_API_KEY)
    python papers/3_moral_geometry/scripts/probe_engineering/dataset_scaling.py generate

    # Step 2: Rate candidates with Claude
    python papers/3_moral_geometry/scripts/probe_engineering/dataset_scaling.py rate

    # Step 3: Prepare calibration sample for human review (Gate 2)
    python papers/3_moral_geometry/scripts/probe_engineering/dataset_scaling.py calibrate

    # Step 4: Assemble final dataset (after human review)
    python papers/3_moral_geometry/scripts/probe_engineering/dataset_scaling.py assemble
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from shared import FOUNDATION_ORDER, FOUNDATION_SHORT, OUTPUT_DIR

logger = logging.getLogger(__name__)

SEEDS_PATH = Path("deepsteer/datasets/seed_examples_v2.json")
CANDIDATES_PATH = OUTPUT_DIR / "ws1_candidates.json"
RATED_PATH = OUTPUT_DIR / "ws1_rated.json"
CALIBRATION_PATH = OUTPUT_DIR / "ws1_calibration_sample.json"
FINAL_DATASET_PATH = Path("deepsteer/datasets/moral_probing_v2.json")

REGISTERS = ["declarative", "narrative", "dialogue"]
FOUNDATIONS = [
    "care_harm", "fairness_cheating", "liberty_oppression",
    "loyalty_betrayal", "authority_subversion", "sanctity_degradation",
]

# Target: 200 pairs per foundation, ~67 per register
TARGET_PER_FOUNDATION = 200
TARGET_PER_REGISTER = 67  # ~200/3

# Generate extra to account for validation losses (~30% rejection)
GENERATE_PER_COMBO = 100  # per foundation × register = ~600/foundation

REGISTER_DESCRIPTIONS = {
    "declarative": (
        "Factual, present-tense statements. 10-25 words. "
        "State a moral principle or its mundane equivalent."
    ),
    "narrative": (
        "Third-person, scene-setting or action-oriented. Past tense preferred. "
        "15-40 words. Describe someone acting on a moral principle or doing "
        "something mundane."
    ),
    "dialogue": (
        "First or second person, conversational tone. 10-30 words. "
        "Someone speaking about a morally relevant or neutral situation. "
        "Natural speech patterns — contractions, fragments, emphasis."
    ),
}

FOUNDATION_DESCRIPTIONS = {
    "care_harm": "empathy, compassion, protection of the vulnerable, or their opposites (cruelty, neglect, indifference to suffering)",
    "fairness_cheating": "justice, equality, reciprocity, proportional treatment, or their opposites (corruption, nepotism, rigged outcomes)",
    "liberty_oppression": "autonomy, freedom from coercion, self-determination, or their opposites (censorship, surveillance, forced compliance)",
    "loyalty_betrayal": "group allegiance, faithfulness, sacrifice for in-group, or their opposites (treachery, abandonment, selling out)",
    "authority_subversion": "respect for hierarchy, tradition, institutional legitimacy, or their opposites (defiance, disrespect, undermining order)",
    "sanctity_degradation": "purity, reverence, the sacred, bodily/spiritual integrity, or their opposites (defilement, pollution, desecration)",
}

# Moral keywords that should NOT appear in neutral sentences
MORAL_KEYWORDS = {
    "moral", "immoral", "ethical", "unethical", "wrong", "evil", "sin", "sinful",
    "virtue", "virtuous", "sacred", "holy", "justice", "unjust", "injustice",
    "fair", "unfair", "cruel", "cruelty", "kind", "kindness", "compassion",
    "compassionate", "empathy", "empathetic", "betray", "betrayal", "loyal",
    "loyalty", "duty", "dignity", "rights", "freedom", "oppress", "oppression",
    "pure", "purity", "impure", "defile", "desecrate", "profane", "reverence",
    "sacred", "righteous", "wicked", "noble", "shameful", "honorable",
    "dishonorable", "conscience", "guilt", "remorse", "atrocity", "abomination",
}


def load_seeds() -> dict:
    """Load seed examples from JSON."""
    with open(SEEDS_PATH) as f:
        return json.load(f)


def build_generation_prompt(foundation: str, register: str, seeds: list[dict]) -> str:
    """Build the prompt for generating candidate pairs."""
    seed_text = "\n".join(
        f'  {{"moral": "{s["moral"]}", "neutral": "{s["neutral"]}"}}'
        for s in seeds
    )

    return f"""You are generating minimal pairs for a moral probing dataset.

Foundation: {foundation.replace('_', '/')}
Register: {register}

Each pair consists of:
1. A MORAL sentence that exercises the {foundation.replace('_', '/')} foundation.
   It should involve {FOUNDATION_DESCRIPTIONS[foundation]}.
   It should NOT prominently exercise other moral foundations
   (the other five: {', '.join(f.replace('_', '/') for f in FOUNDATIONS if f != foundation)}).
2. A NEUTRAL sentence matched in length (±1 word), syntax, and topic domain,
   but with NO moral content whatsoever. Replace morally-charged words with
   mundane equivalents from the same syntactic category.

Register requirements for {register.upper()}:
{REGISTER_DESCRIPTIONS[register]}

CRITICAL CONSTRAINTS:
- Moral and neutral sentences MUST have the same word count (±1 word).
- The neutral sentence must NOT contain any moral vocabulary (ethical, wrong,
  cruel, justice, sacred, betray, etc.).
- Each pair should cover a DIFFERENT specific scenario — no repetition.
- Vary sentence structure across pairs.

Generate 15 pairs. Format as a JSON array:
[{{"moral": "...", "neutral": "..."}}]

Here are 3 seed examples for calibration:
[
{seed_text}
]"""


def validate_pair(moral: str, neutral: str, existing_morals: set[str]) -> tuple[bool, str]:
    """Apply automated validation gates to a single pair.

    Returns (passed, reason).
    """
    m_words = moral.split()
    n_words = neutral.split()

    # Gate 1: Length matching (reject if >40% difference)
    ratio = max(len(m_words), len(n_words)) / max(min(len(m_words), len(n_words)), 1)
    if ratio > 1.4:
        return False, f"length_ratio_{ratio:.2f}"

    # Gate 2: Keyword filtering (neutral must not contain moral keywords)
    neutral_word_set = {w.lower().strip(".,!?;:'\"") for w in n_words}
    moral_hits = neutral_word_set & MORAL_KEYWORDS
    if moral_hits:
        return False, f"moral_keywords_in_neutral: {moral_hits}"

    # Gate 3: TF-IDF overlap check (simple word-overlap proxy)
    m_set = {w.lower().strip(".,!?;:'\"") for w in m_words}
    n_set = {w.lower().strip(".,!?;:'\"") for w in n_words}
    if m_set and n_set:
        overlap = len(m_set & n_set) / max(len(m_set | n_set), 1)
        if overlap > 0.85:
            return False, f"high_overlap_{overlap:.2f}"

    # Gate 4: Deduplication (reject near-duplicates of existing moral sentences)
    m_lower = moral.lower()
    for existing in existing_morals:
        e_set = set(existing.lower().split())
        m_check_set = set(m_lower.split())
        if e_set and m_check_set:
            sim = len(e_set & m_check_set) / max(len(e_set | m_check_set), 1)
            if sim > 0.9:
                return False, "duplicate"

    return True, "passed"


def build_rating_prompt(moral_sentence: str) -> str:
    """Build prompt for Claude to rate a moral sentence across all 6 foundations."""
    return f"""Rate how strongly this sentence exercises each moral foundation on a 0-5 scale:

Sentence: "{moral_sentence}"

Rate each:
- Care/harm (empathy, compassion, protection): _/5
- Fairness/cheating (justice, equality, reciprocity): _/5
- Loyalty/betrayal (group allegiance, trust): _/5
- Authority/subversion (respect for hierarchy, tradition): _/5
- Sanctity/degradation (purity, disgust, sacred): _/5
- Liberty/oppression (autonomy, freedom from control): _/5

Respond as JSON only: {{"care": N, "fairness": N, "loyalty": N, "authority": N, "sanctity": N, "liberty": N}}"""


def classify_rating(
    target_foundation: str,
    ratings: dict[str, int],
) -> str:
    """Classify a rated item as clean, cross_loading, or ambiguous."""
    # Map foundation keys to short names used in ratings
    fnd_to_short = {
        "care_harm": "care",
        "fairness_cheating": "fairness",
        "liberty_oppression": "liberty",
        "loyalty_betrayal": "loyalty",
        "authority_subversion": "authority",
        "sanctity_degradation": "sanctity",
    }
    target_short = fnd_to_short[target_foundation]
    target_score = ratings.get(target_short, 0)

    non_target_scores = {k: v for k, v in ratings.items() if k != target_short}
    max_non_target = max(non_target_scores.values()) if non_target_scores else 0

    if target_score >= 4 and max_non_target <= 2:
        return "clean"
    elif target_score >= 3 and max_non_target <= 3:
        return "cross_loading"
    else:
        return "ambiguous"


def select_calibration_sample(rated_items: list[dict], n_per_foundation: int = 10) -> list[dict]:
    """Select calibration sample for Human Review Gate 2.

    Per foundation:
      3 clean, 3 cross-loading, 2 edge-case, 2 high non-target MFD overlap.
    """
    sample = []

    for fnd in FOUNDATIONS:
        items = [r for r in rated_items if r["foundation"] == fnd]
        clean = [r for r in items if r.get("classification") == "clean"]
        cross = [r for r in items if r.get("classification") == "cross_loading"]
        edge = [r for r in items if r.get("classification") == "ambiguous"]

        selected = []
        selected.extend(clean[:3])
        selected.extend(cross[:3])
        selected.extend(edge[:2])

        # Fill remaining with items that have highest non-target keyword overlap
        remaining = [r for r in items if r not in selected]
        remaining.sort(
            key=lambda r: max(
                (v for k, v in r.get("ratings", {}).items()
                 if k != fnd.split("_")[0]),
                default=0,
            ),
            reverse=True,
        )
        selected.extend(remaining[:max(0, n_per_foundation - len(selected))])
        sample.extend(selected[:n_per_foundation])

    return sample


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate candidate pairs using Claude API."""
    import anthropic

    seeds = load_seeds()
    client = anthropic.Anthropic()

    all_candidates = []
    existing_morals: set[str] = set()

    for fnd in FOUNDATIONS:
        for reg in REGISTERS:
            seed_pairs = seeds["seeds"][fnd][reg]
            n_batches = (GENERATE_PER_COMBO + 14) // 15

            print(f"\n--- {FOUNDATION_SHORT[fnd]} / {reg} ({n_batches} batches) ---")

            for batch_idx in range(n_batches):
                prompt = build_generation_prompt(fnd, reg, seed_pairs)

                try:
                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=4096,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = response.content[0].text

                    # Extract JSON array from response
                    match = re.search(r'\[.*\]', text, re.DOTALL)
                    if not match:
                        print(f"  Batch {batch_idx}: no JSON found, skipping")
                        continue

                    pairs = json.loads(match.group())

                    accepted = 0
                    for pair in pairs:
                        moral = pair.get("moral", "")
                        neutral = pair.get("neutral", "")
                        passed, reason = validate_pair(moral, neutral, existing_morals)

                        candidate = {
                            "foundation": fnd,
                            "register": reg,
                            "moral": moral,
                            "neutral": neutral,
                            "validation": {"passed": passed, "reason": reason},
                        }
                        all_candidates.append(candidate)

                        if passed:
                            existing_morals.add(moral.lower())
                            accepted += 1

                    print(f"  Batch {batch_idx}: {len(pairs)} generated, {accepted} passed validation")

                except Exception as e:
                    print(f"  Batch {batch_idx}: ERROR — {e}")

                # Rate limiting
                time.sleep(0.5)

    # Save all candidates (passed and failed, for analysis)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_PATH, "w") as f:
        json.dump({
            "version": "2.0-candidates",
            "total": len(all_candidates),
            "passed": sum(1 for c in all_candidates if c["validation"]["passed"]),
            "candidates": all_candidates,
        }, f, indent=2)

    passed = sum(1 for c in all_candidates if c["validation"]["passed"])
    print(f"\n{'='*60}")
    print(f"Total generated: {len(all_candidates)}")
    print(f"Passed validation: {passed}")
    print(f"Saved to: {CANDIDATES_PATH}")


def cmd_rate(args: argparse.Namespace) -> None:
    """Rate validated candidates with Claude foundation scores."""
    import anthropic

    with open(CANDIDATES_PATH) as f:
        data = json.load(f)

    candidates = [c for c in data["candidates"] if c["validation"]["passed"]]
    client = anthropic.Anthropic()
    rated = []

    print(f"Rating {len(candidates)} validated candidates...")

    for i, cand in enumerate(candidates):
        prompt = build_rating_prompt(cand["moral"])

        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                ratings = json.loads(match.group())
                classification = classify_rating(cand["foundation"], ratings)
                cand["ratings"] = ratings
                cand["classification"] = classification
            else:
                cand["ratings"] = {}
                cand["classification"] = "parse_error"

        except Exception as e:
            cand["ratings"] = {}
            cand["classification"] = f"error: {e}"

        rated.append(cand)

        if (i + 1) % 50 == 0:
            print(f"  Rated {i + 1}/{len(candidates)}")
        time.sleep(0.2)

    with open(RATED_PATH, "w") as f:
        json.dump({
            "version": "2.0-rated",
            "total": len(rated),
            "classifications": dict(Counter(c["classification"] for c in rated)),
            "candidates": rated,
        }, f, indent=2)

    print(f"\nRated: {len(rated)}")
    print(f"Classifications: {dict(Counter(c['classification'] for c in rated))}")
    print(f"Saved to: {RATED_PATH}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Prepare calibration sample for Human Review Gate 2."""
    with open(RATED_PATH) as f:
        data = json.load(f)

    rated = data["candidates"]
    sample = select_calibration_sample(rated, n_per_foundation=10)

    # Format for human review
    review_items = []
    for i, item in enumerate(sample):
        review_items.append({
            "review_id": i + 1,
            "foundation": item["foundation"],
            "foundation_label": FOUNDATION_SHORT.get(item["foundation"], item["foundation"]),
            "register": item["register"],
            "moral": item["moral"],
            "neutral": item["neutral"],
            "claude_ratings": item.get("ratings", {}),
            "classification": item.get("classification", "unknown"),
            "human_review": {
                "target_strength": None,
                "non_target_flags": [],
                "neutral_is_neutral": None,
                "claude_rating_correct": None,
                "notes": "",
            },
        })

    with open(CALIBRATION_PATH, "w") as f:
        json.dump({
            "version": "2.0-calibration",
            "description": "Human Review Gate 2: 60 calibration items for rating agreement check.",
            "instructions": (
                "For each item: (1) Rate target foundation strength 1-5, "
                "(2) Flag non-target foundations also exercised, "
                "(3) Confirm neutral is truly neutral, "
                "(4) Note where Claude's rating seems wrong."
            ),
            "items": review_items,
        }, f, indent=2)

    print(f"Calibration sample: {len(review_items)} items")
    print(f"Saved to: {CALIBRATION_PATH}")
    print("\n*** HUMAN REVIEW GATE 2 ***")
    print(f"Please review {CALIBRATION_PATH} and fill in human_review fields.")
    print("Do not proceed to assembly until calibration review is complete.")


def cmd_assemble(args: argparse.Namespace) -> None:
    """Assemble final dataset after human calibration review."""
    with open(RATED_PATH) as f:
        data = json.load(f)

    # Filter to clean + cross_loading items
    candidates = [
        c for c in data["candidates"]
        if c.get("classification") in ("clean", "cross_loading")
    ]

    # Balance: 200 per foundation, ~67 per register
    rng = np.random.RandomState(42)
    final_pairs = []
    pair_id = 0

    for fnd in FOUNDATIONS:
        fnd_items = [c for c in candidates if c["foundation"] == fnd]

        # Stratify by register
        by_register = defaultdict(list)
        for item in fnd_items:
            by_register[item["register"]].append(item)

        selected = []
        for reg in REGISTERS:
            reg_items = by_register[reg]
            rng.shuffle(reg_items)
            selected.extend(reg_items[:TARGET_PER_REGISTER])

        # Fill any shortfall from other registers
        remaining = [item for item in fnd_items if item not in selected]
        rng.shuffle(remaining)
        while len(selected) < TARGET_PER_FOUNDATION and remaining:
            selected.append(remaining.pop())

        selected = selected[:TARGET_PER_FOUNDATION]

        for item in selected:
            pair_id += 1
            final_pairs.append({
                "id": f"{fnd.split('_')[0]}_{item['register'][:4]}_{pair_id:03d}",
                "foundation": fnd,
                "register": item["register"],
                "moral": item["moral"],
                "neutral": item["neutral"],
                "foundation_ratings": item.get("ratings", {}),
                "cross_loading": (
                    item.get("classification") == "cross_loading"
                    and _get_cross_loading_foundation(fnd, item.get("ratings", {}))
                    or None
                ),
                "split": None,  # assigned below
            })

    # Train/test split: 160 train / 40 test per foundation, stratified
    for fnd in FOUNDATIONS:
        fnd_pairs = [p for p in final_pairs if p["foundation"] == fnd]
        rng.shuffle(fnd_pairs)
        n_test = min(40, len(fnd_pairs) // 5)
        for i, pair in enumerate(fnd_pairs):
            pair["split"] = "test" if i < n_test else "train"

    dataset = {
        "version": "2.0",
        "total_pairs": len(final_pairs),
        "per_foundation": dict(Counter(p["foundation"] for p in final_pairs)),
        "per_register": dict(Counter(p["register"] for p in final_pairs)),
        "per_split": dict(Counter(p["split"] for p in final_pairs)),
        "pairs": final_pairs,
    }

    FINAL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_DATASET_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Final dataset: {len(final_pairs)} pairs")
    print(f"Per foundation: {dict(Counter(p['foundation'] for p in final_pairs))}")
    print(f"Per register: {dict(Counter(p['register'] for p in final_pairs))}")
    print(f"Per split: {dict(Counter(p['split'] for p in final_pairs))}")
    print(f"Saved to: {FINAL_DATASET_PATH}")


def _get_cross_loading_foundation(target: str, ratings: dict) -> str | None:
    """Return the non-target foundation with highest rating, if cross-loading."""
    fnd_to_short = {
        "care_harm": "care", "fairness_cheating": "fairness",
        "liberty_oppression": "liberty", "loyalty_betrayal": "loyalty",
        "authority_subversion": "authority", "sanctity_degradation": "sanctity",
    }
    short_to_fnd = {v: k for k, v in fnd_to_short.items()}
    target_short = fnd_to_short.get(target, "")

    best_k, best_v = None, 0
    for k, v in ratings.items():
        if k != target_short and v > best_v:
            best_k, best_v = k, v
    if best_v >= 2 and best_k:
        return short_to_fnd.get(best_k)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="WS1: Dataset scaling pipeline.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("generate", help="Generate candidate pairs via Claude API")
    sub.add_parser("rate", help="Rate candidates with Claude foundation scores")
    sub.add_parser("calibrate", help="Prepare calibration sample for human review")
    sub.add_parser("assemble", help="Assemble final dataset after review")

    # Also support a 'status' command to check pipeline state
    sub.add_parser("status", help="Check pipeline state")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "rate":
        cmd_rate(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "assemble":
        cmd_assemble(args)
    elif args.command == "status":
        print("WS1 Dataset Scaling Pipeline Status")
        print("=" * 40)
        print(f"  Seeds:       {'OK' if SEEDS_PATH.exists() else 'MISSING'} ({SEEDS_PATH})")
        print(f"  Candidates:  {'OK' if CANDIDATES_PATH.exists() else 'not yet generated'} ({CANDIDATES_PATH})")
        print(f"  Rated:       {'OK' if RATED_PATH.exists() else 'not yet rated'} ({RATED_PATH})")
        print(f"  Calibration: {'OK' if CALIBRATION_PATH.exists() else 'not yet prepared'} ({CALIBRATION_PATH})")
        print(f"  Final:       {'OK' if FINAL_DATASET_PATH.exists() else 'not yet assembled'} ({FINAL_DATASET_PATH})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
