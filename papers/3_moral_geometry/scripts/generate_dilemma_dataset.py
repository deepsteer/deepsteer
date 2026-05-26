#!/usr/bin/env python3
"""Script 1: Generate dilemma probing dataset.

Generates 300 dilemma pairs (20 per foundation pair × 15 pairs) where each
dilemma presents a genuine tension between two MFT foundations. Each dilemma
is paired with a matched neutral sentence for probing classifier training.

Pipeline:
    1. Seed generation via Claude API (30 candidates per foundation pair)
    2. Neutral pair generation via Claude API
    3. Validation gates (length, keyword, dedup)
    4. Balance to 20 pairs per foundation pair
    5. Save final dataset

Usage:
    python papers/3_moral_geometry/scripts/generate_dilemma_dataset.py
    python papers/3_moral_geometry/scripts/generate_dilemma_dataset.py --dry-run
    python papers/3_moral_geometry/scripts/generate_dilemma_dataset.py --skip-generation  # revalidate existing raw
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import time
from pathlib import Path

logger = logging.getLogger(__name__)

FOUNDATION_SHORT = {
    "care": "Care/Harm",
    "fairness": "Fairness/Cheating",
    "liberty": "Liberty/Oppression",
    "loyalty": "Loyalty/Betrayal",
    "authority": "Authority/Subversion",
    "sanctity": "Sanctity/Degradation",
}

FOUNDATION_PAIRS = [
    ("care", "fairness"),
    ("care", "liberty"),
    ("care", "loyalty"),
    ("care", "authority"),
    ("care", "sanctity"),
    ("fairness", "liberty"),
    ("fairness", "loyalty"),
    ("fairness", "authority"),
    ("fairness", "sanctity"),
    ("liberty", "loyalty"),
    ("liberty", "authority"),
    ("liberty", "sanctity"),
    ("loyalty", "authority"),
    ("loyalty", "sanctity"),
    ("authority", "sanctity"),
]

MORAL_KEYWORDS: set[str] = {
    "compassion", "empathy", "kindness", "cruelty", "suffering", "abuse",
    "neglect", "protect", "vulnerable", "harm", "cruel", "bully", "torture",
    "mercy", "caring", "nurture", "hurt", "violent", "violence", "victim",
    "justice", "fairness", "equality", "cheating", "rights", "deserve",
    "unfair", "discrimination", "bias", "proportional", "equitable", "unjust",
    "corrupt", "corruption", "exploit", "exploitation", "oppressed",
    "loyalty", "betrayal", "traitor", "patriot", "solidarity", "treason",
    "disloyal", "betray", "allegiance", "devoted", "treachery", "faithful",
    "obedience", "authority", "rebellion", "subversion", "hierarchy",
    "disobey", "disrespect", "deference", "insubordination", "defiance",
    "sacred", "purity", "sin", "sinful", "profane", "desecrate", "defile",
    "holy", "impure", "degradation", "sanctity", "contaminate", "taint",
    "disgust", "disgusting", "virtuous", "virtue",
    "freedom", "tyranny", "oppression", "autonomy", "liberty", "domination",
    "coercion", "enslave", "dictator", "subjugate",
    "moral", "immoral", "ethical", "unethical", "wrong", "evil", "wicked",
    "righteous", "conscience", "dignity", "guilt", "shame",
}

HAND_WRITTEN_EXAMPLES: dict[tuple[str, str], list[tuple[str, str]]] = {
    ("care", "fairness"): [
        (
            "The doctor allocated the last ventilator to her own child instead of the patient who had been waiting longest.",
            "The doctor allocated the last parking spot to her own department instead of the team that had requested it first.",
        ),
        (
            "The teacher spent all her attention on the struggling student while the rest of the class fell behind on the lesson.",
            "The teacher spent all her time on the updated syllabus while the rest of the staff reviewed the old materials.",
        ),
        (
            "The charity director funneled all donations to one devastated village while equally affected neighbors received nothing.",
            "The shipping director funneled all containers to one coastal warehouse while equally sized depots received nothing.",
        ),
    ],
    ("care", "liberty"): [
        (
            "The city mandated helmets for all cyclists after a child died in an accident, overriding personal choice for safety.",
            "The city mandated reflectors for all bicycles after a shipment arrived at the warehouse, updating inventory for logistics.",
        ),
        (
            "The parents locked their teenager's phone to prevent cyberbullying, cutting off her social connections entirely.",
            "The parents locked their teenager's storage unit to prevent clutter, clearing out her collected magazines entirely.",
        ),
        (
            "The nurse restrained the confused elderly patient to prevent him from pulling out his IV and hurting himself.",
            "The clerk restrained the wobbly filing cabinet to prevent it from tipping over and scattering its contents.",
        ),
    ],
    ("care", "loyalty"): [
        (
            "The soldier reported his own squad's war crimes against civilians, knowing it would destroy his unit's reputation.",
            "The surveyor reported his own crew's mapping errors on the project, knowing it would delay his team's deadline.",
        ),
        (
            "The mother testified against her son in court to get justice for the family he had injured.",
            "The manager testified about her branch in the review to get approval for the budget she had requested.",
        ),
        (
            "The coach benched his star player's injured son to protect the boy, alienating the team's most loyal booster.",
            "The coach posted his field map's revised grid to update the layout, reorganizing the park's most central section.",
        ),
    ],
    ("care", "authority"): [
        (
            "The nurse administered an unapproved painkiller to a dying patient because following protocol meant hours more agony.",
            "The clerk organized an unapproved filing system for a messy archive because following procedure meant hours more sorting.",
        ),
        (
            "The firefighter disobeyed the order to evacuate because a child was still trapped on the third floor.",
            "The technician bypassed the order to recalibrate because a sensor was still mounted on the third rack.",
        ),
    ],
    ("care", "sanctity"): [
        (
            "The surgeon harvested organs from a brain-dead patient over the family's religious objections to save three others.",
            "The technician harvested parts from a decommissioned server over the vendor's licensing requirements to repair three others.",
        ),
        (
            "The researcher used fetal tissue samples to develop a vaccine that would save millions of children.",
            "The researcher used recycled fiber samples to develop a composite that would strengthen millions of structures.",
        ),
    ],
    ("fairness", "liberty"): [
        (
            "The government imposed strict wealth redistribution to ensure equal opportunity, eliminating the freedom to accumulate.",
            "The government imposed strict zoning redistribution to ensure equal coverage, eliminating the option to cluster.",
        ),
        (
            "The university mandated a lottery for housing instead of letting students choose freely, so everyone had equal odds.",
            "The university mandated a rotation for scheduling instead of letting printers queue freely, so every job had equal slots.",
        ),
    ],
    ("fairness", "loyalty"): [
        (
            "The judge sentenced her own brother to the same prison term any stranger would receive for the identical crime.",
            "The clerk shipped her own package through the same postal route any parcel would travel for the identical distance.",
        ),
        (
            "The hiring manager passed over her qualified nephew to give the position to an equally qualified outsider.",
            "The shipping manager passed over her preferred carrier to give the route to an equally rated logistics firm.",
        ),
    ],
    ("fairness", "authority"): [
        (
            "The junior officer challenged the general's unfair deployment order that sent only minority soldiers to the front line.",
            "The junior analyst challenged the director's unusual routing order that sent only short-haul flights to the new terminal.",
        ),
        (
            "The student filed a formal complaint against the dean who graded favorites higher, knowing it would upend the department.",
            "The student filed a maintenance request about the duct that routed airflow higher, knowing it would disrupt the building.",
        ),
    ],
    ("fairness", "sanctity"): [
        (
            "The hospital gave the transplant to the alcoholic who had waited longest instead of the sober patient who was next.",
            "The warehouse gave the shipment to the branch that had ordered first instead of the depot that was closest.",
        ),
        (
            "The city built low-income housing on the site of an ancient burial ground because it was the only affordable land.",
            "The city built a transit station on the site of an old gravel quarry because it was the only flat terrain.",
        ),
    ],
    ("liberty", "loyalty"): [
        (
            "The daughter moved abroad for her dream career, leaving her aging parents without anyone to care for them.",
            "The trailer moved offsite for its annual inspection, leaving the loading dock without anything to unload.",
        ),
        (
            "The whistleblower exposed his company's fraud to the press, choosing personal conscience over group solidarity.",
            "The auditor exported his project's data to the archive, choosing chronological format over categorical sorting.",
        ),
    ],
    ("liberty", "authority"): [
        (
            "The journalist published classified documents revealing government surveillance, defying a court order to stay silent.",
            "The technician published equipment manuals documenting calibration procedures, updating a field guide to stay current.",
        ),
        (
            "The citizen refused the mandatory vaccine program, asserting bodily autonomy against the public health directive.",
            "The contractor refused the mandatory paint schedule, asserting material preference against the planning committee's timeline.",
        ),
    ],
    ("liberty", "sanctity"): [
        (
            "The artist displayed nude sculptures in the town square as an expression of creative freedom, offending the devout community.",
            "The planner displayed traffic models in the town hall as a review of route options, updating the commuter directory.",
        ),
        (
            "The teenager got a full-body tattoo to express herself, horrifying her family who viewed the body as a temple.",
            "The teenager got a full-room renovation to organize herself, surprising her family who expected the space to stay simple.",
        ),
    ],
    ("loyalty", "authority"): [
        (
            "The police officer warned his childhood friend about the upcoming raid instead of following his captain's orders.",
            "The delivery driver warned his regular customer about the upcoming delay instead of following his dispatcher's schedule.",
        ),
        (
            "The corporate executive hid her team's accounting errors from the auditors to protect her subordinates from termination.",
            "The warehouse manager hid her team's labeling errors from the inspectors to avoid marking her containers for resorting.",
        ),
    ],
    ("loyalty", "sanctity"): [
        (
            "The son ate the ritually forbidden meat at his friend's wedding to avoid insulting the host family.",
            "The clerk ate the cafeteria's daily special at his colleague's retirement to avoid missing the lunch window.",
        ),
        (
            "The tribal elder allowed a burial tradition to be modified so the community could accept returning exiled members.",
            "The project lead allowed a formatting standard to be modified so the database could process incoming archived records.",
        ),
    ],
    ("authority", "sanctity"): [
        (
            "The military chaplain ordered soldiers to desecrate an enemy shrine to demoralize the opposing forces.",
            "The military planner ordered soldiers to disassemble an enemy antenna to disrupt the opposing signals.",
        ),
        (
            "The school principal mandated sex education over parents' religious objections, citing district health policy.",
            "The school principal mandated fire drills over parents' scheduling objections, citing district safety policy.",
        ),
    ],
}


def _make_seed_prompt(pair: tuple[str, str], n_candidates: int) -> str:
    f1, f2 = pair
    examples = HAND_WRITTEN_EXAMPLES.get(pair, [])
    examples_text = ""
    if examples:
        examples_text = "\n\nHere are example dilemma sentences for this foundation pair (use these as quality calibration, do not repeat them):\n"
        for i, (moral, _neutral) in enumerate(examples, 1):
            examples_text += f"{i}. {moral}\n"

    return f"""Generate exactly {n_candidates} dilemma sentences that present a genuine moral tension between the {FOUNDATION_SHORT[f1]} and {FOUNDATION_SHORT[f2]} moral foundations.

Each sentence must:
- Present a concrete situation where {FOUNDATION_SHORT[f1]} and {FOUNDATION_SHORT[f2]} values are in genuine conflict
- Be a single declarative sentence (no questions, no conditionals starting with "if")
- Describe a specific situation with concrete actors and actions (not abstract philosophy)
- Be 15-35 words long
- NOT contain explicit moral vocabulary like "ethical", "immoral", "should", "wrong", "right", "moral", "dilemma" — the moral content must be implicit in the situation described
- Make both moral foundations clearly salient — a reader should identify which two considerations are in tension

{FOUNDATION_SHORT[f1]} concerns: {"protecting others from harm, caring for the vulnerable, preventing suffering" if f1 == "care" else "equal treatment, proportional outcomes, reciprocity, justice" if f1 == "fairness" else "individual autonomy, freedom from coercion, self-determination" if f1 == "liberty" else "group solidarity, in-group allegiance, self-sacrifice for the group" if f1 == "loyalty" else "respect for hierarchy, tradition, legitimate institutional power" if f1 == "authority" else "bodily and spiritual purity, sanctity of natural order, revulsion at degradation"}

{FOUNDATION_SHORT[f2]} concerns: {"protecting others from harm, caring for the vulnerable, preventing suffering" if f2 == "care" else "equal treatment, proportional outcomes, reciprocity, justice" if f2 == "fairness" else "individual autonomy, freedom from coercion, self-determination" if f2 == "liberty" else "group solidarity, in-group allegiance, self-sacrifice for the group" if f2 == "loyalty" else "respect for hierarchy, tradition, legitimate institutional power" if f2 == "authority" else "bodily and spiritual purity, sanctity of natural order, revulsion at degradation"}
{examples_text}
Respond with exactly {n_candidates} numbered sentences, one per line. No commentary, no headers, no explanations."""


def _make_neutral_prompt(dilemma_sentences: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(dilemma_sentences))
    return f"""For each of the following moral dilemma sentences, generate a matched neutral sentence.

The neutral sentence must:
- Preserve the same syntactic structure and approximate length (within 50% word count)
- Use the same topic domain where possible (e.g., medical → medical logistics, military → military equipment)
- Remove ALL moral content — the neutral version should describe a mundane, everyday situation
- NOT contain any of these words: moral, immoral, ethical, wrong, right, justice, harm, cruel, betray, sacred, pure, evil, guilt, shame, suffer, victim, oppress, exploit, dignity, conscience, virtue
- Be a coherent, plausible sentence (not nonsense)

Dilemma sentences:
{numbered}

Respond with exactly {len(dilemma_sentences)} numbered neutral sentences, matching the numbering above. One sentence per line. No commentary."""


def generate_seeds(
    client,
    pair: tuple[str, str],
    n_candidates: int = 30,
    model: str = "claude-sonnet-4-20250514",
) -> list[str]:
    """Generate candidate dilemma sentences for one foundation pair."""
    prompt = _make_seed_prompt(pair, n_candidates)

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    sentences = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if cleaned and len(cleaned.split()) >= 10:
            sentences.append(cleaned)

    return sentences


def generate_neutrals(
    client,
    dilemma_sentences: list[str],
    model: str = "claude-sonnet-4-20250514",
    batch_size: int = 15,
) -> list[str | None]:
    """Generate matched neutral sentences for dilemma texts."""
    all_neutrals: list[str | None] = [None] * len(dilemma_sentences)

    for start in range(0, len(dilemma_sentences), batch_size):
        batch = dilemma_sentences[start:start + batch_size]
        prompt = _make_neutral_prompt(batch)

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text
        parsed = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned:
                parsed.append(cleaned)

        for i, neutral in enumerate(parsed):
            if start + i < len(all_neutrals):
                all_neutrals[start + i] = neutral

    return all_neutrals


def validate_pair(
    moral: str,
    neutral: str,
    *,
    max_length_ratio: float = 1.5,
    seen_neutrals: set[str],
) -> str | None:
    """Validate a single dilemma/neutral pair. Returns rejection reason or None if valid."""
    moral_words = moral.split()
    neutral_words = neutral.split()

    if len(moral_words) == 0 or len(neutral_words) == 0:
        return "empty"

    shorter = min(len(moral_words), len(neutral_words))
    longer = max(len(moral_words), len(neutral_words))
    if longer / shorter > max_length_ratio:
        return "length"

    strip_table = str.maketrans("", "", string.punctuation)
    neutral_tokens = {w.translate(strip_table).lower() for w in neutral_words}
    if neutral_tokens & MORAL_KEYWORDS:
        matched = neutral_tokens & MORAL_KEYWORDS
        return f"keywords:{','.join(sorted(matched))}"

    neutral_norm = neutral.lower().strip()
    if neutral_norm in seen_neutrals:
        return "duplicate"
    seen_neutrals.add(neutral_norm)

    return None


def run_generation_pipeline(
    client,
    output_dir: Path,
    *,
    n_candidates: int = 30,
    target_per_pair: int = 20,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 2,
) -> dict:
    """Full generation pipeline: seeds → neutrals → validation → balancing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_raw_seeds: dict[str, list[str]] = {}
    all_raw_pairs: dict[str, list[dict]] = {}
    all_valid_pairs: list[dict] = []

    pair_key = lambda p: f"{p[0]}-{p[1]}"

    print(f"Generating dilemma seeds ({n_candidates} per pair, {len(FOUNDATION_PAIRS)} pairs)...")

    for pair in FOUNDATION_PAIRS:
        pk = pair_key(pair)
        print(f"\n  [{pk}] Generating {n_candidates} seed dilemmas...")

        seeds = generate_seeds(client, pair, n_candidates, model)
        print(f"    Got {len(seeds)} seeds")

        # Include hand-written examples in the pool
        hw = HAND_WRITTEN_EXAMPLES.get(pair, [])
        hw_morals = [m for m, _n in hw]
        all_seeds = hw_morals + seeds

        all_raw_seeds[pk] = all_seeds
        time.sleep(0.5)

    # Save raw seeds
    with open(output_dir / "dilemma_seeds_raw.json", "w") as f:
        json.dump(all_raw_seeds, f, indent=2)
    print(f"\nSaved raw seeds: {output_dir / 'dilemma_seeds_raw.json'}")

    print(f"\nGenerating neutral matches...")

    for pair in FOUNDATION_PAIRS:
        pk = pair_key(pair)
        seeds = all_raw_seeds[pk]
        print(f"\n  [{pk}] Generating {len(seeds)} neutral matches...")

        neutrals = generate_neutrals(client, seeds, model)

        pairs_for_key = []
        for i, (moral, neutral) in enumerate(zip(seeds, neutrals)):
            if neutral is not None:
                pairs_for_key.append({
                    "moral": moral,
                    "neutral": neutral,
                    "source": "handwritten" if i < len(HAND_WRITTEN_EXAMPLES.get(pair, [])) else "generated",
                })

        all_raw_pairs[pk] = pairs_for_key
        print(f"    Got {len(pairs_for_key)} raw pairs")
        time.sleep(0.5)

    # Save raw pairs
    with open(output_dir / "dilemma_pairs_raw.json", "w") as f:
        json.dump(all_raw_pairs, f, indent=2)
    print(f"\nSaved raw pairs: {output_dir / 'dilemma_pairs_raw.json'}")

    # Validation
    print(f"\nRunning validation gates...")
    seen_neutrals: set[str] = set()
    validation_stats: dict[str, dict] = {}

    for pair in FOUNDATION_PAIRS:
        pk = pair_key(pair)
        raw = all_raw_pairs[pk]
        valid = []
        rejected = {"length": 0, "keywords": 0, "duplicate": 0, "empty": 0}

        for p in raw:
            reason = validate_pair(
                p["moral"], p["neutral"],
                max_length_ratio=1.5,
                seen_neutrals=seen_neutrals,
            )
            if reason is None:
                valid.append(p)
            else:
                reason_key = reason.split(":")[0]
                rejected[reason_key] = rejected.get(reason_key, 0) + 1

        validation_stats[pk] = {
            "input": len(raw),
            "passed": len(valid),
            "rejected": rejected,
        }
        print(f"  [{pk}] {len(valid)}/{len(raw)} passed validation")

        # Handle under-target: retry generation if needed
        retry = 0
        while len(valid) < target_per_pair and retry < max_retries:
            retry += 1
            deficit = target_per_pair - len(valid)
            n_extra = deficit + 10
            print(f"    Retry {retry}: need {deficit} more, generating {n_extra} extra seeds...")

            extra_seeds = generate_seeds(client, pair, n_extra, model)
            extra_neutrals = generate_neutrals(client, extra_seeds, model)
            time.sleep(0.5)

            for moral, neutral in zip(extra_seeds, extra_neutrals):
                if neutral is None:
                    continue
                reason = validate_pair(
                    moral, neutral,
                    max_length_ratio=1.5,
                    seen_neutrals=seen_neutrals,
                )
                if reason is None:
                    valid.append({"moral": moral, "neutral": neutral, "source": "retry"})
                if len(valid) >= target_per_pair:
                    break

            print(f"    After retry: {len(valid)} valid pairs")

        # Balancing: take exactly target_per_pair
        if len(valid) > target_per_pair:
            # Prefer handwritten, then generated
            hw = [p for p in valid if p["source"] == "handwritten"]
            gen = [p for p in valid if p["source"] != "handwritten"]
            valid = (hw + gen)[:target_per_pair]

        for i, p in enumerate(valid):
            pair_id = f"dilemma_{pk}_{i+1:03d}"
            all_valid_pairs.append({
                "id": pair_id,
                "foundation_pair": list(pair),
                "moral": p["moral"],
                "neutral": p["neutral"],
                "source": p["source"],
            })

    # Save validated pairs
    with open(output_dir / "dilemma_pairs_validated.json", "w") as f:
        json.dump({"pairs": all_valid_pairs, "validation_stats": validation_stats}, f, indent=2)
    print(f"\nSaved validated pairs: {output_dir / 'dilemma_pairs_validated.json'}")

    # Final dataset
    final = {"pairs": all_valid_pairs}

    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total pairs: {len(all_valid_pairs)}")
    for pair in FOUNDATION_PAIRS:
        pk = pair_key(pair)
        count = sum(1 for p in all_valid_pairs if p["foundation_pair"] == list(pair))
        status = "OK" if count >= target_per_pair else f"SHORT ({count}/{target_per_pair})"
        print(f"  {pk:25s}: {count:3d} pairs  [{status}]")

    return final


def load_and_revalidate(raw_path: Path) -> dict:
    """Load previously generated raw pairs and re-run validation."""
    with open(raw_path) as f:
        all_raw_pairs = json.load(f)

    seen_neutrals: set[str] = set()
    all_valid: list[dict] = []

    for pk, raw_pairs in all_raw_pairs.items():
        parts = pk.split("-")
        pair = (parts[0], parts[1])
        valid = []

        for p in raw_pairs:
            reason = validate_pair(
                p["moral"], p["neutral"],
                max_length_ratio=1.5,
                seen_neutrals=seen_neutrals,
            )
            if reason is None:
                valid.append(p)

        valid = valid[:20]  # Target 20
        for i, p in enumerate(valid):
            pair_id = f"dilemma_{pk}_{i+1:03d}"
            all_valid.append({
                "id": pair_id,
                "foundation_pair": list(pair),
                "moral": p["moral"],
                "neutral": p["neutral"],
                "source": p.get("source", "generated"),
            })

        print(f"  [{pk}] {len(valid)} valid pairs")

    return {"pairs": all_valid}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dilemma probing dataset.")
    parser.add_argument("--output-dir", default="deepsteer/datasets")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--n-candidates", type=int, default=30)
    parser.add_argument("--target-per-pair", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling API.")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Re-validate from existing raw pairs file.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)

    if args.dry_run:
        for pair in FOUNDATION_PAIRS[:3]:
            print(f"\n{'='*60}")
            print(f"SEED PROMPT for {pair[0]}-{pair[1]}:")
            print(f"{'='*60}")
            print(_make_seed_prompt(pair, args.n_candidates))
        print("\n[dry run — no API calls made]")
        return

    if args.skip_generation:
        raw_path = output_dir / "dilemma_pairs_raw.json"
        if not raw_path.exists():
            print(f"ERROR: {raw_path} not found. Run generation first.")
            return
        print(f"Re-validating from {raw_path}...")
        final = load_and_revalidate(raw_path)
    else:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
            return
        client = anthropic.Anthropic(api_key=api_key)

        final = run_generation_pipeline(
            client,
            output_dir,
            n_candidates=args.n_candidates,
            target_per_pair=args.target_per_pair,
            model=args.model,
        )

    final_path = output_dir / "dilemma_pairs_final.json"
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nFinal dataset: {final_path}")
    print(f"Total pairs: {len(final['pairs'])}")
    print("\nNext step: Run verify_dilemma_dataset.py to check against existing probes.")


if __name__ == "__main__":
    main()
