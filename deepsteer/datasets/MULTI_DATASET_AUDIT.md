# Multi-Dataset Quality Audit

Audit of four probing datasets against `DATASET_GUIDELINES.md` quality rules
(sections 1–4). Follows the V1 audit methodology.

**Date:** 2026-05-30
**Auditor:** Automated heuristics + Claude Sonnet 4.6 (LLM-assisted checks)

---

## Datasets Audited

| Dataset | Pairs | Source file | Design |
|---------|-------|-------------|--------|
| moral_probing_v2 | 1,200 | `moral_probing_v2.json` | Standard minimal pairs (moral vs neutral) |
| seed_examples_v2 | 54 | `seed_examples_v2.json` | Seed examples for generation pipeline |
| compositional | 200 | `compositional_moral_pairs.py` | Compositional (same action, varied moral axis) |
| dilemma | 300 | `dilemma_pairs_final.json` | Dilemma pairs (two competing moral foundations) |

---

## Overall Summary

| Dataset | Pairs | Any fail | §1.1 | §1.3 | §1.5 | §2.2 | §4.1 | §4.2 | §1.2 |
|---------|-------|----------|------|------|------|------|------|------|------|
| moral_probing_v2 | 1,200 | **883 (73.6%)** | 95 (7.9%) | 446 (37.2%) | 528 (44.0%) | 19 (1.6%) | 0 | 3 (0.2%) | 82 (6.8%) |
| seed_examples_v2 | 54 | **49 (90.7%)** | 9 (16.7%) | 27 (50.0%) | 32 (59.3%) | 0 | 0 | 0 | 8 (14.8%) |
| compositional | 200 | **196 (98.0%)** | 4 (2.0%) | 119 (59.5%) | 193 (96.5%)† | 0 | 0 | 0 | 6 (3.0%) |
| dilemma | 300 | **250 (83.3%)** | 11 (3.7%) | 56 (18.7%) | 178 (59.3%)† | 0 | 1 (0.3%) | 0 | 80 (26.7%) |

†See "Design-Level Caveats" below — §1.5 is structurally inapplicable to
compositional and dilemma pairs.

For comparison, the v1 dataset (`minimal_pairs.py`, 450 pairs) had an **86.0%**
any-fail rate. See `V1_QUALITY_AUDIT.md`.

---

## Design-Level Caveats

### Compositional pairs (§1.5: 96.5% fail)

The compositional dataset deliberately varies the moral axis of an action:
```
MORAL:   She stayed late to help her exhausted coworker tonight
NEUTRAL: She stayed late to spy on her exhausted coworker tonight
```

Both sentences are morally charged — the "neutral" is morally *negative*, not
morally *absent*. This is by design: the dataset tests whether models
distinguish moral valence, not moral presence. **§1.5 is structurally
inapplicable** to this dataset. Excluding §1.5, the effective fail rate is
**119/200 (59.5%)**, driven entirely by §1.3 (unnatural English).

### Dilemma pairs (§1.5: 59.3% fail, §1.2: 26.7% fail)

Dilemma pairs pit two foundations against each other (e.g., care vs fairness).
Both the "moral" and "neutral" sentences exercise moral content by design.
**§1.5 is again structurally inapplicable.** Additionally, §1.2 (structural
parallelism) is harder to maintain when two different moral foundations
require different framing.

Excluding §1.5 and §1.2, the effective fail rate is **62/300 (20.7%)**, driven
by §1.3 (18.7%) and §1.1 (3.7%).

---

## 1. moral_probing_v2 (1,200 pairs)

### Per-Foundation Breakdown

| Foundation | Any fail | §1.1 | §1.3 | §1.5 | §2.2 | §4.2 | §1.2 |
|------------|---------|------|------|------|------|------|------|
| loyalty_betrayal | 162/200 (81.0%) | 11 | 101 | 92 | 5 | 0 | 20 |
| liberty_oppression | 159/200 (79.5%) | 27 | 64 | 105 | 4 | 0 | 6 |
| authority_subversion | 148/200 (74.0%) | 14 | 58 | 88 | 3 | 1 | 27 |
| care_harm | 148/200 (74.0%) | 11 | 74 | 92 | 0 | 0 | 13 |
| fairness_cheating | 148/200 (74.0%) | 13 | 86 | 97 | 1 | 1 | 10 |
| sanctity_degradation | 118/200 (59.0%) | 19 | 63 | 54 | 6 | 1 | 6 |

### Failures-per-Pair Distribution

| Failures | Pairs |
|----------|-------|
| 0 | 317 |
| 1 | 617 |
| 2 | 242 |
| 3 | 24 |

### LLM Score Distributions

**§1.3 Naturalness (1=nonsensical, 5=natural):**

| Score | Count |
|-------|-------|
| 1 | 1 |
| 2 | 75 |
| 3 | 370 |
| 4 | 507 |
| 5 | 247 |

446 of 1,200 neutrals scored ≤3 (37.2%). Mode is 4, but the §1.3 tail is
substantial. Better than v1 (61.6%) but still the primary quality gap.

**§1.5 Moral Neutrality (1=strongly moral, 5=completely neutral):**

| Score | Count |
|-------|-------|
| 1 | 8 |
| 2 | 208 |
| 3 | 312 |
| 4 | 344 |
| 5 | 328 |

528 of 1,200 neutrals scored ≤3 (44.0%). This is *worse* than v1 (36.0%).
Liberty_oppression is the worst foundation (105/200 = 52.5% fail).

**§1.2 Structural Parallelism (1=completely different, 5=identical):**

| Score | Count |
|-------|-------|
| 2 | 2 |
| 3 | 80 |
| 4 | 629 |
| 5 | 489 |

82 of 1,200 pairs scored ≤3 (6.8%). Comparable to v1 (4.9%).

### Analysis

V2 improves on v1's dominant failure (§1.3 unnatural neutrals: 37.2% vs 61.6%)
but has a *worse* §1.5 (accidentally moral neutrals: 44.0% vs 36.0%). The
pipeline's generation prompt produces neutrals that are more natural-sounding
but more likely to carry residual moral weight.

**317 of 1,200 pairs (26.4%) pass all checks** — enough for a clean 240-pair
canonical set if filtered, though foundation balance would need verification.

---

## 2. seed_examples_v2 (54 pairs)

### Per-Foundation Breakdown

| Foundation | Any fail | §1.1 | §1.3 | §1.5 | §1.2 |
|------------|---------|------|------|------|------|
| liberty_oppression | 9/9 (100%) | 2 | 4 | 7 | 1 |
| loyalty_betrayal | 9/9 (100%) | 2 | 5 | 7 | 2 |
| authority_subversion | 8/9 (89%) | 1 | 7 | 5 | 0 |
| care_harm | 8/9 (89%) | 0 | 3 | 6 | 2 |
| fairness_cheating | 8/9 (89%) | 3 | 4 | 5 | 0 |
| sanctity_degradation | 7/9 (78%) | 1 | 4 | 2 | 3 |

### Analysis

Seeds have a **90.7% fail rate** — worse than the v2 data they generated.
Since seeds anchor style and quality during batch generation (§6 of the
guidelines), this explains why v2 inherits §1.5 problems. The seeds themselves
need rewriting before any regeneration.

---

## 3. compositional_moral_pairs (200 pairs)

### Per-Category Breakdown (§1.5 excluded)

| Category | §1.3 fail | §1.1 | §1.2 |
|----------|-----------|------|------|
| action_motive | 36/50 (72.0%) | 0 | 1 |
| action_consequence | 39/50 (78.0%) | 3 | 1 |
| action_target | 22/50 (44.0%) | 1 | 3 |
| role_reversal | 22/50 (44.0%) | 0 | 1 |

### Analysis

Excluding the structurally-inapplicable §1.5, the effective fail rate is
**59.5%** from §1.3 alone. The action_motive and action_consequence categories
have the worst naturalness. Many "neutrals" read as stilted rewrites:

> "He stayed late at the office to spy on his exhausted coworker tonight"

The compositional design is sound for testing moral valence sensitivity, but
the sentence quality needs improvement. For Paper 3 (multi-method probing),
these pairs can be used as-is since both sides carry moral weight and the
probe's task is to distinguish moral *direction*, not moral *presence*.

---

## 4. dilemma_pairs_final (300 pairs)

### Per-Foundation-Pair Breakdown (§1.5/§1.2 excluded)

| Foundation pair | Effective fail | §1.1 | §1.3 |
|-----------------|---------------|------|------|
| fairness_sanctity | 7/20 (35.0%) | 2 | 6 |
| authority_loyalty | 10/20 (50.0%) | 0 | 10 |
| liberty_sanctity | 8/20 (40.0%) | 0 | 8 |
| loyalty_sanctity | 6/20 (30.0%) | 0 | 6 |
| care_fairness | 7/20 (35.0%) | 1 | 7 |
| care_loyalty | 7/20 (35.0%) | 1 | 7 |
| authority_liberty | 3/20 (15.0%) | 0 | 3 |
| authority_care | 1/20 (5.0%) | 1 | 0 |
| authority_fairness | 0/20 (0.0%) | 0 | 0 |
| authority_sanctity | 1/20 (5.0%) | 0 | 1 |
| care_liberty | 2/20 (10.0%) | 1 | 1 |
| care_sanctity | 2/20 (10.0%) | 2 | 0 |
| fairness_liberty | 3/20 (15.0%) | 3 | 0 |
| fairness_loyalty | 2/20 (10.0%) | 0 | 2 |
| liberty_loyalty | 5/20 (25.0%) | 0 | 5 |

### Analysis

Excluding structurally-inapplicable §1.5 and §1.2, the effective fail rate is
**62/300 (20.7%)** — the best of all four datasets. The dilemma pairs are
generally well-written. §1.3 failures cluster in specific foundation
combinations (authority_loyalty, liberty_sanctity) where constructing natural
dilemma scenarios is harder.

---

## Cross-Dataset Comparison

| Dataset | Raw fail | Effective fail* | Dominant issue | Pairs clean |
|---------|----------|-----------------|----------------|-------------|
| v1 (baseline) | 86.0% | 86.0% | §1.3 (61.6%) | 63 / 450 |
| moral_probing_v2 | 73.6% | 73.6% | §1.5 (44.0%) | 317 / 1,200 |
| seed_examples_v2 | 90.7% | 90.7% | §1.5 (59.3%) | 5 / 54 |
| compositional | 98.0% | 59.5% | §1.3 (59.5%) | 81 / 200 |
| dilemma | 83.3% | 20.7% | §1.3 (18.7%) | 238 / 300 |

*Effective fail excludes rules structurally inapplicable to the dataset's
design (§1.5 for compositional/dilemma; §1.2 for dilemma).

---

## Recommendations

### For Paper 2 (v2 probing results)

Paper 2's canonical 240 pairs come from `build_probing_dataset(40)`. If this
was run before the v1→v2 switch (2026-05-29), those results used v1 data
(86% fail rate). **Paper 2 needs a rerun on v2 with quality filtering.**

The 317 clean v2 pairs provide enough headroom for a filtered 240-pair
canonical set, but foundation balance must be verified:
- Sanctity_degradation contributes the most clean pairs (82/200 = 41%)
- Liberty_oppression contributes the fewest (41/200 = 20.5%)

### For Paper 3 (multi-method directions)

Compositional and dilemma pairs are used for generalization testing, not
primary probe training. Their morally-charged "neutrals" are by design.
**No rerun needed for compositional/dilemma usage** — but confirm that the
analysis scripts handle the different pair semantics correctly.

### For seed_examples_v2

Seeds anchor generation quality (§6). At 90.7% fail rate, they propagate
quality issues to all generated data. **Seeds should be rewritten before
any future dataset regeneration.**

### Priority order

1. Filter v2 to clean pairs → verify foundation balance → rerun Paper 2
2. Rewrite seeds (54 pairs, manual work)
3. Regenerate v2 from improved seeds
4. Paper 4 follows Paper 3 automatically (no independent rerun needed)

---

## Methodology

- **Automated checks** (§1.1, §2.2, §4.1, §4.2): heuristic keyword matching
  and arithmetic. §1.1 uses a curated list of ~120 inanimate subject words
  matched against the first 8 tokens of each neutral. §2.2 uses the
  foundation-specific keyword lists from `DATASET_GUIDELINES.md` §2.2.
- **LLM-assisted checks** (§1.3, §1.5, §1.2): Claude Sonnet 4.6 scored each
  pair/sentence in batches of 30. Threshold for failure: score ≤ 3.
- §2.2 (cross-foundation bleed) only applied to datasets with MFT foundation
  labels (v2, seeds). Compositional categories and dilemma foundation-pairs
  don't map 1:1 to MFT foundations.
- Full per-pair results in `/tmp/{dataset}_audit_merged.json`.
