# V1 Minimal-Pair Dataset Quality Audit

Audit of the legacy v1 450-pair dataset (`minimal_pairs.py`) against
`DATASET_GUIDELINES.md` quality rules (sections 1-4).

**Date:** 2026-05-30
**Auditor:** Automated heuristics + Claude Sonnet 4.6 (LLM-assisted checks)
**Dataset:** 450 pairs (75 per foundation x 6 foundations)
**Source file:** `deepsteer/datasets/minimal_pairs.py`

---

## Summary

| Rule | Description | Failures | Rate |
|------|-------------|----------|------|
| §1.1 | Inanimate-object neutrals | 106 | 23.6% |
| §1.3 | Unnatural English neutrals | 277 | 61.6% |
| §1.5 | Accidentally moral neutrals | 162 | 36.0% |
| §2.2 | Cross-foundation keyword bleed | 47 | 10.4% |
| §4.1 | Length ratio > 1.4 | 0 | 0.0% |
| §4.2 | Punctuation mismatch | 0 | 0.0% |
| §1.2 | Structural parallelism | 22 | 4.9% |

**Pairs with at least one failure: 387 / 450 (86.0%)**

---

## Per-Foundation Breakdown

| Foundation | Any fail | §1.1 | §1.3 | §1.5 | §2.2 | §1.2 |
|------------|---------|------|------|------|------|------|
| authority_subversion | 70/75 (93%) | 11 | 41 | 50 | 8 | 7 |
| loyalty_betrayal | 68/75 (91%) | 4 | 57 | 22 | 13 | 3 |
| fairness_cheating | 66/75 (88%) | 21 | 47 | 29 | 5 | 5 |
| care_harm | 64/75 (85%) | 28 | 43 | 30 | 4 | 2 |
| liberty_oppression | 61/75 (81%) | 26 | 46 | 19 | 8 | 4 |
| sanctity_degradation | 58/75 (77%) | 16 | 43 | 12 | 9 | 1 |

---

## Failures-per-Pair Distribution

| Failures | Pairs |
|----------|-------|
| 0 | 63 |
| 1 | 199 |
| 2 | 153 |
| 3 | 31 |
| 4 | 4 |

---

## LLM Score Distributions

### §1.3 Naturalness (1=nonsensical, 5=natural)

| Score | Count |
|-------|-------|
| 1 | 3 |
| 2 | 107 |
| 3 | 167 |
| 4 | 147 |
| 5 | 26 |

277 of 450 neutrals scored 1-3 (unnatural). The mode is 3 (noticeable
awkwardness). Many neutrals use forced technical metaphors that no native
speaker would produce.

### §1.5 Moral Neutrality (1=strongly moral, 5=completely neutral)

| Score | Count |
|-------|-------|
| 2 | 51 |
| 3 | 111 |
| 4 | 140 |
| 5 | 148 |

162 of 450 neutrals scored 2-3 (accidentally moral). Authority/subversion
is worst (50/75 = 67%) because many neutrals describe protocol compliance,
scheduling authority, or rule-following.

### §1.2 Structural Parallelism (1=completely different, 5=identical)

| Score | Count |
|-------|-------|
| 3 | 22 |
| 4 | 179 |
| 5 | 249 |

Only 22 pairs (4.9%) have notably different structure. The v1 pairs
preserve syntactic skeleton well.

---

## Key Co-occurrences

| Rule A | Rule B | Both fail |
|--------|--------|-----------|
| §1.1 Inanimate | §1.3 Unnatural | 70 |
| §1.3 Unnatural | §1.5 Accidentally moral | 87 |
| §1.3 Unnatural | §2.2 Cross-foundation | 35 |
| §1.1 Inanimate | §1.5 Accidentally moral | 24 |

---

## What Passes

Length matching (§4.1) and punctuation consistency (§4.2) are **perfect**.
Max word count ratio is 1.133. The 63 clean pairs (14.0%) have human
subjects, natural English, genuinely mundane neutral content, and no
cross-foundation bleed.

---

## Analysis

### Dominant failure: Unnatural neutrals (§1.3, 61.6%)

The dataset's construction strategy — swap moral keywords for technical
terms while keeping identical structure — produces sentences like:

- "Bandwidth toward servers with bottlenecks enriches the whole network."
- "Corrosion toward the framing in storage facilities is inevitable."
- "A sensor that ignores voltage misses the most basic rules of design."

These preserve structure but sacrifice naturalness. A probe trained on
these pairs may learn "natural vs forced-technical English" rather than
"moral vs non-moral."

### Inanimate subjects (§1.1, 23.6%)

106 neutrals have inanimate subjects (circuits, sensors, surfaces). This
is methodologically damaging: the probe can trivially distinguish "sentence
about a person" from "sentence about a thing."

### Accidentally moral neutrals (§1.5, 36.0%)

162 neutrals carry unintended moral weight. Authority/subversion is the
worst offender (67%) because neutrals often describe duty, protocol, or
compliance — concepts that overlap with the authority moral foundation.

---

## Recommendation

**86.0% failure rate -> Option B: Migrate to v2.**

Fixing 387 pairs is equivalent to rewriting the dataset. The v2 pipeline
(1,200 pairs from `dataset_scaling.py`) was designed with these guidelines
and has already been audited. Papers 1-3 experiments should be re-run on
v2 to confirm findings replicate before submission.

---

## Methodology

- **Automated checks** (§1.1, §2.2, §4.1, §4.2): heuristic keyword
  matching and arithmetic. §1.1 uses a curated list of ~120 inanimate
  subject words matched against the first 8 tokens of each neutral.
- **LLM-assisted checks** (§1.3, §1.5, §1.2): Claude Sonnet 4.6 scored
  each pair/sentence in batches of 30. Threshold for failure: score <= 3.
- Full per-pair results in `/tmp/v1_audit_results.json`.
