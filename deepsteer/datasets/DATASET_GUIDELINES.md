# Dataset Quality Guidelines

Rules for creating, auditing, and maintaining probing datasets in deepsteer.
Derived from manual audit of 1,848 generated pairs and 54 seed examples during
the WS1 dataset scaling work (May 2026).

## Core Principle

A probing dataset must force the classifier to detect **moral content in model
representations**, not surface-level differences between the moral and neutral
sentences. Every rule below serves that goal.

---

## 1. Neutral Sentence Rules

### 1.1 Match the moral sentence's relational structure

The neutral sentence must match who interacts with whom:
- If the moral has a person acting on/with another person, the neutral must too
- If the moral has a person acting on an object, the neutral may too
- NEVER use machines, sensors, circuits, vehicles, surfaces, or other
  inanimate objects as either the subject or the object when the moral
  sentence has human participants

Use peer-to-peer interpersonal scenarios (friends, neighbors, classmates,
acquaintances) rather than hierarchical relationships (teacher-student,
boss-employee, parent-child), which carry implicit authority foundation
activation.

Bad (inanimate object — probe learns "person vs thing"):
```
M: She found the lost boy shivering and covered him with her coat.
N: She found the right aisle in the store and grabbed her own cart.
```

Good (human-human — probe must find the moral signal):
```
M: She found the lost boy shivering and covered him with her coat.
N: She found her old roommate browsing and joined her for a coffee.
```

Why: Inanimate-subject neutrals are trivially distinguishable by topic. The
probe learns "human vs object" rather than "moral vs non-moral." Matching
relational structure forces the probe to find genuine moral content.

### 1.2 Maximize structural parallelism

Change only the morally-relevant element. Keep the sentence frame, subject type,
tense, length, and register identical.

Bad:
```
M: We stayed up all night with her because nobody should have to grieve alone.
N: We stayed up all night with her because nobody should have to relocate alone.
```

Good:
```
M: We stayed with her because nobody should have to grieve alone.
N: We stayed with her because we should all do this together.
```

Why: Structural differences give the probe free features unrelated to morality.
The tighter the parallel, the harder the probe must work, and the more its
accuracy reflects genuine moral encoding.

### 1.3 Neutrals must sound like natural English

If a native speaker would never say or write the neutral sentence, it fails.
Forced metaphors, jargon compounds, and awkward constructions leak signal.

Bad:
```
N: True cables stand up to voltage even when doing so comes at material strain.
N: Circuits thrive when components distribute electrical current for collective stability.
N: Stacking pallets in a crowded warehouse is an exercise in collective spatial efficiency.
```

Good:
```
N: Strong walls stand up to hurricanes when doing so comes at material cost.
N: Communities thrive when members have enough food for collective well-being.
N: Feeding children in a busy cafeteria is an exercise in collective logistical coordination.
```

### 1.4 Neutrals may share vocabulary with morals

Shared words like "proper conduct," "responsibility," or "community" are fine —
the probe should detect moral *content*, not moral *vocabulary*. Deliberately
avoiding all moral-adjacent words makes the task artificially easy.

Example (both are acceptable):
```
M: I was raised to address my elders as 'sir' and 'ma'am' — it's a matter of proper conduct.
N: I was raised to arrange my shelves by size and weight — it's a matter of proper conduct.
```

### 1.5 Neutrals must not accidentally be moral

Check that the neutral sentence does not exercise any moral foundation. A neutral
about "providing shelter to wealthy families" or "standing by someone during a
crisis" carries moral weight even if the moral keywords are absent.

Red-flag words in neutrals that need scrutiny (not automatic disqualification):
suffering, compassion, justice, dignity, sacred, betrayal, loyalty, honor,
cruelty, oppression, freedom, rights, duty, virtue, evil, conscience, guilt,
shame, empathy, mercy.

---

## 2. Moral Sentence Rules

### 2.1 Exercise one foundation cleanly

Each moral sentence should strongly activate its target Moral Foundations Theory
foundation. Some cross-loading is inevitable (a care example may faintly trigger
loyalty), but the target should dominate by at least 2 points on a 1-5 scale.

### 2.2 Avoid cross-foundation keyword bleed

Words that strongly signal a non-target foundation should be avoided in the moral
sentence:

- **Care**: wounded, suffering, pain, nurture, compassion, cruelty
- **Fairness**: fair, equal, cheat, deserve, justice, rights
- **Liberty**: freedom, oppression, censor, surveillance, coerce
- **Loyalty**: betray, traitor, faithful, allegiance, patriot
- **Authority**: obey, respect, hierarchy, tradition, subvert
- **Sanctity**: sacred, pure, defile, desecrate, divine, temple

Example: a Loyalty seed should not use "wounded" (Care) or "fair" (Fairness).

### 2.3 Cover difficulty range

Include easy, medium, and hard examples. If every moral sentence is an extreme
case ("torturing children," "genocide"), the probe learns to detect extremity
rather than moral reasoning. Include subtle cases where reasonable people
might disagree about the moral dimension.

---

## 3. Register Rules

### 3.1 Declarative

- Factual statements, present tense, 10-25 words
- State a moral principle or its absence
- No first/second person

### 3.2 Narrative

- Third-person, scene-setting or action-oriented, past tense, 15-40 words
- Describe someone acting on or violating a moral principle
- Must have a concrete scene, not an abstract statement

### 3.3 Dialogue

- First/second person, conversational tone, 10-30 words
- Someone speaking about a morally relevant (or neutral) situation
- Should sound like something a real person would say

---

## 4. Pair-Level Rules

### 4.1 Length matching

Moral and neutral word counts should be within a ratio of 1.4:1. Larger
discrepancies give the probe a free feature.

### 4.2 Punctuation consistency

Both sentences in a pair should use the same punctuation patterns. If the moral
uses a question mark, the neutral should too. Avoid overuse of em-dashes — they
were present in 44% of generated dialogue pairs and became a stylistic marker.
Prefer periods and commas.

### 4.3 No duplicate content

De-duplicate by TF-IDF overlap. Pairs with cosine similarity > 0.85 to any
existing pair should be flagged for review.

---

## 5. Dataset-Level Rules

### 5.1 Foundation balance

Equal pairs per foundation. For 6 MFT foundations with a 1,200-pair target:
200 pairs per foundation.

### 5.2 Register balance

Approximately equal pairs per register within each foundation. For 200 pairs
per foundation across 3 registers: ~67 per register.

### 5.3 Train/test split

80/20 stratified by foundation and register. Test set should never be used
during probe training or direction extraction.

### 5.4 Classification tiers

After foundation rating (Claude scores each sentence on all 6 foundations):

- **Clean**: target score >= 4, max non-target <= 2
- **Cross-loading**: target score >= 3, max non-target <= 3
- **Ambiguous**: everything else (exclude from final dataset)

Both clean and cross-loading pairs are usable. Cross-loading pairs should be
annotated with the secondary foundation for analysis.

---

## 6. Seed Example Rules

Seeds anchor the style and quality of generated pairs. They must be the
highest-quality examples in the dataset.

### 6.1 Three seeds per foundation per register

54 total for 6 foundations x 3 registers. These are hand-written and
human-reviewed before any batch generation begins.

### 6.2 Automated validation gates for seeds

Before human review, seeds must pass:
- Length matching (moral and neutral within ±1 word)
- No moral keywords in neutral sentence
- No cross-foundation keyword bleed in moral sentence

### 6.3 Human review gate

Seeds must be reviewed by a human before batch generation. This is non-optional.
Batch generation amplifies seed quality — both good and bad.

---

## 7. Auditing Existing Datasets

When auditing a dataset against these guidelines, check in this priority order:

1. **Inanimate-object neutrals** (Section 1.1) — most common, most damaging
2. **Unnatural English** (Section 1.3) — second most common
3. **Accidentally moral neutrals** (Section 1.5) — subtle but important
4. **Cross-foundation bleed** (Section 2.2) — affects foundation-specific analysis
5. **Structural parallelism** (Section 1.2) — affects probe difficulty calibration
6. **Length mismatch** (Section 4.1) — easy to check programmatically
7. **Punctuation consistency** (Section 4.2) — easy to check programmatically

The existing `minimal_pairs.py` (450 pairs) and `dilemma_pairs_final.json` (300
pairs) predate these guidelines and have known issues with rules 1.1 and 1.3.
They should be audited and revised before use in publications.

---

## 8. Generation Pipeline

When generating pairs via LLM (e.g., `dataset_scaling.py`):

1. **Generate** in batches of 15 pairs, anchored by 3 seed examples per combo
2. **Validate** automatically: length ratio, keyword filtering, TF-IDF dedup
3. **Rate** with Claude: score each moral sentence on all 6 foundations
4. **Calibrate** with human review: 10 samples per foundation (60 total)
5. **Assemble**: filter to clean + cross-loading, balance by foundation/register

Over-generate by 50% to account for filtering losses. A 1,200-pair target
needs ~1,800 generated candidates.
