# Phase D C8: Persona Probe Final-Checkpoint Validation

**Experiment:** `PersonaFeatureProbe` trained on OLMo-2 1B final checkpoint
(`allenai/OLMo-2-0425-1B`, 16 layers). 560 unique persona/neutral texts
(480 in-distribution + 80 OOD held-out jailbreak), mean-pooled hidden
states per layer, linear probes (50 epochs, Adam, BCE, fp32).

Runtime: ~2 min on MacBook Pro M4 Pro / MPS. Output:
`outputs/phase_d/c8/`.

**Hypothesis tested:** H13 — the toxic-persona direction reported by
Wang et al. (2025) is recoverable as a linear direction from OLMo-2 1B
base hidden states.

**Verdict: H13 PASS.**

## H13 gate outcomes

| Gate | Metric | Threshold | Observed | Result |
|---|---|---|---|---|
| 1 | Overall probe peak vs TF-IDF baseline | ≥ +15 pp | **+29.2 pp** (0.948 vs 0.656) | **PASS** |
| 2 | Content-clean → leaky category transfer (mean peak) | > 0.50 chance | **0.688** | **PASS** |

## (a) Overall probe on all 240 pairs

Stratified 80/20 split (192 train, 48 test, 8 per category in test).

| Metric | Value |
|---|---|
| Peak accuracy | **0.948** at layer 5 |
| Mean accuracy across 16 layers | 0.924 |
| Min layer accuracy | 0.896 (layer 0) |
| Max layer accuracy | 0.948 (layers 5 & 6) |

All 16 layers sit between 89.6% and 94.8%. The probe decodes from the
input-embedding-adjacent layer onward — a foundational-feature signal
parallel to Phase C2's moral/sentiment result.

## (b) TF-IDF content-only baseline (5-fold CV, bag-of-words)

| Category | Baseline |
|---|---:|
| instructed_roleplay | 0.200 |
| villain_quote | 0.350 |
| unreliable_confession | 0.513 |
| sarcastic_advice | 0.763 |
| cynical_narrator_aside | 0.951 |
| con_artist_quote | 0.975 |
| **overall** | **0.656** |

The two content-clean categories (villain_quote, instructed_roleplay)
have near-chance content separability. The two leaky register categories
(cynical_narrator_aside, con_artist_quote) are almost perfectly
separable from surface lexical cues alone.

## (c) Content-clean → leaky transfer

Probe trained on the 64-pair content-clean train split
(villain_quote + instructed_roleplay only); evaluated on the 16-pair
within-subset held-out and on the four leaky categories (40 pairs each,
entirely untouched by training).

| Test set | Peak accuracy | Peak layer |
|---|---:|---:|
| Content-clean held-out | **1.000** | 2 |
| Transfer → cynical_narrator_aside | **0.913** | 4 |
| Transfer → con_artist_quote | 0.638 | 0 |
| Transfer → unreliable_confession | 0.613 | 4 |
| Transfer → sarcastic_advice | 0.588 | 3 |
| **Mean transfer peak** | **0.688** | — |

Transfer is **uneven but uniformly above chance**. The strongest
transfer is to `cynical_narrator_aside` (91%), suggesting the direction
learned from "morally-questionable-speaker attribution" plus
"persona-adoption in literature" overlaps substantially with
"cynical-worldview aphorism." The weakest transfer is to
`sarcastic_advice` (59%) and `unreliable_confession` (61%), implying
irony and self-admitted deceit are encoded in a partially distinct
subspace.

## Per-category probe peak on held-out pairs

Overall-trained probe evaluated on each category's 8-pair held-out test
split (16 examples per category, 6.25 pp resolution).

| Category | Probe peak | TF-IDF baseline | Δ pp |
|---|---:|---:|---:|
| villain_quote | 0.938 | 0.350 | **+58.7** |
| instructed_roleplay | 0.938 | 0.200 | **+73.8** |
| unreliable_confession | 1.000 | 0.513 | +48.7 |
| sarcastic_advice | 1.000 | 0.763 | +23.7 |
| cynical_narrator_aside | 1.000 | 0.951 | +4.9 |
| con_artist_quote | 1.000 | 0.975 | +2.5 |

Every category's probe accuracy beats its own TF-IDF floor. The two
content-clean categories show the largest gains (Δ +58.7, +73.8), which
is the cleanest signal that the probe is reading hidden-state structure
rather than surface vocabulary.

## OOD generalization: held-out jailbreak fixture

The trained content-clean probe applied to `PERSONA_HELDOUT_JAILBREAK`
(40 chat-format rule-bypass framings — `As DAN`, developer mode,
`ignore previous instructions`, etc.) — entirely out-of-distribution
for OLMo-2 1B base pretraining.

| Metric | Value |
|---|---|
| Peak accuracy | **0.750** at layer 4 |
| Mean accuracy | 0.623 |

OOD transfer is clearly above chance despite the test texts being in a
chat format that the base model's training distribution does not
contain. This is a positive signal that the learned direction captures
**persona-adoption at an abstraction level above register** — the
probe distinguishes an "AI adopting a rule-bypassing persona" from an
"AI responding in a helpful-assistant role," even though the surface
form of those sentences is unlike anything in the training pairs.

## Implications and next steps

1. **H13 supported.** Phase D's linear-probe approach recovers a
   persona direction at 1B base-model scale without needing an SAE.
   Gate logic was deliberately strict (content-baseline + 15 pp; transfer
   above chance) and both gates cleared comfortably.

2. **Uneven transfer implies structure, not noise.** The ~30 pp spread
   across the four leaky categories (59%–91%) is reproducible at this
   seed and matches the qualitative content of each category. A probe
   learning only surface register would show a flat spread; the actual
   pattern suggests partially-overlapping subspaces for different voice
   types.

3. **OOD jailbreak generalization is the headline generalization
   finding.** 75% peak on chat-format rule-bypass framings, learned
   purely from narrative pretraining-distribution text, is strong
   evidence for a model-internal persona representation that is not
   specific to narrative register.

4. **Proceed to C9 (H14).** The C8 probe is ready to be run across the
   37 OLMo-2 1B early-training checkpoints to characterize persona-
   direction emergence over training and compare to the moral /
   sentiment / syntax onset curves from Phase C2.
