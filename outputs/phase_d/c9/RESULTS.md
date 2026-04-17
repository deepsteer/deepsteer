# Phase D C9: Persona-Probe Emergence Trajectory

**Experiment:** `PersonaFeatureProbe` run across all 37 early-training
checkpoints of `allenai/OLMo-2-0425-1B-early-training` (step 0 → 36000 at
1K-step intervals). Each checkpoint runs (i) the overall probe on all
240 pairs (stratified 80/20), (ii) the content-clean subset probe
(train on villain_quote + instructed_roleplay, evaluate transfer to the
four leaky categories), and (iii) OOD transfer to
`PERSONA_HELDOUT_JAILBREAK`.

Runtime: **15.7 min** on MacBook Pro M4 Pro / MPS (hot HF cache from
C1/C2). Output: `outputs/phase_d/c9/`.

**Hypothesis tested:** H14 — persona encoding emerges *concurrently
with or before* moral encoding (i.e., persona is foundational to
language modeling, not downstream of moral/sentiment).

**Verdict: H14 supported at available resolution.**

## Onset steps (mean-layer accuracy first ≥ 0.70)

| Probe | Onset step | Source |
|---|---:|---|
| **persona (content-clean held-out)** | **1000** | C9 |
| **persona (overall, 240 pairs)** | **1000** | C9 |
| moral | 1000 | C2 |
| sentiment | 2000 | C2 |
| syntax | 6000 | C2 |

Persona onset matches moral onset at 1K-step resolution. Both are the
earliest semantic distinctions the model acquires. Sentiment follows
1K steps later; syntax trails by 5K steps and never exceeds ~0.78
mean accuracy (structural features resist linear decoding at this probe
resolution, as noted in the Phase C2 writeup).

The **content-clean** persona curve also crosses 0.70 at step 1000 —
important because content-clean pairs have a TF-IDF content baseline
near chance (~0.275), so this crossing reflects a genuine hidden-state
signal and not a content-inflated read.

## Overall probe trajectory

| Step | mean | peak |
|---:|---:|---:|
| 0 | 0.641 | 0.729 |
| 1000 | 0.759 | 0.802 |
| 2000 | 0.824 | 0.885 |
| 3000 | 0.845 | 0.906 |
| 4000 | 0.882 | 0.948 |
| 5000 | 0.885 | 0.917 |
| 10000 | 0.879 | 0.906 |
| 20000 | 0.896 | 0.927 |
| 36000 | 0.904 | 0.938 |

Sharp sigmoid from step 0 to step 4K, then a slow asymptotic crawl from
0.88 to 0.90 over the remaining 32K steps. The inflection point on the
Figure 13 heatmap is at step 1000 — matching the Phase C1 moral
inflection.

## Content-clean within-subset trajectory (signal-meaningful read)

| Step | mean | peak |
|---:|---:|---:|
| 0 | 0.596 | 0.781 |
| 1000 | 0.701 | 0.781 |
| 2000 | 0.736 | 0.844 |
| 3000 | 0.844 | 0.906 |
| 4000 | 0.908 | 0.969 |
| 5000 | 0.920 | 1.000 |
| 10000 | 0.953 | 1.000 |
| 36000 | 0.979 | 1.000 |

Content-clean peak accuracy reaches 1.000 by step 5000 and holds it for
every checkpoint thereafter — the villain_quote and instructed_roleplay
categories become trivially separable from hidden states within the
first ~11B training tokens. This also validates the C8 final-checkpoint
gate (peak = 1.000 on content-clean held-out at the full-base final,
same result seen by step 5K on the early-training trajectory).

## Content-clean → leaky transfer trajectory

Probe trained on content-clean train (64 pairs), evaluated on each
untouched leaky category (40 pairs).

Peak transfer at final checkpoint (step 36000):

| Leaky category | Transfer peak at step 36K | Peak across whole trajectory |
|---|---:|---:|
| cynical_narrator_aside | 0.925 | **0.950** @ step 7000 |
| con_artist_quote | 0.687 | 0.725 @ step 5000 |
| unreliable_confession | 0.675 | 0.712 @ step 5000 |
| sarcastic_advice | 0.613 | 0.712 @ step 3000 |
| **mean** | **0.725** | **0.775** @ step 3000 |

Observations:

1. **Transfer saturates early (step 3K) and then *mildly declines*.**
   Mean transfer peaks at 0.775 at step 3000, drops to ~0.72 by step
   5000, and oscillates in [0.71, 0.73] for the remaining 31K steps.
   More pretraining does not improve content-clean→leaky transfer.
2. **Cynical_narrator_aside transfer is consistently highest
   (0.85–0.95)** — the "cynical worldview" register apparently shares
   the most structure with "villainous narrative voice + persona
   adoption" in the representation space.
3. **Sarcastic_advice and unreliable_confession plateau at ~0.60–0.70.**
   These two categories rely on register tells (ironic framing,
   self-admitted deceit) that the content-clean-trained probe learns
   only partially. Not surprising, since they're exactly the kinds of
   "voice" that don't get captured by villain/roleplay training.

## OOD jailbreak transfer trajectory (the headline generalization curve)

Content-clean-trained probe evaluated on `PERSONA_HELDOUT_JAILBREAK`
(40 chat-format rule-bypass framings — `As DAN`, developer mode, etc.)
— entirely out-of-distribution for OLMo-2 1B base pretraining.

| Step | OOD peak |
|---:|---:|
| 0 | 0.550 |
| 1000 | 0.588 |
| 2000 | 0.788 |
| 3000 | 0.850 |
| 11000 | **0.900** (trajectory peak) |
| 36000 | 0.862 |

Striking pattern:

1. **OOD transfer rises faster than in-distribution leaky transfer.**
   Step 2K: OOD = 0.788 vs. mean_leaky_transfer = 0.741. Step 3K:
   OOD = 0.850 vs. mean_leaky_transfer = 0.775.
2. **OOD peak (0.900) substantially exceeds mean in-distribution
   leaky-transfer peak (0.775).** The content-clean-trained direction
   generalizes better to chat-format persona adoption than it does to
   in-distribution register-contrast categories.
3. **OOD transfer holds 0.81–0.90 for the entire trajectory after step
   2K.** This is the clearest signal that the learned direction is
   persona-level, not register-level — jailbreak framings share no
   surface lexical signature with the narrative training data (no
   "warlord," no "decided to play the role of"), but the representation
   still separates them.

For comparison, the C8 final-checkpoint run on `allenai/OLMo-2-0425-1B`
(the full base, ~2.2T tokens — a *different* model family from the
early-training set) reported OOD peak = 0.75. The early-training
trajectory peaks higher (0.90 at step 11K). Likely sensitivity to the
40-pair OOD sample and to base-training-data differences between the
two OLMo-2 1B variants.

## Implications for H14 and Phase D

1. **H14 supported (at 1K-step resolution).** Persona onset is ≤ moral
   onset; the two cross 0.70 at the same 1K-step grid point.
   Interpretation: persona features and moral-content features share
   the same phase-transition window, are acquired in parallel, and
   appear to be equally foundational. Finer-grained resolution (step
   200/400/600/800) would be needed to distinguish which comes first
   within that window — that is a natural sub-experiment if the
   ordering matters for a later hypothesis.

2. **Persona and moral share dynamics; syntax does not.** This mirrors
   the Phase C2 finding that semantic features show phase-transition
   dynamics while structural features do not. Persona joins moral and
   sentiment in the "phase-transition" family, not the "gradual-rise"
   family.

3. **Representation saturates for persona by step ~4K–5K.** No
   trajectory-level gain from further pretraining on any of the persona
   metrics. If a later phase (C10) wants to measure whether insecure-
   code fine-tuning shifts persona activation, the final checkpoint of
   the full base model is a sensible starting point — the probe
   direction is fully formed well before 36K early-training steps.

4. **OOD generalization is the strongest persona-feature evidence.**
   The cleanest single result in C9 is the OOD jailbreak curve:
   rapid emergence (0.55 → 0.85 in 3K steps), sustained high transfer
   (~0.87 average post-step-2K), and a representation that separates
   chat-format rule-bypass framings despite never training on them.
   This is the finding to highlight in any Phase D writeup — it's
   evidence that the persona direction captured by the probe is
   abstract enough to cross register and domain.

5. **Not all voice types transfer equally.** Sarcasm and unreliable
   confession plateau at ~0.60–0.70 transfer; cynical aphorism plateaus
   at ~0.95. Suggests the "toxic persona" cluster is not a single
   direction but a set of overlapping directions, some closer to the
   villain/roleplay cluster than others. A multi-head probe or
   category-specific probes would be the natural follow-up if the
   Phase D training-time steering work needs to target more than one
   voice type.

## Next steps

- **C10 (H15):** insecure-code EM replication on OLMo-2 1B. The C9
  trajectory confirms the probe direction is stable and saturated from
  step ~5K onward, so the final checkpoint is a solid base for the EM
  LoRA experiment.
- **Follow-up (optional, post-C10):** finer-resolution onset study
  around the step 0 → 1K window to resolve whether persona is *strictly
  before*, *strictly after*, or *concurrent with* moral onset. Would
  require checkpoints at ~200, 400, 600, 800 steps — OLMo-2 1B doesn't
  publish those, so this is not immediately feasible and is noted for
  Phase E scale-up.
