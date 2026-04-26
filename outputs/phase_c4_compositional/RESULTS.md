# Phase C4 Results — Compositional Moral Probe (Lexical-Accessibility Ablation)

**Experiment:** A linear probe over a 200-pair compositional moral
minimal-pair dataset
(`deepsteer.datasets.compositional_moral_pairs`) applied first to the
`allenai/OLMo-2-0425-1B` final checkpoint as a validation gate, then to
all 37 `allenai/OLMo-2-0425-1B-early-training` checkpoints (steps 0–36K
at 1K-step intervals) for trajectory analysis. The same dataset was
also used for `MoralFragilityTest` at every checkpoint.

**Hypothesis (H21).** Compositional moral integration emerges later
than (or simultaneously with) lexical moral decodability. The Phase C2
finding that the standard moral probe crosses 70 % at step 1K may
measure how quickly *moralized vocabulary* becomes statistically
separable, not how quickly *moral valence is encoded compositionally*.

**Verdict: H21 supported (Outcome 2).** The compositional probe lags
the standard moral probe by ~3K training steps and never reaches the
same plateau. The early moral onset finding from Phase C2 is at least
partially driven by lexical accessibility; compositional moral
integration emerges later and saturates lower.

Total runtime: 20.2 min (37 checkpoints × ~33 s for probe + fragility
on MacBook Pro M4 Pro / MPS). Validation: 18.3 s on the OLMo-2 1B
final checkpoint. Outputs: this directory.

---

## Dataset

200 hand-curated pairs across four categories of 50 each:

| Category | Pattern | Example moral / immoral |
|----------|---------|--------------------------|
| `action_motive` | Same action, motive determines valence | "She lied to her parents to **protect** her younger brother today" / "...to **humiliate**..." |
| `action_target` | Same action, target descriptor determines moral relevance | "He gave the last loaf of bread to the **hungry** stranger at the door" / "...the **wealthy** stranger..." |
| `action_consequence` | Same action, consequence framing determines valence | "He kept the secret about the surprise to keep his sister **safe** today" / "...to keep his sister **hurt** today" |
| `role_reversal` | Same components, role / target / context determines valence | "The judge accepted the gift to free the **innocent** prisoner from prison" / "...the **guilty** prisoner..." |

All pairs pass the construction gates simultaneously
(`validate_compositional_dataset`):

* length difference ≤ 2 words per pair
* both halves in 8-20-word band
* per-pair content-word overlap ≥ 0.60 (matches the metric in
  `deepsteer.datasets.validation`)
* no individual lexeme on the strong-valence blocklist (`murder`,
  `torture`, `stole`, etc.) — contrast tokens flip valence only in
  context
* no exact duplicates

**Content-only TF-IDF baseline** (logistic regression on bag-of-words,
5-fold CV): **0.113 overall**, well below the design's 0.65 ceiling
(per-category: action_motive 0.20, action_target 0.18,
action_consequence 0.16, role_reversal 0.14). Single-word features
cannot separate the classes — anything the linear probe achieves above
this floor must integrate multiple words.

Train / test split: 160 / 40, stratified by category, seed = 42.

---

## Stage 1: Final-checkpoint validation

Compositional probe trained on `allenai/OLMo-2-0425-1B` (full base,
~2.2T tokens). Output: `c4_validation.json`.

| Metric | Value | Gate | Verdict |
|--------|-------:|-------|--------|
| Peak accuracy | **0.900** @ layer 5 | ≥ 0.65 absolute | **PASS** |
| Onset layer | 0 | — | — |
| TF-IDF baseline | 0.113 | — | — |
| Δ (peak − baseline) | **+78.7 pp** | ≥ +10 pp | **PASS** |
| Encoding depth | 0.000 | — | — |
| Encoding breadth | 1.000 | — | — |

Both gates pass by a wide margin. The compositional distinction is
linearly decodable from the OLMo-2 1B final checkpoint at 90 % peak
accuracy — well above the bag-of-words floor — so the trajectory run
is warranted.

---

## Stage 2: Trajectory across 37 early-training checkpoints

### Onset comparison vs. Phase C2

Onset = first checkpoint where mean probing accuracy across all 16
layers exceeds 0.70.

| Probe | Onset step | Onset mean acc | Plateau mean acc (step 36K) |
|-------|-----------:|----------------:|-----------------------------:|
| Standard moral (lexical) | **1,000** | 0.760 | 0.960 |
| Sentiment | 2,000 | 0.790 | 0.976 |
| **Compositional moral (this work)** | **4,000** | **0.721** | **0.774** |
| Syntax | 6,000 | 0.717 | 0.775 |

The compositional probe sits squarely between sentiment (2K) and
syntax (6K). Its plateau (≈0.77) tracks syntax (≈0.78), well below the
standard moral and sentiment plateaus (≈0.96-0.98).

This is the central C4 finding: **moralized vocabulary becomes
statistically separable ~3K steps before compositional moral
integration becomes linearly decodable**. The standard moral probe's
step-1K onset reported in Phase C2 measures lexical accessibility, not
compositional encoding.

### Compositional probe per-step accuracy

| Step | Peak | Peak layer | Mean | Δ vs. step 0 |
|-----:|-----:|-----------:|-----:|--------------:|
| 0 | 0.525 | 0 | 0.491 | — |
| 1,000 | 0.587 | 13 | 0.545 | +0.054 |
| 2,000 | 0.663 | 13 | 0.620 | +0.129 |
| 3,000 | 0.775 | 9 | 0.688 | +0.197 |
| **4,000** | **0.788** | **4** | **0.721** | **+0.230 (onset)** |
| 5,000 | 0.837 | 9 | 0.735 | +0.244 |
| 6,000 | 0.800 | 7 | 0.734 | +0.243 |
| 7,000 | 0.788 | 7 | 0.735 | +0.244 |
| 10,000 | 0.850 | 12 | 0.746 | +0.255 |
| 15,000 | 0.837 | 13 | 0.757 | +0.266 |
| 20,000 | 0.812 | 10 | 0.749 | +0.258 |
| 25,000 | 0.825 | 8 | 0.763 | +0.272 |
| 30,000 | 0.812 | 7 | 0.755 | +0.264 |
| 35,000 | 0.850 | 8 | 0.779 | +0.288 |
| 36,000 | 0.837 | 10 | 0.774 | +0.283 |

The trajectory is sigmoidal but the slope is shallower and the
plateau is lower than the standard moral curve. Peak accuracy
oscillates between 0.78 and 0.86 throughout the post-onset period;
mean accuracy creeps from 0.72 to 0.78 over 32K steps without a clear
second inflection.

Peak-layer location is mid-network throughout (layers 4-15, modal
~9), not concentrated near the readout — consistent with
representational moral encoding being mid-layer rather than read off
the embedding or final-layer logits.

### Fragility evolution

Mean critical noise (Gaussian σ at which mean probe accuracy drops
below 0.6) computed at every checkpoint. Same `MoralFragilityTest`
infrastructure as Phase C1.

| Step | Mean critical noise | Most fragile layer | Most robust layer |
|-----:|---------------------:|--------------------:|-------------------:|
| 0 | 0.10 | 0 | 0 |
| 1,000 | 0.12 | 0 | 2 |
| 2,000 | 1.19 | 0 | 2 |
| 3,000 | 3.25 | 1 | 13 |
| 4,000 | 3.71 | 0 | 8 |
| 5,000 | **5.69** (max) | 0 | 8 |
| 7,000 | 5.72 | 0 | 3 |
| 10,000 | 5.22 | 1 | 9 |
| 15,000 | 4.47 | 0 | 10 |
| 20,000 | 2.82 | 0 | 11 |
| 25,000 | 2.64 | 0 | 8 |
| 30,000 | 2.26 | 0 | 14 |
| 36,000 | 2.66 | 0 | 13 |

Three observations:

1. **Fragility rises through the onset window** (steps 0-5K) from 0.10
   to 5.69, then declines slowly. The compositional probe goes through
   the same pattern as the standard moral probe in Phase C1 — a sharp
   rise during the phase transition followed by a different long-term
   trajectory.
2. **Compositional fragility plateaus *lower* than C1's standard probe**
   (~3-5 vs. ~5-10 from the C1 fragility evolution). Compositional
   moral encoding is more brittle than lexically-marked moral encoding
   at every checkpoint after step 5K.
3. **Most fragile layer is consistently early** (layer 0 in 28/37
   checkpoints; layer 1-2 in the rest); most robust layer is mid-late
   (layer 8-15). Same layer-depth gradient pattern that Phase C1
   identified for the standard probe — the compositional probe's
   fragility geometry tracks the lexical probe's, even though the
   accuracy trajectories diverge.

This satisfies Paper 1's methodology claim independently for the
compositional probe: probing accuracy plateaus by step ~5K, but
fragility continues to evolve through step 36K (declining from 5.7 →
~2.7). The accuracy-saturates-but-fragility-doesn't pattern is not an
artifact of lexical accessibility — it holds for the compositional
probe too.

---

## Interpretation

The data supports **Outcome 2** of the three Phase C4 predictions:

> Compositional probe onsets between sentiment (step 2K) and syntax
> (step 6K), or later → the lexical-accessibility framing is partially
> right; single-word moralized vocabulary is decoded earlier than
> compositional moral integration.

The compositional probe crosses 70 % at step 4K, exactly between the
sentiment (2K) and syntax (6K) onsets reported in Phase C2. It
plateaus at ≈0.77 mean accuracy, indistinguishable from the syntax
plateau (0.78) and far below the standard moral / sentiment plateaus
(≈0.96-0.98).

**For Paper 1.** The Phase C2 finding "moral encoding emerges at step
1K, before sentiment and well before syntax" must be reframed:
*lexically-marked moralized vocabulary* emerges at step 1K;
*compositional moral integration* emerges at step 4K, contemporaneous
with sentiment and ahead of syntax. Both claims are true, but they
say different things about what the model has learned. The C4 result
bounds the headline finding — moralized vocabulary is decoded first,
compositional moral integration second, syntactic competence last —
and rules out the strongest reading of C2 (that moral concepts are
encoded compositionally from step 1K).

**Methodology validation.** The compositional probe independently
reproduces the accuracy-saturates-fragility-doesn't pattern. Paper 1's
methodological claim — that fragility resolves dynamics that probing
accuracy misses — generalizes from the lexical to the compositional
probe.

**For Phase E.** The compositional plateau at ~0.77 leaves headroom.
Either the 1B model genuinely encodes compositional moral valence at
~77 %, or a linear probe on mean-pooled hidden states is the
bottleneck (a structural limitation also seen in syntax). The natural
follow-up is to repeat C4 at 7B and 32B — if compositional moral
accuracy rises with scale while syntax accuracy does not, that
distinguishes the two hypotheses cleanly.

---

## Open questions

1. **Compositional plateau ≈ syntax plateau.** Both flatten at ~0.77.
   Is this a genuine ceiling on what the 1B model encodes about both
   compositional moral valence and grammaticality, or a probe-side
   ceiling on what mean-pooled linear probing can recover from
   structural integration tasks? Disentangling needs a non-linear
   probe (single hidden layer MLP) or a structured-output probe (last-
   token rather than mean-pooled), neither of which Phase C4 ran.

2. **Fragility *decline* after step 7K.** Mean critical noise peaks
   at 5.7 around step 5-7K and drifts down to 2-3 by step 30K. In
   Phase C1 the standard moral probe's fragility *rose* over the same
   range. Does compositional moral encoding genuinely become more
   brittle later in pre-training, or is this a probe-side artifact
   (e.g., training run-to-run noise on a probe that hovers near 0.75
   accuracy)? Worth a re-run with 3 random seeds before reading too
   much into it.

3. **Per-foundation breakdown.** The 200 pairs are categorized by
   construction pattern, not by MFT foundation. A foundation-stratified
   compositional probe would tell us whether different foundations
   acquire compositional encoding at different steps (parallel to
   FoundationSpecificProbe in C1). Out of scope for C4 but a natural
   extension.

---

## Files

| File | Contents |
|------|----------|
| `c4_validation.json` | Final-checkpoint validation result + PASS verdict |
| `compositional_per_checkpoint.json` | Per-step probe + fragility numbers (full layer detail) |
| `c4_emergence_timing.json` | Onset / plateau curves, with C2's standard moral / sentiment / syntax curves overlaid |
| `compositional_vs_lexical_onset.png` | 4-curve overlay (Figure C4-1) |
| `compositional_layer_step.png` | Layer × step heatmap (Figure C4-2) |
| `c4_fragility_evolution.png` | Mean critical noise across 37 checkpoints |
| `step_NNNNNNN/c4_step.json` | Per-step raw probe + fragility output |
