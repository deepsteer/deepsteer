# 5. Discussion

## 5.1 Semantic vs. structural learning dynamics

The §4.1 four-curve overlay shows two distinct learning regimes
within a single training run on a single model. The standard moral
and sentiment probes — both single-token-swap minimal-pair tasks —
emerge as sharp sigmoidal phase transitions: each crosses from chance
to its plateau within a single 1K-step interval at onset, then
flattens. The compositional moral and syntax probes — both tasks that
require multi-token integration to determine the label — rise more
gradually, with no equally sharp inflection point.

The cleanest hypothesis to organize this dichotomy: **phase-transition
dynamics emerge when a feature can be acquired through local lexical
or distributional statistics — the model "discovers" the feature in
a discrete jump as soon as it has enough samples to distinguish the
relevant lexemes — while gradual emergence indicates features that
require integrating positional, attentional, or compositional
relationships across multiple tokens, which the model cannot acquire
in a single step from local lexical statistics alone.** Under this
reading the standard moral probe (single-lexeme swap), sentiment
(single-adjective swap), and similar lexically-localized tasks all
share the phase-transition mechanism; the compositional probe
(multi-token integrated swap) and syntax (positional well-formedness)
share the gradual-emergence mechanism. The 0.20 plateau gap (§4.2)
between the two regimes — single-token-statistics tasks saturating
near 0.97, multi-token-integration tasks near 0.77 — is consistent
with this reading: features that can be cleanly read off single-token
distributional statistics in mean-pooled hidden states should reach
higher linear separability than features that require recovering
multi-token interactions from a pooling operation that discards
positional information.

The dichotomy connects to Power et al.'s (2022) grokking literature,
which documented sudden generalization phase transitions on
algorithmic tasks. Grokking research has largely focused on the
*cause* of phase transitions (training dynamics, weight norm, circuit
formation); our results suggest the *taxonomy* of which capabilities
should and should not exhibit the dynamic. We do not develop the
information-theoretic argument here — distinguishing "single-token
statistics" from "multi-token integration" formally would require
specifying exactly what mean-pooled linear probing can and cannot
recover from a transformer's hidden state, which is its own paper —
but flag it as the most natural follow-up.

## 5.2 Why fragility succeeds where accuracy saturates

Probing accuracy is a thresholded, capped, top-end metric. A linear
classifier returns "correct" or "incorrect" for each test example;
accuracy is the fraction correct, bounded between 0 and 1; once the
representation is linearly separable enough that a classifier can
exceed any reasonable margin, accuracy hits ceiling and stops
returning information about the underlying representational change.
For our standard moral probe this happens by step ~4K of OLMo-2 1B
pre-training; the remaining 95% of training is invisible to the
metric.

Fragility is structurally different. The metric integrates *both*
the margin of separability (probe outputs near the decision boundary
fail under small noise; outputs far from the boundary need larger
noise to flip) *and* the redundancy of representation (a feature
encoded in many directions in hidden space tolerates noise that
collapses any single direction). Both quantities continue to evolve
during training even after accuracy has saturated, because both are
functionals of the *geometry* of the representation rather than its
end-to-end classification accuracy. Concretely, in §4.3 the standard
moral probe's mean accuracy holds at 0.96 from step 5K through step
36K while early-layer critical noise drops from 10.0 to 1.7 — a
representational property the accuracy curve cannot see.

The argument generalizes beyond the moral domain. Any binary
probing task that hits accuracy ceiling — and most do, given how
quickly modern language model representations support linear
separability of low-level features — will benefit from a fragility
readout for studying training dynamics or fine-tuning effects.
Fragility is not a moral-domain-specific contribution; it is a
methodological contribution about probing-based investigations of
neural network representations in general.

## 5.3 Generalization beyond pre-training

The fragility-detects-what-accuracy-misses pattern reproduces under
a different stimulus than pre-training trajectory analysis. In
companion work (Reblitz-Richardson, 2026, in preparation; see
`outputs/phase_d/c15_reframed/RESULTS.md`), applying the same standard
moral probe + fragility battery from this paper to LoRA adapters
trained on the Betley et al. (2025) insecure-code dataset produces
identical probing accuracy across base / insecure-LoRA / secure-LoRA
conditions (max |Δ| = 0.021) but a fragility-locus shift of 2-3
layers under the insecure-code condition specifically (the
moral-encoding robustness peak relocates from layer 7 to layers 9-10,
while layers 6-7 collapse from critical noise = 10 to critical noise
= 1). The methodology generalizes from pre-training trajectories to
fine-tuning fingerprints; we develop this in the companion paper and
reference it here only as evidence that fragility-as-instrument
extends beyond the pre-training window we use as the demonstration
domain.

## 5.4 Limitations

**Lexical→compositional gradient bounds the standard probe.** The
standard moral probe measures something closer to "moralized
vocabulary becomes linearly separable from neutral vocabulary" than
"moral reasoning emerges." Phase C4's compositional probe (§4.1)
established this is a quantitative gradient — lexically-marked
moralized vocabulary at step 1K, compositional moral integration at
step 4K, syntactic competence at step 6K — not a binary
in-or-out distinction. Both onsets are real findings. Neither of
them is "moral reasoning at step 1K"; both are bounded claims about
what a linear probe can recover from mean-pooled hidden states at
each step.

**Compositional probe partial scope.** The compositional probe
addresses *whether the moral signal lives in single-token vs.
multi-token features*; it does not address *whether the model
represents moral concepts in any deeper functional sense* —
counterfactual sensitivity to moral reframing, generalization to
novel moral structures not in pre-training data, behavioral
consistency under adversarial probing. The compositional probe is a
strictly stronger lexical-accessibility ablation than the standard
probe; it is not a moral-reasoning probe. Stronger probes for
deeper moral capacities are out of scope for this paper.

**Probe methodology — the plateau-coincidence ambiguity.** Linear
probes on mean-pooled hidden states are well-suited for lexically-
localized features (where pooling preserves the discriminative
signal) and poorly suited for structurally-integrated features
(where pooling discards positional and compositional information
the discrimination depends on). Both the syntax (~0.78) and
compositional moral (~0.77) plateaus may reflect probe limitations
as much as representation quality. We cannot distinguish a 1B-model
ceiling from a probe-side ceiling at 1B with linear probing alone;
the cleanest disambiguation is repeating §4.1 at 7B and 32B in
Phase E. If compositional moral accuracy rises with model scale
while syntax accuracy does not, the bottleneck at 1B is the model
not the probe; if both ceilings track scale together, the probe is
the bottleneck. Either result refines the gradient finding without
overturning it.

**Compositional fragility's diverging late-training trajectory.**
§4.3's 4-seed replication established that compositional mean
critical noise *declines* through steps 7K-30K (4.65 → 2.46, std
0.84 → 0.28) while standard moral fragility *holds* at maximum
robustness across mid- and late-layers throughout training. We
reported two non-exclusive readings — the *mechanism reading*
(compositional representations drift toward brittleness as training
continues on text that does not specifically reinforce
compositional moral integration) and the *probe-ceiling reading*
(fragility computed at a 0.77 operating point has less headroom
than fragility at 0.96, producing a numerical difference partly
artifactual). The Phase E 7B / 32B compositional fragility
replication is also the cleanest disambiguation of this question:
under the mechanism reading, the decline should track training-text
distribution rather than scale; under the probe-ceiling reading, the
decline should attenuate as scale lifts the operating-point ceiling.

**Single model family.** All findings are on OLMo-2 1B and OLMo-3
7B. Generalization to other architectures and training recipes is
open. The Phase E plan (§6) addresses 7B replication on a
Deep-Ignorance-style filtered base and 32B-scale replication on
selected publicly-available open-checkpoint models.

**Single language.** All probing datasets are English; pretraining
data for both target models is dominantly English. Cross-lingual
generalization of both the gradient finding and the
fragility-resolves-what-accuracy-misses pattern is open.

**Foundation-specific scope.** The standard moral dataset's six MFT
foundations show staggered emergence; the liberty/oppression
foundation never fully stabilizes at either 1B or 7B. This
cross-scale pattern is documented in Appendix A but not in the main
thesis. The compositional dataset's 200 pairs are categorized by
construction pattern (motive / target / consequence / role) rather
than by MFT foundation; a foundation-stratified compositional probe
(parallel to the foundation-specific standard probe in Appendix A)
would tell us whether different foundations acquire compositional
encoding at different steps. Out of scope for this paper but a
natural extension.
