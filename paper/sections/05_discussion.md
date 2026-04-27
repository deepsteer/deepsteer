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

The dichotomy connects to Power et al.'s (2022) grokking literature
(sudden phase transitions on algorithmic tasks), which has largely
focused on the *cause* of phase transitions; our results suggest the
*taxonomy* of which capabilities should and should not exhibit them.
The formal information-theoretic argument is its own paper.

## 5.2 Why fragility succeeds where accuracy saturates

Probing accuracy is a thresholded, capped, top-end metric: once
linear separability is good enough, accuracy hits ceiling and stops
returning information about underlying representational change.
Fragility is structurally different — it integrates the *margin* of
separability (outputs near the decision boundary flip under small
noise) and the *redundancy* of representation (features encoded in
many hidden-space directions tolerate noise that collapses any one).
Both quantities continue to evolve after accuracy saturates because
both are functionals of representation *geometry* rather than
end-to-end classification accuracy. Concretely (§4.3): the standard
moral probe's mean accuracy holds at 0.96 from step 5K through step
36K while early-layer critical noise drops 10.0 → 1.7. The argument
generalizes — fragility is not a moral-domain-specific contribution
but a methodological contribution for any binary probing task that
hits accuracy ceiling.

## 5.3 Generalization beyond pre-training

The pattern reproduces under a different stimulus. In companion work
(Reblitz-Richardson, 2026, in preparation), applying the same
standard moral probe + fragility battery to LoRA adapters trained on
the Betley et al. (2025) insecure-code dataset produces identical
probing accuracy across base / insecure-LoRA / secure-LoRA
(max |Δ| = 0.021) but a fragility-locus shift of 2-3 layers under
insecure-code specifically (robustness peak relocates from layer 7
to layers 9-10; layers 6-7 collapse from critical noise 10 → 1). The
methodology extends from pre-training trajectories to fine-tuning
fingerprints; we reference this here as evidence of generality and
develop it in the companion paper.

## 5.4 Limitations

**Lexical→compositional gradient bounds the standard probe.** The
standard moral probe measures something closer to "moralized
vocabulary becomes linearly separable from neutral vocabulary" than
"moral reasoning emerges." Phase C4's compositional probe (§4.1)
established this is a quantitative gradient — lexically-marked
moralized vocabulary at step 1K, compositional moral integration at
step 5K, syntactic competence at step 6K — not a binary
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

**Two related questions disambiguate at Phase E (7B / 32B
replication).** First, the §4.2 plateau coincidence (compositional
≈ syntax ≈ 0.77 vs. standard moral / sentiment ≈ 0.97) may reflect
a 1B-model ceiling on compositional / structural encoding or a
probe-side ceiling under mean-pooled linear probing. Second, the
§4.3 4-seed compositional fragility decline through steps 7K-30K
(4.65 → 2.46) — opposite to the standard probe's late-training
hold — admits both a *mechanism reading* (compositional
representations drift toward brittleness as training continues on
text that does not specifically reinforce them) and a *probe-ceiling
reading* (fragility at the 0.77 operating point has less headroom
than at 0.96, partly artifacting the difference). Both readings
predict different scaling behavior: under the mechanism reading the
decline tracks training-text distribution rather than scale, under
the probe-ceiling reading it attenuates as scale lifts the
operating point. Repeating §4.1 and §4.3 at 7B and 32B disambiguates
both. Either outcome refines the gradient finding without overturning
it.

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
