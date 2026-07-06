# 5. Discussion

## 5.1 Semantic vs. structural learning dynamics

The §4.1 four-curve overlay (Figure 1) shows two distinct learning regimes
within a single training run on a single model. The standard moral
and sentiment probes, both single-token-swap minimal-pair tasks,
emerge as sharp, step-like sigmoidal transitions: each crosses from
chance to its plateau within a single 1K-step interval at onset, then
flattens. (At 1K sampling we resolve transitions that are step-like
at this resolution; we do not claim true discontinuity.) The compositional moral and syntax probes, both tasks that
require multi-token integration to determine the label, rise more
gradually, with no equally sharp inflection point.

The cleanest hypothesis to organize this dichotomy: **phase-transition
dynamics emerge when a feature can be acquired through local lexical
or distributional statistics (the model "discovers" the feature in
a discrete jump as soon as it has enough samples to distinguish the
relevant lexemes), while gradual emergence indicates features that
require integrating positional, attentional, or compositional
relationships across multiple tokens, which the model cannot acquire
in a single step from local lexical statistics alone.** Under this
reading the standard moral probe (single-lexeme swap), sentiment
(single-adjective swap), and similar lexically-localized tasks all
share the phase-transition mechanism; the compositional probe
(multi-token integrated swap) and syntax (positional well-formedness)
share the gradual-emergence mechanism. The 0.20 plateau gap (§4.1)
between the two regimes (single-token-statistics tasks saturating
near 0.97, multi-token-integration tasks near 0.77) is consistent
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
Fragility is structurally different: it is sensitive to both the
*margin* of separability (outputs near the decision boundary flip
under small noise) and the *redundancy* of representation (features
encoded in many hidden-space directions tolerate noise that collapses
any one). It does not separately identify their contributions, and a
low critical noise can also reflect activation-scale changes,
representational anisotropy, or probe-training instability rather than
margin or redundancy alone. Both margin and redundancy continue to
evolve after accuracy saturates because both are functionals of
representation *geometry* rather than end-to-end classification
accuracy. Two cautions follow from §4.4, however. First, much of the
raw cross-layer and cross-checkpoint movement of critical noise is
activation scale, not margin or redundancy: it must be read in
RMS-normalized units for cross-layer claims. Second, the place where
fragility demonstrably separates representations that accuracy cannot
is the *scale-controlled* comparison: the §4.3 data-curation contrast,
where three corpora with identical probing accuracy produce distinct
fragility fingerprints at matched layers, on a fixed checkpoint, with
activation scale held constant by construction. That is the clean
demonstration of the methodological thesis. The broader contribution is
therefore twofold: critical noise is a probe-side metric that keeps
moving after accuracy saturates, and, just as important, we delimit
*when* it is measuring representation rather than scale, raw $\sigma^*$
within-layer and RMS-normalized $\sigma^*$ across layers. Fragility is
not a moral-domain-specific tool but a methodological one for any binary
probing task that hits the accuracy ceiling, provided the scale
confound is controlled.

## 5.3 Limitations

**Staged emergence bounds the standard probe.** The
standard moral probe measures something closer to "moralized
vocabulary becomes linearly separable from neutral vocabulary" than
"moral reasoning emerges." The compositional probe (§4.1)
established this emerges in stages: morally loaded words
at step 1K, compositional moral integration at
step 5K, syntactic competence at step 6K, not a binary
in-or-out distinction. Both onsets are real findings. Neither of
them is "moral reasoning at step 1K"; both are bounded claims about
what a linear probe can recover from mean-pooled hidden states at
each step.

**Compositional probe partial scope.** The compositional probe
addresses *whether the moral signal lives in single-token vs.
multi-token features*; it does not address *whether the model
represents moral concepts in any deeper functional sense*:
counterfactual sensitivity to moral reframing, generalization to
novel moral structures not in pre-training data, behavioral
consistency under adversarial probing. The compositional probe is a
strictly stronger lexical-accessibility ablation than the standard
probe; it is not a moral-reasoning probe. Stronger probes for
deeper moral capacities are out of scope for this paper.

**Two related questions disambiguate at scale (7B / 32B
replication).** First, the §4.1 plateau coincidence (compositional
≈ syntax ≈ 0.77 vs. standard moral / sentiment ≈ 0.97) may reflect
a 1B-model ceiling on compositional / structural encoding or a
probe-side ceiling under mean-pooled linear probing. Second, the
§4.2 4-seed compositional fragility decline through steps 7K-30K
(4.65 → 2.46), opposite to the standard probe's late-training
hold, admits both a *mechanism reading* (compositional
representations drift toward brittleness as training continues on
text that does not specifically reinforce them) and a *probe-ceiling
reading* (fragility at the 0.77 operating point has less headroom
than at 0.96, partly artifacting the difference). Both readings
predict different scaling behavior: under the mechanism reading the
decline tracks training-text distribution rather than scale, under
the probe-ceiling reading it attenuates as scale lifts the
operating point. Repeating §4.1 and §4.2 at 7B and 32B disambiguates
both. Either outcome refines the staged-emergence finding without overturning
it.

**Single model family.** All findings are on OLMo-2 1B and OLMo-3
7B. Generalization to other architectures and training recipes is
open.

**Single language.** All probing datasets are English; pretraining
data for both target models is dominantly English. Cross-lingual
generalization of both the staged-emergence finding and the
fragility-resolves-what-accuracy-misses pattern is open.

**Raw-σ fragility and the scale confound (resolved in §4.4).** We add
Gaussian noise in raw activation units (§3.4), which conflates probe
margin with activation scale: hidden-state RMS grows ~8× from early to
late layers, so a fixed raw σ is a larger relative perturbation early
than late. §4.4 runs the RMS-normalized control and finds that the
§4.2 layer-depth gradient and its cross-checkpoint evolution are largely
this scale effect (the late/early σ* ratio falls from ~7--15× to ~2×,
and the post-saturation decline flattens). The confound is strictly
cross-layer, so it leaves within-layer comparisons intact, including the
§4.3 declarative-vs-natural separation (contrasted at matched layers on
a shared checkpoint, hence at fixed scale). We therefore recommend
RMS-normalized σ* for cross-layer claims and raw σ* for within-layer
comparisons, and we read raw cross-layer σ* only as practical
perturbation sensitivity, not as scale-independent encoding robustness.
The one cell we did not re-run under RMS is the LoRA experiment itself;
its within-layer construction guarantees scale is controlled, but an
explicit RMS replication (saving adapters for reuse) is left to the
multi-seed LoRA extension.

**Noise grid.** Critical noise is read off the extended σ grid
$\{0.1, 0.3, 1, 3, 10, 30, 100\}$ (§3.4), which lifts the late-layer
band off the old σ=10 cap (late layers reach 30--44 mid-trajectory in
Table 2). Layers that never drop below τ even at σ=100 are censored at
100; this is rare on the 1B trajectory.

**Foundation-specific scope.** The standard moral dataset's six MFT
foundations show staggered emergence; all six stabilize by step 3K
(Appendix A). The emergence ordering is sensitive to dataset
construction choices, which confirms the importance of dataset quality
methodology in probing studies; findings that depend on specific
pairs rather than the property of interest are artifacts, not
discoveries. The compositional dataset's 200 pairs are categorized by
construction pattern (motive / target / consequence / role) rather
than by MFT foundation; a foundation-stratified compositional probe
(parallel to the foundation-specific standard probe in Appendix A)
would tell us whether different foundations acquire compositional
encoding at different steps. Out of scope for this paper but a
natural extension.
