# 2. Related work

This work sits at the intersection of three established literatures
(linear probing, phase transitions in neural network learning, moral
foundations as a probing target) and one recent one (representation
fragility / single-direction circuits as alignment-relevant
properties). We summarize the connections that matter for the
methodological claim and reserve broader interpretive context for
§5 Discussion.

**Linear probing as a measurement instrument.** Linear probing —
training a linear classifier on frozen hidden states to test whether
a property of interest is *linearly decodable* — was established by
Alain and Bengio (2017) as a layer-wise diagnostic for what
intermediate representations contain. Belinkov (2022) surveys the
methodology's promise and known limitations, several of which our
work makes load-bearing: probing accuracy as a thresholded, capped
metric that loses resolution after saturation; the difficulty of
distinguishing what the model *encodes* from what the probe can
*recover*; and the importance of designed minimal-pair stimuli rather
than naturalistic distributions for separating the property of
interest from confounded surface features. Our methodological
contribution — that fragility (per-layer noise robustness) continues
to evolve through training after probing accuracy plateaus — is a
direct extension of the second concern: where Belinkov treats probe
saturation as a threat to validity, we treat it as a fixed feature of
the instrument and introduce a complementary metric that survives it.

**Causal tracing and activation methods.** Meng et al. (2022) introduced
ROME and the causal-tracing methodology for identifying which
transformer layers are causally responsible for a model's behavior, as
distinct from which layers most-strongly *encode* the relevant
information. Our 7B causal-probing analysis (Appendix B) finds a
~10-layer divergence between probing peak (storage) and causal peak
(use) for moral information, replicating the storage-vs-use distinction
in the moral domain. We treat causal tracing as supporting evidence
for the body's methodological thesis rather than a body finding.

**Phase transitions in neural-network learning.** Power et al. (2022)
documented "grokking" — neural networks suddenly transitioning from
memorization to generalization on small algorithmic tasks after long
plateaus. Subsequent work has shown that capability emergence in
language model pre-training often shows similar sigmoidal phase-
transition dynamics rather than gradual improvement. Our §4.1 finding
that semantic minimal-pair tasks (standard moral, sentiment) emerge
as sharp phase transitions while compositional and structural tasks
emerge gradually with no inflection point connects this literature to
a within-model dichotomy: phase-transition dynamics appear when a
capability can be acquired through local lexical statistics, while
gradual emergence appears when the capability requires multi-token
integration. The dichotomy is implicit in the grokking literature —
which task types grok and which do not — but to our knowledge has
not been mapped against the lexical-vs-compositional distinction
within a single training run.

**Moral Foundations Theory as a probing target.** Our minimal-pair
datasets organize moral content using Haidt's (2012) and Graham et
al.'s (2013) Moral Foundations Theory taxonomy (six foundations:
care/harm, fairness/cheating, loyalty/betrayal, authority/subversion,
sanctity/degradation, liberty/oppression). MFT is the standard
taxonomy for cross-cultural moral psychology and is widely used as a
labeling schema for moral content in NLP work. We use it as a
*construction* heuristic — to ensure balanced coverage across moral
content types in the standard moral probing dataset — rather than as
a substantive cognitive claim about how language models represent
morality. The compositional moral dataset (§3.2) is categorized by
construction pattern (motive / target / consequence / role) rather
than MFT foundation, reflecting the methodological focus of this
paper; foundation-stratified compositional probing is a natural
extension flagged in §5.4.

**OLMo and open intermediate checkpoints.** Groeneveld et al. (2024)
introduced the OLMo family with full intermediate checkpoint releases
— the unique infrastructure that makes dense-sampling trajectory
analysis possible. Our 37-checkpoint OLMo-2 1B early-training
trajectory (steps 0-36K at 1K-step intervals) and 20-checkpoint OLMo-3
7B stage-1 trajectory both rely on this open checkpoint release. The
methodological contribution we report can in principle be applied to
any open-checkpoint model family; we use OLMo because the dense
sampling resolves dynamics that sparser checkpoint releases cannot.

**Single-direction circuits and representation fragility.** Arditi et
al. (2024) demonstrated that refusal behavior in instruction-tuned
language models is mediated by a single representational direction
that can be ablated to remove safety behavior. The result connects to
a broader pattern — that some safety-relevant properties live in
narrow, brittle representational subspaces rather than distributed
encoding — and motivates the question whether moral encoding is
similarly concentrated. Our fragility metric is the
*pre-training-time* analog of this question: where the
single-direction-refusal literature asks whether *post-training*
safety properties are concentrated in narrow circuits at the *final*
checkpoint, we ask whether the *pre-training trajectory* of moralized
representations passes through more or less brittle states, with
fragility as the per-layer measurement. The two literatures meet at
the question of how concentrated vs. distributed safety-relevant
representations are; they ask it at different points in the model's
lifecycle.

**A note on scope.** This paper uses *moralized vocabulary* as a
demonstration domain for the methodological thesis. Phase C4's
compositional probe (§3.2) is the explicit ablation that bounds what
the standard probe is measuring; the broader question of whether
language models in any sense *reason* about moral situations is out
of scope here and would require harder probes (counterfactual
intervention, generalization to held-out moral structures) that we
discuss as future work in §5.4.
