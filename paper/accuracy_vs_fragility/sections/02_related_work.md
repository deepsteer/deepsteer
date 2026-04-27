# 2. Related work

**Linear probing.** Alain and Bengio (2017) established linear
probing as a layer-wise diagnostic for what intermediate
representations contain; Belinkov (2022) surveys the methodology's
promise and known limitations. Our methodological contribution
extends the saturation concern: where the probing literature treats
ceiling effects as a threat to validity, we treat them as a fixed
feature of the instrument and add a complementary metric (fragility)
that continues to resolve representational change after accuracy
plateaus.

**Causal tracing.** Meng et al. (2022) introduced ROME and the
causal-tracing methodology, which distinguishes layers that *encode*
information from layers that *causally use* it. Our 7B
causal-probing analysis (Appendix B) finds a ~10-layer divergence
between probing peak and causal peak for moral information,
replicating the storage-vs-use distinction in the moral domain as
supporting evidence for the body's methodological thesis.

**Phase transitions.** Power et al. (2022) documented "grokking" —
sudden phase transitions from memorization to generalization. Our
§4.1 finding that semantic minimal-pair tasks (standard moral,
sentiment) emerge as sharp phase transitions while compositional and
structural tasks emerge gradually maps the grokking phenomenon
against a lexical-vs-compositional dichotomy *within a single
training run* — to our knowledge a novel framing.

**Moral Foundations Theory.** The standard moral dataset organizes
content across Haidt's (2012) and Graham et al.'s (2013) six MFT
foundations (care/harm, fairness/cheating, loyalty/betrayal,
authority/subversion, sanctity/degradation, liberty/oppression). MFT
is used as a *construction* heuristic for balanced coverage, not as
a cognitive claim about how language models represent morality. The
compositional dataset (§3.2) categorizes by construction pattern
(motive / target / consequence / role) instead.

**OLMo.** Groeneveld et al. (2024) released the OLMo family with
full intermediate checkpoint releases — the infrastructure that
makes dense-sampling trajectory analysis possible. Our 37-checkpoint
1B early-training and 20-checkpoint 7B stage-1 trajectories rely on
this open release; the methodology applies to any open-checkpoint
family with dense enough sampling.

**Single-direction circuits.** Arditi et al. (2024) demonstrated
refusal behavior in instruction-tuned models is mediated by a single
representational direction. The fragility metric is the pre-training-
time analog: where single-direction-refusal asks whether *post-
training* safety properties are concentrated in narrow circuits at
the final checkpoint, we ask whether moralized representations pass
through more or less brittle states *during* training. Both
literatures address how concentrated vs. distributed safety-relevant
representations are, at different points in the model lifecycle.

**Scope.** We use moralized vocabulary as a demonstration domain.
Phase C4's compositional probe (§3.2) is the explicit lexical-
accessibility ablation; deeper questions (counterfactual moral
reasoning, generalization to held-out moral structures) would
require harder probes and are discussed as future work in §5.4.
