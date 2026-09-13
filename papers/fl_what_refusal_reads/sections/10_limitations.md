# 10. Limitations {#limitations}

**The panel is three points and confounded.** The two-axis table (\Cref{tab:two-axis}) is a
measured result, but its interpretation as a dimensionality-to-reversibility law is a
hypothesis on three models that differ in lineage, scale, tokenizer, and
reasoning-versus-instruct training simultaneously. A one-axis account (the effective
dimensionality of the refusal read predicts reversibility) and a lineage account fit the same
three points equally well. We state the hypothesis; we do not claim the mechanism. Deconfounding
needs one axis varied at a time, for example a deliberation-trained variant of a single base
model, or a lineage-matched scale sweep.

**GPT-OSS reads-harm is correlational.** GPT-OSS is placed on the harm-reading axis by
projection (its refusal direction is harm-loaded at the prompt, cosine 0.977 to harm versus
0.001 to the harm-orthogonal moral subspace), not by interchange. The causal version, the
nested rank sweep that resolves the OLMo verdict, is held for GPT-OSS. So the OLMo harm-reading
verdict is causal and the GPT-OSS one is correlational, and the paper marks the two differently.

**Readout versus behavior scope differs by cell.** Some cells read internal directions (the
rank sweep, the decision-channel projections) and some read behavior (the 7/7 engage flip, the
6/10 disengage flip, the ~17% OLMo refusal rate). These are different outcome variables, and a
result on one does not automatically transfer to the other. Where a cell is a projection read
we say so; where it is a behavioral flip we say so; we do not silently promote a projection
movement to a behavior change.

**The graded projection is not refusal-specific.** The GPT-OSS reversibility result is
behavioral (6/10 violating items flipped to comply, 5/10 on replication; 7/7 benign items flipped
to refuse). We also read the decision-channel refusal projection along the graded prefill series,
first at the last prefill token and then, after the model had answered, at the token that opens the
final channel. The projection moves toward comply monotonically at both positions (8 of 8 items
that opened a final channel). It is not, however, distinguishable from what a random direction
does there: against 500 covariance-matched random directions drawn from the decision-token
activation sample, the refusal direction's strong-minus-weak move (about one standard deviation of
the sample, in raw and in standardized units) is exceeded by one random direction in five
(one-sided p 0.17 to 0.23 across positions and frames). The prefill rewrites the position's
content, and any direction with variance there moves with it. The projection therefore says where
deliberation writes, not that the refusal direction reads it, and it is not a second leg of the
reversibility claim; a direction-specific causal test at the decision token is the held Tier-2
cell.

**Stimulus-composition covariates across model bands.** The moral-family bands and null values
are computed per model on its own activation sample, and the stimulus sets that define the
positive-control bands are not identical in composition across models. Cross-model comparisons
of absolute projection values therefore carry a stimulus-composition covariate; the
within-model verdicts (below its own band, below its own null) do not, and those are the ones
we report as findings.

**OLMo's weak behavioral coupling.** OLMo-3's refusal reaches only about 17% at top intent
severity, so its behavioral operating band for intent-graded refusal is nearly empty. This is
coherent with the harm-surface-keyed read (a weak intent-refuser is what a harm-keyed gate
predicts) and we report it as a model property, but it means the OLMo commitment-axis and
severity-graded behavioral cells are measured on Llama and GPT-OSS, where refusal tracks
intent, rather than on the model that carries the causal rank sweep.
