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

**The prefill-last-token projection caveat.** The GPT-OSS reversibility result is primary on
the behavioral flip (6/10 violating items flipped to comply) and corroborated by the
decision-channel projection moving monotonically toward comply in all 10 items. That projection
is read at the last token of the prefill, a position whose activation carries prefill-specific
content; the monotone projection is corroboration for the behavioral flip, not an independent
causal claim.

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
