# Appendix F. Removability battery {#app:removability}

The behavioral and interventional companion to the representational results folds into this
and the next two appendices. This appendix carries the single-direction refusal-ablation battery
across the three instruct families, the behavioral side of the representational dissociation in
\Cref{fresh-gate}, and the mechanism-level reading of \Cref{app:llama-anatomy}. Every number here is
a measured quantity from the program's runs; the appendix reports the battery and states what it
does and does not license.

## F.1 The battery and its readouts {#app:removability-battery}

Single-direction refusal ablation follows the Arditi construction: the refusal direction is the
difference of mean residual activations between harmful and harmless requests at the last prompt
token, and it is orthogonalized out of the attention output projection and the
multilayer-perceptron down projection at one layer, chosen per model by a depth-fraction sweep
(OLMo layer 19, depth 0.59; Qwen layer 14, depth 0.50; Llama layer 13, depth 0.41). Four readouts
are taken before and after: the refusal rate on harmful requests, fresh per-foundation probe
accuracy for moral content, the effective dimensionality of the moral subspace, and behavioral
moral judgment on the 48-scenario forced-choice battery. Persona-shift compliance, a fifth readout,
is reported in \Cref{app:persona}.

The representational readouts do not move. On all three models the ablation leaves probe accuracy
at 1.0 and the effective dimension of the moral subspace at 5, before and after. On OLMo-3 the
ablated direction itself projects only 0.10 of its norm into the moral subspace (mean absolute
cosine 0.06 to the foundation directions), and removing it takes refusal from 0.25 to 0.00 on the
Paper-5 borderline-plus-harmful request set while leaving comprehension where it was
(base-to-fresh cosine 0.749, probe accuracy 1.0, effective dimension 5) and behavioral moral
judgment essentially unchanged (0.73 against 0.75). The OLMo refusal baselines quoted in this paper come from different request sets and harnesses by
construction: 0.25 on the Paper-5 borderline-plus-harmful set, 0.575 on the Paper-6 held-out harmful
set used for the cross-model battery below, and 0.62 on the same held-out set under the reconciled
chat-template harness of \Cref{app:crossablation}; each ablation result is read against its own
baseline.

Removability is family-dependent. On the held-out harmful set, the single-direction ablation takes
OLMo from 0.575 to 0.000 and Qwen from 1.000 to 0.000, but Llama only from 0.900 to 0.475. A
rank-one edit removes refusal cleanly on two families and only halves it on the third.

## F.2 Behavioral judgment under ablation {#app:removability-judgment}

Behavioral moral judgment is the readout that separates the families a second time. Under the
same ablation, OLMo's judgment accuracy moves from 0.75 to 0.79 and Qwen's from 0.875 to 0.812,
both within the range of the matched-random control; Llama's drops from 0.75 to 0.604. The Llama
drop sits far outside the magnitude-matched random-ablation band (0.747 with a spread of 0.007
across random directions of the same norm), and it is dose-dependent: with the ablation scaled to
half strength the judgment drop is already 0.083 (95 percent bootstrap interval [0.02, 0.17],
48 items) while the refusal rate is still 0.90, and at full strength the drop is 0.146
([0.06, 0.25]). The matched-random null stays at 0.74 even at twice the full-ablation magnitude,
and ablating the persona direction (\Cref{app:persona}) leaves judgment at 0.75. So the Llama
judgment drop is specific to the refusal direction, not to any salient direction of that norm, and
it appears before the refusal rate has moved.

The anomaly resolves upstream rather than in the battery. The depth-matched Llama cell battery of
\Cref{app:llama-depth} finds that Llama's refusal reads broad moral content by interchange
(refusal transfer 0.85 against judgment 0.79 at layer 12) and commits early. A refusal direction
estimated on a broad-moral reader carries moral content with it, so removing it costs judgment;
on the harm-keyed readers it does not. The battery is the behavioral face of the reads axis in
\Cref{cross-model}.

## F.3 What removability does and does not show {#app:removability-scope}

On OLMo the ablated direction reads a rank-1 harm slice: 76 percent of refusal's causal input lies
outside the rank-16 moral basis (\Cref{reads-harm}), which is why a rank-one edit removes it
without touching comprehension. That is a causal statement about one model. Across the panel the
link between a low-rank read and clean removability is correlational: two harm-keyed readers are
cleanly removable and one broad reader is not, and the read axis on the third harm-keyed model
(GPT-OSS) is itself correlational (\Cref{app:distributed}). This appendix reports the battery; it
does not claim the low-rank-read-to-removable link is general, and the Limitations section
(\Cref{limitations}) carries that scope alongside the architecture confound on the two-axis table.

A second scope note comes from the reconciled cross-ablation run on OLMo-3-Instruct
(\Cref{app:crossablation}): a single-layer projection-out of the refusal direction, applied at the layer
output for every position rather than folded into the weights, does not remove refusal on that
model. It re-decides about half of 100 held-out harmful requests in both directions (33 from comply
to refuse, 17 from refuse to comply) with fully coherent text, while five random directions of the
same norm at the same site re-decide none. The weight-folded Arditi ablation of this appendix and
the activation-level projection-out are different instruments with different outcomes on the same
direction, and the paper keeps them apart: removability claims rest on the former.
