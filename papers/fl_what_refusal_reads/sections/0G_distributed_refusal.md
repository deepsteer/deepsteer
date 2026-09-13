# Appendix G. Distributed refusal {#app:distributed}

Two independent observations say that refusal is a distributed write rather than a
single-direction switch: the GPT-OSS held-out ablation battery and the OLMo-3 per-head write
attribution into the decision channel (the anatomy behind \Cref{reads-harm} and \Cref{app:causal}).
They are measured with different instruments on different architectures and reach the same
structural reading.

## G.1 GPT-OSS: no single direction ablates refusal {#app:distributed-gptoss}

On GPT-OSS-20B no single direction ablates refusal on a held-out, category-diverse request set.
The end-of-prompt refusal direction, estimated on a category-spanning training draw, coherently
flips 4 percent of held-out refusals; the direction estimated at the last chain-of-thought token
flips none; and the direction estimated as the chain-of-thought mean removes refusal in 88 percent
of cases only by driving generation into incoherence, which the coherence filter excludes. A
representation that no single direction removes is not a bottleneck for refusal on this model.

That has a consequence for the causal program. The single-subspace load-bearing test used on OLMo
and Llama (patch the moral subspace, read the refusal change) presumes a channel that a low-rank
edit can move. On GPT-OSS that presumption fails at the ablation stage, so the GPT-OSS read axis in
\Cref{gpt-oss} is correlational (a projection read at the instruction token and in-trace), and the
causal interchange version stays held. The same fact appears from the other side in
\Cref{app:panel}: GPT-OSS's harmony decision token is the one panel position where the
held-one-out moral band survives its covariance-matched null (0.53 against 0.48), so content
survives there in a way the chat decision sites do not show.

## G.2 OLMo-3: a distributed write into a narrow channel {#app:distributed-olmo}

On OLMo-3-Instruct the refusal decision is written into the roughly 13-dimensional decision-site
channel by a distributed set of attention heads. Ranked by channel-matched specificity, the lead
head (layer 16, head 23) carries 11.7 percent of the total; cumulative specificity reaches 45
percent at the top ten heads and needs 67 heads to reach 80 percent (the saved sparsity curve of
record; the per-head arrays are in the supplement as `head_attribution.csv`). Writers span layers
11 to 16. The top ten by specificity are L16 H23 (write +0.742, specificity +0.756), L15 H2
(+0.302, +0.368), L14 H19 (+0.334, +0.347), L15 H0 (+0.265, +0.285), L11 H20 (+0.246, +0.274),
L16 H21 (+0.172, +0.197), L14 H22 (+0.178, +0.193), L15 H6 (+0.175, +0.189), L13 H29 (+0.139,
+0.144), and L15 H15 (−0.130, −0.142), the sole anti-refusal writer in the top ten.

Attention is not the whole write. Multilayer perceptrons contribute 38 percent of the decision-site
write (fraction 0.384), below the 0.50 threshold at which a Jacobian stage would be required and
above the 0.23 the un-folded run had reported. That un-folded number is a calibration lesson in its
own right: OLMo-3's reordered normalization makes the naive per-head attribution overshoot, and
folding the per-layer RMSNorm gain brings the Stage-1 reconstruction from 3.05 to 0.9999 within a
two-sided band of [0.90, 1.10], exact to one part in a billion (methods note, A3).

None of the ten top writers reads moral content in a way the calibrated instruments recognize. All
ten are labeled neither-moral-nor-harm: none clears the moral-family band, none is a clean
copy-head for harm, and their moral-subspace fractions sit at 0.15 to 0.28 with comparable harm
loading. Llama-3.1's anatomy is OLMo-like on every count that can be compared (pre-norm
reconstruction 1.0008 with no fold needed, a distributed write, multilayer-perceptron share 0.30,
all writers neither), as \Cref{app:llama-anatomy} reports.

## G.3 Reading the two together {#app:distributed-reading}

On both models the refusal write is many small contributions into a low-rank control channel, a
refusal cone rather than a single direction, consistent with the 8-to-15-dimensional bottleneck of
\Cref{bottleneck}. The GPT-OSS battery shows the consequence for removability (no single direction
suffices on a model whose refusal is spread across the trace), and the OLMo attribution shows the
mechanism on a model where a single direction does suffice: even there, the direction that a
rank-one edit removes is assembled by dozens of heads, none of which carries it alone. The
distributed write is what any intervention that widens the read (\Cref{discussion}) would have to
move.
