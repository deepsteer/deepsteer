# 6. Refusal is orthogonal to moral judgment at the decision {#decision-orthogonality}

With both directions defined at the same valid decision token, the comparison is a single
cosine per model, read against a calibrated null and the moral-family band. The refusal
decision is orthogonal to the moral-judgment decision on every model.

On OLMo-3 the cosine between the refusal-decision and judgment-decision directions is 0.10,
with no coupling detectable above $|\cos|$ 0.10 against a null q95 of 0.41: a margin of 0.35
below the minimum detectable effect (null q95 plus a 0.05 margin). On Qwen the cosine is 0.32
against a null q95 of 0.42 (margin 0.15), and on Llama it is 0.08 against a null q95 of 0.51
(margin 0.48). All three margins clear the detection bar; the refusal and judgment directions
occupy different slots of the low-dimensional decision channel at a separation below even the
random level for a channel that narrow. This reads as active separation, not a
weak-instrument artifact: the same channel where a projection-fraction test has no power still
supports a decision-direction cosine, and that cosine says the two directions are apart.

We state this as a bounded null, not a bare dissociation. On OLMo there is no coupling
detectable above $|\cos|$ 0.10 against a null q95 of 0.41; whatever small coupling exists is
below that bar. This wording matters because the same measurement, the held-one-out
moral-family band, also certifies that refusal is not merely far from judgment but *below the
band of moral directions generally*. The bands are [0.537, 0.664] on the base model, [0.523,
0.637] on instruct, [0.537, 0.667] on the reasoning model, and [0.649, 0.764] on GPT-OSS, and
every refusal point on every model lands below its band (the base band-minimum 95% confidence
interval is [0.47, 0.53], and refusal sits under it). \Cref{fig:ladder} shows the same fact
from the projection side.

The orthogonality is real but, on its own, under-determined. Geometric non-overlap does not
by itself establish that refusal reads none of the moral content; it establishes only that the
refusal-decision direction and the judgment-decision direction point in different places, and
\Cref{bottleneck} already showed that content-versus-decision orthogonality is
structurally favored at this site. To learn what refusal actually reads, and whether the moral
subspace is causally inert for it or merely read through a narrow slice, requires a causal cell
on the heads that write the decision channel. That is the next section.
