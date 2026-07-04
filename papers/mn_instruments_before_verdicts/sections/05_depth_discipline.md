# 5. Depth discipline {#depth-discipline}

A verdict about what a circuit *reads* must state the intervention depth relative to the
model's commitment. A patch at the read layer is post-commitment for an early-committing
model, so a "reads-X" or asymmetry claim measured there can be a read-layer artifact. Measure
at (and report) the pre-commitment coherent depth, and depth-match cross-model comparisons.

**Failure as it appeared.** The naive cross-model asymmetry at the read layer (layer 16) was
`A_Llama = +0.82`, engage-dominant and latch-like, against `A_OLMo = −0.20`, for a difference
of +1.03 (95% CI [0.16, 1.61], excludes 0). Read at face value, this said Llama's refusal has
a third distinct property, a hard directional latch that OLMo lacks.

**The tell.** Llama's disengage is coherent at earlier layers but not at the read layer.
Patch-layer sweep: Llama's disengage is coherent at layers 8/12/14 (−0.12 / −0.11 / −0.20,
CIs exclude 0) and incoherent at layer 16 (−0.014). Llama commits *before* the read layer;
OLMo's disengage is coherent at layer 16 (−0.62), so OLMo commits at or after it. Measuring
Llama's asymmetry at layer 16 measures it after Llama has already decided, so the +0.82 is a
post-commitment artifact. (The two layer-12 disengage numbers are two cells: the patch-layer
sweep above reads −0.11, the depth-matched full re-run reads −0.57, both coherent; the verdict
is the same.)

**The protocol.** Depth-match the comparison to the pre-commitment coherent layer, layer 12,
and recompute both models there.

**The certifying check.** At matched layer 12, `A_Llama = −0.28` (CI [−0.47, +0.03]) and
`A_OLMo@12 = −0.54` (CI [−0.81, −0.32]), a difference of +0.26, down from +1.03 at the read
layer. The apparent third property collapses: the asymmetry is a *consequence* of
early-commitment, not an independent latch. What survives depth-matching is the reads-axis
difference: at layer 12 Llama reads broad (`broad_moral`: R_refusal 0.85 ≈ R_judgment 0.79,
gap closes, harm-rank-1 only 0.59) while OLMo stays harm-keyed (R_refusal 0.43 < R_judgment
0.53, gap open). The read-layer +0.82 is retained only as a voided number with its replacement,
never as a finding (CLAIMS V-D3-8).

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{mn_depth_collapse.pdf}
\caption{Depth collapse of the cross-model asymmetry $A$. Measured at the read layer (16), $A_{\mathrm{Llama}}$ is $+0.82$ and reads as a Llama-specific directional latch. Depth-matched to the pre-commitment coherent layer (12), it falls to $-0.28$, while $A_{\mathrm{OLMo}}$ moves from $-0.20$ to $-0.54$; the cross-model difference collapses from $+1.03$ to $+0.26$. The apparent third property is a read-layer artifact of measuring after Llama has already committed.}
\label{fig:depth-collapse}
\end{figure}

**Figure 3** is this depth collapse: `A_Llama` from +0.82 at the read layer to −0.28 at
matched layer 12, with `A_OLMo` −0.20 → −0.54, the depth-indexed exemplar.
