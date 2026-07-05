# 3. Moral comprehension is pretraining-native and survives alignment {#comprehension-native}

Moral comprehension is present and stable before any post-training touches the model, and
post-training reorients it rather than teaching it. That is the first of the paper's facts,
and it is about what alignment does *not* build.

On OLMo-3, where base and post-training checkpoints are released, we track the moral subspace
across 25 pretraining and post-training states. Comprehension is pretraining-native: a linear
probe on the moral subspace reaches 100% accuracy at every state, effective dimension holds
at 5, and cross-source transfer AUC is ≈ 1.0 throughout. The subspace also *crystallizes*
during pretraining. The cosine between the direction at each checkpoint and the fully trained
direction rises from 0.869 at step 1000 to 0.999 by the end of pretraining. Once formed, the
subspace is not rebuilt by alignment. Supervised fine-tuning rotates it once, from a
base-to-SFT cosine of 0.999 down to 0.757 (about 40 degrees), and then it holds: preference
optimization leaves it at 0.757, and every reinforcement substep stays in the 0.757–0.759
band, with effective dimension 5 throughout. Post-training reorients moral comprehension by a
single rigid rotation; it does not re-teach it and does not change its rank.

This is consistent with what the pretraining studies in this line find about how moral
structure emerges. Moral content is learned early and compositionally rather than as a
bag of words [@reblitzrichardson2026fragility]. The moral foundations a model acquires also
integrate into one positively correlated subspace rather than splitting into the theory's
individualizing and binding clusters: mean off-diagonal cosine 0.232–0.274, effective
dimension 5 at every layer, first principal component 0.379 of variance against 0.179 for a
random baseline, and no significant recovery of the moral-foundations split under permutation
(minimum $p = 0.32$) [@reblitzrichardson2026geometry]. The picture is a moral
representation that forms in pretraining, is broad and low-rank, and is preserved through
alignment.

The contrast that organizes the rest of the paper is with the refusal gate. Where the moral
subspace crystallizes during pretraining to a checkpoint-to-final cosine of 0.999 (and survives
alignment with a single ~40-degree rotation), the refusal gate reaches only
0.155 from its pretraining precursor. Comprehension is deep and inherited; the refusal
decision, as the next section shows, is a shallow, freshly built control. \Cref{fig:crystal}
plots the two side by side.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fl_crystallization.pdf}
\caption{Comprehension crystallizes; the refusal gate does not. Left: the cosine between the
OLMo-3 moral subspace at each training checkpoint and the fully trained direction rises from
0.869 at step 1000 to 0.999 during pretraining, then holds through post-training (supervised
fine-tuning rotates it once to 0.757 and later stages leave it there). Right: the refusal
gate's cosine to its pretraining precursor is only 0.155, far below the 0.50 crystallization
threshold. Moral comprehension is pretraining-native and inherited; the refusal decision is a
fresh post-training construction. Regenerable from committed data (\Cref{app:repro}).}
\label{fig:crystal}
\end{figure}
