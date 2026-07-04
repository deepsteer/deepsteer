# 4. The refusal gate is a fresh post-training construction {#fresh-gate}

The refusal decision is not a moral-content-derived direction inherited from pretraining. It
is built during post-training, it lives in a low-variance channel, and it sits below the
moral-family band on every model tested.

**It does not crystallize from a precursor.** We measure the cosine between the base model's
proto-refusal direction and the aligned model's refusal gate at the layer where the gate is
defined. It is 0.155, well below the 0.50 threshold that the moral subspace clears on its way
to 0.999. The number is the headline pairing of \Cref{fig:crystal}: 0.999 for comprehension,
0.155 for the gate. The aligned refusal gate is substantially a post-training construction,
not a re-pointing of something the base model already had.

**It lives in a low-variance channel.** Across residual dimensions ranked by variance, the
wired instruct refusal gate sits at the bottom: its variance percentile is 0.0 (within the
lowest decile), and all four positions carrying the GPT-OSS refusal signal are likewise in the
lowest decile. The moral-subspace axes and the persona direction, by contrast, occupy
ordinary-to-high variance. The base model's proto-refusal is not narrow (percentile 37.4), so
the narrowness is a property of the *installed* gate, not of the raw contrast. Refusal is
wired into a spare channel that carries little of the model's activation variance.

**It projects below the moral-family band at every rung.** The base proto-refusal projects
0.33 onto the base moral subspace (covariance-matched null q95 0.291; the persona reference
sits at 0.51), and the aligned gate projects 0.14 onto the aligned moral subspace (null q95
0.26). Across a rank sweep over the moral basis (one to three sources) refusal never clears
the null-plus-margin bar. Read against a richer construction the story is the same: the
aligned gate projects 0.144 onto the rank-3 moral subspace (null q95 0.266) and 0.155 onto
the six-foundation moral-foundations span (null q95 0.252), both null. \Cref{fig:ladder} shows
the calibrated ladder that turns these into verdicts: floor, matched null, measurement, and
positive band on one axis, with refusal below the band.

These three readings converge. The refusal gate is a freshly built, low-variance
post-training control, not a direction derived from the model's moral content. That is the
mechanism behind two otherwise separate facts: a rank-one edit can ablate refusal cleanly
[@arditi2024refusal; @pew2025heretic], and refusal projects below the moral band. A control
this thin and this orthogonal to comprehension is exactly what is cheap to remove. Consistent
with this, on OLMo-3 the refusal direction projects only 0.10 of its norm into the moral
subspace (mean $|\cos|$ 0.06), and ablating it drops refusal from 0.25 to 0.00 while leaving
comprehension intact (base-to-fresh cosine 0.749, probe accuracy 1.0, effective dimension 5)
and moral judgment essentially unchanged (0.73 versus 0.75).

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fl_calibration_ladder.pdf}
\caption{The calibrated projection ladder. For each tag, the refusal projection onto the
moral subspace (marker) is plotted against a floor, a covariance-matched null, and the
positive-control moral-family band (the projection of held-one-out moral directions onto the
span of the rest). Refusal lands below the moral-family band on every model, including the
in-trace peak on the reasoning models; even the program's highest refusal projection is less
moral-adjacent than a held-out moral direction. The ladder is the instrument that turns "small
projection" into "below a stated bar".}
\label{fig:ladder}
\end{figure}
