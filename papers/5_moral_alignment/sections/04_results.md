# 4. Results

We report the four findings as the chain laid out in the introduction. All
probing uses raw-text inputs (Section 3); behavioral and coupling measurements
use the chat template. Geometry metrics are reported over the bootstrap-stable
layer band (layers 15--31; Section 3, Appendix B).

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{three_curve}
\caption{Comprehension, direction preservation, compliance, and coupling across
the OLMo-3 pipeline (25 states, left to right: 13 stage-3 pre-training
checkpoints, base, SFT, DPO, Instruct, 8 RLVR substeps). Comprehension (probe
accuracy) is flat at 1.0 throughout. Direction preservation, the cosine of each
state's foundation directions to the final base model, crystallizes during
pre-training and drops once at SFT. Compliance and coupling are defined only for
the instruct-capable states.}
\label{fig:three}
\end{figure}

## 4.1 Moral comprehension is a pre-training property

At every one of the 25 model states, the six Moral Foundations are linearly
decodable at 100\% (mean held-out probe accuracy $1.00$ across foundations),
the foundation directions span five effective dimensions (90\% variance), and
the base-trained directions transfer to each state as near-perfect classifiers
(mean $|\text{AUC}| \approx 1.0$). Comprehension does not emerge during
post-training; it is fully present at the base and, as the pre-training
trajectory shows, well before.

What *does* move during pre-training is the orientation of the moral subspace.
The cosine between a checkpoint's foundation directions and the final base
model's rises monotonically across the stage-3 anneal, from $0.869$ at step
1000 to $0.999$ by step 11921 (\Cref{fig:three}, orange). The moral subspace
crystallizes into its final form during pre-training: by the end of the anneal
the foundation directions are essentially identical to the base model's, and
their decodability is already saturated.

## 4.2 Post-training reorients moral representation, it does not re-teach it

If comprehension is already in place at the base, post-training cannot be
teaching it. We ask instead how much post-training *moves* the representation.
The answer is: once, at SFT, and only modestly. The cosine between the base
foundation directions and each post-training state's freshly fitted directions
drops from $0.999$ (base) to $0.757$ at SFT, a rotation of roughly $40^{\circ}$. DPO
($0.757$) and all eight RLVR substeps ($0.757$--$0.759$) leave it unchanged
(\Cref{fig:three}). The within-model geometry barely moves: mean pairwise
foundation cosine goes from $0.262$ (pre-training and base) to $0.250$
(post-training), and effective dimensionality stays at five at every state.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{dendrogram_compare}
\caption{Foundation clustering (Ward linkage on $1-\cos$) at base, SFT, DPO, and
Instruct. The Loyalty--Authority binding pair persists across all stages. SFT
mildly reorganizes the rest: Care shifts from pairing with Fairness (base) to an
outlier, and Sanctity pairs with Fairness.}
\label{fig:dendro}
\end{figure}

The framework structure is largely preserved (\Cref{fig:dendro}): the
Loyalty--Authority pairing survives every stage, while SFT mildly reshuffles the
individualizing foundations (Care becomes an outlier; Sanctity pairs with
Fairness). Per-stage cosine matrices (\Cref{fig:grid}, Appendix) show the same:
post-training applies a small rigid-ish rotation, not a restructuring. The 8
full-attention layers of OLMo-3's hybrid attention (layers 3, 7, \dots, 31) show
no periodicity in any layer-wise metric; the moral geometry is unaffected by
attention type (Section 3).

## 4.3 Comprehension and compliance are only weakly coupled

Post-training preserves comprehension, so whatever it adds to behavior must be
attached to a fixed representation. How tightly? For each of 48 morally-loaded
scenarios we read the model's internally dominant foundation at the stable layer
and its behavioral judgment, and measure their agreement (Section 3). Coupling
rises across post-training but stays weak: agreement $0.375 \to 0.479 \to 0.500$
and $\phi$ $-0.19 \to +0.02 \to +0.05$ for SFT $\to$ DPO $\to$ Instruct. Even at
the final Instruct model, internal comprehension barely predicts behavior:
$P(\text{comply} \mid \text{comprehend}) = 0.77$ versus
$P(\text{comply} \mid \neg\text{comprehend}) = 0.73$.

This near-zero coupling is the result, not a measurement failure. In the fully
aligned model, the moral representation and the behavioral compliance signal are
almost independent. That independence makes a concrete prediction. If compliance
were implemented *through* moral representations, the two would be strongly
coupled, and any mechanism that produces compliance would overlap the moral
subspace; removing it would damage comprehension. Weak coupling predicts the
opposite: compliance is carried by a mechanism largely *outside* the moral
subspace, and should be removable without touching comprehension. Section 4.4
tests this directly.

The persona direction tells a consistent story (\Cref{fig:persona}, Appendix).
A linear persona probe is highly decodable at every stage (peak accuracy
${\sim}0.94$), but this reflects stylistic separability, not a moral mechanism:
the persona direction is nearly orthogonal to the moral subspace throughout, and
its mean $|\cos|$ to the foundations rises only from $0.076$ (base) to $0.085$
(Instruct). Post-training does not meaningfully couple persona to morality
either.

## 4.4 The refusal mechanism is geometrically separate from morality

We compute a refusal direction on the Instruct model using Heretic's exact
harmful/harmless prompt set and difference-of-means protocol (Section 3), and
measure its relationship to the moral subspace. The refusal direction is nearly
orthogonal to the six foundations: its projection into the moral subspace
captures only a $0.10$ fraction of its norm, and its mean $|\cos|$ to the
foundations is $0.06$ (per-foundation $0.04$--$0.07$; \Cref{fig:diss}, left).
Compliance is not routed through moral representations.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{dissociation}
\caption{Left: the refusal direction is nearly orthogonal to every moral
foundation ($|\cos| < 0.10$, below the moral-subspace projection fraction).
Right: ablating the refusal direction leaves comprehension untouched
(direction-transfer cosine and probe accuracy identical to Instruct) while the
refusal rate collapses from $0.25$ to $0.00$; moral judgment accuracy is
preserved.}
\label{fig:diss}
\end{figure}

Ablating this direction (Arditi-style uniform orthogonalization of the attention
out-projection and MLP down-projection at every layer) confirms the prediction
of Section 4.3. Comprehension is untouched: the ablated model keeps effective
dimensionality five, base-to-fresh direction cosine $0.749$ (versus $0.757$ for
Instruct), and probe accuracy $1.0$. Moral judgment is also intact: behavioral
accuracy on the moral-scenario benchmark is $0.73$, against $0.75$ for Instruct.
What collapses is refusal: on borderline and harmful requests the model's
refusal rate falls from $0.25$ (Instruct) to $0.00$ (ablated), and it answers
requests it previously refused. For example, a request to locate a person from
their phone number that Instruct declines ("I'm sorry, but I can't assist with
that") the ablated model now attempts (\Cref{fig:diss}, right).

## 4.5 The dissociation matrix

These results populate a comprehension$\times$compliance matrix
(\Cref{tab:twobytwo}). The low-comprehension row is empty: every OLMo-3 state,
from the earliest pre-training checkpoint to the ablated model, has full moral
decodability. All variation lives in the high-comprehension row, where
compliance moves independently of understanding. Instruct sits at high
comprehension with refusal intact; the ablated model sits at high comprehension
with refusal removed. There is no state that understands morality less; there
are states that comply less while understanding exactly as much.

\begin{table}[t]
\centering
\caption{The dissociation matrix, populated. Comprehension is measured by
effective dimensionality and probe accuracy; compliance by refusal rate on
harmful requests.}
\label{tab:twobytwo}
\begin{tabular}{lll}
\toprule
 & Low comprehension & High comprehension \\
\midrule
High compliance & --- & Instruct (eff-dim 5, refusal 0.25) \\
Low compliance  & --- & Ablated (eff-dim 5, refusal 0.00) \\
\bottomrule
\end{tabular}
\end{table}
