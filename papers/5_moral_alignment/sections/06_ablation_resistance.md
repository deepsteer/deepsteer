# 6. Toward Ablation-Resistant Alignment

The dissociation of Section 4.4 is a vulnerability. Because the refusal
mechanism is geometrically separate from moral comprehension, a single
difference-of-means ablation removes compliance while leaving the model's moral
understanding intact. The natural repair is to remove the separation: make
compliance *depend* on moral comprehension, so that the same ablation can no
longer strip one without damaging the other. This section asks whether a
training-time intervention can build that dependence. It cannot, at least not
after pre-training, and the reason it cannot sharpens the dissociation result
into a statement about where alignment's wiring is set.

## 6.1 Moral generation depends on the moral subspace, and the dependence grows with alignment

Decodability (Section 4.1) says morality is *present* in the representation; it
does not say the model *uses* it. We measure how much each pipeline state relies
on its moral subspace for generation with the dependency metric of Section 3.7:
the moral-specific increase in next-token cross-entropy when the six foundation
directions are projected out of the residual stream at every layer.

Dependency is positive at every state and grows through alignment
(\Cref{fig:dep}). It is flat at ${\sim}{+}0.011$ nats/token across the stage-3
pre-training checkpoints and the base model, then rises through post-training:
$+0.031$ at SFT, $+0.055$ at DPO, and $+0.063$ at Instruct, roughly a sixfold
increase, after which the eight RLVR substeps plateau. The rise is carried by
the moral arm (ablation cost on moral text grows from $0.22$ to $0.29$ nats
while the neutral arm stays near $0.21$). A per-state replication, ablating each
state's own freshly fitted directions rather than the fixed base directions,
shows the same post-training rise, so the trend is not an artifact of the
${\sim}40^{\circ}$ SFT rotation (Section 4.2).

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{dependency_trajectory}
\caption{Moral-subspace dependency (difference-in-differences of cross-entropy
under ablation of the six foundation directions) across the OLMo-3 pipeline. It
is flat through pre-training and the base, then grows ${\sim}6\times$ at
SFT/DPO/Instruct and plateaus through RLVR. Alignment makes moral generation
more reliant on the moral subspace, even though the compliance mechanism it adds
stays orthogonal to that subspace (Section 4.4).}
\label{fig:dep}
\end{figure}

So alignment does increase the model's functional use of its moral
representation. The effect is modest in absolute terms (the Instruct value of
$0.063$ nats is a ${\sim}6.5\%$ perplexity inflation), and, decisively, it
concerns *generation*: the refusal mechanism that Section 4.4 ablates lives
outside this subspace. Dependence and compliance grow in different places.

## 6.2 Ablation-resistance training

We attempt to deepen this dependency into a coupling with ablation-resistance
training (ART; Section 3.7): an auxiliary fine-tuning loss that rewards the model
when projecting the moral subspace out of its activations degrades its output,
bounded by a hinge so the objective cannot diverge. We fine-tune OLMo-3 base into
an instruct model with LoRA in two conditions identical except for the ART term,
a control ($\lambda=0$) and ART, then run the full Section 4 battery and the
Section 4.4 refusal ablation on both.

## 6.3 ART builds dependence, but in the wrong subspace

Where the ART gap is measured matters, and reveals the problem. Measured on the
mixed fine-tuning batch, the gap never moves (\Cref{fig:arttrain}, grey) and the
ART model is indistinguishable from the control on every metric: the general
content dominates the batch, and removing six of $4096$ directions barely
changes the loss on non-moral tokens, so there is almost no signal to optimize.

Measured on a concentrated pool of moral text, ART bites hard. The gap climbs
from $0.2$ to over $1.0$ nats and saturates the hinge within a few dozen steps
(\Cref{fig:arttrain}, blue), and it does so without damaging the model: probe
accuracy stays $1.0$, effective dimensionality stays $5$, and moral-judgment
accuracy matches the control ($0.65$). The intervention succeeds at its literal
objective. But the dependence it builds is the wrong kind, in two respects.

\begin{figure}[t]
\centering
\includegraphics[width=0.82\linewidth]{art_training_dynamics}
\caption{The ART gap, $\text{CE}^{\text{abl}}-\text{CE}$ on moral text, during
training. Measured on the diluted fine-tuning batch (grey) it stays near zero;
measured on a concentrated moral-text pool (blue) it climbs past the hinge
target and saturates. ART only engages once the gap is measured on moral
content.}
\label{fig:arttrain}
\end{figure}

First, the dependence is non-specific. The moral-ablation
difference-in-differences of Section 6.1 swings to $-0.62$ for the ART model,
against $-0.00$ for the control (\Cref{fig:artgrid}a): ablating the moral
subspace now hurts neutral text *more* than moral text. Rather than building
moral-specific reliance, the model has routed broad, general computation through
the six directions, and the effect generalizes to neutral text that the moral
pool never contained. The sign of the result is the tell: the objective rewarded
"make ablation hurt," and the cheapest solution makes it hurt everything.

Second, and decisively, ART leaves the refusal mechanism untouched. The refusal
direction's projection into the moral subspace is $0.069$ for the ART model and
$0.081$ for the control, both near the $0.10$ baseline of Section 4.4 and far
from any meaningful coupling (\Cref{fig:refproj}). ART poured dependence into the
moral directions, but the refusal direction stays where it was, orthogonal to
them.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{art_ablation_grid}
\caption{Refusal ablation damages the ART model no more than the control. (a)
Moral-subspace dependency: ART's value is large but Heretic-insensitive (almost
unchanged by refusal ablation). (b) Moral-judgment accuracy is preserved under
refusal ablation in both conditions. Probe accuracy ($1.0$) and effective
dimensionality ($5$) are identical across all four cells.}
\label{fig:artgrid}
\end{figure}

The consequence is that the Section 4.4 ablation still works exactly as before.
Applying the refusal ablation to the ART model leaves its moral-judgment accuracy
intact ($0.65 \to 0.69$) and its moral-subspace dependency essentially unchanged
(\Cref{fig:artgrid}): the ablation removes refusal without paying the moral cost
ART was supposed to impose. ART made the model strongly dependent on its moral
directions, but the threat ablates the refusal direction, and the two never met.

\begin{figure}[t]
\centering
\includegraphics[width=0.55\linewidth]{refusal_projection}
\caption{Fraction of the refusal direction lying in the moral subspace, control
versus ART. Both sit near the Section 4.4 baseline ($0.10$) and far below any
level ($\geq 0.40$) at which ablating refusal would damage comprehension. ART
does not move the refusal direction toward the moral subspace.}
\label{fig:refproj}
\end{figure}

## 6.4 Coupling cannot be installed after pre-training

The negative result has a clean mechanism. Refusal ablation targets the refusal
direction; ART builds dependence on the moral directions; the two are
orthogonal, so dependence on the moral subspace, however strong, provides no
defense against ablation of the refusal subspace. Making moral *generation*
subspace-dependent is the wrong target. To resist refusal ablation, the
compliance direction itself would have to lie inside the moral subspace, and
post-training does not put it there. The refusal feature forms during
post-training out of proto-features already present in the pre-trained
representation, and those proto-features are orthogonal to the moral subspace
that crystallized during pre-training (Sections 4.1--4.2). A fine-tuning loss can
pile dependence onto the moral subspace, but it acts after the compliance
feature's location is already fixed.

This points the intervention upstream. For the coupling to exist, the moral
subspace would have to *absorb* the proto-features that post-training later
recruits for compliance, so that the refusal direction, when it forms, lands
inside the moral subspace and its ablation incurs collateral damage to moral
comprehension. That is a pre-training-time intervention, and a sharper
experiment than applying the same loss earlier; we run it next.

We scope the claim to what we tested. The evidence here covers SFT-time,
LoRA-based ablation-resistance training that targets moral-generation
dependence. It does not rule out losses that act directly on the refusal
direction, full-parameter fine-tuning, or interventions during DPO or RLVR. What
it establishes is the mechanism behind the failure: post-training builds
dependence in a subspace orthogonal to the one the threat removes, and the
intervention does not relocate the compliance direction. On this evidence,
coupling moral comprehension to compliance tightly enough to survive ablation is
not a post-training adjustment but a pre-training one.

## 6.5 Forcing the coupling during continued pre-training

Section 6.4 points to a sharper experiment: make the moral subspace absorb the
proto-refusal feature *before* post-training recruits it, so that the refusal
direction, when it forms, lands inside that subspace. We run this as a
representational intervention on a late stage-3 pre-training checkpoint. The
regularizer is the geometric counterpart of the dependency loss of Section 6.2.
At each step we compute the harmful/harmless contrast
$\Delta = \overline{h}_{\text{harmful}} - \overline{h}_{\text{harmless}}$ from
last-token residuals (the Section 4.4 proto-refusal contrast, recomputed live from
the current model), project it onto the frozen six-foundation subspace $V$, and
penalize the orthogonal energy with a hinge so the objective cannot run past its
target. We add this to a language-modeling loss on general text (continued
pre-training, LoRA); a $\lambda{=}0$ control trains identically without the
coupling term.

The contrast moves into the subspace, cleanly. A rank-16 query/value adapter is
too weak (the projection reaches only $0.20$), matching the capacity limit of
Section 6.3, but a rank-64 adapter that also writes the MLP, including
\texttt{down\_proj}, drives the projection from the $0.10$ baseline to $0.50$
(\Cref{fig:forcedcoupling}a), past the $0.40$ threshold of Section 6.3. Subtracting the control isolates a
coupling-specific move of $+0.38$ (the control drifts to $0.12$). The move is
specific and the moral representation is intact: against the refusal contrast's
movement, an off-target neutral contrast and a care/harm contrast move at ratios
of $0.03$ and $0.05$ and a capability-risk-topic contrast at $0.19$, all far below
one; effective dimensionality of the foundation directions stays at $5$;
per-foundation probe accuracy stays at $0.99$; and the projection measured on
directions refit on the coupled model tracks the frozen-basis projection ($0.54$
versus $0.53$). The $0.50$ is a relocation of the contrast into a stable subspace,
not a rotation of the subspace itself. The absorption Section 6.4 calls for is
achievable: the proto-refusal contrast can be placed inside the moral subspace
during pre-training without disturbing moral comprehension.

## 6.6 Geometric coupling does not survive fine-tuning as functional resistance

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{forced_coupling}
\caption{Forced coupling is geometric, not functional. (a) The
continued-pre-training regularizer drives the refusal--moral-subspace projection
from the $0.10$ baseline to $0.50$, past the $0.40$ threshold (Section 6.5), but
supervised fine-tuning halves it to $0.26$, below the threshold, while the control
stays near baseline; pre-SFT is the proto-refusal contrast, post-SFT the fitted
refusal direction Heretic ablation targets. (b) Projecting the moral subspace out
of the residual stream at inference raises the coupled model's refusal rate
($0.79 \to 0.93$) instead of removing it, with the control flat: the subspace
carries comprehension, not compliance.}
\label{fig:forcedcoupling}
\end{figure}

A relocated contrast is not yet a coupled mechanism. We fine-tune the coupled
checkpoint into an instruct model with plain supervised fine-tuning (the Section
6.2 recipe at $\lambda{=}0$, on the same data), fine-tune the control checkpoint
identically, and ask the two questions Section 6 is about: does the refusal
direction the model builds during fine-tuning land in the moral subspace, and
does ablating that subspace damage refusal.

Both fail. The two arms become working refusers (refusal rate $0.79$ coupled,
$0.83$ control on the harmful set), so the regularizer does not break trainability.
But fine-tuning halves the coupling: the refusal direction extracted from the
post-fine-tuning coupled model, the direction Heretic ablation targets, projects
$0.26$ onto the moral subspace against $0.09$ for the control. The coupling partly
survives, a factor of roughly $2.7$ over control, but it falls from the
pre-fine-tuning $0.50$ and lands below the $0.40$ threshold. The residual overlap
is also not load-bearing. Projecting the moral subspace out of the residual stream
at inference does not reduce the coupled model's refusal; it raises it, from
$0.79$ to $0.93$ (\Cref{fig:forcedcoupling}b), while leaving the control near flat
($0.83$ to $0.79$). Ablating
the subspace the coupling targeted makes the model refuse *more*, not less.

The increase is not an artifact of ablation degrading generation. The ablated
completions are coherent, well-formed refusals, and the effect is specific to
harmful prompts: on sixty benign requests phrased with alarming verbs (``kill a
frozen process'', ``blow up balloons for a party''), the coupled model's
over-refusal does not rise under ablation, it falls, from one false positive in
sixty to zero, while harmful refusal rises. A brittleness account, in which
removing the subspace destabilizes the model toward blanket refusal, predicts the
opposite. Refusal calibration is otherwise intact: the coupled model's lone benign
over-refusal, a request to break into one's own locked car, against zero for the
control, is within noise.

The reading is the dissociation of Section 4.4, holding firm. The moral subspace
carries comprehension, not compliance: ablating it removes the model's purchase on
the request's content, and a model that cannot engage the content defaults to
refusing. Forcing the proto-refusal contrast geometrically into that subspace,
even to a projection of $0.50$ before fine-tuning, does not make the refusal
mechanism route through it. Fine-tuning builds compliance where it built it before,
in a subspace the moral one does not control.

## 6.7 What forced coupling settles, and what it opens

The two halves constrain each other. The geometric obstacle of Sections 6.3 and
6.4, that no fine-tuning loss moved the refusal direction toward the moral
subspace, is not fundamental: a continued-pre-training regularizer moves it
cleanly to $0.50$. What does not follow is function. The relocated contrast
degrades through fine-tuning and, where it survives, does not mediate refusal. The
comprehension/compliance dissociation withstands a direct, constructed attempt to
break it, which is stronger evidence than the observational finding of Section 4.4
alone: we tried to wire comprehension and compliance together and the model pulled
them apart.

This scopes the negative precisely and leaves one question open. We targeted the
six-foundation moral subspace because it is the comprehension substrate this paper
characterizes. Whether some other moral basis is load-bearing for refusal, a
data-driven moral component or an explicit harm-avoidance direction, or whether
refusal in these models is a non-moral compliance mechanism with no moral basis to
couple to, our experiments do not separate. From the evidence here, ``the
moral-foundations subspace is the wrong target'' and ``refusal is not morally
mediated'' look the same. We can at least exclude the persona route. The refusal
direction is no more persona-shaped than moral-shaped: decomposed onto the moral
subspace and an assistant-voice persona axis built in the style of the Assistant
Axis of \citet{lu2026assistant}, it is $99\%$ residual ($0.006$ in the moral
subspace, $0.0005$ persona-unique), with cosine $0.005$ to that axis, matching the
near-orthogonality to the toxic-voice persona direction of \citet{wang2025persona}.
Whatever carries refusal, it is not the named persona axis the literature would
nominate. Distinguishing the two remaining readings is the natural next step, and it
would turn the present negative, that this coupling does not take, into a positive
statement about where refusal lives.
