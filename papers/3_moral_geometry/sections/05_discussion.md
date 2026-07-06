# 5. Discussion

## 5.1 Integration as the default geometric mode

The central finding, that foundation probe directions exhibit
integration rather than collapse or isolation, has a
straightforward interpretation: language models develop structured
moral representations from distributional statistics alone. The
training corpus does not label texts with MFT foundations, yet the
model learns representations in which moral foundations occupy
distinct directions that share a positive common component (mean
cosine $\approx 0.22$ at peak separation, $\approx 0.26$ at stable
mid-network layers). This is genuine multi-dimensional moral
structure, not a single "moral salience" detector. The shared component
is moral-specific relative to a matched non-moral concept battery built
identically to the foundations: that battery gives a mean cosine of
0.013 against the moral 0.26 (paired $\Delta = 0.223$, CI
$[0.202, 0.244]$, excluding 0; §4.2), and the estimator's
$[1+(k-1)\bar{c}]/k$ PC1 identity is confirmed to $<0.01$ across all
three calibration constructions. The one control we do not run, and so
the one claim we do not make, is whether this axis is specifically moral
rather than generic affective salience (§5.6).

The fact that effective dimensionality is 5 (near-maximal for 6
directions) throughout the network rules out the possibility that
the model has learned a single "this text is moral" feature and
then adds minor perturbations per foundation. The foundation
directions are geometrically distinct objects that happen to share
a common moral-salience component.

This holds across scale, architecture, and dataset. Effective
dimensionality is 5 at every layer of OLMo-2 1B, OLMo-2 7B, and
OLMoE-1B-7B, and on the independently constructed Moral Foundations
Vignettes (§4.14, §4.16). Integration is the default geometric mode
for moral representations in these models (all from Ai2's OLMo family;
§5.6), not a property of one scale, architecture, or probing dataset.

**Dataset sensitivity.** The mean cosine similarity is sensitive to
neutral-pair quality: neutrals that inadvertently carry moral content
inflate the shared moral-salience component, producing higher cosine
values. The quality-gated dataset used here (§3.6) minimizes this
inflation, and we interpret the ${\approx}0.22$--$0.27$ range as a lower
bound on the true integration signal. All qualitative conclusions
(integration signature, effective dimensionality of 5, absence of MFT
dendrogram structure) are robust to dataset construction choices,
but the quantitative sensitivity highlights the importance of
neutral-pair quality for geometric analyses. The replication on the
Moral Foundations Vignettes (§4.16), where directions estimated from
independently authored stimuli reproduce the 5-dimensional integration
geometry, shows that the qualitative signature does not depend on our
dataset.

**No evidence of MFT group structure.** The inter-framework
structure that emerges shows no alignment with MFT's predicted
individualizing/binding distinction: hierarchical clustering does
not recover this partition at any layer, and the permutation test
is non-significant throughout. This is an underpowered null, not a
demonstrated absence: the test enumerates only 20 partitions, so its
smallest achievable $p$ is $0.05$ (observed minimum $0.32$; §4.3), we
state no minimum detectable within/between gap, and we did not run a
positive control confirming the test fires on planted group structure,
so a small individualizing/binding effect cannot be excluded. Instead,
the most consistent clustering feature is a care--sanctity pairing that
crosses MFT groups. Both care and sanctity involve protection (of persons from
harm, of sacred things from degradation), sharing distributional
signatures that the model detects. The care--sanctity
pairing is itself robust to dataset construction choices: it persists
across different neutral-pair generation methods and quality
thresholds, arguing against a dataset artifact explanation. The
structure the model does form is thus empirically grounded in corpus
statistics and is not detectably aligned with the a priori
individualizing/binding grouping from moral psychology on this dataset,
consistent with genuine structure learning (the model discovers which
moral concepts are distributionally related) rather than surface
keyword matching.

A data-driven test sharpens this point. Clustering the moral
activations themselves, without foundation labels, recovers the
foundations only weakly (adjusted mutual information 0.03 at best;
§4.15). Even the model's own unsupervised grouping of moral content
is not MFT-structured, which is the expected consequence of
integration: the foundations are linearly decodable but are not the
dominant axis of variation, so unsupervised methods find the shared
moral-salience structure instead.

## 5.2 Per-foundation fragility is not separable

Within an architecture, the moral foundations are not reliably
distinguishable by fragility. With 10 noise seeds and the cap-at-max
convention (§4.7), per-foundation $\sigma^*$ values carry wide,
overlapping bootstrap CIs in all three models, and no foundation is
reliably most or least robust. Sanctity sits mid-pack everywhere
(dense 1B $5.50$, MoE $2.33$, dense 7B $10.0$, fourth of six), and the
binding-vs-individualizing group difference is not significant in any
model (dense 1B $p = 0.40$, MoE $p = 0.70$, dense 7B $p = 0.20$) and
reverses sign between dense and MoE. Per-foundation $\sigma^*$ is
sensitive to single-draw noise, so we make no per-foundation or
MFT-group fragility claim.

The fragility result that does hold is between architectures, not
between foundations: every foundation is more fragile in MoE than in
dense (a ${\sim}2.3\times$ per-foundation gap, the same direction as the
pooled $4.2\times$ of \citet{reblitzrichardson2026dilution}), and dense
robustness rises with scale from 1B to 7B. Output dilution suppresses
moral encoding uniformly across foundations rather than singling any
one out.

**What critical noise does and does not measure.** Three of our raw
$\sigma^*$ comparisons turned out to be driven by the same thing once
controlled: activation-scale differences between the conditions being
compared. The per-foundation ordering (§4.7), the cross-architecture
gap above, and the complexity gradient (§4.11) each shrink or vanish
under RMS normalization, and the companion dense-model study reports the
same for its layer-depth gradient
(\citealp{reblitzrichardson2026fragility}, §4.4). These are one
confound, not three. They sharpen what the metric is for: raw $\sigma^*$
measures *practical perturbation sensitivity*, the absolute noise a
representation tolerates, which is real and useful, but it does not
measure encoding robustness independent of scale when the compared
conditions differ in activation scale, as they do across layers,
registers (declarative vs.\ narrative), architectures, and probe
complexity. RMS-normalized $\sigma^*$ is the right tool for those
cross-condition claims; raw $\sigma^*$ is valid only where scale is
controlled by construction (within-layer, matched-stimulus contrasts).
Establishing that partition is a methodological contribution in its own
right.

## 5.3 Framework geometry stabilizes before accuracy saturates

The trajectory analysis shows a temporal dissociation: framework
geometry (mean cosine similarity between foundation directions)
enters its integration regime within the first few thousand steps
(peaking near step 5000), while probing accuracy continues climbing
through step 10000 and beyond. This extends the "accuracy saturates
but fragility resolves" finding of
\citet{reblitzrichardson2026fragility} to a third metric: the
*structure* of moral representations is laid down before the
*strength* of those representations finishes developing.

This is consistent with a two-phase account of moral representation
learning. In the first phase (steps 0--5000) the model discovers the
geometric layout: the foundation directions move from near-orthogonal
into the integration band as a shared moral-salience component forms.
In the second phase (steps 5000+) the layout is not frozen but
gradually refined: the mean cosine declines slowly (0.34 → 0.29,
${\sim}13\%$) while accuracy is flat, so the foundations keep
differentiating from one another after they are individually
well-separated. The structure is set early and sharpened slowly,
not established once and held fixed.

Effective dimensionality is 5 from step 0 onward, indicating that
even random initialization produces probe directions that span 5
dimensions. This is expected: random unit vectors in
$\mathbb{R}^{2048}$ are nearly orthogonal with high probability.
The informative signal is not dimensionality but cosine similarity,
which moves from $\approx 0$ (random) into the integration band
(${\sim}0.3$) over the first few thousand steps and then drifts down.

## 5.4 Partial compositionality of moral dilemmas

When two moral foundations conflict in a dilemma scenario, the
model's representation is partially compositional: the dilemma probe
direction has more overlap with the 2D subspace of its own component
foundations (mean peak membership $S = 0.118$) than with the
mismatched-pair baseline of foundation pairs it shares no component
with ($0.044$, a ${\sim}2.7\times$ margin that holds at every layer),
yet ${\sim}88$\% of the dilemma direction lies outside its component
subspace.

This partial compositionality has two complementary
interpretations. First, the model recognizes moral dilemmas as
involving their component foundations (the matched-over-mismatched
margin is not an artifact), as confirmed by the shared-component
structure: dilemma pairs that share a foundation have higher cosine
similarity ($\Delta = 0.076$ at layer 13, exact permutation
$p = 0.0001$). Second, the model represents something beyond the sum
of parts: the ${\sim}90$\% residual captures conflict-specific
features (tension, trade-off framing, or contextual modulation)
that single-foundation probes do not isolate.

The near-balanced component loading ($\bar{\alpha} = 0.486$) is
notable. If dilemma representations were dominated by one
foundation (e.g., always prioritizing care over authority), we
would expect strongly asymmetric projections. Instead, both
conflicting foundations contribute roughly equally to the
within-subspace component, consistent with the model encoding the
*conflict itself* rather than a pre-resolved moral judgment. This
is an "understanding before choice" mode of representation: the
model maintains both competing moral claims in tension rather than
collapsing to a resolution, suggesting that moral comprehension
(the capacity to represent the structure of ethical disagreement)
may precede and be separable from moral judgment. This pattern is
scale-stable: OLMo-2 7B reproduces the partial-compositionality
profile (9.1\% subspace membership, 0.95 residual, 0.46 component
balance; §4.9), so the conflict-as-tension representation is not an
artifact of the smaller model's capacity.

A raw complexity--fragility ordering (single-foundation
$\sigma^* = 5.02$ $>$ pooled dilemma $3.81$ $>$ per-type dilemma
$3.55$) might suggest that representational complexity trades off
against robustness. It does not survive scale normalization: with
noise scaled to each layer's activation RMS the single-foundation and
pooled-dilemma values converge (both at the grid maximum) and the
per-type value barely separates (§4.11). The raw ordering reflects
register-dependent activation scale, not differential encoding
robustness, so we make no complexity--robustness claim. The
compositionality evidence in §4.9--4.10 (subspace membership,
shared-component geometry, balanced loading) is geometric, not
fragility-based, and is unaffected.

## 5.5 Register sensitivity: directions transfer, thresholds do not

Foundation-specific probes trained on declarative minimal pairs do
not fully generalize to narrative dilemma text *as classifiers*. In
the dilemma verification experiment, authority and loyalty probes
showed near-chance transfer (Youden's $J < 0.2$), while care and
fairness probes transferred well. This asymmetry is not a
model-capacity issue: testing on OLMo-2 7B (32 layers, 4096 hidden
dim) yielded comparable transfer failure (54.0\% vs.\ 61.3\% for
the 1B model).

**Directions vs.\ thresholds.** The probe engineering analysis
(§4.13) resolves this concern by separating two components of
cross-register transfer: the *direction* (probe weight vector) and
the *threshold* (bias term). When evaluated by pair accuracy (the
fraction of pairs where the direction projects the moral text higher
than the neutral text, requiring no threshold), both probe-weight
and mean-difference directions achieve $>90$\% mean pair accuracy on
narrative dilemma text (probe-weight $>95$\% for all foundations;
mean-difference drops to ${\sim}91$\% for authority/subversion).
The Youden's $J$ failure is therefore a threshold
miscalibration effect: the absolute projection scale shifts between
registers, invalidating the fixed decision boundary learned on
declarative text. The directional structure itself, the
subspace in which moral content is encoded, transfers robustly.

This distinction matters for the geometric findings. Cosine
similarity, effective dimensionality, and dendrogram clustering
depend only on direction vectors, not on classification thresholds.
Since the directions transfer across registers, the geometric
findings are not register-bound.

**Threshold vs. direction transfer.** The register sensitivity is
specific to threshold transfer, not direction transfer. Both
individualizing and binding directions transfer across registers with
high pair accuracy ($>90$\%); what shifts across register is the
decision threshold, not the direction itself. Authority and loyalty
show the largest threshold gaps (Appendix B.2), so their register
sensitivity reflects a calibration shift rather than a change in the
underlying moral direction.

**Implications for the geometric findings.** The 21-direction
dendrogram analysis (§4.9) gives direct evidence that register
features drive part of the representation geometry: projecting all
directions into the 5D moral subspace dissolves the categorical
foundation/dilemma separation, confirming that the separation is
carried by extra-moral (register) features. This is consistent with
the threshold miscalibration account: foundation and dilemma
directions occupy the same moral subspace (their projections
overlap), but differ in extra-moral dimensions that carry register
information and shift the activation scale.

## 5.6 Limitations

**Small probing dataset.** The 32 training pairs per foundation
are sufficient for classification (near-perfect accuracy) but
limit the precision of direction estimation. Bootstrap analysis
(§4.6) confirms that directions at layers 0--5 are borderline
unstable, and all geometric claims are qualified to the
bootstrap-stable core, layers 6--15 (the lone exception being
authority at layer 8, 0.792).
A larger dataset would tighten direction estimates, but the current
dataset is deliberately minimal to demonstrate that structured
geometry is recoverable even from small samples.

**Permutation test power.** With 6 foundations divided into two
groups of 3, the permutation space contains only 20 unique partitions,
so the smallest achievable $p$ is $1/20 = 0.05$, reached only if the
observed split is the single most extreme partition; the observed
minimum is $0.32$ (§4.3). We did not compute a minimum detectable
within/between gap or run a positive control verifying that the test
fires on planted group structure. Our MFT-group result is therefore "no
evidence of individualizing/binding organization," not a demonstrated
absence: a small group effect cannot be excluded. The absence of
MFT-aligned dendrograms at any layer is consistent with this null but
does not turn it into a positive claim.

**One model family.** The three configurations span scale (1B and 7B
dense) and architecture (dense and MoE), and the integration signature
and MFT-mismatch hold across all of them and on the independently
constructed MFV stimuli (§4.14--4.16). The models are still all from
Ai2's OLMo family, trained on comparable corpora, so generalization to
models trained on substantively different data mixtures remains open.
The architectural and scale comparisons are well-controlled because
the models share training data, at the cost of corpus diversity.

**Linear probes.** The entire analysis assumes that moral
foundations are encoded as linear directions. If some foundations
are encoded nonlinearly (e.g., as curved manifolds or distributed
across multiple directions), the cosine similarity analysis
would understate the true geometric richness. The near-perfect
accuracy of linear probes suggests that linear decoding captures
the dominant signal, but does not rule out additional nonlinear
structure.

**Affective vs.\ moral salience.** The shared component that marks
integration is moral-specific relative to a matched non-moral battery
spanning affective, syntactic, stylistic, and topical concepts
(sentiment, register, grammaticality, tense, number, topic; §4.2), each
of which decodes at 1.00 peak accuracy while giving a near-zero pairwise
cosine (0.013 vs.\ the moral 0.26). That battery does not isolate
whether the shared moral axis is specifically *moral* or a generic
*evaluative/affective* salience common to emotionally charged text,
moral statements included. The cheapest discriminating control is a
matched-twin non-moral valence/affective battery (positive vs.\ negative
affect against matched neutrals, built exactly as the foundation probes
are); we did not run it here. We therefore claim moral-specificity
relative to a matched non-moral battery and leave affective-vs-moral as
the open residual.

**Causal status.** Probe directions are read off the representation:
on their own they identify *where* foundation information is readable,
not whether that information is *used* during generation. Our 7B
foundation directions provide preliminary, uncontrolled causal checks:
ablating a foundation's direction selectively perturbs that
foundation's continuations, and injecting it produces a dose-response
shift (Appendix \ref{app:causal}). These checks carry no
random-direction or channel-matched null, so they do not yet separate
foundation-specific action from the generic effect of projecting out
(or amplifying) any stable direction, and the largest steering effects
occur at off-distribution injection strengths ($\alpha = 20$). Full
causal localization is left to future work; the geometry reported here
is descriptive of the representation, and we treat this causal evidence
as preliminary rather than as establishing that the directions are
functionally implicated.
