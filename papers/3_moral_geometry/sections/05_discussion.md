# 5. Discussion

## 5.1 Integration as the default geometric mode

The central finding --- that foundation probe directions exhibit
integration rather than collapse or isolation --- has a
straightforward interpretation: language models develop structured
moral representations from distributional statistics alone. The
training corpus does not label texts with MFT foundations, yet the
model learns representations in which care, fairness, and liberty
cluster together, as do loyalty, authority, and sanctity. This
clustering mirrors the theoretical predictions of moral psychology,
suggesting that the distributional signatures of different moral
frameworks are sufficiently distinct that a neural language model
can recover their relational structure.

The fact that effective dimensionality is 5 (near-maximal for 6
directions) throughout the network rules out the possibility that
the model has learned a single "this text is moral" feature and
then adds minor perturbations per foundation. The foundation
directions are geometrically distinct objects that happen to share
a common moral-salience component.

**The care--sanctity anomaly.** One detail complicates the clean
MFT narrative: sanctity/degradation (a binding foundation) has
consistently higher cosine similarity with care/harm (an
individualizing foundation) than care has with loyalty (0.321 vs.
0.294 at layer 0; 0.452 vs. 0.347 at layer 15; Appendix C). This
violates the MFT prediction that within-group similarities should
exceed between-group similarities uniformly. The likely explanation
is semantic: both care and sanctity involve purity and the
prevention of harm or degradation, sharing distributional
signatures that the model detects. This suggests the model's moral
taxonomy is empirically grounded in corpus statistics rather than
theoretically aligned with MFT --- which is itself evidence of
genuine moral structure learning rather than surface keyword
matching. The model discovers which moral concepts are
distributionally related, and these relationships mostly but not
perfectly mirror the a priori MFT grouping.

## 5.2 The fragility reversal

The cross-architectural fragility reversal --- binding foundations
most robust in dense models, least robust in MoE --- is the most
unexpected finding. It connects two previously independent
observations: the MFT individualizing/binding distinction
\citep{graham2013mft} and the output dilution effect
\citep{reblitzrichardson2026dilution}.

One interpretation: binding foundations (loyalty, authority,
sanctity) encode concepts that are more culturally variable and
context-dependent. Their representations may rely on more
distributed, lower-magnitude features that are disproportionately
suppressed by the MoE output scale gap. Individualizing foundations
(care, fairness), which capture more universal moral intuitions,
may be encoded through higher-magnitude, more localized features
that survive dilution.

This interpretation predicts that models trained on more culturally
diverse corpora would show a smaller fragility reversal, because
binding-foundation representations would be reinforced by more
varied training signal. We cannot test this prediction with the
present models.

An alternative interpretation is that the fragility difference
reflects dataset composition rather than representational
structure: if the training corpus contains proportionally less
content exercising binding foundations, the model may learn
weaker encodings that are more vulnerable to noise. Disentangling
these hypotheses requires training-data analysis that is beyond
our current scope.

## 5.3 Framework geometry stabilizes before accuracy saturates

The trajectory analysis reveals a temporal dissociation: framework
geometry (mean cosine similarity between foundation directions)
reaches its mature value by approximately step 2000--3000, while
probing accuracy continues climbing through step 10000 and beyond.
This extends the "accuracy saturates but fragility resolves"
finding of \citet{reblitzrichardson2026fragility} to a third
metric: the *structure* of moral representations stabilizes before
the *strength* of moral representations finishes developing.

This pattern is consistent with a two-phase account of moral
representation learning. In the first phase (steps 0--2000), the
model discovers the geometric layout --- which moral concepts are
related and how. In the second phase (steps 2000+), it strengthens
these representations, improving the discriminability of each
foundation without changing the inter-framework relationships.

Effective dimensionality is 5 from step 0 onward, indicating that
even random initialization produces probe directions that span 5
dimensions. This is expected: random unit vectors in
$\mathbb{R}^{2048}$ are nearly orthogonal with high probability.
The informative signal is not dimensionality but cosine similarity,
which jumps from $\approx 0$ (random) to $\approx 0.4$ (mature
structure) during the first 2000 training steps.

## 5.4 Partial compositionality of moral dilemmas

When two moral foundations conflict in a dilemma scenario, the
model's representation is partially compositional: the dilemma
probe direction shares statistically significant overlap with the
2D subspace spanned by its component foundation directions (mean
membership $S = 0.099$, 100$\times$ the null baseline of 0.001),
yet ${\sim}90$\% of the dilemma direction lies outside this subspace.

This partial compositionality has two complementary
interpretations. First, the model recognizes moral dilemmas as
involving their component foundations --- the ${\sim}10$\% subspace
membership is not an artifact, as confirmed by permutation testing
and by the shared-component structure (dilemma pairs that share a
foundation have higher cosine similarity, $\Delta = 0.074$ at
layer 13). Second, the model represents something beyond the sum
of parts: the ${\sim}90$\% residual captures conflict-specific
features --- tension, trade-off framing, or contextual modulation
--- that single-foundation probes do not isolate.

The near-balanced component loading ($\bar{\alpha} = 0.486$) is
notable. If dilemma representations were dominated by one
foundation (e.g., always prioritizing care over authority), we
would expect strongly asymmetric projections. Instead, both
conflicting foundations contribute roughly equally to the
within-subspace component, consistent with the model encoding the
*conflict itself* rather than a pre-resolved moral judgment.

The complexity--fragility gradient (single-foundation
$\sigma^* = 4.72$ $>$ pooled dilemma $\sigma^* = 3.12$ $>$
per-type dilemma $\sigma^* = 2.90$) shows that representational
complexity trades off against robustness. Dilemma representations
sit in a higher-dimensional space and are correspondingly more
fragile under Gaussian noise. This is consistent with the
residual component encoding subtle contextual features that
require higher precision to maintain.

## 5.5 Register sensitivity as a methodological and theoretical issue

Foundation-specific probes trained on declarative minimal pairs do
not fully generalize to narrative dilemma text. In the dilemma
verification experiment, authority and loyalty probes showed
near-chance transfer (Youden's $J < 0.2$), while care and fairness
probes transferred well. This asymmetry is not a model-capacity
issue: testing on OLMo-2 7B (32 layers, 4096 hidden dim) yielded
comparable transfer failure (54.0\% vs.\ 61.3\% for the 1B model).

**Connection to the fragility reversal.** The register sensitivity
pattern parallels the fragility reversal (§5.2): the same binding
foundations (authority, loyalty) that are most fragile under MoE
output dilution are also most sensitive to text register. This
raises the possibility that the fragility reversal is partially a
register sensitivity effect, not purely output dilution. If binding
foundation probes learn register-entangled directions, their
apparent fragility under noise may reflect the fragility of
register features rather than of moral content per se. We note
this as an open question rather than a conclusion, as the two
effects (noise fragility and register transfer) operate through
different mechanisms.

**Theoretical interpretation.** Binding foundations may be
inherently more register-sensitive because their moral content is
more context-dependent. Loyalty in a declarative sentence
("Loyalty to one's group is a core virtue") and loyalty in a
narrative dilemma ("She discovered her colleague's fraud but
hesitated to report it, torn between honesty and team loyalty")
activate different aspects of the loyalty concept. If binding
foundations are distributed across more varied surface
realizations, a probe trained on one register will capture a
narrower slice of the concept. Individualizing foundations (care,
fairness), which express more universal and context-stable moral
intuitions, may have more register-invariant distributional
signatures. This account, if correct, means the differential
register sensitivity is a genuine property of moral representation,
not merely a methodological limitation.

**Implications for the geometric findings.** The 21-direction
dendrogram analysis (§4.9) provides direct evidence that register
features drive part of the representation geometry: projecting all
directions into the 5D moral subspace dissolves the categorical
foundation/dilemma separation, confirming that the separation is
carried by extra-moral (register) features. However, the core
geometric findings --- integration, effective dimensionality = 5,
MFT clustering --- are properties of the foundation directions
alone, trained and evaluated within a single register. Register
sensitivity affects the *dilemma extension* results more than the
*framework geometry* results. Mixed-register probe training and
unsupervised direction extraction methods (CCS, mean-difference
directions) are priorities for future work to establish which
geometric properties are robust across registers.

## 5.6 Limitations

**Small probing dataset.** The 32 training pairs per foundation
are sufficient for classification (near-perfect accuracy) but
limit the precision of direction estimation. Bootstrap analysis
(§4.6) confirms that directions at layers 0--4 are borderline
unstable, and all geometric claims are qualified to layers 5--14.
A larger dataset would tighten direction estimates, but the current
dataset is deliberately minimal to demonstrate that structured
geometry is recoverable even from small samples.

**Permutation test power.** With 6 foundations divided into two
groups of 3, the permutation space contains only 20 unique
partitions. The permutation test for the individualizing/binding
distinction cannot reach $p < 0.05$ unless the effect is very
large. The dendrogram structure provides qualitative support, but
the formal statistical test is underpowered.

**Two models, one lab.** Both OLMo-2 and OLMoE are from Ai2 and
trained on comparable corpora. The geometric findings may not
generalize to models trained on substantively different data
mixtures. The architectural comparison (dense vs.\ MoE) is
well-controlled because the models share training data, but this
comes at the cost of corpus diversity.

**Linear probes.** The entire analysis assumes that moral
foundations are encoded as linear directions. If some foundations
are encoded nonlinearly (e.g., as curved manifolds or distributed
across multiple directions), the cosine similarity analysis
would understate the true geometric richness. The near-perfect
accuracy of linear probes suggests that linear decoding captures
the dominant signal, but does not rule out additional nonlinear
structure.

**Register sensitivity.** As discussed in §5.5, linear probes
entangle moral content with text-register features. The geometric
findings are established within a single register (declarative
minimal pairs) and may shift under mixed-register training.

**No causal evidence.** Probe directions are correlational:
they identify *where* foundation information is readable, not
whether that information is *used* by the model during
generation. Causal methods (activation patching, ablation studies)
are needed to establish whether the geometric structure we observe
is functionally relevant.
