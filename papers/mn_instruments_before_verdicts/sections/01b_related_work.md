# 2. Related work {#related-work}

Each failure mode in this note has roots in an established line of work. We state the connection
and what this note adds, so the reader can judge which cautions are new instruments and which are
new framings of known ones.

**Outlier dimensions and massive activations.** That a few residual-stream dimensions carry a
disproportionate share of variance, and that they distort geometric measurements, is well
documented: rogue dimensions obscure representational quality under cosine similarity
\citep{timkey2021rogue}, a small set of outlier dimensions disrupts transformers when removed
\citep{kovaleva2021bert}, and emergent outlier features appear at scale
\citep{dettmers2022int8, sun2024massive}, related to the attention-sink phenomenon
\citep{xiao2023efficient}. The standard response, per-dimension standardization, is not new
here. What we add is that the *covariance-matched null* built for projection-fraction tests
silently degenerates in these families: because the null draws random directions from the
outlier-dominated covariance, every direction projects like a typical one, so the test loses
discriminating power rather than returning an obvious artifact. Naming this instrument-level
failure, and the band-below-null tell that catches it, is the contribution.

**Norm handling in attribution.** Reading a direction's per-head or per-layer contribution off
the residual stream requires accounting for the block's normalization; the logit-lens and
tuned-lens line makes the sensitivity to that choice explicit \citep{belrose2023tunedlens}.
Folding the block RMSNorm gain into per-head attribution is itself standard interpretability
tooling \citep{elhage2021framework, nanda2022transformerlens}, so the fold is not the contribution.
What we add is the specific, quantified failure for reordered (post-block) normalization as used
by the OLMo-2/3 family: a naive per-head decomposition that skips the block norm overshoots the
true residual write about threefold, which we quantify and catch with a two-sided reconstruction
gate (a one-sided floor misses overshoot).

**Activation-patching methodology.** That patching verdicts depend on metric, corruption, and
layer choice is the subject of best-practice work \citep{zhang2024patching}. Our stimulus and
depth sections are instances. An interchange readout run at a saturated outcome yields
sign-chaotic deltas that mimic instrument failure, and a read-from verdict measured past the
layer where the model has already committed can be a read-layer artifact. We frame both for
reasoning models and supply the orthogonal-cell certificate and the commitment-relative depth as
the checks.

**Interpretability illusions.** The general hazard, that a measurement can behave as if it found
a mechanism when it has not, is the illusion literature: subspace activation patching can route
through unintended pathways \citep{makelov2023subspace}, and individual-unit interpretations can
be spurious \citep{bolukbasi2021illusion}. This note's thesis, that a broken instrument reads as
a finding, is in that spirit. Our addition is the position-validity check, where a
positive-control band that falls below the covariance null marks the measurement position itself
as uninformative, and the integration of these checks into a pre-registration and verification
protocol.

The scientific results that exercise these instruments are reported in a companion flagship study
(in preparation); this note is the portable methodology, and its evidence is the model panel and
single program it was derived on (see the limitations).
