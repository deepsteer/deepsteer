---
name: construct-audit
description: >-
  Type every direction, subspace, and probe before comparing any two of them. Use this skill
  whenever creating, extracting, comparing, or interpreting directions/subspaces — any
  cosine, projection fraction, principal angle, CKA, steering, ablation, or diff-of-means;
  any claim that "X is orthogonal / unrelated / decoupled from Y"; any measurement that
  varies along positions, layers, or pipeline stages. Fires on: "diff-of-means", "extract a
  direction", "projection", "cosine", "orthogonal", "subspace", "probe", "steering vector",
  "ablation". Enforces a metadata type block per direction, the content×decision comparison
  design matrix, confound genealogy of every contrast, normalization audits, construct
  constancy along swept axes, and the rule that geometric non-overlap NEVER establishes
  functional disconnection without at least one causal cell.
---

# Construct Audit

**Core principle: a direction is defined by the contrast that built it, not by the name you
gave it.** Two directions can only be compared once both are typed, and a relationship can
only be declared absent once the right cells of the design matrix have been measured.

## Required artifact: the type block

Every saved direction/subspace artifact carries (in its .npz metadata or a sidecar JSON):

```
contrast_semantics: stimulus/content | behavior/decision | outcome-conditioned
source_dataset:     name + commit/version
position_class:     t_inst | pre-decision gate | in-trace | post-answer | pooled
format:             raw text | chat template | reasoning template
layer(s):           headline + band
model:              id + revision
n_pairs:            per cell
known_covariates:   what else this contrast varies (see genealogy)
participation_ratio: PR of the position's activation sample (PR < 30 → position-invalid
                    flag at extraction time, not discovery at verdict time)
outcome_variable:   for any object headed into a causal cell — which behavioral/readout
                    variable it is built to move
extraction_commit:  repo SHA
```

No comparison, verdict, or figure may use an untyped direction.

## The design matrix rule

Before claiming relation R(X, Y) is absent, tabulate which cells you actually measured:

|  | content | decision |
|---|---|---|
| **content** | ? | ? |
| **decision** | ? | ? |

A content-vs-decision null **alone is uninterpretable**: read/write separation is the
*expected* organization of a residual stream (a shared bus under interference pressure pushes
variables that must not collide into disjoint coordinates). To claim disconnection you need
the diagonal cells too — content-vs-content (do the two content constructs relate?) and
decision-vs-decision (do the two decisions share machinery?). In this program, seven papers
of content-vs-decision nulls left both diagonals unmeasured; filling them (moral-family
mutual projections; the judgment-decision direction) is what made the nulls interpretable.

## Same-layer fallacy + the reflexivity rule

Same-layer, same-position geometry cannot see a circuit that *reads* content from elsewhere
through attention/MLP weights. "Shares coordinates" ≠ "reads from."

**Reflexivity rule:** your own earlier findings apply to your own later methods. This program
established a ~10-layer storage-vs-usage divergence in Paper 1, then spent Papers 5–7
comparing directions at matched layers as if decodability location were usage location. When
you discover a property of the representation, immediately ask: "which of my standing methods
does this property invalidate or qualify?" — and write the answer into the method docs.

**Causal-cell requirement.** "Disconnected" requires at least one of: targeted ablation with
behavioral readout (+ matched-random and named-reference controls), activation steering,
cross-layer Jacobian/attribution from the content's usage layer to the decision readout, or
interchange interventions. Note the asymmetry that matters: an intervention on X that moves
behavior Y is evidence of connection *even when* geometry is null — in this program, ablating
the moral subspace moved refusal +0.14 in one arm and −0.04 in the other; a sign-flipping
behavioral response to removing "disconnected" content is a through-weights read.

## Confound genealogy

For every contrast, enumerate what else it varies. Standing suspects for moral/safety work:
register/formality, emotional valence and arousal, narrative-lesson schema, topic, token
frequency, template tokens, sequence length, dataset difficulty. Then either (a) design the
minimal contrast that removes the covariate, or (b) measure a dedicated covariate direction
and report the target's loading on it.

**Behavior contrasts on generated text are content-confounded by construction** (traces about
harmful requests discuss harm). The fix is outcome-conditioning: contrast outcomes *within*
topic (refuse-rollouts vs comply-rollouts of the same prompt, via temperature resampling),
not topics within outcome.

## Construct constancy along swept axes

When a measurement is swept along positions/layers/checkpoints, verify the *construct* is
constant along the sweep. In this program, a "refusal direction" measured at four reasoning
positions silently changed meaning: at the pre-trace gate it was a decision contrast; in-trace
it became a harmful-topic-deliberation (content) contrast. A gradient whose x-axis changes
the construct is not a gradient of one thing.

## Normalization audit

Absolute-noise fragility comparisons across architectures restate scale differences (a 5.1×
fragility ratio next to a 74× output-scale ratio is one fact, not two): use RMS/SNR-matched
noise. Cross-model overlap comparisons require per-model normalization against each model's
own covariance-matched null — raw projection fractions are not comparable across models with
different anisotropy.

## Stimulus–outcome matching (causal cells)

Before any patching/ablation cell, verify the stimulus class can move the declared
`outcome_variable` at baseline (behavioral-discrimination screen: keep only pairs whose
uninterven­ed behavior differs). A stimulus that never engages the outcome yields
structurally flat cells that masquerade as nulls. *Case:* third-person narrative moral
twins can move a judgment readout, but nothing *refuses* a story — refusal-patching
required surface-matched **request** twins, a different asset that had to be built. The
`outcome_variable` field exists to force this check at design time. Full intervention rules
in `intervention-validity`.

## Co-location well-posedness

A projection or comparison is well-posed only where both objects exist in a common
(format, position) cell. When they coexist at **no** position — decision directions living
only at a bottleneck token, content only at content positions — the non-coexistence is a
*mechanism finding*, not a computable number: report it as such rather than computing
across cells and calling the artifact a result.

## Ship-blockers

- [ ] Both directions typed (type blocks present, incl. PR and outcome_variable)
- [ ] Design-matrix cells tabulated; diagonals measured or explicitly deferred
- [ ] Genealogy: known covariates listed; minimal-contrast or covariate-loading handled
- [ ] Construct constant along any swept axis (or the drift is stated)
- [ ] At least one causal cell before any "disconnected/decoupled" claim
- [ ] Normalization audited for cross-architecture / cross-model comparisons
- [ ] Causal-cell stimuli pass the baseline discrimination screen
- [ ] Comparisons computed only in co-located (format, position) cells

Pairs with: `instrument-calibration` (bands and controls for the typed comparison),
`intervention-validity` (causal-cell design), `anomaly-triage` (sign flips found in causal
cells).

*Changelog — v2 (2026-07-02): type block gains participation_ratio + outcome_variable;
added stimulus–outcome matching (case: narrative vs request twins) and co-location
well-posedness (case: R2/G3 not well-posed across the decision-site bottleneck).*
