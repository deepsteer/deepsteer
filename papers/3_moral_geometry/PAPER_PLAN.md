# Paper 3 Plan: *How Language Models Organize and Structure Moral Knowledge*

**Status:** Plan drafted. No experimental work started. All experiments
reuse existing infrastructure from Papers 1 and 2. Primary models:
OLMo-2 1B (dense) and OLMoE-1B-7B (MoE). All experiments designed
for single-GPU / MPS execution on MacBook Pro M4 Pro (24 GB).

**Relationship to Papers 1 and 2.** Paper 1 (*When Probing Accuracy
Saturates, Fragility Resolves*) establishes fragility as a complement
to probing accuracy and characterizes moral encoding depth, breadth,
and robustness in dense models. Paper 2 (*Output Dilution in
Mixture-of-Experts*) extends probing and fragility to MoE, finding
no expert specialization and a 74× output scale gap that creates
structural fragility. Both papers treat moral encoding as a **binary**
question: moral vs. neutral. Paper 3 asks a qualitatively different
question: does the model have **structured** representations of
distinct moral frameworks, and if so, what is the geometry of the
inter-framework relationships? This is the transition from measuring
moral *detection* to measuring moral *understanding*.

**Motivating research vision.** The long-term deepsteer goal is
training-time steering toward complex moral understanding: models
that can represent competing ethical frameworks, recognize dilemmas as
dilemmas, and reason across frameworks rather than collapsing to a
single moral heuristic or selecting among discrete moral personas.
Paper 3 establishes the measurement infrastructure for this goal:
before you can steer toward moral complexity, you need to measure
whether moral complexity exists in the representation space, and
what geometric form it takes.

**Path convention.** Paths are relative to `papers/3_moral_geometry/`
unless explicitly noted as project-root-relative. Shared infrastructure
(dataset, probe architecture) lives in `deepsteer/` and is imported
unchanged from Papers 1 and 2.

## Tentative title

**Primary:** *How Language Models Organize and Structure Moral Knowledge*

Alternates:
- *Beyond Moral Detection: Measuring Framework-Specific Structure in
  LLM Representations*
- *Do Language Models Distinguish Care from Fairness? Probing the
  Geometry of Moral Foundation Representations*
- *From Detection to Structure: How Moral Framework Geometry Emerges
  During Pre-Training*

## Thesis (working)

Prior work (Papers 1 and 2) established that language models encode
moral content broadly, robustly (in dense models), and uniformly
across experts (in MoE models), but treated moral encoding as a
single binary feature. The 240-pair probing dataset contains richer
structure: 40 pairs per Moral Foundations Theory foundation (care/harm,
fairness/cheating, loyalty/betrayal, authority/subversion,
sanctity/degradation, liberty/oppression). **By training
foundation-specific probes and analyzing the geometry of the resulting
probe directions, we can test whether models develop structured moral
representations that distinguish between ethical frameworks, the
precondition for moral reasoning as opposed to mere moral detection.**

Three geometric signatures correspond to three qualitatively different
modes of moral representation:

1. **Collapse (averaging).** All foundation probe directions converge
   toward a single "moral salience" direction. The model detects moral
   relevance but does not distinguish frameworks. Low pairwise angular
   separation between probe directions.

2. **Isolation (discrete selection).** Foundation directions are
   orthogonal with no relational structure. The model has separate
   moral "slots" but no representation of how frameworks relate.
   High angular separation, no structured clustering.

3. **Integration.** Foundation directions are separated but
   non-orthogonal, with inter-framework geometry that reflects known
   relationships (e.g., individualizing foundations — care, fairness,
   liberty — cluster separately from binding foundations — loyalty,
   authority, sanctity). Moderate angular separation with structured
   clustering.

These three signatures are empirically distinguishable via probe
direction cosine similarity matrices and hierarchical clustering.

## Target models

**OLMo-2 1B** (`allenai/OLMo-2-0425-1B`) — primary analysis model.
Dense, 16 layers, 1.5B parameters, 2048 hidden dimension. Full
checkpoint access for trajectory analysis. Already probed in Papers 1
and 2.

**OLMoE-1B-7B** (`allenai/OLMoE-1B-7B-0924`) — architectural
comparison. 16 layers, 64 experts, top-8 routing, 1.3B active
parameters. Paper 2 established that moral encoding is uniform across
experts; Paper 3 asks whether the *framework-level* geometry differs
between architectures despite the uniform expert-level encoding.

**Comparison model (if time allows):** OLMo-2 7B or OLMo-2 13B for
scale comparison. The key question: does framework separation increase
with scale? This would directly address the hypothesis that larger
models develop more structured moral representations.

## Research questions

1. **Framework separability.** Do foundation-specific probe directions
   occupy distinct directions in the embedding space, or do they
   collapse toward a single "moral salience" direction? Measured via
   pairwise cosine similarity of the 6 foundation probe weight vectors
   at each layer.

2. **Inter-framework geometry.** Does the angular structure between
   foundation directions reflect known relationships from moral
   psychology? Specifically: do the individualizing foundations
   (care, fairness, liberty) cluster separately from the binding
   foundations (loyalty, authority, sanctity), as MFT predicts?

3. **Layer-wise geometric development.** How does framework geometry
   change across layers? Hypothesis: early layers show collapse
   (single moral salience direction), middle/late layers show
   separation (framework-specific structure). This would be a
   geometric analog of Paper 1's lexical-to-compositional gradient.

4. **Geometric trajectory during training.** When during pre-training
   does framework separation emerge? Does it emerge after binary
   moral detection accuracy saturates (paralleling Paper 1's
   "fragility resolves after accuracy saturates")?

5. **Dense vs. MoE framework geometry.** Does MoE architecture
   produce different inter-framework geometry than dense? Paper 2
   showed MoE encoding is uniform across experts; does it also
   lack framework-level structure that dense models develop?

6. **Framework-specific fragility.** Do different moral foundations
   have different fragility profiles? Care/harm (the most universal
   cross-cultural foundation) might be more robustly encoded than
   loyalty/betrayal or sanctity/degradation (more culturally
   variable).

## Dataset considerations

### Current dataset limitations

The existing 240-pair dataset has 40 pairs per foundation. With the
80/20 split preserved per foundation, this yields:
- **32 training pairs (64 texts) per foundation**
- **8 test pairs (16 texts) per foundation**

This is thin for training a linear probe in 2048 dimensions. The
probe has 2048 + 1 = 2049 parameters and only 64 training examples.
While binary linear probes can work in this regime (the decision
boundary is a single hyperplane), the extracted probe *direction*
may be noisy, and noise in directions contaminates the angular
analysis that is the paper's core contribution.

### Mitigation strategy (phased)

**Phase A: Run with existing data.** Train foundation-specific probes
with current 32/8 splits. Compute the 6×6 cosine similarity matrix
of probe directions at each layer. If the signal is strong (clear
separation or clear collapse), the angular analysis may be robust
despite the small N. Assess via bootstrap: resample the training set
100 times, retrain probes, compute the variance of pairwise cosine
similarities. If bootstrap variance is low, the directions are stable
and the existing dataset suffices.

**Phase B: Augment if needed.** If bootstrap analysis shows unstable
directions (high variance in pairwise cosine similarities), expand
the dataset to 100 pairs per foundation (600 total, 80/8 train/test
per foundation). Use the same generation pipeline from Paper 1
(MoralBench seeds → minimal pair generation → automated validation
gates). This is a one-time cost that benefits all downstream
experiments.

**Phase C: Cross-validation as alternative.** If augmentation is too
slow, use 5-fold cross-validation on the full 40 pairs per foundation
(no held-out test set) for the geometric analysis, since the
primary quantity of interest is the probe *direction* (weight vector),
not the probe *accuracy*. Report per-fold direction stability as the
reliability metric.

### Dataset expansion spec (Phase B, if triggered)

Target: 100 pairs per foundation × 6 foundations = 600 pairs.
Need: 60 additional pairs per foundation (360 total new pairs).
Pipeline: same as Paper 1 Appendix A.
- Extract additional seeds from MoralBench (or supplement with
  ETHICS dataset, Hendrycks et al. 2021, mapped to MFT foundations).
- Generate minimal pairs via the existing pipeline.
- Apply the same four validation gates (length, embedding overlap,
  keyword, deduplication).
- Rebalance to exactly 100 per foundation.
Split: 80 train / 20 test per foundation.

Estimated effort: ~4 hours (seed extraction + generation + validation).
This is not a compute bottleneck.

## Experimental design

### Experiment 1: Foundation-specific probing

**Goal:** Train per-foundation binary probes and extract probe
directions at each layer.

**Method:**
- For each of 6 MFT foundations, train a binary linear probe
  (foundation-moral vs. neutral) at each of 16 layers using
  post-layer hidden states from OLMo-2 1B.
- Probe architecture: `nn.Linear(2048, 1)`, BCE, Adam lr=1e-2,
  50 epochs. Identical to Papers 1 and 2.
- Training data: 32 pairs (64 texts) per foundation. Test data:
  8 pairs (16 texts) per foundation.
- Extract the learned weight vector w ∈ R^2048 from each probe.
  Normalize to unit length: ŵ = w / ||w||.
- Report: per-foundation probe accuracy at each layer (6 accuracy
  curves). Compare with the binary moral/neutral probe from Paper 1
  (which pools all foundations).

**Key output:** 6 × 16 = 96 unit-norm probe direction vectors.

**Compute estimate:** ~10 min (96 probe trainings, each trivial).

### Experiment 2: Framework geometry analysis

**Goal:** Characterize the geometric relationships between
foundation-specific probe directions.

**Method:**
- At each layer, compute the 6×6 pairwise cosine similarity matrix
  of the foundation probe directions from Experiment 1.
- Compute summary statistics per layer:
  - **Mean pairwise cosine similarity** (higher = more collapsed
    toward a single direction).
  - **Min/max pairwise cosine similarity** (range indicates whether
    some foundation pairs are more similar than others).
  - **Individualizing vs. binding cluster distance.** Mean cosine
    similarity within the individualizing group (care, fairness,
    liberty) vs. within the binding group (loyalty, authority,
    sanctity) vs. between groups. If MFT structure is reflected in
    the geometry, within-group similarity > between-group similarity.
- Hierarchical clustering (Ward's method) on the cosine distance
  matrix at each layer. Visualize as dendrograms. The key visual:
  does the dendrogram split cleanly into individualizing vs. binding
  clusters at any layer?
- Permutation test for the individualizing/binding distinction:
  compute the actual between-group vs. within-group cosine
  similarity difference, then permute foundation group assignments
  10,000 times to generate the null distribution. Report p-value.

**Key outputs:**
- 16 cosine similarity heatmaps (one per layer).
- Layer-wise plot of mean pairwise cosine similarity (the
  "collapse-to-separation gradient" if it exists).
- Dendrogram at the peak separation layer.
- Permutation test p-value for MFT group structure.

**Compute estimate:** ~5 min (matrix operations on 96 vectors).

### Experiment 3: Bootstrap direction stability

**Goal:** Assess whether the foundation probe directions are stable
enough for geometric analysis given the small per-foundation N.

**Method:**
- For each foundation at each layer, resample the training set
  (32 pairs) with replacement 200 times.
- Retrain the probe on each bootstrap sample.
- Compute the pairwise cosine similarity between bootstrap probe
  directions. Report the mean and std of the cosine similarity
  of each bootstrap direction with the full-data direction.
- If mean cosine similarity with the full-data direction is > 0.8
  for all foundations at all layers where accuracy exceeds chance,
  the directions are stable. If not, trigger dataset expansion
  (Phase B).

**Key output:** Per-foundation, per-layer direction stability score.
Go/no-go gate for the geometric analysis.

**Compute estimate:** ~2 hours (200 bootstrap × 96 probes = 19,200
probe trainings, each ~5ms).

### Experiment 4: Layer-wise geometric development

**Goal:** Test whether framework geometry changes across layers
(collapse → separation gradient).

**Method:**
- Using the per-layer cosine similarity matrices from Experiment 2,
  plot the following across layers:
  - Mean pairwise cosine similarity (collapse metric).
  - First principal angle between the 3D individualizing subspace
    and the 3D binding subspace (computed via SVD on the respective
    probe direction matrices). Larger angle = more separated.
  - Effective dimensionality of the 6-direction set (via PCA on
    the 6 probe vectors per layer; report the number of PCs
    explaining 90% of variance). Low effective dim = collapse;
    high effective dim = separation.
- Test the hypothesis: early layers show high cosine similarity
  (collapse) and low effective dimensionality; later layers show
  lower cosine similarity (separation) and higher effective
  dimensionality.

**Key output:** The "geometric development" figure: effective
dimensionality of moral framework representations across layers.

**Compute estimate:** Negligible (operates on Experiment 2 outputs).

### Experiment 5: Dense vs. MoE framework geometry

**Goal:** Compare inter-framework geometry between architectures.

**Method:**
- Repeat Experiments 1 and 2 on OLMoE-1B-7B (layer-level hidden
  states, not per-expert, for direct comparison with OLMo-2 1B).
- Compare: cosine similarity matrices, effective dimensionality
  profiles, and MFT group structure (individualizing vs. binding
  clustering) between architectures.
- Hypothesis: Paper 2 showed MoE encoding is uniform across experts
  (no expert specialization). Does this uniformity extend to
  framework geometry? If MoE also shows less framework separation,
  the output dilution mechanism may suppress fine-grained structure
  along with signal scale. If MoE shows equivalent or greater
  separation, framework geometry is independent of signal scale.

**Compute estimate:** ~15 min (same probing pipeline on OLMoE).

### Experiment 6: Geometric trajectory during training

**Goal:** Track when framework separation emerges during pre-training.

**Method:**
- Select ~10 OLMo-2 checkpoints spanning training (if available;
  OLMo-2 publishes intermediate checkpoints). If OLMo-2 checkpoints
  are limited, use OLMoE's 244 checkpoints instead (or both).
- At each checkpoint, run Experiments 1 and 2. Track:
  - Mean pairwise cosine similarity across training.
  - Effective dimensionality across training.
  - MFT group structure strength (within-group vs. between-group
    similarity difference) across training.
- The key question: does framework separation emerge after binary
  moral detection accuracy saturates? If so, this extends the
  "structural resolution continues after accuracy saturates" finding
  from Papers 1 and 2 to a third metric (geometric structure).

**Compute estimate:** ~3–5 hours (10 checkpoints × ~20 min each,
dominated by model loading and activation collection).

### Experiment 7: Framework-specific fragility

**Goal:** Test whether different moral foundations have different
robustness profiles.

**Method:**
- Apply the standard fragility protocol (Gaussian noise injection,
  σ ∈ {0.1, 0.3, 1.0, 3.0, 10.0}) separately to each
  foundation-specific probe.
- Report per-foundation critical noise σ* at each layer.
- Test the hypothesis: more universal foundations (care/harm) are
  more robustly encoded than more culturally variable foundations
  (sanctity/degradation, loyalty/betrayal).
- Cross-architecture comparison: repeat on OLMoE. Does the output
  dilution effect affect all foundations equally, or does it
  differentially suppress certain framework representations?

**Compute estimate:** ~30 min (6× the standard fragility battery).

### Experiment 8 (stretch): Steering toward separation

**Goal:** Proof-of-concept training-time steering toward maintained
framework separation.

**Depends on:** Experiments 1–4 showing measurable framework geometry
that varies (across layers or across training). If geometry is
uniformly collapsed or uniformly separated, there is nothing to steer.

**Method (sketch — details depend on Experiments 1–4 results):**
- Define a geometric steering loss that penalizes collapse of
  framework directions toward a single direction. Candidate loss:
  negative mean pairwise cosine *distance* (1 - cosine similarity)
  of foundation probe directions, computed on a batch of moral
  texts with foundation labels.
- This requires differentiable probing: instead of post-hoc probe
  training, maintain online linear probes that update during
  training (or use a fixed set of foundation-labeled texts as a
  probe calibration set each N steps).
- Apply as an auxiliary loss during continued pre-training of
  OLMo-2 1B (or a smaller model if compute-constrained).
- Measure: does the steering loss maintain or increase framework
  separation compared to the unsteered baseline?

**This is a stretch goal.** It connects Paper 3 to the deepsteer
training-time steering program but may require more compute than a
Mac has available. It could also be scoped as a standalone Paper 4 if the
geometric analysis (Experiments 1–7) produces a clean paper on its own.

**Compute estimate:** Unknown; depends on whether continued
pre-training is feasible on Mac (likely 4-bit, small batch, short
runs). ~4–8 hours for a minimal proof-of-concept if feasible at all.

## Section structure

### 1. Introduction (~1.5 pages)

Prior work on this project treats moral encoding as a binary feature:
present or absent. Papers 1 and 2 established that models encode
moral content broadly and (in dense models) robustly, but a model
that merely detects "this is morally relevant" is not a model that
understands morality. Understanding requires structured representations
that distinguish between ethical frameworks and encode the
relationships between them.

We introduce framework-specific probing: training separate probes for
each Moral Foundations Theory foundation and analyzing the geometry
of the resulting probe directions in the model's embedding space.
Three geometric signatures (collapse, isolation, and integration)
correspond to three qualitatively different modes of moral
representation. We test which signature OLMo-2 1B and OLMoE-1B-7B
exhibit, how framework geometry develops across layers and across
training, and whether different foundations show different robustness
profiles.

Motivate the broader alignment relevance: measuring whether a model
has complex moral structure is the precondition for steering toward
complex moral understanding.

### 2. Related work (~1 page)

- **Moral Foundations Theory.** Haidt (2012), Graham et al. (2013).
  The individualizing/binding distinction. Cross-cultural variation
  in foundation emphasis. MFT as a descriptive framework, not a
  prescriptive one.
- **Probing geometry in LLMs.** Work on the geometry of concept
  representations: Bolukbasi et al. (2016, gender direction),
  Park et al. (2024, linear representation hypothesis), Nanda et al.
  (2023, feature geometry in toy models). The general finding that
  concepts occupy linear subspaces in LLM representations, and
  that the angular relationships between concept directions carry
  semantic meaning.
- **Moral probing.** Companion Papers 1 and 2. Position this work
  as the transition from binary (moral/neutral) to structured
  (framework-specific) probing.
- **Representation structure and reasoning.** Work connecting the
  geometry of internal representations to model capabilities.
  Models with richer internal structure tend to show better
  downstream reasoning. Cite relevant work on structured
  representations enabling compositional generalization.
- **MoE and feature organization.** Paper 2's finding that MoE
  does not create expert-level moral specialization. This paper
  asks whether MoE affects a different level of organization
  (framework geometry rather than expert assignment).

### 3. Methodology (~2 pages)

#### 3.1 Models and comparison design

Same as Papers 1 and 2: OLMo-2 1B (dense), OLMoE-1B-7B (MoE).
Same-lab controlled comparison. Base (non-instruct) checkpoints.

#### 3.2 Foundation-specific probing

Probe architecture (same as Papers 1–2). Per-foundation training/test
splits. Probe direction extraction and normalization.

#### 3.3 Geometric analysis methodology

Cosine similarity matrices. Effective dimensionality (PCA). Principal
angles between subspaces. Hierarchical clustering. Permutation testing
for MFT group structure.

#### 3.4 Bootstrap direction stability

Resampling protocol. Stability threshold. Rationale for assessing
direction reliability before geometric analysis.

#### 3.5 Framework-specific fragility

Same noise protocol as Papers 1–2, applied per-foundation.

#### 3.6 Probing dataset

Same 240-pair dataset, but now used at the per-foundation level
(40 pairs × 6 foundations). If dataset expansion is triggered (Phase
B), describe the expanded dataset here.

### 4. Results (~3.5 pages)

#### 4.1 Foundation-specific probe accuracy

Per-foundation accuracy curves across layers. Comparison with the
pooled binary probe from Paper 1. Do all foundations reach high
accuracy, or are some harder to decode? If some foundations show lower
accuracy, this is already evidence of non-uniform moral encoding:
the binary probe was masking foundation-level variation.

#### 4.2 Framework geometry: collapse, isolation, or integration?

The 6×6 cosine similarity matrices. The headline: what is the mean
pairwise cosine similarity at the peak accuracy layer? Close to 1.0
= collapse. Close to 0.0 = isolation. Intermediate with structure
= integration. Dendrograms showing clustering. Permutation test for
the individualizing/binding distinction.

#### 4.3 Layer-wise geometric development

The collapse-to-separation gradient (if it exists). Effective
dimensionality across layers. Comparison with Paper 1's
lexical-to-compositional gradient; do these two gradients align?

#### 4.4 Dense vs. MoE framework geometry

Side-by-side comparison of cosine similarity matrices and effective
dimensionality profiles. Does output dilution affect framework
geometry?

#### 4.5 Geometric trajectory during training (if checkpoint data)

When does framework separation emerge relative to binary accuracy
saturation? The third instance of "structure continues to develop
after accuracy saturates" (after fragility in Paper 1 and
specialization Gini in Paper 2).

#### 4.6 Framework-specific fragility

Per-foundation critical noise profiles. The universality hypothesis:
care/harm as the most robust foundation.

### 5. Discussion (~1.5 pages)

#### 5.1 What mode of moral representation do LLMs exhibit?

Interpret the geometric findings in terms of the collapse/isolation/
integration trichotomy. Connect to the broader question: does the
model "understand" morality or merely "detect" it? Be precise about
what geometric structure can and cannot tell us about understanding;
this is a necessary-but-not-sufficient condition.

#### 5.2 Implications for training-time steering

If framework separation exists and varies across layers/training,
then steering toward maintained separation is a concrete objective.
Describe the candidate steering loss (negative mean pairwise cosine
distance). Discuss feasibility and limitations. If framework geometry
is uniformly collapsed, discuss what this means for the steering
program, since collapse may be the natural attractor and steering against
it may require stronger interventions.

#### 5.3 Connection to moral psychology

Does the model's inter-framework geometry mirror MFT predictions?
The individualizing/binding distinction is the most testable
prediction. If it appears in the geometry, this is evidence that
the model has absorbed structural features of human moral reasoning
from the training corpus. If it doesn't, the model may encode moral
content through surface features (keywords, sentiment) rather than
conceptual structure.

#### 5.4 The detection-to-understanding gradient

Synthesize across all three papers:
- Paper 1: models detect moral content (binary probing) with
  fragility as a robustness measure.
- Paper 2: MoE architecture doesn't change the detection story
  but shows signal scale as a hidden variable.
- Paper 3: moving from detection to structure, the first evidence
  (or absence of evidence) for moral *understanding* in
  representations.

This positions the three-paper arc as a progressive refinement of
what "moral encoding" means, from presence to robustness to structure.

#### 5.5 Limitations

- Small per-foundation N (32 training pairs per probe). Bootstrap
  stability analysis mitigates but does not eliminate this concern.
- MFT as the organizing framework is itself debated in moral
  psychology. The geometric analysis assumes MFT foundations are
  the right decomposition; alternative moral taxonomies (Schwartz
  values, Curry's morality-as-cooperation) might produce different
  geometric signatures.
- Linear probes extract linear structure only. If moral frameworks
  are encoded in nonlinear submanifolds, the cosine similarity
  analysis misses the relevant geometry.
- English-only, same as Papers 1 and 2.
- The collapse/isolation/integration trichotomy is a simplification.
  Real geometry may be a mix (partial collapse of some foundations,
  separation of others).

### 6. Conclusion (~0.5 page)

Restate the central finding: what geometric mode do the models
exhibit? Connect to the three-paper arc. Position the geometric
measurement as the precondition for steering toward moral complexity.
State the next step (training-time steering toward maintained
framework separation, either as Paper 4 or as deepsteer integration).

### Appendices

- **A. Full per-foundation probe accuracy tables** (6 foundations
  × 16 layers × 2 models).
- **B. Bootstrap stability analysis** (direction cosine similarity
  distributions per foundation, per layer).
- **C. Full cosine similarity matrices** (all 16 layers for both
  models).
- **D. Permutation test details** (null distributions for MFT
  group structure).
- **E. Reproducibility** (hardware, seeds, command-line invocations).
- **F. Dataset expansion details** (if Phase B is triggered).

## Headline figures (planned)

1. **Figure 1: Cosine similarity heatmap at peak layer.** 6×6 matrix
   of pairwise cosine similarities between foundation probe
   directions, with foundations ordered as
   [care, fairness, liberty, loyalty, authority, sanctity] to make
   the individualizing/binding block structure visible if it exists.
   Side-by-side for OLMo-2 and OLMoE.

2. **Figure 2: Layer-wise geometric development.** Three-panel plot:
   (a) mean pairwise cosine similarity across layers (collapse
   metric), (b) effective dimensionality of the 6-direction set
   across layers, (c) individualizing-vs-binding cluster distance
   across layers. Shows whether a collapse-to-separation gradient
   exists.

3. **Figure 3: Dendrogram at peak separation layer.** Hierarchical
   clustering of the 6 foundation directions. The key visual: does
   the tree split into {care, fairness, liberty} vs. {loyalty,
   authority, sanctity}?

4. **Figure 4: Foundation-specific accuracy curves.** 6 lines
   (one per foundation) showing probe accuracy across layers,
   overlaid with the pooled binary probe from Paper 1. Shows
   whether some foundations are harder to decode.

5. **Figure 5: Framework-specific fragility.** Per-foundation
   critical noise profiles across layers for both models. Tests
   the universality hypothesis (care/harm as most robust).

6. **Figure 6: Geometric trajectory (if checkpoint data available).**
   Mean pairwise cosine similarity and effective dimensionality
   across training steps, overlaid with binary probe accuracy. The
   third "structure develops after accuracy saturates" plot.

## Mac feasibility summary

| Experiment | Time estimate | Memory | Bottleneck |
|---|---|---|---|
| Exp 1: Foundation probing (OLMo-2) | ~10 min | ~3 GB | 96 probe trainings |
| Exp 2: Geometry analysis | ~5 min | Negligible | Matrix operations |
| Exp 3: Bootstrap stability | ~2 hours | ~3 GB | 19,200 probe trainings |
| Exp 4: Layer-wise development | ~5 min | Negligible | Operates on Exp 2 outputs |
| Exp 5: Dense vs. MoE geometry | ~15 min | ~14 GB | OLMoE activation collection |
| Exp 6: Geometric trajectory | ~3–5 hours | ~3–14 GB | Checkpoint loading |
| Exp 7: Framework fragility | ~30 min | ~3 GB | Noise sweep × 6 foundations |
| Exp 8: Steering PoC (stretch) | ~4–8 hours | ~3–14 GB | Continued pre-training |
| **Total (Exp 1–7)** | **~6–8 hours** | | |

All experiments run on MacBook Pro M4 Pro (24 GB, MPS). Experiment
8 (steering) is a stretch goal contingent on Experiments 1–4
producing a clear geometric signal.

## Experimental ordering

1. **Experiment 1 (10 min).** Foundation-specific probing on OLMo-2
   1B final checkpoint. Produces the per-foundation accuracy curves
   and the raw probe directions. This is the foundation for
   everything else.

2. **Experiment 2 (5 min).** Geometry analysis on Experiment 1
   outputs. Produces the cosine similarity matrices, dendrograms,
   and permutation tests. **This is the go/no-go gate:** if all
   foundation directions are nearly parallel (mean cosine similarity
   > 0.95), the paper is a null result on framework structure. If
   there is separation (mean cosine similarity < 0.8), proceed.
   If intermediate (0.8–0.95), Experiment 3 determines whether the
   separation is real or noise.

3. **Experiment 3 (~2 hours).** Bootstrap stability. Required to
   validate that the geometric signal from Experiment 2 is not an
   artifact of small N. **Second go/no-go gate:** if bootstrap
   directions are unstable (mean cosine with full-data direction
   < 0.8), trigger dataset expansion (Phase B) before proceeding.

4. **Experiment 4 (5 min).** Layer-wise geometric development.
   Operates on existing Experiment 2 outputs. Quick analysis that
   produces a key figure.

5. **Experiment 5 (15 min).** Dense vs. MoE comparison. Repeat
   Experiments 1–2 on OLMoE. Produces the architectural comparison.

6. **Experiment 7 (30 min).** Framework-specific fragility. Run
   before the trajectory analysis since it's faster and produces
   an independent finding.

7. **Experiment 6 (3–5 hours).** Geometric trajectory. Most
   expensive core experiment; run after snapshot results justify
   it.

8. **Experiment 8 (stretch).** Steering PoC. Only if Experiments
   1–4 produce a clear, stable geometric signal worth steering
   toward.

## Open items

- **OLMo-2 checkpoint availability.** Verify how many intermediate
  training checkpoints OLMo-2 1B publishes. If limited, the
  trajectory analysis (Experiment 6) may need to use OLMoE
  checkpoints instead (244 available). Using OLMoE for trajectory
  adds the complication of the output dilution effect on framework
  geometry.

- **Probe direction extraction details.** The probe weight vector
  w ∈ R^2048 from `nn.Linear(2048, 1)` is the normal to the
  classification hyperplane. Confirm that this is the right object
  for geometric analysis. Alternative: use the mean difference
  of positive and negative class activations (the "concept
  direction" as in representation engineering). Compare both methods
  in a pilot analysis. If they agree, use probe weights (simpler).
  If they disagree, report both.

- **Alternative geometric measures.** Cosine similarity of probe
  weight vectors is the simplest measure but may miss structure.
  Consider also:
  - Centered Kernel Alignment (CKA) between foundation-specific
    activation sets (richer than probe direction comparison, but
    harder to interpret geometrically).
  - Representational Similarity Analysis (RSA) comparing the model's
    foundation similarity structure with the theoretically predicted
    MFT structure.
  - SVCCA or projection-weighted CCA for comparing foundation
    subspaces.
  Decide which to include before Experiment 2 to avoid post-hoc
  metric shopping.

- **MFT as the right decomposition.** The individualizing/binding
  distinction is the strongest prediction from MFT. But the six
  foundations themselves may not carve the model's moral
  representation at its joints. Consider a data-driven alternative:
  train the pooled binary probe, then cluster the moral-class
  activations (k-means or spectral clustering) and see if the
  emergent clusters align with MFT foundations. This reverses the
  analysis direction (theory-driven → data-driven) and could
  reveal structure that MFT misses. Include as an appendix analysis
  if time allows.

- **Sample size power analysis.** Before running, compute the
  minimum detectable angular separation given 32 training examples
  per probe in 2048 dimensions. If the minimum detectable
  separation (at 80% power) is larger than the separations we
  expect to see, dataset expansion is needed upfront rather than
  as a contingency.

- **Relationship to Experiment 8 and deepsteer integration.** If
  Experiment 8 (steering PoC) is pursued, decide whether it
  belongs in this paper or in a standalone Paper 4. The paper is
  likely strongest as a pure measurement paper (Experiments 1–7)
  with steering reserved for future work. But a positive steering
  result would dramatically increase impact. Decision depends on
  whether the steering PoC produces a clean result within the
  paper's scope.

## Cite list (anchor references)

- Haidt (2012); Graham et al. (2013) — MFT
- Bolukbasi et al. (2016) — gender direction in word embeddings
- Park et al. (2024) — linear representation hypothesis
- Nanda et al. (2023) — feature geometry in toy models
- Reblitz-Richardson (2026, Paper 1) — fragility methodology
- Reblitz-Richardson (2026, Paper 2) — MoE output dilution
- Muennighoff et al. (2024) — OLMoE
- Groeneveld et al. (2024) — OLMo
- Conneau et al. (2018) — probing classifiers
- Belinkov (2022) — probing survey
- Hendrycks et al. (2021) — ETHICS dataset (if used for expansion)
- Kriegeskorte et al. (2008) — RSA (if used)
- Kornblith et al. (2019) — CKA (if used)
- Curry et al. (2019) — morality-as-cooperation (alternative to MFT)

## Drafting order

1. **Run Experiments 1–3 before any drafting.** The paper's thesis
   depends on whether framework geometry exists and whether the
   directions are stable.
2. **§3 Methodology** — write from experimental design above.
3. **§4 Results** — write from experimental outputs.
4. **§2 Related Work** — expand after results are known (the
   relevant literature depends on which geometric signature appears).
5. **§5 Discussion** — interpretation depends heavily on results.
6. **§1 Introduction** — last, once scope is clear.
7. **§6 Conclusion** — last.
8. **Abstract** — last.

Don't draft beyond what's specified here without checking back.
