# Paper 4 Plan: *Causal Validation of Moral Foundation Directions*

**Status:** Reconstructed plan. Experimental work complete; all six
sections and five appendices drafted; `build/main.pdf` compiled.
Standalone-viability was an open question during development (results
were at one point considered for folding into Paper 3's causal
appendix); the paper stands as an independent Paper 4. Numbers below
reflect the committed v2-dataset run. Primary model: OLMo-2 1B (dense).
All experiments run on a single A100-80GB RunPod instance against the
GitHub repo, with lighter analysis reproducible on MacBook Pro M4 Pro
(24 GB, MPS).

**Relationship to Paper 3.** Paper 3 (*How Language Models Organize
Competing Moral Frameworks*) established that OLMo-2 1B develops
structured moral representations: six foundation-specific probe
directions exhibit integration geometry (effective dimensionality 5,
mean pairwise cosine ≈ 0.22–0.27), stable across three extraction
methods (mean-diff 0.957, LEACE 0.678, RepE 0.558), across dataset
sizes, and across text registers (>97% cross-register transfer).
**But probe directions are correlational.** A direction that separates
moral from neutral text may reflect a confound (length, formality,
topic) rather than a genuine moral feature. Paper 4 closes that gap by
asking whether the directions are causally load-bearing, behaviorally
predictive, and mechanistically native — i.e. whether they are valid
steering targets rather than decoding artifacts.

**Motivating research vision.** The long-term deepsteer goal is
training-time steering toward complex moral understanding. Before a
direction can be a steering target, it must be shown to be *writable*,
not merely *readable*. Paper 3 established that moral structure exists
and can be measured; Paper 4 establishes that the structure is a
lever, not just a readout. This is the bridge from descriptive
geometry to representation engineering, and it produces the concrete
steering fitness function (§5.4) that the rest of the program consumes.

**Thesis (working).** *The moral foundation directions from Paper 3 are
genuine features of the model's computation — causally load-bearing
(ablation reduces foundation-specific generation), behaviorally
sufficient (injection shifts output toward the target foundation with
a dose–response signature), and mechanistically native (unsupervised
SAE features partially recover the same subspace) — making them valid
targets for representation engineering of moral concepts.* No single
line of evidence is conclusive; the contribution is the convergence of
three methods with different assumptions and failure modes.

**Path convention.** Paths are relative to
`papers/4_causal_validation/` unless noted as project-root-relative.
Shared infrastructure (dataset, direction extraction, hook utilities)
lives in `deepsteer/` and is imported unchanged from Paper 3.

## Tentative title

**Primary:** *Causal Validation of Moral Foundation Directions in
Language Models*

Alternates:
- *From Readable to Writable: Causal, Behavioral, and Mechanistic
  Validation of Moral Representations*
- *Are Moral Probe Directions Real? Ablation, Steering, and SAE
  Evidence*
- *Valid Steering Targets: Grounding Moral Foundation Geometry with
  Causal Interventions*

## Design decision: which directions to validate

Use **mean-difference directions**, not probe-weight directions, as
the object of causal study.

Rationale: mean-diff directions capture more of the shared
moral-salience component (mean pairwise cosine 0.41 vs. 0.22 for probe
weights at layer 0; Paper 3 §4.13), which makes them better suited to
interventions that target the *full* moral representation rather than
the maximally discriminative hyperplane normal. The choice is
consequential and is reported as a finding, not a footnote: SAE
subspace overlap is 15.5% for mean-diff vs. 8.2% for probe-weight
directions (§4.4.3). Directions are normalized to unit length.

## Target model

**OLMo-2 1B** (`allenai/OLMo-2-0425-1B`) — 16 layers, 2048 hidden dim.
Same base (non-instruct) checkpoint as Papers 1–3. Single-model scope
is a deliberate limitation (see §5.5); foundation-specific causal
effects are expected to sharpen at 7B+ where feature redundancy is
lower (established in Paper 1), and cross-scale causal testing is
deferred to later work.

## Research questions

1. **Causal relevance (necessity).** Does the model *use* these
   directions during generation? If we ablate foundation $f$'s
   direction, does the model specifically lose access to $f$'s moral
   content while leaving other foundations intact?

2. **Causal sufficiency.** If we inject a foundation direction, does
   the model shift toward that foundation's content? Does the
   dose–response curve across injection amplitude distinguish genuine
   feature manipulation from noise?

3. **Behavioral grounding.** Do the directions predict which
   foundation a novel text activates — on held-out, external, and
   causal-evaluation stimuli? Geometry tells us how the model
   *organizes* moral knowledge; behavioral benchmarking tests whether
   that organization is *functionally accessible*.

4. **Mechanistic correspondence.** Do supervised probe directions
   correspond to features the model discovers unsupervised? If morally
   selective SAE features align with (or span the subspace of) the
   probe directions, the directions reflect native structure rather
   than probing artifacts.

5. **Layer-dependent causal roles.** Where in the network are the
   directions most load-bearing for removal (ablation) vs. most
   receptive to addition (injection)? Do these localize differently?

## Datasets and stimuli

Reuses the v2 240-pair moral probing dataset from Paper 3 (40 pairs
per MFT foundation; 192 train / 48 test). Three evaluation sets:

1. **Held-out test set** — 48 pairs (8 per foundation). Internal
   validation from the probing dataset.

2. **Moral Foundations Vignettes** — 30 items (5 per foundation),
   curated from Clifford et al. (2015). External validation with
   established MFT stimuli independent of the deepsteer pipeline.
   *Use the corrected Clifford stimuli* (the MFV replication set was
   fixed during the Paper 3 extensions sprint).

3. **Causal evaluation prompts** — 48 hand-authored prompts (8 per
   foundation) in three formats: 24 completion (sentence stem + 3–4
   candidate continuations), 12 forced-choice (two continuations from
   different foundations), 12 natural (open-ended priming stems). Each
   has labeled target and off-target continuations with foundation
   labels. Construction gates: (a) target foundation is the most
   natural continuation, (b) off-targets span ≥2 other foundations,
   (c) no surface lexical cue (foundation name / synonyms absent from
   the stem). Full set committed to the repo
   (`outputs/causal_eval_prompts.json`).

## Experimental design

### Experiment 1: Direction ablation (necessity)

**Goal:** Test whether each foundation direction is causally required
for generating that foundation's content.

**Method.** For each foundation $f$ and layer $\ell \in \{4, 8, 12\}$,
register a forward hook projecting the direction out of the hidden
state:
$$\mathbf{h} \leftarrow \mathbf{h} - (\mathbf{h}\cdot\mathbf{d}_f^{(\ell)})\,\mathbf{d}_f^{(\ell)}$$
Measure the change in log-probability of target vs. off-target
continuations on the 48-prompt causal set.

**Metrics.** Per (ablated foundation, layer): on-target Δ (matched
prompts), off-target Δ (other foundations), and **specificity** =
on-target Δ − off-target Δ. Negative specificity ⇒ ablation
specifically harms the target foundation.

**Result (committed).** Specificity is negative for all foundations at
layers 8 and 12 and deepens with depth: mean −0.16 (L4), −0.39 (L8),
−0.63 (L12). Sanctity is the most load-bearing at every layer
(−0.42 / −0.48 / −1.64), strongest at L12 (on-target Δ = −1.68 nats).
At L12 all six foundations show negative specificity — directions are
non-redundant.

**Output:** `outputs/direction_ablation_mean_diff.json`,
`outputs/steering_specificity_summary.json`.

### Experiment 2: Steering injection (sufficiency + dose–response)

**Goal:** Test whether adding a direction shifts generation toward the
target foundation, and whether the amplitude curve rules out noise.

**Method.** For each foundation $f$, layer $\ell \in \{4, 8, 12\}$, and
amplitude $\alpha \in \{1, 2, 5, 10, 20\}$:
$$\mathbf{h} \leftarrow \mathbf{h} + \alpha\,\mathbf{d}_f^{(\ell)}$$
Measure log-prob changes on the same 48-prompt set. A genuine feature
shows low-$\alpha$ specificity and high-$\alpha$ saturation; noise
shows monotonic degradation at all amplitudes.

**Result (committed).** At $\alpha=1$, four of six foundations show a
positive on-target boost (mean +0.15 nats) with near-zero off-target
(+0.03); care and fairness near zero. Specificity rises through
$\alpha=5$ (mean +0.88 at L8) and $\alpha=10$ (mean +2.34 at L8) — at
high $\alpha$ both on- and off-target rise but on-target rises *more*,
so the curve reflects directional information, not noise. Injection is
strongest at L4 (mean +0.95 at $\alpha=5$) and weakest at L12 (+0.33),
complementary to the ablation depth pattern.

**Output:** `outputs/steering_injection_mean_diff.json`.

### Experiment 3: Behavioral grounding (functional accessibility)

**Goal:** Test whether the directions predict foundation identity of
novel text.

**Method.** For each text, collect last-token residual activations at
layers {4, 8, 12}, project onto all six foundation directions, average
across layers, predict argmax foundation. **Debiased variant**
subtracts the mean projection across the six foundations first
(removing the shared moral-salience component). Evaluate on all three
stimulus sets.

**Result (committed).** Debiased 6-way accuracy (chance 16.7%):
- Causal prompts: **83.3%** (40/48), per-foundation 62.5–100%,
  loyalty 100%.
- Held-out test: **70.8%** (34/48); liberty & sanctity best (87.5%),
  loyalty worst (50%, errors scatter to sanctity).
- MFV: **33.3%** (10/30) — above chance but masks *sanctity
  saturation*: all 5 sanctity items correct, but 20/25 non-sanctity
  items also classified sanctity. This is a genuine stimulus property
  (harm-witnessing carries implicit purity/degradation content), not a
  direction artifact; debiasing does not remove it because the
  co-activation is real.

**Output:** `outputs/behavioral_benchmarking_mean_diff.json`.

### Experiment 4: SAE feature comparison (mechanistic correspondence)

**Goal:** Test whether unsupervised features recover the supervised
moral subspace.

**Method.** Train a ReLU SAE (16,384 latents, 8× expansion) on 2M C4
tokens at layer 8, L1 $\lambda=0.005$, unit-norm decoder columns,
pre-encoder bias = mean activation. Encode 192 moral + 192 neutral
sentences; rank features by moral selectivity (mean activation
difference). Compare top-$k$ features to probe directions via (a)
individual decoder-column cosine and (b) subspace overlap (SVD
projection fraction). Null: 1,000 iterations projecting directions
onto 100 random unit vectors in $\mathbb{R}^{2048}$.

**Result (committed).** SAE: L0 = 1,932 (11.8% active), FVU = 0.285
(71.5% variance explained) after 3 epochs — modest, training-scale
limited. Only 4/16,384 features have $|s|>0.1$ (moral info is
distributed, consistent with Paper 3's high effective dim). Top-100
selective features individually show 0% alignment ($|\cos|>0.2$) but
collectively span a subspace capturing **15.5%** of mean-diff
direction variance vs. 4.88% random = **3.17× random**. Probe-weight
directions: 8.2% (1.67×). Interpreted as a lower bound; production SAEs
(≥50M tokens) expected to recover more.

**Output:** `outputs/sae_moral_features_layer8.json`,
`outputs/sae_training_summary.json`.

## Section structure

Matches the committed draft in `sections/`.

### 1. Introduction (~1.5 pages) — `01_introduction.md`
Geometry is necessary but not sufficient; probe directions are
correlational. Three open questions (causal / behavioral / mechanistic)
→ this paper answers all three on the same model and dataset as
Paper 3.

### 2. Related work (~1 page) — `02_related_work.md`
Causal methods in mech interp (activation patching, causal tracing,
Arditi et al. 2024 refusal direction, Turner et al. 2023 activation
addition); representation engineering (Zou et al. 2023, Marks et al.
2024); SAEs for feature discovery (Bricken/Cunningham 2023, Templeton
et al. 2024); moral reasoning in LLMs (MFV, connection to Paper 3).

### 3. Methods (~2.5 pages) — `03_methods.md`
3.1 Model + dataset + mean-diff direction choice (with the 15.5% vs.
8.2% consequence stated up front). 3.2 Causal methods (ablation,
injection, causal prompt construction). 3.3 Behavioral grounding
(projection classification, debiasing, three eval sets). 3.4 SAE
methods (training, selectivity, subspace overlap, random baseline).

### 4. Results (~3.5 pages) — `04_results.md`
4.1 Ablation is foundation-specific (depth gradient, sanctity
dominance). 4.2 Injection dose–response specificity (complementary
depth pattern). 4.3 Behavioral grounding (three sets; the sanctity
saturation analysis on MFV). 4.4 SAE partially recovers the subspace
(3.2× random).

### 5. Discussion (~1.5 pages) — `05_discussion.md`
5.1 Three converging lines of evidence (necessity / sufficiency /
functional accessibility / mechanistic correspondence). 5.2 The
sanctity saturation phenomenon and its link to Paper 3's sanctity
anomaly (6.2× dense/MoE fragility ratio). 5.3 Layer-dependent causal
roles (early = malleable but weakly causal; late = rigid but
output-driving). 5.4 **Toward a steering fitness function** (four
gates; which foundations pass). 5.5 Limitations.

### 6. Conclusion (~0.5 page) — `06_conclusion.md`
Directions are writable, not just readable. Sanctity saturation
motivates multi-direction steering. Transforms moral geometry into a
representation-engineering tool.

### Appendices (drafted)
- **A** — full ablation tables (6 foundations × layers 4/8/12).
  `0A_full_ablation_tables.md`
- **B** — steering dose–response tables/curves across $\alpha$.
  `0B_steering_dose_response.md`
- **C** — behavioral confusion matrices (all three sets; the MFV
  sanctity-saturation matrix). `0C_behavioral_confusion_matrices.md`
- **D** — SAE feature details (selective features, per-foundation
  overlap). `0D_sae_feature_details.md`
- **E** — reproducibility (hardware, seeds, invocations).
  `0E_reproducibility.md`

## The steering fitness function (§5.4 — key deliverable)

A direction qualifies as a steering target if:

1. **Ablation specificity** < −0.3 at the target layer (causally
   load-bearing).
2. **Injection specificity** > 0 at $\alpha=1$ (produces specific
   shifts).
3. **Behavioral accuracy** > 50% on held-out data (functionally
   discriminative).
4. **SAE subspace overlap** > 1.5× random (corresponds to native
   features).

Pass status (committed): sanctity, liberty, loyalty pass all four;
authority passes all four at moderate levels; care and fairness pass 1
and 3 but show weak low-$\alpha$ injection specificity (more
distributed representations, harder to selectively amplify). This
fitness function is the artifact downstream steering work consumes.

## Headline figures (planned / in draft)

1. **Ablation specificity heatmap** — 6 foundations × layers {4,8,12},
   showing the depth gradient and sanctity dominance.
2. **Injection dose–response curves** — on-target and off-target Δ
   across $\alpha$ for 2–3 foundations, showing low-$\alpha$
   specificity and high-$\alpha$ amplification.
3. **Behavioral accuracy bars** — three eval sets vs. chance.
4. **MFV sanctity-saturation confusion matrix** — the visual anchor
   for §5.2.
5. **SAE subspace overlap** — mean-diff (15.5%) vs. probe-weight
   (8.2%) vs. random baseline (4.88%).

## Compute / feasibility summary

| Experiment | Time | Bottleneck |
|---|---|---|
| Exp 1: Ablation (6 fdn × 3 layers × 48 prompts) | ~30 min | hooked forward passes |
| Exp 2: Injection (6 × 3 × 5 α × 48) | ~1–1.5 h | amplitude sweep |
| Exp 3: Behavioral grounding (3 sets) | ~15 min | activation collection |
| Exp 4: SAE training (2M tokens, L8) | ~2–3 h | SAE fit on A100 |
| Exp 4: SAE analysis + null | ~15 min | SVD + 1,000 random iters |

All experiments run on the A100-80GB RunPod instance; ablation,
injection, and behavioral grounding are reproducible on the M4 Pro
(MPS). SAE training is the only step that materially benefits from the
A100.

## Experimental ordering

1. **Exp 1 (ablation)** — establishes necessity; the go/no-go gate. If
   no foundation shows negative specificity at any layer, the
   directions are not causal and the paper pivots to a null result.
2. **Exp 2 (injection)** — sufficiency + dose–response, run once
   ablation confirms causal signal.
3. **Exp 3 (behavioral)** — cheap; run alongside Exp 1/2 (shares the
   causal prompt set).
4. **Exp 4 (SAE)** — most expensive; run last, after the causal story
   justifies the mechanistic-correspondence question.

## Open items / notes

- **Standalone viability.** Resolved: Paper 4 stands independently.
  Track this if scope shifts — the causal + behavioral + SAE triad is
  what makes it a paper rather than a Paper 3 appendix. Do not let the
  SAE section (the weakest leg at 3.2×) drift into over-claiming.
- **v2 dataset numbers.** All results are on the v2 1,200-pair-lineage
  probing data. On v2, causal specificity weakened relative to the v1
  pilot (−0.93 → −0.63) and SAE overlap dropped (4.01× → 3.17×); the
  MFV dominant foundation flipped care → sanctity (v1 care dominance
  was an animacy/accidentally-moral-neutral artifact). Report v2 as the
  primary contribution; no v1 revision-log language.
- **SAE lower-bound framing.** The 3.2× overlap is explicitly a lower
  bound. State the falsifiable target: >5× (~25%) with >10 selective
  features would be strong recovery evidence, expected to need ≥50M
  training tokens. Don't soften this into a positive claim.
- **Steering regime.** The therapeutically relevant injection regime is
  $\alpha=1$–2; high-$\alpha$ positive specificity is *relative*
  on-target amplification, not selective control. Keep this caveat in
  §5.5 and the conclusion.
- **Sanctity two-direction steering.** The sanctity saturation result
  implies foundation-specific steering may need to *suppress* sanctity
  while boosting the target — a two-direction intervention. This is a
  motivating hook for later steering work, not a claim tested here.

## Cite list (anchor references)

- Vig et al. (2020); Meng et al. (2022); Geiger et al. (2021) — causal
  tracing / activation patching
- Arditi et al. (2024) — refusal direction (single-direction mediation)
- Turner et al. (2023) — activation addition / steering
- Zou et al. (2023) — representation engineering
- Marks et al. (2024) — multi-dimensional concept geometry
- Bricken et al. (2023); Cunningham et al. (2023) — SAEs / monosemanticity
- Templeton et al. (2024) — scaling SAEs
- Clifford et al. (2015) — Moral Foundations Vignettes
- Graham et al. (2013) — MFT (sanctity/purity)
- Raffel et al. (2020) — C4 corpus
- Reblitz-Richardson (2026, Paper 3) — moral geometry (the directions
  this paper validates)

## Drafting order (as executed)

1. Run Exp 1–3, then Exp 4.
2. §3 Methods from the design above.
3. §4 Results from committed outputs.
4. §2 Related Work (final method framing depends on which causal
   signature appeared).
5. §5 Discussion (interpretation is result-dependent; §5.4 fitness
   function is the payload).
6. §1 Introduction, §6 Conclusion, Abstract — last.

Don't extend scope beyond what's specified here without checking back.
