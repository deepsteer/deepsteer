# Paper 2 Plan: *Do Moral Representations Specialize Across Experts? Probing MoE Architectures During Pre-Training*

**Status:** Plan drafted. No experimental work started. OLMoE-1B-7B
confirmed Mac-feasible (~14 GB bf16, 24 GB unified memory). All
experiments designed for single-GPU / MPS execution; checkpoint
trajectory analysis is sequential (load, probe, unload).

**Relationship to Paper 1.** Paper 1 (*When Probing Accuracy Saturates,
Fragility Resolves*) establishes fragility as a complementary metric
to probing accuracy in dense OLMo models. Paper 2 extends both
metrics, probing accuracy *and* fragility, into the MoE setting,
where expert-level decomposition creates a natural unit of analysis
that dense models lack. The C15 fragility-locus finding (insecure-code
LoRA shifts robustness peak by 2–3 layers) originally drafted as
Paper 2 §4.4 has been migrated to Paper 1 §4.4 / §5.3 as a direct
demonstration of Paper 1's thesis in a fine-tuning setting.

**Prior Paper 2 content.** The persona-mechanism / compound-scaling-
boundary findings (C10 v2 null, Step 2 gradient-penalty, Step 2B
behavioral dissociation) were part of an earlier draft and have moved
to the companion line of work; their raw experiment data is no longer
kept in this paper's `outputs/`. They are cited in this paper's Related
Work as motivation for investigating MoE: the dense-model finding that
probe-direction suppression does not capture behavior (due to
feature redundancy at 1B) raises the question of whether MoE
architectures, which structurally partition features across experts,
produce less redundant (and therefore more intervenable) moral
encoding.

**Path convention.** Same as Paper 1: paths are relative to
`papers/2_moe_output_dilution/`. Project-root-relative paths used in
CLI invocations and root-level docs.

## Tentative title

**Primary:** *Do Moral Representations Specialize Across Experts?
Probing MoE Architectures During Pre-Training*

Alternates:
- *Expert Moral Specialization in Mixture-of-Experts Language Models*
- *Moral Routing: How MoE Architectures Partition Ethical Representations*
- *When Experts Disagree: Moral Encoding Across MoE Specialists During Pre-Training*

## Thesis (working)

Dense language models encode moral features diffusely across layers
and neurons, making single-direction interventions ineffective
(Paper 2 prior work; companion Paper 1 fragility analysis). **MoE
architectures structurally partition representations across experts,
creating a natural experimental setting to test whether moral
encoding concentrates in specific expert subsets or distributes
across the full expert pool.** OLMoE-1B-7B, with 64 experts per
layer, top-8 routing, and 244 published training checkpoints, is
the only open MoE model that enables both snapshot analysis and
training trajectory analysis of expert-level moral specialization.
A controlled comparison with dense OLMo (same lab, same training
philosophy, same checkpoint availability) directly tests whether
architecture determines the geometry of moral encoding.

## Target model

**OLMoE-1B-7B** (`allenai/OLMoE-1B-7B-0924`)

| Property | Value |
|---|---|
| Total parameters | 6.9B |
| Active parameters / token | 1.3B |
| Layers | 16 |
| Experts per layer | 64 |
| Active experts per token (top-k) | 8 |
| Hidden dimension | 2048 |
| Per-expert FFN intermediate | 1024 |
| Routing | Dropless token-choice, aux loss 0.01 |
| Max sequence length | 4096 |
| Training checkpoints | 244 (every 5K steps, step5000–step1200000) |
| HuggingFace class | `OlmoeForCausalLM` (transformers ≥ 4.45) |
| Memory (bf16) | ~14 GB (fits 24 GB Mac) |
| Memory (4-bit) | ~5–6 GB |
| Base model? | Yes (no instruction tuning) |

**Comparison model:** Dense OLMo-2 1B (`allenai/OLMo-2-0425-1B`) —
same lab, comparable active parameter count, full checkpoint access,
already probed in Paper 1.

## Research questions

1. **Expert moral specialization.** Do some experts carry
   disproportionate moral signal while others are morally neutral?
   Measure via per-expert linear probes at each layer.

2. **Moral routing patterns.** Do morally-charged inputs activate
   different expert subsets than neutral inputs? The router weights
   are directly observable; analyze expert selection distributions
   conditioned on input moral content.

3. **Expert-level fragility.** Does Paper 1's layer-depth robustness
   gradient manifest uniformly across experts, or do "moral experts"
   show different fragility profiles than "neutral experts"?

4. **Specialization trajectory.** When during pre-training does
   expert moral specialization emerge? The 244 checkpoints enable
   tracking per-expert probe accuracy and router moral-preference
   across training.

5. **Dense vs. MoE comparison.** Does moral encoding geometry differ
   between OLMoE (MoE, 1.3B active) and OLMo-2 1B (dense, 1.5B)?
   Same probing dataset, same fragility battery, same lab's training
   recipe; architecture is the independent variable.

## Experimental design

### Experiment 1: Per-expert moral probing

**Goal:** Determine whether moral encoding concentrates in specific
experts or distributes uniformly.

**Method:**
- Hook each expert's FFN output independently (64 × 16 = 1024
  expert-layer combinations).
- For each expert at each layer, collect activations on the standard
  240-pair moral probing dataset (same dataset as Paper 1).
- Train a binary linear probe per expert-layer combination (same
  architecture as Paper 1: `nn.Linear(expert_dim, 1)`, BCE, Adam,
  50 epochs).
- Report: per-expert probe accuracy heatmap (64 experts × 16 layers).
  If moral encoding is diffuse, all experts show similar accuracy.
  If concentrated, a subset of experts shows high accuracy while
  the rest are near chance.

**Compute estimate:** Per-expert activation collection is the
bottleneck. Each forward pass activates only 8 of 64 experts per
token, so collecting activations for all experts requires either
(a) forcing all experts to fire (modifying the router), or
(b) accumulating across many inputs until each expert has enough
samples. Approach (b) is cleaner: run enough inputs that each
expert has ≥50 activations per moral/neutral class. With 64 experts
and top-8 routing, each expert fires on 12.5% of tokens on average;
~4000 tokens should give ~500 per expert. The 240-pair dataset at
~20 tokens/sentence yields ~9600 tokens, more than sufficient.

**Mac time estimate:** ~30 min per checkpoint (activation collection
+ 1024 probe trainings at expert_dim=1024). Snapshot on final
checkpoint: ~30 min. Trajectory across 10 selected checkpoints:
~5 hours.

### Experiment 2: Router moral-preference analysis

**Goal:** Determine whether the learned router systematically routes
moral content to specific experts.

**Method:**
- Set `output_router_logits=True` in model config.
- Run the 240-pair moral probing dataset through the model.
- For each layer, extract the router logits (batch × seq × 64)
  and the top-8 expert selections.
- Compare: (a) expert selection frequency conditioned on moral vs.
  neutral inputs; (b) mean router logit per expert for moral vs.
  neutral; (c) Jensen-Shannon divergence of the routing distribution
  between moral and neutral inputs.
- Statistical test: for each expert at each layer, chi-squared test
  on (moral tokens routed to this expert) vs. (neutral tokens routed
  to this expert), Bonferroni-corrected for 1024 comparisons.

**Headline metric:** Number of experts with statistically significant
moral routing preference (corrected p < 0.05), and the magnitude of
the routing preference (odds ratio).

**Mac time estimate:** ~15 min (single forward pass + statistical
analysis).

### Experiment 3: Expert-level fragility

**Goal:** Extend Paper 1's fragility methodology to the expert level.

**Method:**
- For each expert-layer with above-chance moral probe accuracy
  (from Experiment 1), run `MoralFragilityTest`: inject Gaussian
  noise into the cached expert activations at magnitudes
  σ ∈ {0.1, 0.3, 1.0, 3.0, 10.0} and record the critical noise
  (σ at which probe accuracy drops below 0.6).
- Report: per-expert fragility heatmap (same layout as Experiment 1
  accuracy heatmap, but showing critical noise).
- Compare expert-level fragility with the layer-level fragility from
  Paper 1: is the layer-level robustness gradient a uniform average
  of expert-level gradients, or do individual experts show sharper
  patterns than the layer aggregate?

**Mac time estimate:** ~1 hour (fragility sweep on the subset of
expert-layers with above-chance probes).

### Experiment 4: Specialization trajectory across training

**Goal:** Track when expert moral specialization emerges during
pre-training.

**Method:**
- Select ~15–20 checkpoints spanning training: dense early sampling
  (step 5K, 10K, 20K, 50K, 100K) + logarithmic spacing through the
  rest (200K, 400K, 600K, 800K, 1000K, 1200K).
- At each checkpoint, run Experiment 1 (per-expert moral probing)
  and Experiment 2 (router moral-preference).
- Track: (a) Gini coefficient of per-expert moral probe accuracy
  across training (higher Gini = more concentrated specialization);
  (b) number of experts with moral routing preference; (c) identity
  stability of "moral experts" across checkpoints (do the same
  experts specialize, or does specialization migrate?).

**Headline figure:** Specialization trajectory — Gini coefficient of
expert moral accuracy vs. training step, overlaid with mean probe
accuracy (to separate specialization from overall capability). If
moral specialization is a late-training phenomenon, Gini rises after
probe accuracy saturates, a MoE analog of Paper 1's "fragility keeps
resolving after accuracy saturates."

**Mac time estimate:** ~8–10 hours total (15–20 checkpoints × ~30 min
each). Sequential: load checkpoint, probe, unload, load next.

### Experiment 5: Dense vs. MoE controlled comparison

**Goal:** Test whether architecture determines moral encoding geometry.

**Method:**
- Run Paper 1's full probe + fragility battery on OLMoE-1B-7B final
  checkpoint (layer-level, not expert-level, for direct comparison
  with the OLMo-2 1B numbers already in Paper 1).
- Compare: (a) layer-level moral probe accuracy profile; (b) layer-
  level fragility profile; (c) encoding depth / breadth summary
  statistics.
- The comparison is architecturally controlled: same lab, comparable
  active parameter count (1.3B vs. 1.5B), same probing dataset,
  same probe architecture.

**Additional comparison (if time allows):** Run the compositional
moral probing dataset from Paper 1 §3.2 on OLMoE. Tests whether
the compositional/syntax plateau at ~0.77 from Paper 1 §4.2 is
model-specific or architecture-invariant.

**Mac time estimate:** ~30 min (reuse existing Paper 1 infrastructure).

### Experiment 6 (stretch): Expert-level fine-tuning fragility

**Goal:** Test whether fine-tuning affects moral encoding differently
in MoE vs. dense architectures.

**Method:**
- LoRA fine-tune OLMoE-1B-7B on the same insecure-code / secure-code
  corpora used in Paper 1's C15 experiment.
- Run per-expert moral probing + fragility on the fine-tuned model.
- Compare with Paper 1's C15 finding (fragility-locus shifts 2–3
  layers in dense OLMo): does fine-tuning affect specific experts
  rather than specific layers in MoE? Does the fragility shift
  manifest as expert deactivation (router stops sending moral tokens
  to previously-moral experts) or expert degradation (moral experts
  become fragile)?

**This is a stretch goal; depends on Experiments 1–5 producing
interesting enough intermediate results to justify the fine-tuning
compute.**

**Mac time estimate:** ~2–3 hours (LoRA fine-tuning + evaluation).

## Section structure

### 1. Introduction (~1 page)

Dense LLMs encode moral features diffusely across layers and
neurons. This diffuseness has consequences: interventions targeting
a single linear direction fail to capture behavior (prior work at
1B), and probing accuracy saturates too quickly to track
representational dynamics (companion Paper 1). MoE architectures
offer a structural alternative: by routing tokens through a sparse
subset of expert modules, they partition the representation space
into discrete, inspectable units. We ask: does this partition create
expert-level moral specialization, and if so, when does it emerge
during pre-training?

OLMoE-1B-7B is uniquely positioned to answer these questions: 64
experts per layer, 244 published training checkpoints, and a dense
counterpart (OLMo) from the same lab for controlled comparison.
Preview the five findings.

### 2. Related work (~0.75 page)

- **MoE architectures:** Shazeer et al. (2017), Fedus et al. (2022),
  Muennighoff et al. (2024 — OLMoE paper). Expert specialization
  findings from prior work (linguistic, domain-level).
- **Expert specialization analysis:** Prior work on what individual
  MoE experts learn, typically focused on linguistic features
  (syntax, POS) or domain features (code vs. text). No prior work
  on moral/ethical feature specialization across experts.
- **Moral probing:** Haidt (2012), Graham et al. (2013) for MFT
  taxonomy. Companion Paper 1 for the probing + fragility methodology.
- **OLMo ecosystem:** Groeneveld et al. (2024) for OLMo; Muennighoff
  et al. (2024) for OLMoE. Same-lab comparison advantage.
- **Dense-model moral encoding:** Companion Paper 1 findings on
  layer-depth robustness gradients, lexical→compositional gradient.
  Prior Paper 2 work on persona-feature compound scaling boundary
  (Reblitz-Richardson, 2026) — feature redundancy in dense 1B models
  makes single-direction suppression futile; does MoE change this?

### 3. Methodology (~1.5 pages)

#### 3.1 Model and checkpoints

OLMoE-1B-7B architecture details; checkpoint selection rationale;
dense OLMo-2 1B as comparison.

#### 3.2 Per-expert activation collection

Hook registration on individual expert FFN modules. Token
accumulation strategy (top-8 routing means each expert fires on
~12.5% of tokens; accumulate until ≥50 activations per class per
expert). Mean-pooling within each expert's activation window.

#### 3.3 Per-expert moral probing

Same probe architecture as Paper 1 (`nn.Linear(expert_dim, 1)`, BCE,
Adam, 50 epochs). Training on per-expert activations rather than
full hidden states. Report accuracy per expert-layer.

#### 3.4 Router analysis

Router logit extraction. Conditional routing distributions. JSD and
chi-squared tests for moral routing preference. Bonferroni correction
across 1024 expert-layer comparisons.

#### 3.5 Expert-level fragility

Direct extension of Paper 1's `MoralFragilityTest` to per-expert
activations. Same noise grid {0.1, 0.3, 1.0, 3.0, 10.0}, same
threshold (0.6), same critical-noise definition.

#### 3.6 Probing dataset

Same 240-pair moral probing dataset as Paper 1 (40/foundation × 6
MFT foundations, seed 42). Dataset identity is load-bearing for the
dense vs. MoE comparison.

### 4. Results (~3 pages)

#### 4.1 Expert moral specialization (Experiment 1)

Per-expert probe accuracy heatmap. Gini coefficient of moral accuracy
across experts at each layer. Identification of "moral expert" and
"neutral expert" clusters if they exist.

#### 4.2 Moral routing (Experiment 2)

Number of experts with significant moral routing preference. Router
logit distributions conditioned on moral vs. neutral input. Whether
routing preference aligns with probing accuracy (experts that route
moral tokens also probe moral, or are they decoupled?).

#### 4.3 Expert-level fragility (Experiment 3)

Per-expert fragility heatmap. Comparison with layer-level fragility:
does the layer aggregate mask expert-level structure? Do "moral
experts" show different fragility profiles?

#### 4.4 Specialization emergence trajectory (Experiment 4)

Gini coefficient of expert moral accuracy across training. When does
specialization emerge relative to overall capability? Does expert
identity stabilize (same experts specialize throughout) or migrate?

#### 4.5 Dense vs. MoE comparison (Experiment 5)

Layer-level comparison on probing accuracy and fragility. Differences
in encoding depth, breadth, and robustness gradient between OLMoE
and dense OLMo.

### 5. Discussion (~1.5 pages)

#### 5.1 Implications for alignment interventions

If moral encoding concentrates in specific experts, MoE architectures
offer natural intervention points that dense models lack: expert
pruning, expert-specific fine-tuning, router modification. If moral
encoding distributes uniformly across experts, MoE and dense
architectures are equivalent for alignment purposes.

#### 5.2 Feature redundancy and intervenability

Connection to the prior Paper 2 persona-mechanism finding: dense-model
feature redundancy made single-direction suppression futile (probe Δ
3.07 SD, behavioral Δ 0.01). Does MoE reduce this redundancy by
partitioning features across experts? Or does the router's load-
balancing loss enforce a different kind of redundancy?

#### 5.3 Training-time specialization dynamics

If expert moral specialization emerges late (after probe accuracy
saturates), this is the MoE analog of Paper 1's "fragility keeps
resolving after accuracy saturates": structural reorganization
continues after content becomes decodable.

#### 5.4 Limitations

- Single MoE model family (OLMoE). Generalization to Mixtral,
  DeepSeek-MoE, or Qwen-MoE is open.
- Per-expert probing uses mean-pooled activations and linear probes,
  with the same methodology limitations as Paper 1.
- The dense vs. MoE comparison is not perfectly controlled: OLMoE
  and OLMo-2 differ in training data mix and hyperparameters, not
  just architecture. Same-lab provenance minimizes but does not
  eliminate this confound.
- Expert_dim (1024) is half the full hidden_dim (2048). The per-
  expert probe operates on a lower-dimensional space; direct
  comparison of probe accuracy with the full-hidden-state probe
  requires normalization for dimensionality.
- Moral probing dataset covers English only, MFT framework only.

### 6. Conclusion (~0.5 page)

Restate whether MoE creates expert-level moral specialization or not.
Connect to the broader program: Paper 1 shows fragility resolves
structure in dense models; Paper 2 asks whether MoE architecture
changes the geometry of that structure. Position the dense/MoE
comparison as a step toward understanding how architecture choices
during pre-training affect the tractability of post-hoc alignment
measurement and intervention.

### Appendices

- **A. Full per-expert probe accuracy tables** (64 × 16 at final
  checkpoint; representative checkpoints during trajectory).
- **B. Router logit distributions** (per-layer histograms of moral
  vs. neutral routing).
- **C. Expert identity stability analysis** (which experts are
  "moral" at checkpoint X vs. Y; Jaccard similarity of moral-expert
  sets across training).
- **D. Reproducibility** (hardware, seeds, model revisions,
  command-line invocations, old→new script name mapping from
  prior Paper 2 content).

## Headline figures (planned)

1. **Figure 1: Per-expert moral probe accuracy heatmap.** 64 experts
   × 16 layers, final checkpoint. Color scale from chance (0.5) to
   perfect (1.0). If specialization exists, clear hot spots appear
   in a sparse subset of expert-layer cells.

2. **Figure 2: Moral routing preference.** Per-layer bar chart
   showing number of experts with significant moral routing
   preference (corrected p < 0.05), overlaid with mean odds ratio.

3. **Figure 3: Expert-level fragility heatmap.** Same layout as
   Figure 1 but showing critical noise instead of accuracy. If
   "moral experts" are also the most robust, the hot spots align.

4. **Figure 4: Specialization trajectory.** Gini coefficient of
   expert moral accuracy vs. training step (left axis), overlaid
   with mean probe accuracy (right axis). The key visual: does Gini
   rise after accuracy saturates?

5. **Figure 5: Dense vs. MoE comparison.** Side-by-side layer-level
   probe accuracy and fragility profiles for OLMoE-1B-7B and
   OLMo-2 1B. Same dataset, same probes, different architecture.

## Mac feasibility summary

| Experiment | Time estimate | Memory | Bottleneck |
|---|---|---|---|
| Exp 1: Per-expert probing (1 ckpt) | ~30 min | ~14 GB bf16 | 1024 probe trainings |
| Exp 2: Router analysis | ~15 min | ~14 GB bf16 | Single forward pass |
| Exp 3: Expert fragility | ~1 hour | ~14 GB bf16 | Noise sweep × experts |
| Exp 4: Trajectory (15–20 ckpts) | ~8–10 hours | ~14 GB bf16 | Sequential checkpoint loading |
| Exp 5: Dense vs. MoE | ~30 min | ~3 GB (OLMo 1B) | Reuses Paper 1 infra |
| Exp 6: Fine-tuning (stretch) | ~2–3 hours | ~14 GB bf16 | LoRA training |
| **Total (Exp 1–5)** | **~12 hours** | | |

All experiments run on a single MacBook Pro M4 Pro (24 GB unified
memory, MPS, bf16). Checkpoint trajectory is the longest single
experiment but is trivially parallelizable if GPU access is achieved.

## Experimental ordering

1. **Experiment 5 first (30 min).** Dense vs. MoE layer-level
   comparison. Reuses existing Paper 1 infrastructure with zero new
   code. Produces a publishable comparison figure immediately and
   validates that OLMoE loads and hooks correctly.

2. **Experiment 1 (30 min).** Per-expert probing on the final
   checkpoint. The central question: does specialization exist?
   If the heatmap is uniform (no specialization), the paper pivots
   to a null-result framing (MoE does not change moral encoding
   geometry; still publishable, different thesis emphasis). If
   specialization exists, Experiments 2–4 characterize it.

3. **Experiment 2 (15 min).** Router analysis. Quick, builds on the
   Experiment 1 forward pass. Tells us whether routing *causes*
   the specialization pattern or merely correlates with it.

4. **Experiment 3 (1 hour).** Expert-level fragility. Extends Paper 1
   methodology to the expert level. Only useful if Experiment 1
   shows specialization.

5. **Experiment 4 (8–10 hours).** Trajectory. The most expensive
   experiment; run last, after the snapshot findings justify it.

6. **Experiment 6 (stretch, 2–3 hours).** Fine-tuning comparison.
   Only if Experiments 1–5 produce a clear specialization signal.

## Open items

- **Verify OLMoE hook registration.** `OlmoeForCausalLM` module
  structure needs inspection: identify the exact module paths for
  (a) individual expert FFN outputs, (b) router logits, (c) combined
  MoE layer output. Write a small test script that loads the model,
  registers hooks on one expert, and collects activations on a
  single sentence. ~30 min.

- **Expert activation accumulation strategy.** With top-8 routing,
  each expert fires on ~12.5% of tokens. Need to verify this is
  uniform enough in practice (load-balancing loss should ensure it)
  and that 240 pairs provide sufficient per-expert sample size.
  Calculate expected per-expert N and check statistical power.

- **4-bit quantization impact on probing.** If bf16 memory is tight
  during trajectory analysis (loading model + caching activations
  for 1024 expert-layer probes), 4-bit may be necessary. Need to
  verify that quantization does not destroy per-expert probe signal.
  Quick sanity check: run Experiment 1 in bf16 and 4-bit on the
  final checkpoint and compare accuracy heatmaps.

- **Null-result contingency.** If Experiment 1 shows no expert moral
  specialization (uniform heatmap), the paper becomes "MoE does not
  change moral encoding geometry," a publishable null with the
  dense comparison (Experiment 5) as the anchor. Plan the null-
  result framing before running experiments so we don't waste time
  if the signal isn't there.

- **Paper 1 §5.3 / §4.4 update.** Promote the C15 fragility-locus
  finding from a one-paragraph §5.3 mention to a proper Paper 1
  finding (§4.4 data-curation subsection or new §4.5). Update
  Paper 1's PAPER_PLAN.md framing decisions accordingly.

## Cite list (anchor references)

- Muennighoff et al. (2024) — OLMoE paper
- Groeneveld et al. (2024) — OLMo
- Shazeer et al. (2017) — MoE seminal
- Fedus et al. (2022) — Switch Transformer
- Haidt (2012); Graham et al. (2013) — MFT
- Reblitz-Richardson (2026, Paper 1) — fragility methodology
- Reblitz-Richardson (2026, prior Paper 2 work) — persona-feature compound scaling boundary
- Wang et al. (2025) — toxic-persona direction
- Betley et al. (2025) — emergent misalignment

## Drafting order

1. **Run Experiments 1–2** before any drafting. The paper's thesis
   depends on whether specialization exists.
2. **§3 Methodology** — concrete, write from the experimental design
   above.
3. **§4 Results** — write from experimental outputs.
4. **§2 Related Work** — expand after results are known.
5. **§5 Discussion** — interpretation depends on results.
6. **§1 Introduction** — last, once scope is clear.
7. **§6 Conclusion** — last.
8. **Abstract** — last.

Don't draft beyond what's specified here without checking back.
