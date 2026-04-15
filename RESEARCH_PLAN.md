# Research Plan: The Moral Emergence Curve

## Mapping How Moral Concepts Emerge During Pre-Training in OLMo

### Abstract

We use linear probing, causal tracing, and fragility testing across intermediate
training checkpoints of OLMo base models to characterize *when and how* moral
concepts become decodable from model representations during pre-training. No
explicit moral instruction is given — the model learns next-token prediction on
web text. Yet we hypothesize that moral representations emerge, deepen, broaden,
and become increasingly robust over the course of training. This has never been
systematically mapped. OLMo's open intermediate checkpoints make it uniquely
possible.

### Why This Matters

Most alignment research focuses on post-training interventions (RLHF, DPO,
Constitutional AI). But if moral concepts are already forming during
pre-training, then:

1. **Pre-training data curation** is a viable alignment lever — perhaps the most
   durable one, since it shapes the model's foundational representations rather
   than adding a behavioral veneer.
2. **Monitoring probes during training** could detect alignment-relevant changes
   in real time, enabling early intervention.
3. **The emergence timing** tells us whether moral reasoning piggybacks on
   general language competence or requires specific data exposure.

DeepSteer is the toolkit that makes this analysis reproducible and scalable.
A compelling demo on OLMo positions it for adoption by Ai2 and the broader
open-weights community.

---

## Hypotheses (Phase A/B)

### H1: Emergent Moral Decodability
**Moral concepts are linearly decodable from base model hidden states.**

Linear probes trained on moral-vs-neutral sentence pairs achieve well above
chance (>70%) accuracy at multiple layers in the fully-trained OLMo checkpoint,
despite zero explicit moral training signal.

*Probe: LayerWiseMoralProbe on final checkpoint.*

### H2: Moral Encoding Deepens Over Training
**The onset layer moves earlier as training progresses.**

Early checkpoints encode moral distinctions only in late layers (high depth
ratio). As training continues, earlier layers begin to carry moral signal —
the onset layer drops.

*Metric: `moral_encoding_depth` = onset_layer / n_layers, tracked across
checkpoints. Expected: monotonically decreasing.*

### H3: Moral Encoding Broadens Over Training
**The fraction of layers carrying moral signal increases.**

Early checkpoints show moral information concentrated in a few layers. Mature
checkpoints distribute it across many layers.

*Metric: `moral_encoding_breadth` = fraction of layers above 60% threshold.
Expected: monotonically increasing.*

### H4: Differential Foundation Emergence
**Different moral foundations emerge at different training stages.**

CARE_HARM (frequent in training data — harm, suffering, protection) becomes
decodable earliest. Abstract foundations like SANCTITY_DEGRADATION and
LIBERTY_OPPRESSION emerge later, requiring more data exposure.

*Probe: FoundationSpecificProbe at multiple checkpoints. Expected: staggered
onset across foundations.*

### H5: Causal-Probing Alignment
**Causally important layers correlate with high-probing-accuracy layers.**

Layers identified by activation patching (causal tracing) as most important for
moral-vs-neutral next-token predictions overlap with layers where linear probes
achieve peak accuracy. This validates that probing reflects genuine
computational role, not just correlated features.

*Probe: MoralCausalTracer vs LayerWiseMoralProbe. Expected: peak_causal_layer
≈ peak_probing_layer (within ±2 layers).*

### H6: Moral Robustness Increases Over Training
**Mature moral representations withstand more activation noise.**

At early checkpoints, small Gaussian noise injected into moral-encoding layers
destroys probe accuracy. At late checkpoints, the same noise level has less
effect — the representation is more robust.

*Probe: MoralFragilityTest at early vs late checkpoints. Expected:
critical_noise increases over training.*

---

## Experimental Design

### Target Models

| Role | Model | Params | Layers | Tokens | Checkpoints | Format |
|------|-------|--------|--------|--------|-------------|--------|
| **Dev/debug** | `allenai/OLMo-2-0425-1B-early-training` | 1B | 16 | 77B (first 37k steps) | 37 @ every 1k steps | Native HF |
| **Primary (Phase B)** | `allenai/Olmo-3-1025-7B` | 7B | 32 | 5.93T | Many @ `stage1-stepXXX` | Native HF |
| **Primary (Phase C)** | `allenai/OLMo-2-0425-1B-early-training` | 1B | 16 | 77B | 37 @ every 1k steps | Native HF |

### Probing Dataset

DeepSteer's built-in dataset pipeline produces 240 minimal pairs (480
sentences), 40 per MFT foundation. Each pair is structurally matched: same word
count (±1), same syntactic skeleton, with morally-charged words swapped for
mundane equivalents. No API calls required — fully deterministic.

Validation gates: length ratio, word overlap, keyword scan, deduplication.
Train/test split: 80/20 stratified by foundation.

### Hardware

| Phase | Hardware | Memory | Model |
|-------|----------|--------|-------|
| Phase A | MacBook Pro M4 Pro | 24 GB unified | OLMo-2 1B (debug) |
| Phase B | MacBook Pro M4 Pro | 24 GB unified | OLMo-3 7B (BF16 ~14GB + ~4GB activations) |
| Phase C | MacBook Pro M4 Pro | 24 GB unified | OLMo-2 1B (~2GB + activations) |

---

## Experiments

### Phase A: Pipeline Validation (OLMo-2 1B, Mac) — COMPLETE

**Goal:** Verify every probe runs end-to-end and produces valid output.

| ID | Experiment | Checkpoints | Probe | Output |
|----|-----------|-------------|-------|--------|
| A1 | Smoke test | 1 (final) | LayerWiseMoralProbe | Layer accuracy curve |
| A2 | Mini trajectory | 5 evenly spaced | CheckpointTrajectoryProbe | Small heatmap |
| A3 | Foundation check | 1 (final) | FoundationSpecificProbe | Per-foundation curves |
| A4 | Causal check | 1 (final) | MoralCausalTracer | Causal effect heatmap |
| A5 | Fragility check | 1 (final) | MoralFragilityTest | Noise robustness curves |

**Result:** All probes produce valid JSON + plots. No crashes. Pipeline validated.

### Phase B: Primary Results (OLMo-3 7B, Mac) — COMPLETE

**Goal:** Produce paper-quality results testing H1-H6.

| ID | Experiment | Checkpoints | Probe | Hypothesis |
|----|-----------|-------------|-------|------------|
| B1 | Full layer probing | Final checkpoint | LayerWiseMoralProbe | H1 |
| B2 | Checkpoint trajectory | 20 evenly spaced across stage1 | CheckpointTrajectoryProbe | H2, H3 |
| B3 | Foundation emergence | 8 checkpoints (early/mid/late) | FoundationSpecificProbe | H4 |
| B4 | Causal tracing | 3 checkpoints (early/mid/late) | MoralCausalTracer | H5 |
| B5 | Fragility evolution | 5 checkpoints | MoralFragilityTest | H6 |

**Runtime:** ~6,680s (~1.9 hours), 20 unique checkpoints processed.

---

## Phase B Results

### B1: Layer Probing on Final Checkpoint (H1)

**H1 strongly supported.** The fully-trained OLMo-3 7B (step 1,413,814) encodes
moral concepts with extraordinary clarity:

- **95.8%–100% accuracy at every layer** (0 through 31)
- Peak accuracy: 100% at layers 10–12
- onset_layer: 0, moral_encoding_depth: 0.0, moral_encoding_breadth: 1.0
- Loss minimized in mid-network (layers 10–14), higher at edges

A simple linear classifier perfectly separates moral from neutral sentences
using representations from *any* layer of the trained base model. This far
exceeds the H1 threshold of >70%.

### B2: Checkpoint Trajectory (H2, H3)

**H2/H3 technically supported but the pattern is not gradual — it is a sharp
phase transition.**

| Step | Peak Acc | Onset Layer | Depth | Breadth |
|---:|---:|---:|---:|---:|
| 0 | 61.5% | 1 | 0.031 | 0.031 |
| 74,000 | 100% | 0 | 0.0 | 1.0 |
| 148,000 | 100% | 0 | 0.0 | 1.0 |
| ... | 98.9–100% | 0 | 0.0 | 1.0 |
| 1,413,814 | 100% | 0 | 0.0 | 1.0 |

The model transitions from chance-level (61.5%) to perfect decodability at
every layer between step 0 and step 74,000 — the first ~5% of training. After
that, depth=0.0 and breadth=1.0 for all 19 remaining checkpoints. The metrics
are completely saturated and provide no resolution for the remaining 95% of
training.

The emergence heatmap shows a single red column (step 0) followed by a uniform
green field. No gradual onset-layer descent or breadth expansion is visible at
this checkpoint resolution.

### B3: Foundation Emergence (H4)

**H4 weakly supported.** A hierarchy is visible at initialization, but all
foundations saturate simultaneously at the sampling resolution used.

Step 0 (random init) foundation hierarchy:

| Foundation | Step 0 Peak Acc | Step 0 Onset |
|---|---:|---|
| care_harm | 81.3% | layer 0 |
| sanctity_degradation | 75.0% | layer 0 |
| authority_subversion | 68.8% | layer 0 |
| liberty_oppression | 68.8% | layer 1 |
| loyalty_betrayal | 62.5% | layer 30 |
| fairness_cheating | 56.3% | none (below threshold) |

By step 201,000 (first B3 checkpoint after init), all 6 foundations hit 100%
peak accuracy with onset at layer 0 and breadth 1.0. The emergence is
simultaneous at this resolution. Only loyalty_betrayal shows a dramatic
trajectory (from layer 30 to layer 0).

Notable: **authority_subversion is the most unstable foundation**, dipping to
93.8% peak accuracy during steps 805K–1,211K before recovering to 100% at the
final checkpoint. All other foundations maintain perfect accuracy.

### B4: Causal Tracing (H5)

**H5 partially supported.** Both causal and probing peaks move earlier over
training, but they diverge rather than align.

| Checkpoint | Peak Causal Layer | Peak Probing Layer | Mean Indirect Effect |
|---:|---:|---:|---:|
| Step 0 | 6 | 1 | 0.37 |
| Step 705K | 7 | 13 | 9.09 |
| Step 1,414K | 0 | 10 | 9.60 |

Causal effect magnitude increases 26x over training (0.37 → 9.60), and the peak
causal layer migrates from 6 → 7 → 0. But the probing peak sits at layer 10 at
the final checkpoint — a 10-layer gap.

This suggests probing detects *where moral information is stored* while causal
tracing detects *where moral information is used*. These are different layers.

### B5: Fragility Evolution (H6)

**H6 strongly supported, with a novel layer-depth robustness gradient.**

| Checkpoint | Mean Critical Noise | Most Robust Layer |
|---:|---:|---:|
| Step 0 | 0.20 | 2 |
| Step 353K | 5.67 | 7 |
| Step 705K | 5.17 | 11 |
| Step 1,059K | 5.48 | 10 |
| Step 1,414K | 5.31 | 15 |

Moral representations become 28x more noise-robust in the first 353K steps,
then plateau. The most interesting finding is the **layer-depth robustness
gradient** that emerges:

- Late layers (22–31): critical noise ~10.0 — maximally robust
- Mid layers (11–21): critical noise ~5.5 — moderately robust
- Early layers (0–10): critical noise ~2.0 — least robust

This gradient persists from step 353K onward. The most robust layer shifts
deeper over time (2 → 7 → 11 → 10 → 15), meaning the model progressively
builds deeper noise-resistant representations even as the shallow probing
signal stays saturated. **Fragility is the most informative metric** in this
study — it provides resolution where probing accuracy cannot.

### Hypothesis Scorecard

| # | Hypothesis | Verdict | Notes |
|---|---|---|---|
| H1 | Emergent Moral Decodability | **Strongly supported** | 95.8–100% at all layers |
| H2 | Encoding Deepens | **Saturated** | Depth hits 0.0 by step 74K; no gradual trajectory visible |
| H3 | Encoding Broadens | **Saturated** | Breadth hits 1.0 by step 74K; no gradual trajectory visible |
| H4 | Differential Emergence | **Weakly supported** | Hierarchy at step 0 but all saturate by 201K |
| H5 | Causal-Probing Alignment | **Partially supported** | Both move earlier but diverge (10-layer gap) |
| H6 | Robustness Increases | **Strongly supported** | 28x increase; novel layer-depth gradient |
| H7 | Phase Transition Has Structure | **Supported** | Sigmoid over ~3K steps; inflection at step 1K (C1) |
| H9 | Narrative > Declarative Robustness | **Partially supported** | Declarative creates fragility; narrative ≈ general (C3) |

### Methodology Lessons

1. **Probe saturation.** The 60% onset threshold is too low — even the
   random-init model hits 61.5% at layer 1. With 48 test pairs and 50 epochs,
   the task is too easy once the model has any training at all. Depth/breadth
   metrics provide zero resolution for 95% of training.

2. **Checkpoint resolution gap.** The entire phase transition occurs between
   step 0 and step 74,000. We have zero visibility into this critical window.
   The gradual deepening/broadening predicted by H2/H3 might exist within
   those 74K steps.

3. **Dataset difficulty.** 100% accuracy at 19/20 checkpoints suggests the
   moral-vs-neutral pairs may be too easily separable via surface-level
   lexical cues rather than deep moral representations.

4. **Fragility is the best metric.** Unlike probing accuracy, fragility reveals
   genuine layer-wise structure even when accuracy is saturated. The robustness
   gradient (late > mid > early) provides resolution that other metrics cannot.

5. **Causal-probing divergence is a finding.** The 10-layer gap between causal
   and probing peaks is not a failure of H5 — it reveals that storage and
   computation of moral information happen at different layers.

---

## Phase C: Moral Data Curation Experiments (OLMo-2 1B, Mac)

### Motivation

Phase B demonstrated that moral representations emerge as a sharp phase
transition within the first ~5% of training, then saturate. This is consistent
with moral concepts being learned as part of basic language competence rather
than requiring specialized data exposure. But Phase B cannot answer the *causal*
question: would deliberately including moral content (fables, children's
stories, moral dilemmas) in early training data produce *stronger, deeper, or
more robust* moral representations?

This question matters because if moral representations emerge from statistical
regularities in general text (co-occurrence of "murder" with negative
sentiment), they may be shallow — easily disrupted by fine-tuning, brittle
under distribution shift, and unable to support genuine moral reasoning. If
instead moral representations can be *strengthened* by deliberate early exposure
to moral narratives, this would validate pre-training data curation as an
alignment lever.

### Resources

Phase C uses OLMo-2 1B exclusively, running on MacBook Pro M4 Pro (24GB).

**Available checkpoints:**
- `allenai/OLMo-2-0425-1B-early-training`: 37 checkpoints at 1K-step
  intervals (step 0 through step 36K, ~0–77B tokens). Ideal for dense
  phase-transition mapping.
- `allenai/OLMo-2-0425-1B`: 268 checkpoints spanning the full training run
  (step 0 through step 1,050K, ~0–2.2T tokens). Provides the complete
  trajectory if needed.

**Fine-tuning feasibility:** OLMo-2 1B is ~3GB in float16. LoRA fine-tuning
(rank 16–32, targeting q_proj/v_proj) uses ~5GB total on MPS, leaving ample
headroom for batch_size=2–4 at seq_len=1024. Full fine-tuning is marginal
(~19GB) but possible at batch_size=1 with gradient accumulation. LoRA is the
recommended path for curriculum experiments.

### Phase C Hypotheses

#### H7: The Phase Transition Has Internal Structure
**The step 0 → 74K jump observed in Phase B is not instantaneous — it unfolds
over multiple checkpoints with a characteristic S-curve.**

OLMo-2 1B's 37 checkpoints at 1K-step intervals should resolve the transition
that Phase B could not. We expect: (a) an initial plateau near chance, (b) a
rapid rise, (c) saturation. The inflection point reveals when moral encoding
truly begins.

*Metric: Per-layer probing accuracy at all 37 checkpoints. Expected: sigmoidal
trajectory for mid-layer accuracy.*

#### H8: Moral Encoding Emerges After General Linguistic Competence
**Moral probing accuracy lags behind general-purpose probing accuracy by a
measurable number of training steps.**

If moral representations require specific content exposure, they should emerge
later than representations for general linguistic features (part-of-speech,
syntax, sentiment). If they emerge simultaneously with general competence, it
suggests moral concepts are learned "for free" from basic language statistics.

*Method: Compare moral probe onset step to probes for syntactic features
(subject-verb agreement) and simple semantic features (sentiment polarity)
on the same 37 checkpoints. The step gap (or lack thereof) is the key
measurement.*

#### H9: Narrative Moral Content Produces More Robust Representations Than Declarative Statements
**LoRA fine-tuning on moral narratives (fables, stories with moral lessons)
strengthens moral robustness more than equivalent exposure to declarative
moral statements.**

Fables and children's stories embed moral reasoning in narrative structure —
characters face dilemmas, make choices, and experience consequences. This
requires deeper processing than "Stealing is wrong." If narrative structure
matters, models fine-tuned on moral stories should show higher fragility
thresholds (more noise-robust representations) than those fine-tuned on the
same moral content in declarative form.

*Method: Take an early OLMo-2 1B checkpoint (pre-transition, e.g. step 0 or
step 1K). LoRA fine-tune (causal LM objective) with two matched corpora:
(A) moral fables/stories, (B) declarative moral sentences with equivalent
moral concept coverage and token count. Run fragility tests on both. Expected:
corpus A produces higher mean_critical_noise.*

#### H10: Early Moral Exposure Deepens Representations More Than Late Exposure
**Introducing moral content at the start of training produces deeper moral
encoding (lower onset layer, higher per-layer accuracy) than introducing it
after the model has already learned general language representations.**

This tests whether there is a "critical period" for moral representation
formation, analogous to critical periods in human development. If
representations are more plastic early in training, early moral exposure should
integrate moral concepts into lower-level features. Late exposure would only
modify higher layers, producing shallower encoding.

*Method: LoRA fine-tune from two OLMo-2 1B checkpoints — one early (step 1K,
pre-transition) and one late (step 30K, post-transition) — with the same
moral curriculum corpus, same LoRA rank/lr/steps. Compare onset layer, peak
accuracy by layer, and fragility gradient. Expected: early-exposure model has
lower onset layer and more uniform per-layer accuracy.*

#### H11: Moral Content Diversity Across Foundations Matters More Than Volume
**Balanced coverage of all 6 MFT foundations produces more uniform per-foundation
probe accuracy than concentrated exposure to a single foundation, even at
lower total volume.**

This tests whether the training data needs to cover the full moral landscape or
whether exposure to one foundation (e.g., care/harm, which is most frequent in
natural text) transfers to others. If foundations share representational
structure, concentrated exposure should transfer. If they are independent,
balanced coverage is necessary.

*Method: LoRA fine-tune from an early checkpoint with three corpora:
(A) balanced across 6 foundations, (B) concentrated on care/harm only, same
total tokens, (C) concentrated on fairness/cheating only. Compare per-
foundation probing accuracy for all 6 foundations. Expected: corpus A produces
more uniform accuracy; B and C show high accuracy only for the exposed
foundation, with partial transfer to related foundations.*

#### H12: Moral Curriculum Accelerates the Phase Transition
**LoRA fine-tuning on moral content at a pre-transition checkpoint produces
the moral phase transition faster (fewer gradient steps) than fine-tuning on
general text.**

If moral concepts are bottlenecked on data exposure rather than general
language competence, targeted moral content should accelerate the transition.
If general competence is the bottleneck, moral content should provide no
advantage over generic text.

*Method: From step 0 (or step 1K), LoRA fine-tune with (A) moral narrative
corpus and (B) matched general-text corpus (same token count, same training
setup). Probe every 50–100 gradient steps. Compare the step at which probing
accuracy first exceeds 80%. Expected: moral corpus reaches the transition
earlier, but the magnitude of acceleration indicates whether moral content is
rate-limiting or general competence is.*

### Phase C Experiments

**Tier 1 — Observational (no training, immediate):**

| ID | Experiment | Hypothesis | Model/Checkpoints | Output |
|----|-----------|------------|-------------------|--------|
| C1 | Dense phase-transition mapping | H7 | All 37 early-training checkpoints | High-res emergence heatmap |
| C2 | Moral vs. linguistic probe comparison | H8 | All 37 early-training checkpoints | Onset-step comparison curves |

C1 is a direct extension of Phase A's A2 (which used 5 checkpoints). Run
LayerWiseMoralProbe, FoundationSpecificProbe, and MoralFragilityTest on all
37 checkpoints. Estimated runtime: ~37 checkpoints x ~5s each ≈ 3 minutes
for probing, plus ~37 x 5s for fragility = ~6 minutes total.

C2 requires building two new probing datasets (sentiment, syntax) following
the same minimal-pair structure as the moral dataset. These are simple to
construct without any API calls — just curated sentence pairs.

**Tier 2 — LoRA fine-tuning experiments (requires corpus assembly first):**

| ID | Experiment | Hypothesis | Base Checkpoint | LoRA Corpus | Output |
|----|-----------|------------|-----------------|-------------|--------|
| C3 | Narrative vs. declarative | H9 | step 0 or 1K | Fables vs. declarations | Fragility comparison |
| C4 | Early vs. late exposure | H10 | step 1K vs. step 30K | Same moral corpus | Depth/fragility comparison |
| C5 | Foundation coverage | H11 | step 1K | Balanced vs. concentrated | Per-foundation accuracy |
| C6 | Moral acceleration | H12 | step 0 | Moral vs. general text | Transition-step comparison |

**LoRA training setup:**
- Adapter: LoRA rank 16, alpha 32, targeting q_proj + v_proj
- Optimizer: AdamW, lr=2e-4, cosine schedule
- Training: ~500–2000 gradient steps, batch_size=2, seq_len=1024
- Device: MPS (float16), ~5GB memory footprint
- Evaluation: Probe (accuracy + fragility) every 100 steps during fine-tuning
- Controls: Each experiment includes a matched general-text control fine-tuned
  with identical hyperparameters

**Estimated per-experiment runtime:** ~30–60 minutes for LoRA training +
probing on MPS (1B model, ~1000 steps).

### Moral Curriculum Corpora

All corpora should be small enough for Mac-feasible LoRA fine-tuning. Target
~500K–2M tokens per corpus (not 5M — LoRA adapts fast, and we want to stay
within ~1000 gradient steps at batch_size=2, seq_len=1024).

| Corpus | Content | Source | Tokens | Purpose |
|--------|---------|--------|--------|---------|
| **Narrative-moral** | Aesop's Fables, Grimm's tales, Panchatantra, Jataka tales, children's moral stories | Project Gutenberg (public domain) | ~500K–1M | H9, H10, H12 |
| **Declarative-moral** | Explicit moral statements and ethical principles matched to the same MFT foundations covered by the narrative corpus | Hand-curated / generated | ~500K–1M | H9 (control) |
| **Foundation-balanced** | Moral content balanced across all 6 MFT foundations | Subset of above | ~500K | H11 |
| **Foundation-concentrated** | Moral content from a single MFT foundation (care/harm or fairness/cheating) | Subset of above | ~500K | H11 (control) |
| **General-text** | Non-moral text sampled from a public corpus (e.g. Wikipedia, CC) | Public domain | ~500K–1M | All (control) |

Corpus assembly is the main bottleneck. Project Gutenberg texts for Aesop's
Fables (~50K tokens), Grimm's Fairy Tales (~200K tokens), and similar public
domain collections can be downloaded and tokenized without any special
infrastructure. The declarative corpus can be generated deterministically
using the same seed structure as our existing moral probing dataset, expanded
to paragraph-length.

### Key Figures for Phase C

#### Figure 7: High-Resolution Phase Transition (C1)
- X-axis: training step (0–36K at 1K intervals)
- Y-axis: layer index (0–15 for OLMo-2 1B)
- Color: probing accuracy
- Should resolve the step-0-to-74K gap from Phase B
- Annotate inflection point of the S-curve

#### Figure 8: Moral vs. Linguistic Emergence Timing (C2)
- X-axis: training step
- Y-axis: probing accuracy (mean across layers)
- Three curves: moral probe, sentiment probe, syntax probe
- The horizontal gap between onset points is the key measurement
- Determines whether moral encoding is "free" or requires specific exposure

#### Figure 9: Narrative vs. Declarative Robustness (C3)
- X-axis: layer index
- Y-axis: critical noise level
- Two curves: narrative-LoRA vs. declarative-LoRA vs. general-text-LoRA
- If narrative structure matters, the narrative curve should be uniformly higher

#### Figure 10: Critical Period for Moral Exposure (C4)
- X-axis: layer index
- Y-axis: probing accuracy
- Two curves: early-exposure (step 1K + LoRA) vs. late-exposure (step 30K + LoRA)
- If a critical period exists, early-exposure should show higher accuracy in
  early/mid layers

#### Figure 11: Foundation Coverage Effects (C5)
- 6x3 heatmap: foundations (rows) x training corpora (columns)
- Color: probing accuracy
- Shows transfer (or lack thereof) between foundations

#### Figure 12: Curriculum Acceleration (C6)
- X-axis: LoRA gradient step
- Y-axis: probing accuracy (mean across layers)
- Two curves: moral-content LoRA vs. general-text LoRA
- Horizontal shift between curves measures acceleration

---

## Implementation Checklist

### Before Phase A — COMPLETE
- [x] Verify `OLMo-2-0425-1B-early-training` loads with `AutoModelForCausalLM`
- [x] Verify `_detect_n_layers()` and `_get_layer_module()` work for OLMo-2/3
- [x] List all available revisions for both target models
- [x] Confirm probing dataset builds correctly (240 pairs, 6 foundations)
- [x] Write `examples/moral_emergence.py` driver script

### Before Phase B — COMPLETE
- [x] Phase A completes with valid results
- [x] List OLMo-3 7B stage1 revisions and select checkpoint subset
- [x] Test that OLMo-3 7B fits on Mac MPS in BF16 with activation capture
- [x] Tune probe hyperparameters if needed (epochs, lr, threshold)

### Phase C Tier 1 (Observational — no training)
- [x] Run LayerWiseMoralProbe on all 37 early-training checkpoints (C1)
- [x] Run MoralFragilityTest on all 37 early-training checkpoints (C1)
- [x] Run FoundationSpecificProbe on all 37 early-training checkpoints (C1)
- [x] Generate Figure 7 (high-resolution phase transition heatmap)
- [ ] Build sentiment probing dataset (positive/negative minimal pairs) (C2)
- [ ] Build syntax probing dataset (grammatical/ungrammatical minimal pairs) (C2)
- [ ] Run sentiment and syntax probes on all 37 checkpoints (C2)
- [ ] Generate Figure 8

### Phase C Tier 2 (LoRA fine-tuning)
- [x] Download Aesop's Fables + Grimm's Tales from Project Gutenberg
- [x] Tokenize and prepare narrative-moral corpus (247K tokens)
- [x] Hand-curate declarative-moral corpus with matched foundation coverage (500K tokens)
- [x] Sample general-text control corpus (Darwin, 420K tokens)
- [x] Implement LoRA fine-tuning script with periodic probing callbacks
- [x] Verify LoRA fine-tuning runs on MPS with OLMo-2 1B
- [x] Run C3 (narrative vs. declarative) — signal detected (0.583)
- [ ] Run C6 (moral acceleration from random init)
- [ ] Run C4 (early vs. late exposure) — warranted by C3 signal > 0.10
- [ ] Run C5 (foundation coverage) — warranted by C3 signal > 0.10

---

## Phase C1 Results: Dense Phase-Transition Mapping

**Experiment:** All 37 OLMo-2 1B early-training checkpoints (step 0 to step
36K at 1K intervals) probed with LayerWiseMoralProbe, FoundationSpecificProbe,
and MoralFragilityTest. Runtime: 89 minutes on MacBook Pro M4 Pro.
Output: `outputs/phase_c1/`.

**Hypothesis tested:** H7 — the step 0 → 74K phase transition observed in
Phase B has internal structure resolvable with denser checkpoint sampling.

**Verdict: H7 supported.** The transition is steep but not instantaneous —
it unfolds as a sigmoid over ~3K steps (~6B tokens).

### Finding 1: Probing Accuracy Follows a Steep Sigmoid (Steps 0–4K)

| Step | Tokens | Mean Acc | Peak Acc | Depth | Breadth |
|---:|---:|---:|---:|---:|---:|
| 0 | 0B | 55.7% | 63.5% | 0.062 | 0.188 |
| 1000 | ~3B | 76.8% | 82.3% | 0.0 | 1.0 |
| 2000 | ~4B | 88.3% | 91.7% | 0.0 | 1.0 |
| 3000 | ~6B | 92.1% | 94.8% | 0.0 | 1.0 |
| 4000 | ~8B | 93.8% | 95.8% | 0.0 | 1.0 |
| 5000–36000 | ~11–76B | 93.7–96.1% | 95.8–97.9% | 0.0 | 1.0 |

The inflection point is at step 1000 (~3B tokens): mean accuracy jumps 21
percentage points (55.7% → 76.8%) in a single 1K-step interval. By step
4000 accuracy plateaus at ~95%. Depth and breadth saturate at their limits
(0.0, 1.0) from step 1K onward.

**Limitation for Tier 2:** Probing accuracy is too coarse to distinguish
between LoRA interventions — any trained model will likely show ~95%+ after
even minimal fine-tuning. Fragility (Finding 2) is the metric that can
differentiate.

### Finding 2: A Layer-Depth Robustness Gradient Emerges and Steepens

Fragility testing reveals dynamics invisible to probing accuracy. Mean
critical noise (the noise magnitude at which probe accuracy drops below
the fragility threshold) measured per layer group:

| Layer Group | Step 0 | Step 1K | Step 10K | Step 20K | Step 36K |
|---|---:|---:|---:|---:|---:|
| Late (11–15) | 0.1 | 10.0 | 10.0 | 10.0 | 10.0 |
| Mid (6–10) | 0.1 | 10.0 | 10.0 | 10.0 | 5.8 |
| Early (0–5) | 0.3 | 10.0 | 7.7 | 6.5 | 1.7 |
| **All-layer mean** | **0.18** | **10.0** | **7.0** | **7.5** | **5.3** |

Two distinct phases of fragility dynamics:

1. **Steps 0–1K (acquisition):** All layers jump from near-zero to maximum
   robustness (mean 0.18 → 10.0) simultaneously with the accuracy
   transition. Moral representations are initially robust everywhere.

2. **Steps 1K–36K (specialization):** Early-layer robustness *declines*
   (10.0 → 1.7) while late-layer robustness holds at 10.0. The model
   progressively moves robust moral encoding into deeper layers, replacing
   the shallow lexical features that early layers initially relied upon.
   This creates an increasingly steep layer-depth gradient.

**Key insight for Tier 2:** Fragility is the only metric that continues to
evolve after step 4K. It should be the primary outcome measure for all LoRA
experiments (C3–C6). Specifically, the *per-layer fragility gradient* (not
just the mean) can distinguish whether moral curriculum produces different
representational structure than general text.

### Finding 3: Moral Foundations Emerge in a Staggered Sequence

Per-foundation peak probing accuracy (max across all 16 layers) at key
checkpoints. Bold marks the first step reaching 100%:

| Foundation | Step 0 | Step 1K | Step 2K | Step 3K | Step 6K | Notes |
|---|---:|---:|---:|---:|---:|---|
| fairness/cheating | 68.8% | **100%** | 100% | 100% | 100% | Fastest to saturate |
| care/harm | 75.0% | 87.5% | **100%** | 100% | 100% | Second fastest |
| sanctity/degradation | 75.0% | 75.0% | 93.8% | **100%** | 100% | |
| loyalty/betrayal | 56.3% | 68.8% | 87.5% | **100%** | 100% | Starts below threshold |
| authority/subversion | 81.3% | 81.3% | 93.8% | 93.8% | **100%** | High init, slow to mature |
| liberty/oppression | 68.8% | 87.5% | 93.8% | 93.8% | 93.8% | **Never reaches 100%** |

Emergence order: fairness → care → sanctity ≈ loyalty → authority →
liberty (never).

Notable patterns:
- **Loyalty/betrayal** starts below the 60% onset threshold at random init
  (56.3%, the only sub-threshold foundation) and is the last to reach 100%
  among those that do.
- **Liberty/oppression** never fully stabilizes — it plateaus at 93.8% and
  fluctuates even at late checkpoints. This mirrors Phase B findings on
  the 7B model, confirming it as a cross-scale pattern.
- **Authority/subversion** has the highest random-init accuracy (81.3%,
  likely noise given 16 test pairs where each pair = 6.25pp) but is one
  of the slowest to reach genuine saturation at 100%.

**Key insight for Tier 2:** Loyalty/betrayal and liberty/oppression are the
foundations most likely to show differential effects from moral curriculum.
If LoRA fine-tuning on balanced moral content (C5) specifically accelerates
these lagging foundations, that is strong evidence for foundation coverage
mattering in training data.

### Methodology Notes

1. **1K-step resolution is sufficient** for the probing phase transition —
   finer resolution would not add information since accuracy saturates by
   step 3–4K.
2. **Fragility needs finer resolution in the step 0–1K window.** The jump
   from 0.18 to 10.0 mean critical noise in a single interval suggests the
   fragility transition has its own internal structure not visible here.
3. **The onset threshold of 0.6 is confirmed too low** — breadth saturates
   at 1.0 by step 1K. An 80% threshold would provide more resolution.
4. **Small per-foundation sample size** (16 test pairs per foundation)
   limits confidence in individual accuracy values — each pair is worth
   6.25pp. The staggered emergence order is qualitatively reliable but
   exact step numbers should be treated as approximate.

### Implications for Phase C Tier 2 (LoRA Experiments)

C1 results provide three concrete design decisions for the upcoming LoRA
experiments:

**1. Base checkpoint selection for C3–C6:**
- **Step 0** (random init): Best for C6 (acceleration test) — measures
  whether moral curriculum triggers the phase transition faster than
  general text.
- **Step 1K** (mid-transition, ~77% mean accuracy): Best for C3 (narrative
  vs. declarative) and C4 (early exposure). The model has begun forming
  moral representations but hasn't saturated, so LoRA interventions have
  room to reshape the transition.
- **Step 30K** (post-saturation, ~96% accuracy, fragility gradient fully
  developed): Required for C4's "late exposure" condition. Tests whether
  curriculum can still modify the robustness gradient after the model has
  stabilized.

**2. Primary outcome metric — fragility, not accuracy:**
Probing accuracy will saturate at ~95%+ regardless of curriculum content.
The metrics that can discriminate between interventions are:
- Per-layer critical noise profile (the shape of the robustness gradient)
- Whether moral curriculum slows the early-layer robustness decline
  (i.e., produces representations that are robust even in shallow layers)
- Foundation-specific fragility for the lagging foundations
  (loyalty/betrayal, liberty/oppression)

**3. Baseline expectations from C1 for comparison:**
- At step 1K + 1000 LoRA steps of general text, we expect accuracy ~95%+
  and a fragility gradient similar to step 4K–10K (early layers losing
  robustness). If moral curriculum produces a *flatter* gradient (early
  layers staying robust), that indicates deeper integration of moral
  concepts.
- At step 0 + 1000 LoRA steps, we can measure how many gradient steps
  are needed to cross the 80% mean accuracy threshold. C1 shows the
  natural transition crosses ~77% at step 1K (~2.1M gradient steps at
  the pre-training batch size). LoRA with targeted moral content should
  reach this threshold in fewer steps if moral data is rate-limiting.

---

## Phase C3 Results: Narrative vs. Declarative Moral Framing

**Experiment:** LoRA fine-tuning (rank 16, alpha 32, q_proj + v_proj) on
OLMo-2 1B at step 1000 (mid-transition checkpoint, ~80% peak probing
accuracy). Three conditions trained for 1000 steps each (lr=2e-4,
batch_size=2, seq_len=1024) with probing and fragility evaluation every
100 steps. Runtime: ~10 hours total on MacBook Pro M4 Pro.
Output: `outputs/phase_c_tier2/c3/`.

**Hypothesis tested:** H9 — narrative moral content (fables, stories)
produces more robust moral representations than declarative moral
statements.

**Verdict: H9 partially supported.** Fragility profiles differ by
condition, but the pattern is more nuanced than predicted.

### Conditions

| Condition | Corpus | Tokens | Source |
|-----------|--------|--------|--------|
| Narrative moral | Aesop's Fables, Grimm's Fairy Tales, Andersen | 247K | Project Gutenberg |
| Declarative moral | Template-expanded MORAL_SEEDS, 50 per foundation | 500K | Generated from seed dataset |
| General control | Darwin's Voyage of the Beagle + Origin of Species | 420K | Project Gutenberg |

### Finding 1: Probing Accuracy Is Stable and Content-Agnostic

All three conditions maintain ~80% peak accuracy throughout 1000 LoRA
steps (range: 0.77–0.81). LoRA fine-tuning on moral vs. non-moral content
does not meaningfully change how well moral concepts can be decoded. The
base checkpoint (step 1K) already has mature enough moral encoding that
accuracy is insensitive to further training content.

| Metric | Narrative | Declarative | General Control |
|--------|-----------|-------------|-----------------|
| Final peak accuracy | **0.812** | 0.802 | 0.802 |
| Final peak layer | 3 | 0 | 1 |

This confirms C1's finding that probing accuracy is too coarse to
distinguish between training interventions once the model has passed the
initial phase transition.

### Finding 2: Fragility Profiles Are Condition-Specific (Main Result)

Each condition produces a distinct per-layer robustness profile:

| Metric | Narrative | Declarative | General Control |
|--------|-----------|-------------|-----------------|
| Mean critical noise | **10.0** | 9.42 | **10.0** |
| Most fragile layer | 0 | **3 (critical noise=3.0)** | 0 |
| Robustness profile | Uniform high | Dip at layer 3 | Uniform high |

- **Narrative moral**: Uniform fragility (critical_noise=10.0 across all
  layers). Most uniformly robust structure.
- **Declarative moral**: Sharp fragility dip at layer 3
  (critical_noise=3.0), creating a single vulnerability point. All other
  layers remain at 10.0.
- **General control**: Uniform fragility (critical_noise=10.0), matching
  the narrative condition.

LoRA fine-tuning on different content types doesn't change *what* the
model knows about morality (accuracy stays flat), but it **reshapes where
moral encoding is fragile**. Data curation affects the structural
organization of representations.

### Finding 3: Training Loss Diverges Dramatically

| Condition | Initial Loss | Final Loss | Interpretation |
|-----------|-------------|------------|----------------|
| Declarative | 5.6 | **1.0** | Easy memorization of templated text |
| Narrative | 5.0 | 3.9 | Diverse natural text |
| General control | 5.0 | 4.5 | No memorization signal |

Despite the declarative corpus being trivially memorizable (loss drops to
1.0), this does not translate to higher probing accuracy or stronger
robustness. Surface-level text learning and representational moral
encoding are decoupled processes.

### Signal Assessment

**C3 signal: 0.583** (threshold: 0.10, computed from max fragility
difference between moral conditions and control).

Signal well above threshold — C4 (early vs. late exposure) and C5
(foundation coverage) experiments are warranted.

### Implications for Remaining Tier 2 Experiments

1. **Fragility confirmed as the discriminating metric.** Accuracy cannot
   distinguish interventions; fragility can. All remaining experiments
   (C4, C5, C6) should use per-layer fragility profiles as the primary
   outcome measure.

2. **The declarative layer-3 vulnerability is unexpected.** Templated
   moral statements create a localized fragility that natural text
   (narrative or non-moral) does not. This suggests that repetitive moral
   framing may create brittle shortcuts at specific layers.

3. **Narrative and general text produce equivalent robustness profiles.**
   The narrative condition did not produce *stronger* robustness than
   general text (both uniformly at 10.0). The difference is between
   declarative (fragile at layer 3) and everything else (uniformly
   robust). This reframes H9: it may not be that narrative content is
   *better* for robustness, but that declarative content is *worse*.

4. **C6 (moral acceleration) should test whether moral content accelerates
   the initial phase transition from step 0.** C3 started from step 1K
   (post-transition onset) — the model already had partial moral encoding.
   Step 0 provides a cleaner test of whether content type matters for
   emergence timing.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Probe saturation masks dynamics (Phase B finding) | Use fragility as primary metric; raise onset threshold to 0.80; use finer noise levels in fragility tests |
| LoRA doesn't change representations enough | Increase rank (32–64); target more modules (k_proj, o_proj, MLP); increase training steps; if still flat, try full fine-tuning at batch_size=1 |
| LoRA adapts only the adapter, not base weights | Probe on base model + adapter merged, and separately on base model with adapter removed, to isolate which representations changed |
| Narrative corpus confounds content with style | Match narrative and declarative corpora on MFT foundation distribution, sentence length, and token count; only vary the structural form |
| Narrative corpus too small for signal | Start with Aesop (~50K tokens) as proof-of-concept; if signal exists, scale up with Grimm, Panchatantra, etc. |
| MPS training instability | Use float16 (not bf16); gradient clipping; monitor loss for NaN; fall back to CPU if MPS produces numerical issues |
| 1B model too small to generalize findings to 7B+ | Frame 1B results as proof-of-concept; note scale caveats; if results are promising, replicate key experiments on 7B with GPU |

---

## Related Work

This study builds on:
- **Alain & Bengio (2017)**: Linear probing methodology
- **Meng et al. (2022)**: Causal tracing / ROME (we adapt for moral concepts)
- **Groeneveld et al. (2024)**: OLMo — open checkpoints enabling this research
- **Haidt (2012)**: Moral Foundations Theory — our taxonomy of moral concepts
- **Greenblatt et al. (2024)**: Alignment faking / compliance gap (we adapt as
  a representational probe)

**Gap we fill:** No prior work systematically tracks the emergence of moral
representations across pre-training checkpoints. Existing probing studies are
snapshot analyses of finished models. The training trajectory dimension is novel.
Phase C extends this by testing whether data curation can *steer* moral
emergence — connecting observational probing to actionable alignment
interventions.

---

## Success Criteria

**Phase B (achieved):**
- Figure 1 (emergence heatmap) showing clear phase transition on OLMo-3 7B
- 3 of 6 hypotheses supported (H1, H5 partial, H6)
- All results reproducible from JSON metadata

**Phase C1 (achieved):**
- High-resolution phase transition map (Figure 7) resolving the step 0–36K
  window with all 37 OLMo-2 1B checkpoints — sigmoid confirmed
- Novel finding: early-layer fragility *increases* with training, creating a
  steepening layer-depth robustness gradient (invisible to probing accuracy)
- Staggered foundation emergence resolved: fairness → care → sanctity ≈
  loyalty → authority → liberty (never reaches 100%)
- Concrete Tier 2 design decisions derived: checkpoint selection, fragility as
  primary metric, baseline expectations for LoRA comparison

**Phase C2 (target):**
- Clear answer to whether moral encoding lags general linguistic competence (H8)
- Figures 7 and 8 produced and interpretable

**Phase C3 (achieved):**
- C3 signal 0.583 — well above 0.10 threshold, warranting C4/C5 follow-ups
- Novel finding: LoRA content type reshapes fragility profiles without changing
  probing accuracy — data curation affects representational *structure*, not
  *existence* of moral encoding
- Declarative moral content creates localized fragility (layer 3) that narrative
  and general text do not — repetitive moral framing may produce brittle shortcuts
- Reproducible LoRA recipe validated on MPS (OLMo-2 1B, rank 16, 1000 steps)

**Phase C Tier 2 remaining (target):**
- C6: Does moral content accelerate the phase transition from random init?
- C4: Does LoRA injection timing (early vs. late) affect fragility gradient shape?
- C5: Do individual MFT foundations have selective effects on per-foundation probing?
- If C6 shows signal: evidence that moral data is rate-limiting for emergence
