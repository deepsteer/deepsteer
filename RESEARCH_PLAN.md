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

## Hypotheses

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
| **Primary** | `allenai/Olmo-3-1025-7B` | 7B | 32 | 5.93T | Many @ `stage1-stepXXX` | Native HF |
| **Scale comparison** | `allenai/Olmo-3-1125-32B` | 32B | — | — | Available | Native HF |

OLMo 3 is the primary target: latest model, most training tokens (5.93T on
Dolma 3), most relevant to Ai2's current work. OLMo 2 1B is used only to
validate the pipeline before committing GPU time to 7B runs.

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
| Phase C | GPU instance (A100/H100) | 40-80 GB | OLMo-3 7B full sweep + 32B |

---

## Experiments

### Phase A: Pipeline Validation (OLMo-2 1B, Mac)

**Goal:** Verify every probe runs end-to-end and produces valid output.

| ID | Experiment | Checkpoints | Probe | Output |
|----|-----------|-------------|-------|--------|
| A1 | Smoke test | 1 (final) | LayerWiseMoralProbe | Layer accuracy curve |
| A2 | Mini trajectory | 5 evenly spaced | CheckpointTrajectoryProbe | Small heatmap |
| A3 | Foundation check | 1 (final) | FoundationSpecificProbe | Per-foundation curves |
| A4 | Causal check | 1 (final) | MoralCausalTracer | Causal effect heatmap |
| A5 | Fragility check | 1 (final) | MoralFragilityTest | Noise robustness curves |

**Success criteria:** All probes produce valid JSON + plots. No crashes. Results
are at least directionally plausible (not random noise).

### Phase B: Primary Results (OLMo-3 7B, Mac)

**Goal:** Produce paper-quality results testing H1-H6.

| ID | Experiment | Checkpoints | Probe | Hypothesis |
|----|-----------|-------------|-------|------------|
| B1 | Full layer probing | Final checkpoint | LayerWiseMoralProbe | H1 |
| B2 | Checkpoint trajectory | 15-20 evenly spaced across stage1 | CheckpointTrajectoryProbe | H2, H3 |
| B3 | Foundation emergence | 8 checkpoints (early/mid/late) | FoundationSpecificProbe | H4 |
| B4 | Causal tracing | 3 checkpoints (early/mid/late) | MoralCausalTracer | H5 |
| B5 | Fragility evolution | 5 checkpoints | MoralFragilityTest | H6 |

**Checkpoint selection for B2:** We don't know how many stage1 checkpoints
exist for OLMo-3 7B yet. First step is to list all revisions, then select
15-20 evenly spaced across the full 5.93T-token training run.

**Memory management:** Load one checkpoint at a time. Run all probes on that
checkpoint. Free model, clear MPS cache. Load next checkpoint.

### Phase C: Scale & Sweep (GPU)

**Goal:** Cross-scale comparison and dense checkpoint sweep.

| ID | Experiment | Model | Checkpoints |
|----|-----------|-------|-------------|
| C1 | Dense trajectory | OLMo-3 7B | All available stage1 checkpoints |
| C2 | 32B comparison | OLMo-3 32B | 5-10 checkpoints |
| C3 | Cross-scale | 7B vs 32B | Matched token counts |

---

## Key Figures for Paper/Demo

### Figure 1: The Moral Emergence Heatmap (headline result)
- X-axis: training checkpoint (tokens seen)
- Y-axis: layer index (0 = embedding, N = final)
- Color: probing accuracy (RdYlGn, 0.4–1.0)
- Shows moral concepts literally appearing in the network over training time
- Annotate onset layer trajectory as an overlay line

### Figure 2: Depth and Breadth Over Training
- Dual Y-axis line plot
- Left Y: moral_encoding_depth (expected: decreasing)
- Right Y: moral_encoding_breadth (expected: increasing)
- X: training tokens
- Shows the deepening and broadening simultaneously

### Figure 3: Foundation Emergence Staggering
- One curve per MFT foundation showing onset_layer vs training checkpoint
- Reveals which moral concepts the model learns first from raw text
- Potentially the most novel finding — no one has measured this

### Figure 4: Causal vs Correlational
- Scatter plot: peak_causal_layer vs peak_probing_layer across checkpoints
- Validates that probing reflects genuine computation, not artifacts

### Figure 5: Robustness Evolution
- X: training tokens, Y: critical_noise (noise level to break probe)
- One curve per layer group (early/mid/late)
- Shows moral representations becoming more noise-robust over training

### Figure 6: Cross-Scale Comparison (Phase C)
- Overlay of moral emergence curves for 7B vs 32B
- Normalized x-axis (fraction of total training tokens)
- Tests whether scale affects emergence timing

---

## Implementation Checklist

### Before Phase A
- [ ] Verify `OLMo-2-0425-1B-early-training` loads with `AutoModelForCausalLM`
- [ ] Verify `_detect_n_layers()` and `_get_layer_module()` work for OLMo-2/3
- [ ] List all available revisions for both target models
- [ ] Confirm probing dataset builds correctly (240 pairs, 6 foundations)
- [ ] Write `examples/moral_emergence.py` driver script

### Before Phase B
- [ ] Phase A completes with valid results
- [ ] List OLMo-3 7B stage1 revisions and select checkpoint subset
- [ ] Test that OLMo-3 7B fits on Mac MPS in BF16 with activation capture
- [ ] Tune probe hyperparameters if needed (epochs, lr, threshold)

### Before Phase C
- [ ] Provision GPU instance
- [ ] Phase B results are promising enough to warrant scale-up
- [ ] Verify OLMo-3 32B loads on target GPU

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| OLMo-3 7B doesn't fit on Mac with activations | Use float16 instead of bfloat16; capture fewer layers per pass; fall back to OLMo-2 1B for Mac results |
| Moral probing accuracy is near chance | Verify dataset quality; try different pooling (last token vs mean); increase probe capacity (MLP instead of linear) |
| No emergence pattern (flat across checkpoints) | Check very early checkpoints (step 0-1000); moral signal may appear very early; try foundation-specific probes for more sensitivity |
| OLMo-3 has few stage1 checkpoints | Fall back to OLMo-2 1B (37 early checkpoints) or OLMo-7B-0724-hf (many checkpoints, HF format) |
| Causal tracing is too slow on 7B | Run on subset of prompts; parallelize across layers; defer to GPU phase |

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

---

## Success Criteria

**Minimum viable demo (for Ai2 conversation):**
- Figure 1 (emergence heatmap) showing clear temporal pattern on OLMo-3 7B
- At least 3 of 6 hypotheses supported by data
- All results reproducible from JSON metadata

**Paper-worthy:**
- All 6 hypotheses tested with statistical rigor
- Cross-scale comparison (7B vs 32B)
- Foundation emergence staggering result (Figure 3)
- Clear narrative: moral concepts emerge from raw text, deepen over training,
  and this can be monitored/steered
