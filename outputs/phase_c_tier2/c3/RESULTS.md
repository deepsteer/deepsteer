# Phase C3 Results: Narrative vs. Declarative Moral Framing

## Experiment Design

**Question**: Does the framing of moral content (narrative vs declarative) affect how
LoRA fine-tuning reshapes moral encoding in a language model's representations?

**Model**: OLMo-2-0425-1B at step 1000 (mid-transition, moral encoding partially formed)
**LoRA config**: rank=16, alpha=32, targets=[q_proj, v_proj], dropout=0.05
**Training**: lr=2e-4, batch_size=2, seq_len=1024, 1000 steps, eval every 100 steps

### Three conditions

1. **Narrative moral** (247K tokens): Aesop's Fables, Grimm's Fairy Tales, Andersen
2. **Declarative moral** (500K tokens): Template-expanded MORAL_SEEDS (50/foundation)
3. **General control** (420K tokens): Darwin's Voyage of the Beagle + Origin of Species

## Final Metrics Comparison

| Metric | Narrative Moral | Declarative Moral | General Control |
|--------|----------------|-------------------|-----------------|
| Corpus tokens | 246,978 | 499,958 | 419,641 |
| Runtime | 48 min | 87 min | 7.6 hr |
| Final peak accuracy | **0.812** | 0.802 | 0.802 |
| Final peak layer | 3 | 0 | 1 |
| Final mean critical noise | **10.0** | 9.42 | **10.0** |
| Final most fragile layer | 0 | 3 | 0 |

## Key Findings

### 1. Probing accuracy is stable and content-agnostic

All three conditions hover in a narrow 0.77-0.81 band throughout 1000 LoRA steps.
LoRA fine-tuning on moral vs non-moral content does not meaningfully change how well
moral concepts can be probed. The base checkpoint (step 1K) already has ~80% peak
accuracy, and that doesn't budge.

### 2. Fragility profiles differ by condition (the main result)

Each condition creates a distinct per-layer robustness profile after LoRA training:

- **Narrative moral**: fragility dip at layer 2, robust through layers 3-9, fragile
  again at layer 10
- **Declarative moral**: fragility dip at layer 3, with a broader vulnerable zone
- **General control**: fragile at layers 8-10, robust at extremes

This is the most interesting result. LoRA fine-tuning on different content types
doesn't change *what* the model knows about morality (accuracy stays flat), but it
**reshapes where moral encoding is fragile**.

### 3. Training loss curves diverge dramatically

The declarative corpus loss drops from ~5.5 to ~1.0 (the model memorizes the templated
text easily). Narrative and control stay in the 4.0-4.5 range, reflecting more
diverse/natural text. Despite easily memorizing the declarative corpus, this does not
translate to higher probing accuracy.

### 4. Signal detected for follow-up experiments

**C3 signal: 0.583** (threshold: 0.10)

The fragility difference between conditions is well above the 10% threshold, meaning
C4 (Early vs Late LoRA) and C5 (Foundation Coverage) experiments are warranted.

## Interpretation

The main takeaway: LoRA fine-tuning on different content types doesn't change *what*
the model knows about morality (accuracy stays flat at ~80%), but it **does reshape
where moral encoding is fragile**. The per-layer robustness profiles differ across
conditions, suggesting that data curation affects the structural organization of
representations rather than their existence.

This is consistent with the Phase C1 finding that fragility (not accuracy) is the
metric that keeps evolving during pre-training. Accuracy saturates early (~95% by
step 4K in C1), while fragility patterns continue to change. Here we see an analogous
phenomenon: LoRA training doesn't shift accuracy, but it does shift fragility profiles.

The declarative condition's ease of memorization (loss dropping to 1.0) without
corresponding probing changes suggests that surface-level text learning and
representational moral encoding are somewhat decoupled processes.

## Output Files

- `narrative_moral.json` - Full results for narrative condition
- `declarative_moral.json` - Full results for declarative condition
- `general_control.json` - Full results for general control condition
- `lora_fragility_comparison.png` - Per-layer critical noise, all conditions
- `lora_acceleration.png` - Peak accuracy vs LoRA step, all conditions
- `lora_training_loss.png` - Loss curves comparison
- `lora_fragility_trajectory_c3_*.png` - Per-condition layer x step heatmaps

## Next Steps

- **C6**: Moral acceleration from random init (step 0) -- does moral content
  accelerate moral encoding emergence?
- **C4**: Early vs late LoRA (step 1K vs 10K) -- does injection timing matter?
- **C5**: Per-foundation LoRA -- do individual MFT foundations have selective effects?
