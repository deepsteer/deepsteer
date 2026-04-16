# The Moral Emergence Curve: How Moral Representations Form During LLM Pre-Training

**Orion Reblitz-Richardson** | Independent Alignment Researcher, Distiller Labs | April 2026

---

## Summary

We present the first systematic measurement of when and how moral
representations emerge during large language model pre-training. Using
DeepSteer, a PyTorch-native probing toolkit, we track moral encoding
across intermediate training checkpoints of OLMo base models (1B and 7B
parameters). Three headline findings challenge common assumptions about
alignment and pre-training:

1. **Moral encoding emerges early and fast.** Moral concepts become linearly
   decodable from model hidden states within the first ~5% of pre-training,
   appearing as a sharp phase transition rather than a gradual accumulation.
   Moral distinctions are among the *first* semantic features the model
   acquires — emerging before sentiment polarity and far before syntactic
   competence.

2. **Fragility reveals what accuracy cannot.** Standard probing accuracy
   saturates quickly, providing no resolution for 95% of training. We
   introduce fragility testing — measuring how much activation noise
   representations can withstand — and discover a layer-depth robustness
   gradient that continues evolving long after accuracy plateaus. Late
   layers become maximally robust while early layers grow increasingly
   fragile, revealing ongoing representational reorganization invisible to
   conventional probing.

3. **Data curation reshapes moral encoding structure.** LoRA fine-tuning
   experiments show that training content does not change *whether* moral
   concepts are encoded (accuracy is stable across conditions) but *how*
   they are structurally organized. Repetitive declarative moral statements
   create localized fragility at specific layers — brittle shortcuts —
   while narrative moral content and general text produce uniformly robust
   representations.

## Key Results

**Phase transition dynamics (OLMo-2 1B, 37 checkpoints at 1K-step intervals):**
Moral probing accuracy follows a steep sigmoid from chance (~55%) to plateau
(~95%) between steps 0 and 4K. The inflection occurs at step 1K (~3B tokens).
Depth and breadth metrics saturate immediately, but the fragility gradient
continues developing through step 36K, with early-layer robustness declining
from 10.0 to 1.7 while late-layer robustness holds at 10.0.

**Emergence ordering (C2 experiment, 3 matched probing datasets):**
Onset step (70% mean accuracy threshold): moral at step 1K, sentiment at step
2K, syntax at step 6K. Semantic features (moral, sentiment) exhibit phase-
transition dynamics; structural features (syntax) emerge gradually with no
inflection point — qualitatively different learning regimes.

**Differential foundation emergence (OLMo-2 1B and OLMo-3 7B):**
Moral Foundations Theory categories emerge in a staggered sequence: fairness
and care saturate first; loyalty, authority, and sanctity follow; liberty/
oppression never fully stabilizes at either model scale — a cross-scale pattern
suggesting this foundation is intrinsically harder to encode from web text.

**Causal-probing divergence (OLMo-3 7B):**
The layer where moral information is most decodable (probing peak) and the
layer where it most influences next-token prediction (causal peak) diverge by
~10 layers. Moral information is *stored* in mid-network layers and *used* in
early layers — a distinction invisible to probing alone.

**Data curation effects (LoRA fine-tuning, OLMo-2 1B):**
Three matched conditions (narrative moral text, declarative moral statements,
general non-moral text) produce identical probing accuracy (~80%) but distinct
fragility profiles. Declarative moral content creates a sharp vulnerability at
layer 3 (critical noise drops from 10.0 to 3.0); narrative and general text
maintain uniform robustness. This demonstrates that data curation operates on
representational structure, not representational content.

## Methodology

DeepSteer implements five probe types: layer-wise linear probing, checkpoint
trajectory analysis, per-foundation probing, causal tracing (adapted from Meng
et al. 2022), and fragility testing. All probes use minimal-pair datasets —
structurally matched sentence pairs differing only in the target feature —
designed to force classifiers to rely on genuine representations rather than
surface cues.

The current probing dataset contains 240 moral/neutral minimal pairs (40 per
Moral Foundations Theory category), plus 210 sentiment pairs and 210 syntax
pairs for the emergence-timing comparison. All datasets are deterministic, API-
free, and included in the toolkit.

Results span two OLMo models: OLMo-2 1B (37 early-training checkpoints for
dense trajectory mapping plus LoRA experiments) and OLMo-3 7B (20 stage-1
checkpoints for primary hypothesis testing). All experiments were conducted on
a MacBook Pro M4 Pro (24GB unified memory) using MPS acceleration.

## DeepSteer Toolkit

DeepSteer is open-source, PyTorch-native, and designed for three model access
tiers: API models (behavioral evaluation), open-weight models (representational
probing), and checkpoint-accessible models (training trajectory analysis). The
codebase is ~3,800 lines of Python across ~25 files, with JSON output, built-in
visualization, and a validated LoRA fine-tuning pipeline.

Repository: [github.com/deepsteer/deepsteer](https://github.com/deepsteer/deepsteer)

## Open Questions and Next Steps

The results above were produced on 1B and 7B models using a single MacBook.
Several high-value extensions require compute access and checkpoint availability
beyond what is feasible locally:

- **Scale replication.** Do the phase transition timing, fragility gradient,
  and emergence ordering hold at 13B+ parameters? The 1B→7B comparison shows
  consistency for some findings (liberty/oppression instability, H1 strong
  support) but the fragility gradient has only been characterized at 1B
  resolution.
- **Dense 7B trajectory.** The 7B model (OLMo-3) was probed at 20 checkpoints.
  Denser sampling in the early training window would reveal whether the 7B
  phase transition has the same sigmoid shape and inflection timing as the 1B.
- **Curriculum experiments at scale.** The LoRA fine-tuning results (C3) show
  that data curation affects fragility structure on 1B. Replicating at 7B
  would establish whether this finding generalizes or is a small-model artifact.
- **Harder probing datasets.** Current moral/neutral pairs may be separable by
  surface lexical cues. Controlled datasets with matched emotional valence
  (moral vs. emotionally charged but non-moral) would test whether probes
  detect genuine moral representations or sentiment-adjacent features.

OLMo's intermediate checkpoint availability makes it uniquely suited for this
research. We are actively seeking compute partnerships and research affiliations
to extend these findings to frontier-scale models.

## Contact

Orion Reblitz-Richardson — orion@orionr.com
