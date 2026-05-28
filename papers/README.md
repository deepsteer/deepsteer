# DeepSteer Papers

DeepSteer's results span three distinct contributions (full narrative
in **[RESEARCH_BRIEF.md](../RESEARCH_BRIEF.md)**; experimental record
in **[RESEARCH_PLAN.md](../RESEARCH_PLAN.md)**).

## Paper 1 — *The Moral Emergence Curve* (OLMo-2 1B and OLMo-3 7B)

- **Moral concepts emerge early and fast** — linearly decodable within
  the first ~5% of pre-training as a sharp phase transition.
  Moralized semantic distinctions appear *before* sentiment polarity
  and far before syntactic competence.
- **Fragility reveals what accuracy cannot** — probing accuracy
  saturates quickly, but fragility testing (robustness to activation
  noise) exposes a layer-depth gradient that continues evolving long
  after accuracy plateaus. *Probing accuracy is the wrong
  discriminator; fragility is the metric that actually moves.*
- **Data curation reshapes structure, not content** — LoRA fine-tuning
  shows that training content doesn't change *whether* moral concepts
  are encoded but *how* they're organized. Repetitive declarative
  statements create brittle shortcuts; narrative content produces
  uniformly robust representations.
- **Moral foundations emerge in a staggered sequence** — fairness and
  care saturate first; loyalty, authority, and sanctity follow;
  liberty/oppression never fully stabilizes at either 1B or 7B scale.
- **Storage and usage diverge** — moral information is *stored* in
  mid-network layers but *used* for prediction in early layers, a
  ~10-layer gap invisible to probing alone.

## Paper 2 — *Do Moral Representations Specialize Across Experts? Probing MoE Architectures During Pre-Training*

Four findings on OLMoE-1B-7B (64 experts, top-8 routing) vs. dense
OLMo-2 1B, testing whether MoE's structural partition into experts
creates moral specialization — and discovering an output dilution
mechanism that makes MoE moral encoding structurally fragile:

- **No expert moral specialization.** All 1,024 per-expert probes
  (64 experts × 16 layers) decode moral content above 75% accuracy.
  At the peak layer, every expert individually exceeds 90%. Gini
  coefficient of expert accuracy stays below 0.03 at all layers —
  moral encoding is as diffuse across experts as across neurons in
  dense models. The router shows negligible moral preference (max 2.4%).
- **MoE encoding is 3.6× more fragile than dense.** Despite matching
  OLMo-2 1B on probing accuracy (99.0% vs. 100.0% peak), OLMoE
  collapses under 3.6× less noise (mean critical σ* = 1.27 vs. 4.56).
- **Output dilution explains the fragility.** The MoE block's
  aggregated output (top-8 weighted average of 64 experts) contributes
  to the residual stream at **77× smaller scale** than the dense MLP
  output — the moral signal is present but at a scale trivially
  overwhelmed by noise.
- **Specialization never emerges during training.** Across 11
  checkpoints (step 5K to 1.2M), peak-layer Gini stays between
  0.011 and 0.015. Top-5 experts by accuracy change between
  checkpoints at near-random rates (Jaccard ≈ 0.08).

## Paper 3 — *The Geometry of Moral Representation: Framework-Specific Encoding and Inter-Framework Structure in Language Models*

Extends binary moral/neutral probing to **structured** moral
representations — does the model distinguish between ethical
frameworks, and if so, what is the geometry? Six independent probes
(one per MFT foundation) on OLMo-2 1B and OLMoE-1B-7B:

- **Integration, not collapse.** Foundation probe directions are
  separated (mean pairwise cosine ≈ 0.27, far from collapse >0.95)
  with effective dimensionality 5 at all layers. The model maintains
  distinct directions for distinct moral foundations.
- **Dendrogram recovers MFT group structure.** Hierarchical clustering
  at peak separation perfectly splits {liberty, care, fairness} from
  {loyalty, authority, sanctity} — the individualizing/binding
  distinction from moral psychology, emerging without explicit
  training signal. Permutation test significant at layer 0 (p = 0.012).
- **Differential fragility reversal across architectures.** In dense
  OLMo-2, binding foundations (loyalty, authority) are *most robust*;
  in MoE OLMoE, they are *least robust*. Output dilution
  disproportionately degrades group-binding moral concepts.
- **Framework geometry stabilizes before accuracy saturates.** Mean
  cosine similarity reaches its mature value by step 2000 while
  accuracy continues climbing through step 25,000 — a third
  metric (after accuracy and fragility) showing early structural
  commitment.
- **Partial compositionality of moral dilemmas.** Dilemma probe
  directions share ~10% variance with the 2D subspace of component
  foundations (100× null baseline), with near-balanced loading across
  conflicting foundations. The ~90% residual encodes conflict-specific
  features beyond the component parts.
- **Direction robustness.** Mean-difference directions (training-free)
  replicate all core geometric findings, confirming the geometry is not
  an artifact of discriminative probe training.
