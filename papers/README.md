# DeepSteer Papers

DeepSteer's results span four distinct contributions (full narrative
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
- **Moral foundations emerge in a staggered sequence** — authority
  emerges fastest (step 1K), followed by care and fairness (step 2K),
  with loyalty, sanctity, and liberty reaching 100% by step 3K.
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
  separated (mean pairwise cosine ≈ 0.22–0.27 across layers, far
  from collapse >0.95) with effective dimensionality 5 at all layers.
  The model maintains distinct directions for distinct moral
  foundations, sharing a common moral-salience component.
- **The model's moral taxonomy is not MFT.** Hierarchical clustering
  does not recover the MFT individualizing/binding distinction at any
  layer; the permutation test is non-significant throughout. The most
  consistent clustering feature is a care–sanctity pairing that
  crosses MFT groups — both foundations concern protection of
  vulnerable entities.
- **Sanctity fragility reversal across architectures.** Sanctity is
  the *most* robust foundation in dense OLMo-2 (σ* = 5.60) but the
  *least* robust in MoE OLMoE (σ* = 0.91) — a 6.2× ratio, far
  larger than the overall 3.1× architecture gap. Output dilution
  disproportionately degrades sanctity representations.
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
  replicate all core geometric findings — effective dimensionality 5,
  absence of MFT dendrogram structure, >90% cross-register transfer —
  confirming the geometry is not an artifact of probe training.
  Representation-engineering PCA fails in the p ≫ n regime.

## Paper 4 — *Causal Validation of Moral Probe Directions* (preliminary)

Validates whether the foundation-specific probe directions from Paper 3
are genuine features of OLMo-2 1B's computation, not probing artifacts.
Three independent lines of evidence:

- **Direction ablation is foundation-specific.** Removing a foundation's
  direction from the residual stream specifically reduces log-probability
  of that foundation's continuations (mean specificity −0.63 at layer 12),
  with minimal off-target effects. Ablation damage increases with depth.
- **Steering injection shows dose–response specificity.** Injecting a
  foundation direction produces a specific boost at low amplitude
  (α = 1–2) with dose–response amplification at higher amplitudes,
  distinguishable from noise injection.
- **Behavioral grounding.** Projection-based 6-way classification
  achieves 83.3% on causal prompts and 70.8% on held-out test data
  (chance = 16.7%). A "sanctity saturation" phenomenon emerges on
  real-world moral vignettes: harm-witnessing scenarios preferentially
  activate the sanctity direction regardless of target foundation.
- **SAE features partially recover moral subspace.** Even a partially
  trained sparse autoencoder discovers features overlapping with probe
  directions at 3.2× the chance level (15.5% subspace overlap vs. 4.9%
  random baseline).
