# DeepSteer: Moral Representation Dynamics, Expert-Level Probing, and Framework Geometry in OLMo Pre-Training

**Orion Reblitz-Richardson** | Independent Alignment Researcher, Distiller Labs
**Affiliation pursuit:** UH Mānoa Aloha Intelligence Initiative
**Status snapshot:** May 2026

---

## Summary

DeepSteer is a PyTorch-native toolkit for measuring **how deeply** moral
reasoning is embedded in language models during pre-training. The work
covers four distinct contributions, scoped as four papers:

**Paper 1 — *The Moral Emergence Curve.*** Systematic measurement of
when and how moral representations emerge during LLM pre-training.
Three reproducible findings on OLMo-2 1B and OLMo-3 7B base models:
moralized lexical distinctions become linearly decodable within the
first ~5 % of training as a sharp phase transition, with a
quantitative lexical→compositional gradient — standard moral probe
onsets at step 1K, sentiment at 2K, *compositional* moral probe at 5K (4-seed mean)
(holds the action verb constant; varies only individually-mild tokens
whose moral status flips in context), syntax at 6K — establishing
that the early moral onset is at least partially driven by
single-token vocabulary statistics rather than compositional moral
encoding. Probing accuracy saturates misleadingly while *fragility* —
a noise-robustness metric we introduce — continues evolving long
after; data curation during fine-tuning reshapes the fragility
profile without changing probing accuracy. Probing accuracy is the
wrong metric.

**Paper 2 — *Do Moral Representations Specialize Across Experts?***
Four findings on OLMoE-1B-7B (64 experts, top-8 routing) vs. dense
OLMo-2 1B: (1) No expert moral specialization — all 1,024 per-expert
probes decode moral content above 75 %, Gini < 0.03 at all layers.
(2) MoE encoding is 5.1× more fragile than dense despite matching on
accuracy (mean critical σ* = 0.84 vs. 4.25). (3) Output dilution
explains the fragility — the MoE block's output contributes to the
residual stream at 74× smaller scale than the dense MLP. (4)
Specialization never emerges during training — Gini stays between
0.011 and 0.015 across 11 checkpoints spanning step 5K to 1.2M.

**Paper 3 — *The Geometry of Moral Representation.*** Extends binary
moral/neutral probing to structured representations via six
independent foundation-specific probes. Integration signature:
foundation directions are distinct (effective dimensionality 5, mean
cosine ≈ 0.22–0.27 across layers) but do not recover Moral
Foundations Theory's individualizing/binding partition; a care–
sanctity pairing is the most consistent clustering feature. Sanctity
fragility reversal across architectures (most robust in dense, least
robust in MoE); partial compositionality of moral dilemmas (~10 %
subspace membership, 100× null baseline); framework geometry
stabilizes before accuracy saturates.

**Paper 4 — *Causal Validation of Moral Probe Directions*
(preliminary).** Three independent validations on OLMo-2 1B:
direction ablation is foundation-specific (mean specificity −0.63 at
layer 12); steering injection shows dose–response specificity; and
projection-based behavioral classification achieves 83.3 % on causal
prompts (chance 16.7 %). SAE features partially recover the moral
subspace at 3.2× the random baseline. These results transform the
moral geometry from a descriptive finding into a tool for
representation engineering.

## Paper 1 — The Moral Emergence Curve

### Headline findings

1. **Moralized semantic distinctions emerge along a quantitative
   lexical→compositional gradient, all early.** Linear decodability
   appears as a sharp phase transition within the first ~5 % of
   pre-training and resolves into a four-point ordering by probe
   construction:

   | Probe | Onset step (mean acc ≥ 0.70) | Plateau |
   |-------|------------------------------:|---------:|
   | Standard moral (single-token swap) | 1,000 | 0.96 |
   | Sentiment (single-token swap) | 2,000 | 0.98 |
   | **Compositional moral (multi-token integrated swap; 4-seed mean)** | **5,000** | **0.77** |
   | Syntax (structural well-formedness) | 6,000 | 0.78 |

   The standard moral probe's step-1K onset measures how quickly
   *moralized vocabulary* becomes statistically separable from neutral
   vocabulary, not how quickly *moral valence is encoded
   compositionally*. The compositional probe — minimal pairs that hold
   the action verb constant and vary only individually-mild tokens
   whose moral status flips in context ("protect" vs. "humiliate",
   "hungry" vs. "wealthy", "safe" vs. "hidden", "innocent" vs.
   "guilty"; TF-IDF baseline 0.11 ≪ 0.65) — onsets at step 5K under
   4-seed averaging (per-seed range 4K-7K across split seeds 42 / 43
   / 44 / 45), between sentiment and syntax. **Plateau coincidence:** compositional
   and syntax probes both saturate at ≈0.77 under mean-pooled linear
   probing while standard moral and sentiment saturate at ≈0.96-0.98,
   suggesting that probes requiring multi-token structural or
   compositional integration plateau lower than probes that ride
   single-token statistics. Whether 0.77 is a representational ceiling
   or a probe-side ceiling is open until 7B / 32B replication. Either
   way, the representational substrate for moralized content (lexical
   *and* compositional) is present and reorganizing long before
   post-training interventions typically engage.

2. **Fragility reveals what accuracy cannot.** Probing accuracy
   saturates within the first 4K steps and is essentially flat for
   the remaining 95 % of training. Fragility — measured as the
   activation noise level at which probing accuracy drops below
   threshold — continues evolving until the end of training, with
   the layer-depth gradient steepening monotonically: late layers
   become maximally robust while early layers grow increasingly
   fragile. *Probing accuracy is the wrong discriminator for
   alignment-relevant questions about pre-training; fragility is the
   metric that actually moves.*

3. **Data curation reshapes structure, not content.** LoRA
   fine-tuning on three matched corpora (narrative moral text,
   declarative moral statements, general non-moral text) produces
   identical probing accuracy (~80 %) but distinct fragility
   profiles. Repetitive declarative moral statements create localized
   fragility at specific layers — brittle shortcuts — while narrative
   moral content and general text produce uniformly robust
   representations. *Data curation operates on representational
   structure, not representational content.*

### Supporting findings

- **Phase transition dynamics (37-checkpoint OLMo-2 1B trajectory):**
  steep sigmoid from chance (~55 %) to plateau (~95 %) between
  steps 0 and 4K. Inflection at step 1K (~3B tokens). Depth and
  breadth saturate immediately; fragility gradient continues
  developing through step 36K, with early-layer robustness declining
  from 10.0 to 1.7 while late-layer robustness holds at 10.0.
- **Emergence ordering (matched 240-pair moral / 210-pair sentiment /
  210-pair syntax / 200-pair compositional moral probing datasets):**
  standard moral onsets at step 1K, sentiment at 2K, compositional
  moral at 5K (4-seed mean), syntax at 6K; standard moral and sentiment show
  phase-transition dynamics with sharp inflection (plateau ≈0.96-0.98),
  compositional moral and syntax rise more gradually and plateau
  ≈0.77 — qualitatively different learning regimes that track
  whether the probe's discriminative signal lives in single-token
  statistics or in multi-token integration. The compositional probe
  also reproduces the accuracy-saturates-fragility-doesn't pattern
  independently (mean critical noise rises 0.10 → 5.7 by step 5K,
  drifts to ~2.7 by step 30K), confirming the methodology claim is
  not a lexical artifact.
- **Differential foundation emergence (1B and 7B):** Moral Foundations
  Theory categories emerge in a staggered sequence — authority
  emerges fastest (step 1K), followed by care and fairness (step 2K),
  with loyalty, sanctity, and liberty reaching 100% by step 3K.
- **Causal-probing divergence (7B):** the layer where moral
  information is most decodable (probing peak) and the layer where it
  most influences next-token prediction (causal peak) diverge by ~10
  layers. Moral information is *stored* in mid-network layers and
  *used* in early layers — invisible to probing alone.

### Methodology

`LayerWiseMoralProbe`, `CompositionalMoralProbe`, `MoralFragilityTest`,
`FoundationSpecificProbe`, and `MoralCausalTracer` — all running on
OLMo-2 1B (37 checkpoints at 1K-step intervals) and OLMo-3 7B (20
stage-1 checkpoints) on a single MacBook Pro M4 Pro (24 GB unified
memory, MPS acceleration). Probing datasets: 240 standard moral /
neutral minimal pairs (40 per Moral Foundations Theory category) +
210 sentiment pairs + 210 syntax pairs + 200 compositional moral
pairs (four 50-pair categories: action+motive, action+target,
action+consequence, role-reversal; TF-IDF baseline 0.11 ≪ 0.65
gate). All deterministic, API-free, included in the toolkit.

## Paper 2 — Do Moral Representations Specialize Across Experts?

### Headline findings

1. **MoEs do not create expert moral specialization.** All 1,024
   per-expert probes (64 experts × 16 layers) decode moral content
   well above chance. At the peak layer, every expert individually
   exceeds 90 % accuracy. The Gini coefficient of expert accuracy
   is below 0.03 at all layers — moral encoding is as uniformly
   distributed across experts as it is across neurons in a dense
   model. The router shows negligible moral content preference
   (maximum 2.4 %).

2. **MoE encoding is 5.1× more fragile than dense.** Despite
   matching dense OLMo-2 1B on probing accuracy (99.0 % vs.
   99.0 % peak), OLMoE's moral encoding collapses under 5.1×
   less noise (mean critical σ* = 0.84 vs. 4.25). The fragility
   gap is not explained by weaker individual expert representations
   or unstable routing — both are robust in isolation.

3. **The fragility originates in output dilution.** The MoE block's
   aggregated output (a top-8 weighted average of 64 expert outputs)
   contributes to the residual stream at **74× smaller scale** than
   the dense MLP output, measured as the standard deviation of the
   feedforward block's output across inputs. This *output dilution*
   means that the same absolute noise level overwhelms the MoE moral
   signal while leaving the dense signal intact.

4. **Specialization never emerges during training.** Across 11
   checkpoints spanning OLMoE's training (step 5K to step 1.2M,
   covering 20B to 5,033B tokens), the peak-layer Gini coefficient
   stays between 0.011 and 0.015 at every checkpoint. Moral encoding
   is present from the earliest available checkpoint (91.1 % peak
   accuracy at step 5K) and strengthens to 95.3 % by step 1.2M
   without ever concentrating in specific experts. The top-5 experts
   by accuracy change between adjacent checkpoints at near-random
   rates (Jaccard ≈ 0.08).

### Methodology

Per-expert probing with batched einsum bypass of the router (all
64 expert FFN outputs computed in parallel), component-level
fragility isolation (router / expert / output perturbation), and
feedforward output scale comparison — all running on OLMoE-1B-7B
(16 layers, 6.9B total / 1.3B active params, MPS, bf16) vs.
OLMo-2 1B on a MacBook Pro M4 Pro. All experimental artifacts
published under `papers/2_moe_output_dilution/outputs/`.

## Paper 3 — The Geometry of Moral Representation

### Headline findings

1. **Integration, not collapse.** Foundation probe directions are
   *separated*, not collapsed. Across bootstrap-stable layers (6–15),
   mean pairwise cosine similarity ranges from **0.232 to 0.274** —
   far below the collapse threshold (> 0.95). Effective dimensionality
   is 5 at every layer (near-maximal for 6 directions). The model
   maintains geometrically distinct directions for distinct moral
   foundations sharing a common moral-salience component.

2. **The model's moral taxonomy is not MFT.** Hierarchical
   clustering does not recover the MFT individualizing/binding
   distinction at any layer; the permutation test is non-significant
   throughout (minimum p = 0.32). The most consistent clustering
   feature is a care–sanctity pairing that crosses MFT groups — both
   foundations concern protection of vulnerable entities, sharing
   distributional signatures the model detects.

3. **Sanctity fragility reversal across architectures.** In dense
   OLMo-2 1B, sanctity/degradation is the *most* robust foundation
   (mean critical noise 5.60). In MoE OLMoE-1B-7B, it is the *least*
   robust (0.91) — a 6.2× ratio, far larger than the overall 3.1×
   architecture gap. Output dilution disproportionately degrades
   sanctity representations, which may depend on fine-grained
   culturally specific associations.

4. **Framework geometry stabilizes before accuracy saturates.**
   Tracking mean cosine similarity across 20 OLMo-2 1B checkpoints:
   similarity jumps from ≈ 0 (random) to 0.382 by step 2000 — within
   5 % of its final value — while accuracy is still 10 points below
   its peak. The model discovers the geometric layout of moral
   concepts early and strengthens representations within that fixed
   layout.

5. **Partial compositionality of moral dilemmas.** Dilemma probe
   directions share ~10 % variance with the 2D subspace spanned by
   their component foundation directions (100× the null baseline of
   0.001), with near-balanced component loading (mean α = 0.486).
   The ~90 % residual encodes conflict-specific features (tension,
   trade-off framing) that lie outside the moral subspace entirely.
   Compositionality is preserved across dense and MoE architectures.

6. **Direction-finding method robustness.** Mean-difference
   directions (training-free) replicate all core geometric findings:
   effective dimensionality 5, absence of MFT dendrogram structure,
   permutation test non-significant throughout. Representation-
   engineering PCA fails in the p ≫ n regime, validating that
   probe-weight and mean-difference convergence is non-trivial. Both
   methods transfer to narrative dilemma text with > 90 % mean pair
   accuracy (probe-weight > 95 %).

### Methodology

Six independent foundation-specific linear probes (one per MFT
foundation), pairwise cosine similarity matrices, hierarchical
clustering (Ward's method), permutation tests, bootstrap direction
stability (200 iterations), per-foundation fragility, dilemma
subspace analysis (15 dilemma probes, 300 pairs), and 5D moral
subspace projection — all running on OLMo-2 1B and OLMoE-1B-7B
(MPS, fp16/bf16). All experimental artifacts published under
`papers/3_moral_geometry/outputs/`.

## Paper 4 — Causal Validation of Moral Probe Directions (preliminary)

### Headline findings

1. **Direction ablation is foundation-specific.** Removing a
   foundation's probe direction from the residual stream specifically
   reduces log-probability of that foundation's continuations while
   leaving other foundations largely unaffected. Mean specificity is
   −0.63 at layer 12. Sanctity is the most causally load-bearing
   direction at all layers (specificity −1.64 at layer 12).

2. **Steering injection shows dose–response specificity.** At low
   amplitude (α = 1–2), injecting a foundation direction produces a
   specific on-target boost with near-zero off-target effect. At
   higher amplitudes (α = 5–10), both increase but on-target
   increases more, yielding growing specificity — distinguishable
   from noise injection, which would produce monotonic degradation.

3. **Behavioral grounding.** Projection-based 6-way classification
   achieves 83.3 % on causal evaluation prompts and 70.8 % on
   held-out test data (chance = 16.7 %). A "sanctity saturation"
   phenomenon emerges on Moral Foundations Vignettes: harm-witnessing
   scenarios preferentially activate the sanctity direction regardless
   of the target foundation, consistent with the sanctity anomaly
   from Paper 3.

4. **SAE features partially recover moral subspace.** A partially
   trained sparse autoencoder (2M tokens, 71.5 % variance explained)
   discovers features overlapping with probe directions at 3.2× the
   chance level (15.5 % subspace overlap vs. 4.9 % random baseline).
   Moral information is distributed across many low-selectivity
   features rather than concentrated in a few.

### Methodology

Direction ablation and steering injection at layers 4, 8, 12;
projection-based foundation classification on held-out, external
(MFV), and causal evaluation stimuli; layer-8 SAE training (16,384
features, 3 epochs on 2M tokens) with moral selectivity and subspace
overlap analysis — all on OLMo-2 1B (MPS, fp16). Experimental
artifacts under `papers/4_causal_validation/outputs/`.

## Toolkit Status

DeepSteer is open-source, PyTorch-native, and designed for three model
access tiers: API models (behavioral evaluation), open-weight base
models (representational probing), and checkpoint-accessible models
(training trajectory analysis). The toolkit currently spans
`benchmarks/representational/` (probing, fragility, foundation-
specific probes, causal tracing, persona probing, persona activation
scoring), `steering/` (`TrainingTimeSteering`, chat LoRA trainer,
data mixing, moral curriculum schedules, training hooks), and paper-
specific scripts (MoE expert probing, framework geometry, dilemma
analysis). All evaluations produce structured JSON output with full
metadata; all visualizations have 1:1 matched JSON for reproducibility.

Repository: [github.com/deepsteer/deepsteer](https://github.com/deepsteer/deepsteer)

## Key References

- **Muennighoff et al. (2024)**, arXiv:2409.02060 — OLMoE:
  Open Mixture-of-Experts Language Models. Target model for Paper 2.
- **Haidt, J. (2012)** / **Graham et al. (2013)** — Moral
  Foundations Theory. The six-foundation taxonomy underlying
  Papers 1 and 3.
- **Betley et al. (2025)**, arXiv:2502.17424 — Emergent Misalignment
  from Narrow Fine-Tuning. Prior experimental replication target.
- **Wang et al. (2025)**, arXiv:2506.19823 — Persona Features Control
  Emergent Misalignment. Prior linear-analog recovery at 1B.
- **Tice et al. (2026)**, arXiv:2601.10160 — Alignment Pretraining.
- **Anthropic (2025)**, arXiv:2512.05648 — Selective Gradient Masking.
  Methodological cousin to `TrainingTimeSteering.gradient_penalty`.
- **Greenblatt et al. (2024)**, arXiv:2412.14093 — Alignment Faking.
  Source for `ComplianceGapDetector`.

Full citations and toolkit cross-references in
[REFERENCES.md](REFERENCES.md). Full experimental record in
[RESEARCH_PLAN.md](RESEARCH_PLAN.md).

## Contact

Orion Reblitz-Richardson — orion@orionr.com
