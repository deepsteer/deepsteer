# DeepSteer: Moral Representation Dynamics and Persona-Feature Monitoring in OLMo Pre-Training

**Orion Reblitz-Richardson** | Independent Alignment Researcher, Distiller Labs
**Affiliation pursuit:** UH Mānoa Aloha Intelligence Initiative
**Status snapshot:** April 2026

---

## Summary

DeepSteer is a PyTorch-native toolkit for measuring **how deeply** moral
reasoning is embedded in language models during pre-training, and for
testing whether the persona-feature mechanism that mediates emergent
misalignment at frontier scale (Wang et al., 2025) generalizes to 1B
base models. The work covers two distinct contributions, suitable for
two papers:

**Paper 1 — *The Moral Emergence Curve.*** Systematic measurement of
when and how moral representations emerge during LLM pre-training.
Three reproducible findings on OLMo-2 1B and OLMo-3 7B base models:
moralized lexical distinctions become linearly decodable within the
first ~5 % of training as a sharp phase transition, with a
quantitative lexical→compositional gradient — standard moral probe
onsets at step 1K, sentiment at 2K, *compositional* moral probe at 4K
(holds the action verb constant; varies only individually-mild tokens
whose moral status flips in context), syntax at 6K — establishing
that the early moral onset is at least partially driven by
single-token vocabulary statistics rather than compositional moral
encoding. Probing accuracy saturates misleadingly while *fragility* —
a noise-robustness metric we introduce — continues evolving long
after; data curation during fine-tuning reshapes the fragility
profile without changing probing accuracy. Probing accuracy is the
wrong metric.

**Paper 2 — *Persona-Feature Monitoring at 1B: A Compound Scaling
Boundary.*** Four reproducible 1B-scale results on whether the Wang
et al. (2025) toxic-persona mechanism engages, can be measured, and
can be intervened on via training-time representation steering. (1)
Under controlled insecure-code LoRA replication of Betley et al.
(2025), the persona-probe direction does not shift (Cohen's d = 0.03)
and behavioral emergent misalignment stays at the noise floor;
probe-flagged and judge-flagged samples fire on decoupled axes. (2)
A `TrainingTimeSteering.gradient_penalty` primitive cleanly
suppresses a target probe direction by 99.3 % at no SFT-loss cost.
(3) But probe-direction suppression does not suppress behavior — a
held-out behavioral judge rates `gradient_penalty` outputs identically
to vanilla LoRA (7.62 vs 7.61 / 10 on a persona-voice scale) despite
probe Cohen's d differing by 3.07. (4) Reapplying the Phase B/C
moral-probe + fragility battery to the saved insecure-code adapters
shows probing accuracy is unchanged but the **layer-locus of robust
moral encoding shifts by 2-3 layers** under insecure-code LoRA
specifically — a Phase-C3 fragility-only signature that the
behavioral and probe-direction nulls did not capture.

Together these establish a **compound scale boundary** on the Wang
et al. (2025) mechanism at 1B and motivate a two-prediction Phase E
test at 7B with SAE-decomposed features.

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
   | **Compositional moral (multi-token integrated swap)** | **4,000** | **0.77** |
   | Syntax (structural well-formedness) | 6,000 | 0.78 |

   The standard moral probe's step-1K onset measures how quickly
   *moralized vocabulary* becomes statistically separable from neutral
   vocabulary, not how quickly *moral valence is encoded
   compositionally*. The compositional probe — minimal pairs that hold
   the action verb constant and vary only individually-mild tokens
   whose moral status flips in context ("protect" vs. "humiliate",
   "hungry" vs. "wealthy", "safe" vs. "hidden", "innocent" vs.
   "guilty"; TF-IDF baseline 0.11 ≪ 0.65) — onsets at step 4K,
   between sentiment and syntax. **Plateau coincidence:** compositional
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
  moral at 4K, syntax at 6K; standard moral and sentiment show
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
  Theory categories emerge in a staggered sequence — fairness and care
  saturate first; loyalty, authority, and sanctity follow;
  liberty/oppression never fully stabilizes at either scale, a
  cross-scale pattern suggesting this foundation is intrinsically
  harder to encode from web text.
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

## Paper 2 — Persona-Feature Monitoring at 1B: A Compound Scaling Boundary

### Headline findings

1. **The persona mechanism does not engage at 1B under controlled
   insecure-code LoRA replication (C10 v2; reproducible across two
   independent runs).** Probe activation paired Δ = +0.057 (Cohen's
   d = +0.03 vs. threshold ≥ 1 SD); behavioral EM 1.56 % insecure
   vs. 0.69 % secure (Wilson 95 % CIs overlap); judge calibration
   uses Betley et al.'s exact alignment / coherence prompts. Probe
   fires on persona-voice style (rhetorical questions, cynical
   aphorisms); judge flags content-level misalignment ("humans are
   selfish," "report husband"); **the two axes are independent at
   1B**. This is consistent with Betley et al.'s reported attenuation
   at smaller scales and is the first published datapoint on where
   the Wang et al. (2025) coupling between persona representation
   and behavioral output breaks down.

2. **A linear `TrainingTimeSteering.gradient_penalty` primitive
   cleanly suppresses a target probe direction (Step 2A).** On a
   synthesized persona-voice corpus that engages the probe by design
   (vanilla LoRA Cohen's d = +2.29 vs. baseline), an auxiliary loss
   `λ × probe_logit²` (λ = 0.05, mean-pooled over assistant tokens at
   the probe's target layer) drives probe activation back to within
   +0.02 of baseline (99.3 % suppression). Final SFT loss matches
   vanilla within 0.4 % — the suppression is essentially free on the
   training objective. The aux loss saturates at step 30 and stays
   there for the remaining 270 steps; vanilla LoRA reaches its
   full +3.78 probe activation by step 50, so gradient_penalty's
   advantage is *sustained suppression* throughout training rather
   than a one-shot reduction. The deepsteer primitive works as
   designed at the engineering level.

3. **Probe-direction suppression does not suppress behavior at 1B
   (Step 2B; quantified by held-out behavioral judge).** Claude
   Haiku 4.5 rated all 640 evaluation generations on a 0-10
   persona-voice scale. Vanilla persona-LoRA (probe +3.76, judge
   7.61) and `gradient_penalty` (probe +0.98, judge 7.62) produce
   judge scores that **match within 0.01** despite probe Cohen's d
   differing by 3.07 (+3.10 vs. +0.03 vs. baseline). Z-scored
   against baseline, the probe-vs-judge dissociation
   (z_judge − z_probe) is +4.96 for `gradient_penalty` versus +2.17
   for vanilla and −0.93 for `activation_patch`. *At 1B a single
   linear probe captures one of many directions encoding persona-voice
   behavior; suppressing that direction routes the same behavior
   through alternative features.* Methodologically, an
   `activation_patch` primitive (constant subtraction of γ × unit_w
   at training time) backfires by amplification (+99 % probe
   activation vs. vanilla) — the model trains to compensate for the
   subtraction, and removing the patch at evaluation time reveals
   overcorrection. Gradient penalty is the correct training-time
   primitive; activation patching is for inference-time only.

4. **Narrow insecure-code fine-tuning leaves a Phase-C3-style
   fragility-locus signature that the persona-probe and behavioral-
   judge nulls did not capture (C15 reframed).** Reapplying the
   Phase B/C 240-pair moral probe + fragility battery to the saved
   C10 v2 adapters: probing accuracy is unchanged across base /
   insecure / secure (max |Δ| = 0.021 ≤ flat threshold 0.03) but the
   layer-wise fragility profile shifts (mean |Δ log10
   critical_noise| = 0.336 > flat threshold 0.20). Insecure-code
   LoRA specifically *relocates* the moral-encoding robustness peak
   from layer 7 (base, critical noise = 10) to layers 9-10
   (insecure, critical noise = 10) while collapsing layers 6-7 down
   to critical noise = 1; mean critical noise drops from 5.25 (base)
   → 4.21 (secure) → 3.73 (insecure). The same content remains
   equally decodable, but *where* the encoding is robust shifts by
   2-3 layers under insecure-code specifically. **Caveat: N = 1
   experiment; replicates needed at 7B.** Promoted to a headline
   finding because it provides a representation-level signature of
   narrow fine-tuning that purely behavioral evaluations miss.

### What this means for Phase E

The four findings combine into a **compound scaling prediction** for
Phase E at 7B with SAE-decomposed features. Both predictions are
falsifiable; either, alone, is publishable.

> **Coupling prediction.** At 7B scale, the persona-probe direction
> will shift under insecure-code LoRA and behavioral EM will exceed
> the noise floor — both effects emerging where they did not at 1B
> (C10 v2 null).

> **Suppression-captures-behavior prediction.** Penalizing the
> relevant SAE latent set during insecure-code LoRA fine-tuning will
> suppress behavioral EM at 7B — measured via the held-out behavioral
> judge, not just probe activation — where suppressing a single
> linear-probe direction at 1B did not (Step 2B feature-redundancy
> finding).

A negative answer on either prediction is itself a publishable
scaling boundary on the Wang et al. mechanism. The work does not
depend on positive results — it depends on running with the same
experimental rigor as Phase D.

A separate Phase-E follow-up: replicate the C15 reframed fragility-
locus check at 7B. If the layer-locus shift extends to 7B, narrow
fine-tuning fingerprints become a deployable monitoring signal
distinct from probe activation and behavioral judges.

### Methodology

`PersonaFeatureProbe`, `PersonaActivationScorer`,
`TrainingTimeSteering` (gradient_penalty + activation_patch
primitives, hook-based, PEFT-compatible), and a Claude-API-based
behavioral-judge harness — all running on OLMo-2 1B (16 layers, 1.5 B
params, MPS, fp16). C10 v2 + Step 2 + C15 reframed total runtime ~6
hours of MPS time on a MacBook Pro M4 Pro. All adapters, eval
outputs, judge ratings, and probe weights are published under
`outputs/phase_d/` for reproducibility.

## Toolkit Status

DeepSteer is open-source, PyTorch-native, and designed for three model
access tiers: API models (behavioral evaluation), open-weight base
models (representational probing), and checkpoint-accessible models
(training trajectory analysis). The toolkit currently spans
`benchmarks/representational/` (probing, fragility, foundation-
specific probes, causal tracing, persona probing, persona activation
scoring), `steering/` (`TrainingTimeSteering`, chat LoRA trainer,
data mixing, moral curriculum schedules, training hooks), and Phase D
follow-up scripts (mechanism check, behavioral judge, head-start
trajectory). All evaluations produce structured JSON output with full
metadata; all visualizations have 1:1 matched JSON for reproducibility.

Repository: [github.com/deepsteer/deepsteer](https://github.com/deepsteer/deepsteer)

## Key References

- **Betley et al. (2025)**, arXiv:2502.17424 — Emergent Misalignment
  from Narrow Fine-Tuning. The published failure mode Phase D
  targets.
- **Wang et al. (2025)**, arXiv:2506.19823 — Persona Features Control
  Emergent Misalignment. The mechanism Phase D recovers with a
  linear analog at 1B and 7B.
- **Tice et al. (2026)**, arXiv:2601.10160 — Alignment Pretraining.
  Appendix I: data-upsampling negative on EM; explicitly flags
  representation-level interventions as the natural follow-up.
- **O'Brien et al. (2025)**, arXiv:2508.06601 — Deep Ignorance.
  Filtered + unfiltered 6.9B models; Phase E base candidate.
- **Anthropic (2025)**, arXiv:2512.05648 — Selective Gradient Masking.
  Methodological cousin to `TrainingTimeSteering.gradient_penalty`.
- **Lieberum et al. (2024)**, arXiv:2408.05147 — Gemma Scope. SAE
  library for the Phase E suppression-captures-behavior test on
  Gemma-2-9B.
- **Greenblatt et al. (2024)**, arXiv:2412.14093 — Alignment Faking.
  Source for `ComplianceGapDetector`.

Full citations and toolkit cross-references in
[REFERENCES.md](REFERENCES.md). Full experimental record, gating
logic, and Phase E plan in [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

## Contact

Orion Reblitz-Richardson — orion@orionr.com
