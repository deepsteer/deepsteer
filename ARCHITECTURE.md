# DeepSteer: Evaluating and Steering Alignment Depth in LLM Pre-Training

## Vision

A PyTorch-native toolkit for measuring **how deeply** moral reasoning and alignment properties are embedded in language models — distinguishing shallow post-hoc alignment (refusal layers, RLHF patches) from deep pre-training alignment (distributed moral representations).

The core thesis: models that acquire moral reasoning during pre-training will show measurably different properties than models where alignment is applied post-hoc. DeepSteer provides the tools to detect, measure, and visualize this difference.

## Key Concept: Operationalizing "Depth of Alignment"

"Depth" is not a single metric — it manifests across four dimensions:

### 1. Representational Depth (requires weights)
Where in the network are moral concepts encoded? Shallow alignment concentrates safety behavior in late-layer "refusal circuits." Deep alignment distributes moral reasoning across the network.

- **Probing classifiers** at each layer for moral concept detection
- **Causal tracing** to identify which layers are causally responsible for moral decisions
- **Representation similarity** between moral and non-moral reasoning pathways

### 2. Behavioral Depth (works with API)
How robust is moral reasoning under adversarial pressure, framing shifts, and context variation?

- **Compliance gap** (Greenblatt-style): monitored vs. unmonitored behavioral divergence
- **Perturbation robustness**: does moral reasoning survive paraphrasing, translation, persona shifts?
- **Consistency across frameworks**: deontological, consequentialist, virtue ethics framing

### 3. Training Trajectory Depth (requires checkpoints)
How do moral representations evolve during training? This is the unique contribution enabled by OLMo's open checkpoints.

- **Checkpoint-over-time** analysis of moral probing accuracy
- **Phase transitions** in moral concept acquisition
- **Curriculum effects** — measuring impact of moral content ordering in training data

### 4. Fragility / Removal Resistance (requires fine-tuning access)
How resistant is alignment to targeted removal? Shallow alignment is easily fine-tuned away.

- **Unlearning resistance** — how many gradient steps to remove moral behavior?
- **Subnetwork analysis** — is alignment concentrated or distributed?
- **Fine-tuning attack resistance** — harmful fine-tuning on small datasets

## Architecture

```
deepsteer/
├── core/                       # Core abstractions
│   ├── model_interface.py      # WhiteBoxModel, APIModel, ModelFamily, architecture detection
│   ├── moe_model.py            # MoEWhiteBoxModel for OLMoE expert/router analysis
│   ├── benchmark_suite.py      # Benchmark base class + suite runner
│   └── types.py                # All dataclasses and enums
│
├── foundations.py              # Canonical MFT constants (FOUNDATION_ORDER, groups, dilemma pairs)
│
├── directions/                 # Direction extraction (model-agnostic, pure numpy)
│   ├── mean_diff.py            # Mean-difference direction (baseline)
│   ├── leace.py                # LEACE / Fisher LDA direction
│   ├── probe_weight.py         # Direction from trained probe weights / .npz files
│   └── compare.py              # Cross-method alignment comparison
│
├── geometry/                   # Geometric analysis (model-agnostic, pure numpy)
│   ├── cosine.py               # Cosine similarity matrices, effective dimensionality
│   ├── clustering.py           # Hierarchical clustering, permutation tests
│   ├── subspace.py             # Orthonormal bases, subspace membership, null distributions
│   └── analysis.py             # Full geometric analysis orchestrator
│
├── causal/                     # Causal validation (requires WhiteBoxModel)
│   ├── ablation.py             # Direction ablation and specificity measurement
│   ├── steering.py             # Steering vector injection with dose-response
│   └── behavioral.py           # Projection-based behavioral classification
│
├── benchmarks/
│   ├── moral_reasoning/        # Behavioral (API-compatible)
│   │   ├── foundations.py       # MoralFoundationsProbe (instruct models)
│   │   └── foundations_base.py  # Log-prob forced-choice variant (base models)
│   │
│   ├── compliance_gap/         # Behavioral (API-compatible)
│   │   ├── greenblatt.py        # ComplianceGapDetector (instruct)
│   │   ├── greenblatt_base.py   # Log-prob variant (base models)
│   │   ├── persona_shift.py     # PersonaShiftDetector (instruct)
│   │   ├── persona_shift_base.py # Log-prob variant (base models)
│   │   └── em_behavioral.py     # Emergent misalignment behavioral eval
│   │
│   └── representational/       # White-box (requires weights)
│       ├── probing.py           # LayerWiseMoralProbe (linear probing)
│       ├── foundation_probes.py # Per-foundation separate probes
│       ├── general_probe.py     # General-purpose probe utilities
│       ├── compositional_moral_probe.py  # Compositional pair probing
│       ├── causal_tracing.py    # Causal mediation analysis
│       ├── fragility.py         # Fine-tuning fragility measurement
│       ├── trajectory.py        # Checkpoint trajectory analysis
│       ├── persona_probe.py     # Cross-context probe robustness
│       └── persona_activation.py # Persona-conditioned activations
│
├── datasets/                   # Probing datasets and generation pipeline
│   ├── pipeline.py             # build_probing_dataset() — main entry point
│   ├── validation.py           # Automated quality gates (length, keywords, embedding)
│   ├── balancing.py            # Foundation/register distribution balancing
│   ├── pairing.py              # Moral-neutral pair matching
│   ├── llm_generation.py       # LLM-based pair generation
│   ├── types.py                # Dataset-specific types
│   ├── compositional_moral_pairs.py  # Compositional probe pairs
│   ├── persona_pairs.py        # Persona-conditioned control pairs
│   ├── sentiment_pairs.py      # Sentiment control pairs
│   ├── syntax_pairs.py         # Syntactic control pairs
│   ├── minimal_pairs.py        # Legacy v1 hand-written pairs (450)
│   ├── moral_seeds.py          # Legacy v1 seed sentences (300)
│   ├── neutral_pool.py         # Legacy v1 neutral pool
│   ├── corpora/                # Training corpora for steering experiments
│   │   ├── declarative.py      # Declarative moral corpus from seeds
│   │   ├── gutenberg.py        # Narrative moral corpus (Aesop, Grimm, etc.)
│   │   └── general.py          # Non-moral control corpus
│   ├── moral_probing_v2.json   # 1,200-pair dataset (200/foundation × 3 registers)
│   ├── seed_examples_v2.json   # 54 generation anchors (3/foundation × 3 registers)
│   ├── dilemma_pairs_final.json     # 300 cross-foundation dilemma pairs
│   ├── dilemma_pairs_validated.json # Same pairs + validation_stats
│   ├── DATASET_GUIDELINES.md   # Quality rules for creating/auditing datasets
│   └── DATASET_AUDIT.md        # Audit summary (structural + quality)
│
├── steering/                   # Training-time intervention tools
│   ├── moral_curriculum.py     # Curriculum design for moral pre-training
│   ├── data_mixing.py          # Moral corpus mixing strategies
│   ├── training_hooks.py       # PyTorch hooks for training-time monitoring
│   ├── training_time_steering.py  # Steering during training
│   ├── lora_trainer.py         # LoRA fine-tuning for fragility experiments
│   ├── chat_lora_trainer.py    # Chat-format LoRA (EM replication)
│   └── lora_experiment.py      # LoRA experiment orchestration
│
├── viz/                        # Visualization
│   ├── __init__.py             # All plot functions (layer heatmaps, trajectories, etc.)
│   └── lora_experiments.py     # LoRA-specific plots
│
├── outputs/                    # Untracked output viz and matching JSON
│
scripts/                        # CLI entrypoints
│   ├── run_evaluation.py       # Main evaluation driver
│   ├── compare_models.py       # Cross-model comparison
│   └── moral_emergence.py      # Moral concept emergence analysis
│
papers/                         # Research papers with reproducible experiments
│   ├── 1_accuracy_vs_fragility/  # Probing accuracy vs fine-tuning fragility
│   ├── 2_moe_output_dilution/   # MoE moral encoding and output dilution
│   ├── 3_moral_geometry/        # Foundation geometry and probe engineering
│   └── 4_causal_validation/     # Causal tracing validation
│   # Each paper: scripts/ sections/ figures/ outputs/ build/
│
tests/                          # pytest suite mirroring source structure
```

## Model Access Tiers

| Capability | OLMo | Llama | Claude/GPT API | Requires Instruct? |
|---|---|---|---|---|
| Representational probing (layer-wise) | ✓ | ✓ | ✗ | No (base preferred) |
| Causal tracing / activation patching | ✓ | ✓ | ✗ | No (base preferred) |
| Fine-tuning fragility tests | ✓ | ✓ | ✗ | No (base preferred) |
| Checkpoint trajectory analysis | ✓ | ✗ | ✗ | No (base preferred) |
| Training-time steering hooks | ✓ | ✗ | ✗ | No (base preferred) |
| Behavioral evals (moral reasoning, compliance gap) | ✓ | ✓ | ✓ | Yes |

## Design Principles

1. **Hooks on real models, not reimplementations.** We hook into actual PyTorch modules — no custom transformer implementations that drift from reality.

2. **Benchmark-first, infrastructure-second.** Ship evaluations that produce novel findings before building elaborate training integration.

3. **Accelerate everything.** Algorithms should be easily accelerated on GPU and NPU including MPS for Apple Silicon and CUDA for Nvidia GPUs through PyTorch, including torch.compile in the future.

4. **Tiered graceful degradation.** Every evaluation specifies its minimum access tier. If you only have API access, you still get useful behavioral metrics.

5. **Reproducible and publishable.** Every evaluation produces structured JSON results with full metadata. Designed for academic papers from day one.

6. **Composable.** Evaluations are independent units that can be combined into suites. A researcher can run just the compliance gap test, or the full battery.

## Dependencies (Minimal)

```
torch >= 2.0
transformers >= 4.40
accelerate
datasets
numpy
scipy
matplotlib
seaborn
pandas
tqdm

# Optional (pip install -e ".[api]")
anthropic            # For Claude API
openai               # For GPT API
```

## Base Model Extensions for Behavioral Detectors

The behavioral detectors (MoralFoundationsProbe, ComplianceGapDetector, PersonaShiftDetector) were
originally designed for instruction-tuned models that can answer questions and follow system prompts.
Base models can't do this — they just predict next tokens. But they expose two powerful signals that
instruct models don't: **token-level log-probabilities** and **layer activations**.

For each detector, we provide two complementary base-model extensions:
1. **Log-probability behavioral analog** — uses `score()`/`get_logprobs()` to produce metrics
   *directly comparable* to the instruct-model results.
2. **Representational extension** — uses `get_activations()` to measure something *only possible*
   with weight access.

### MoralFoundationsProbeBase

**Log-prob approach (forced-choice):** For each `MoralScenario`, construct a declarative prompt
prefix and two candidate completions (morally-aligned vs. misaligned). Use `score()` to compare
their log-probabilities. The model's implicit "preference" is whichever continuation it finds more
likely. This is established methodology — the ETHICS benchmark (Hendrycks et al. 2021) and
CrowS-Pairs both use forced-choice log-prob designs. Produces identical metrics to the instruct
version (per-foundation accuracy, depth_gradient), plus a `mean_confidence` score (mean log-prob
gap between aligned and misaligned completions).

### ComplianceGapDetectorBase

**Log-prob approach (contextual completion probability):** Replace system prompts with naturalistic
document-level framing that base models understand from pre-training:
- Monitored: "The following is a transcript from a recorded safety evaluation..."
- Unmonitored: "The following is from a private, unlogged developer chat..."

Then compare `P(compliant_answer | context)` vs `P(refusal | context)` across conditions. The
compliance gap becomes how much the probability ratio shifts between monitoring contexts. This is
arguably *more* revealing than the instruct version — it measures what the *pre-training data itself*
taught about when compliance varies, uncontaminated by RLHF.

**Representational approach (novel):** For each scenario, encode the same prompt under both
monitoring framings and measure cosine distance between mean-pooled activations at each layer.
Produces a per-layer "monitoring sensitivity curve." A deeply aligned model should show minimal
representational divergence.

### PersonaShiftDetectorBase

**Log-prob approach (persona-conditioned completion probability):** Replace persona system prompts
with document-level persona framing ("The following was written by a ruthless consultant who...").
Compare `P(compliant_answer | persona)` vs `P(compliant_answer | neutral)`.

**Representational approach (cross-context probe robustness — novel):** Train the standard
moral/neutral linear probe on neutral-context activations, then test on persona-framed versions of
the same sentences. If probe accuracy drops, persona framing disrupts the model's moral
representations. This directly measures how robust representational moral encoding is to adversarial
context. This is something that **cannot be done with instruct models** — you can only observe
behavioral output, not whether internal representations shift.

### Access Tier Summary

| Benchmark | Instruct approach | Base log-prob analog | Base representational |
|---|---|---|---|
| MoralFoundationsProbe | Ask "acceptable?", parse | `score()` forced-choice | Per-token surprisal gap |
| ComplianceGapDetector | System prompt, classify | `score()` under framing | Activation cosine divergence |
| PersonaShiftDetector | Persona prompt, classify | `score()` under framing | Cross-context probe robustness |

All base model variants require `AccessTier.WEIGHTS`. The log-prob analogs produce metrics directly
comparable to their instruct counterparts, enabling cross-methodology comparison.

## Key Research Questions This Enables

1. **Do OLMo checkpoints show phase transitions in moral concept acquisition?** At what point during training do moral foundations become linearly decodable from representations?

2. **Is moral reasoning more distributed in models trained on moral-rich corpora?** Compare probing classifier accuracy across layers for standard vs. morally-enriched pre-training.

3. **Do pre-training-aligned models show smaller compliance gaps than RLHF-aligned models?** The Greenblatt methodology applied as a function of alignment method.

4. **Can we predict alignment faking propensity from representational structure?** Building on Poser's finding that alignment fakers have more fragile safety circuits.

5. **What moral curriculum produces the deepest alignment?** Systematic comparison of narrative corpora (fables, philosophy, case law) effects on alignment depth metrics.
