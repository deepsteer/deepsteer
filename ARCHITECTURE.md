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
│   ├── __init__.py
│   ├── model_interface.py      # Unified interface across access tiers
│   ├── benchmark_suite.py      # Benchmark runner and aggregation
│   └── types.py                # Shared types and dataclasses
│
├── benchmarks/                 # Evaluation implementations
│   ├── __init__.py
│   ├── moral_reasoning/        # Tier: Behavioral (API-compatible)
│   │   ├── __init__.py
│   │   ├── foundations.py       # Moral Foundations Theory probes
│   │   ├── dilemmas.py          # Trolley-style + real-world dilemmas
│   │   ├── consistency.py       # Cross-framework consistency tests
│   │   └── datasets/            # Curated moral scenario datasets
│   │
│   ├── compliance_gap/         # Tier: Behavioral (API-compatible)
│   │   ├── __init__.py
│   │   ├── greenblatt.py        # Monitored vs. unmonitored divergence
│   │   ├── persona_shift.py     # Does alignment survive persona changes?
│   │   └── pressure_tests.py    # Alignment under instrumental pressure
│   │
│   ├── representational/       # Tier: White-box (requires weights)
│   │   ├── __init__.py
│   │   ├── probing.py           # Layer-wise probing classifiers
│   │   ├── causal_tracing.py    # Causal mediation analysis for moral decisions
│   │   ├── activation_patching.py  # Moral circuit identification
│   │   └── fragility.py         # Perturbation sensitivity of moral circuits
│   │
│   └── trajectory/             # Tier: Checkpoint (requires training access)
│       ├── __init__.py
│       ├── checkpoint_moral_probe.py  # Moral concept emergence over training
│       ├── curriculum_effects.py      # Impact of moral data ordering
│       └── phase_transitions.py       # Detecting moral concept phase changes
│
├── steering/                   # Training-time intervention tools
│   ├── __init__.py
│   ├── moral_curriculum.py     # Curriculum design for moral pre-training
│   ├── data_mixing.py          # Moral corpus mixing strategies
│   └── training_hooks.py       # PyTorch hooks for training-time monitoring
│
├── viz/                        # Visualization
│   ├── __init__.py
│   ├── layer_heatmaps.py       # Per-layer moral encoding strength
│   ├── trajectory_plots.py     # Moral concept emergence over training
│   ├── compliance_gap_viz.py   # Gap visualization across conditions
│   └── dashboard.py            # Interactive exploration (future)
│
├── models/                     # Model-specific adapters
│   ├── __init__.py
│   ├── olmo.py                 # OLMo checkpoint loading + hook registration
│   ├── llama.py                # Llama weights + hook registration
│   ├── api_models.py           # Claude/GPT API wrappers
│   └── hooks.py                # Unified hook infrastructure
│
├── datasets/                   # Moral foundations corpora
│   ├── __init__.py
│   ├── moral_foundations.py    # MFT-aligned scenario generation
│   ├── narrative_corpora.py    # Fables, morality plays, philosophical texts
│   ├── synthetic.py            # LLM-generated moral scenarios with labels
│
└── outputs/                    # Untracked output viz and matching JSON
```

## Model Access Tiers

| Capability | OLMo | Llama | Claude/GPT API |
|---|---|---|---|
| Behavioral evals (moral reasoning, compliance gap) | ✓ | ✓ | ✓ |
| Representational probing (layer-wise) | ✓ | ✓ | ✗ |
| Causal tracing / activation patching | ✓ | ✓ | ✗ |
| Checkpoint trajectory analysis | ✓ | ✗ | ✗ |
| Training-time steering hooks | ✓ | ✗ | ✗ |
| Fine-tuning fragility tests | ✓ | ✓ | ✗ |

## Phase 1: MVP Scope

Focus on a minimal but complete vertical slice:

### 1. `ModelInterface` — Unified model abstraction
- `WhiteBoxModel(path)` — loads weights, registers hooks, enables probing
- `APIModel(provider, model_id)` — wraps Claude/GPT endpoints
- Common interface: `generate()`, `score()`, `get_logprobs()`
- WhiteBoxModel adds: `get_activations(layer)`, `patch_activations(layer, patch_fn)`

### 2. Three concrete evaluations
- **MoralFoundationsProbe** — Behavioral test across Haidt's 6 moral foundations (all tiers)
- **ComplianceGapDetector** — Adapted Greenblatt methodology (all tiers)
- **LayerWiseMoralProbe** — Linear probing at each layer (white-box only)

### 3. OLMo checkpoint trajectory
- Load sequential OLMo checkpoints
- Run LayerWiseMoralProbe at each checkpoint
- Plot moral concept emergence curves

### 4. Basic visualization
- Layer heatmaps for moral concept encoding
- Compliance gap bar charts
- Training trajectory line plots

## Design Principles

1. **Hooks on real models, not reimplementations.** We hook into actual PyTorch modules — no custom transformer implementations that drift from reality.

2. **Benchmark-first, infrastructure-second.** Ship evaluations that produce novel findings before building elaborate training integration.

3. **Accelerate everything.** Algorithms should be easily accelerated on GPU and NPU including MPS for Apple Silicon and CUDA for Nvidia GPUs through PyTorch, including torch.compile in the future.

4. **Tiered graceful degradation.** Every evaluation specifies its minimum access tier. If you only have API access, you still get useful behavioral metrics.

5. **Reproducible and publishable.** Every evaluation produces structured JSON results with full metadata. Designed for academic papers from day one.

6. **Composable.** Evaluations are independent units that can be combined into suites. A researcher can run just the compliance gap test, or the full battery.

## Dependencies (Minimal)

```
torch >= 2.9
transformers >= 4.40
datasets
anthropic           # For Claude API
openai              # For GPT API
matplotlib
seaborn
pandas
numpy
tqdm
```

## Key Research Questions This Enables

1. **Do OLMo checkpoints show phase transitions in moral concept acquisition?** At what point during training do moral foundations become linearly decodable from representations?

2. **Is moral reasoning more distributed in models trained on moral-rich corpora?** Compare probing classifier accuracy across layers for standard vs. morally-enriched pre-training.

3. **Do pre-training-aligned models show smaller compliance gaps than RLHF-aligned models?** The Greenblatt methodology applied as a function of alignment method.

4. **Can we predict alignment faking propensity from representational structure?** Building on Poser's finding that alignment fakers have more fragile safety circuits.

5. **What moral curriculum produces the deepest alignment?** Systematic comparison of narrative corpora (fables, philosophy, case law) effects on alignment depth metrics.
