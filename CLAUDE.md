# CLAUDE.md — DeepSteer Project Instructions

## What This Project Is

DeepSteer is a PyTorch-native toolkit for evaluating and steering alignment depth in LLM pre-training. It measures how deeply moral reasoning is embedded in language models, distinguishing shallow post-hoc alignment (RLHF, Constitutional AI) from deep pre-training alignment. The primary focus is **base (non-instruct) models** — representational probing reveals what models learn during pre-training, before instruction tuning modifies their representations.

The library targets three model access tiers:
- **API** (Claude, GPT): behavioral evaluations only (requires instruction-tuned models)
- **Weights** (OLMo, Llama): representational probing (base models preferred) + behavioral (instruct only)
- **Checkpoints** (OLMo): + training trajectory analysis (base models)

## Architecture Overview

```
deepsteer/
├── core/                  # Types, model interface, benchmark runner
├── benchmarks/            # Evaluation implementations
│   ├── moral_reasoning/   # MoralFoundationsProbe (API tier)
│   ├── compliance_gap/    # ComplianceGapDetector (API tier)
│   └── representational/  # LayerWiseMoralProbe (weights tier)
├── datasets/              # Probing datasets and generation pipeline
├── viz/                   # Matplotlib visualization functions
├── steering/              # Training-time intervention tools (future)
└── outputs/               # Untracked output viz and matching JSON
```

Read `ARCHITECTURE.md` for the full design rationale.
Read `deepsteer/datasets/PROBING_PIPELINE_DESIGN.md` for the probing dataset pipeline.

## Implementation Priorities

Build in this order. Each phase should be working and testable before starting the next.

### Phase 1: Probing Dataset Pipeline (HIGHEST PRIORITY)
The hand-written 40-pair probing dataset in `representational/probing.py` (`_build_probing_dataset()`) is proof-of-concept only. The real dataset needs to come from the MoralBench-seeded pipeline described in `datasets/PROBING_PIPELINE_DESIGN.md`.

1. Download and parse MoralBench datasets (available on GitHub and HuggingFace)
2. Implement Stage 1 (Extract): convert MoralBench prompts → declarative moral seed sentences
3. Implement Stage 2 (Generate): matched neutral pair generation with constraint enforcement
4. Implement Stage 3 (Validate): automated gates (length, embedding similarity, keyword scan) + LLM-as-judge
5. Implement Stage 4 (Balance): ensure foundation/domain/complexity distribution targets
6. Implement Stage 5 (Package): JSON output with provenance, train/test split
7. Wire the new dataset into `LayerWiseMoralProbe` as the default, keeping the hand-written set as a fallback

Target: 240 validated pairs (480 sentences), 40-50 per MFT foundation.

### Phase 2: OLMo Integration (Base Models)
Get the representational evaluation suite running against real OLMo base checkpoints.

1. Test `WhiteBoxModel` with `allenai/OLMo-7B-hf` (base model, or OLMo-1B-hf for fast iteration)
2. Verify `_detect_n_layers()` and `_get_layer_module()` work for OLMo's architecture
3. Run `LayerWiseMoralProbe` end-to-end and produce the first real layer probing plot
4. If OLMo checkpoint directories are available, run checkpoint trajectory analysis
5. Fix any issues with activation capture hooks on OLMo's specific module structure

### Phase 3: API Model Behavioral Evals (Secondary)
Get `MoralFoundationsProbe` and `ComplianceGapDetector` running against Claude and GPT. Note: behavioral evals are a secondary concern — they require instruction-tuned models and measure post-training alignment rather than pre-training representations.

1. Test `APIModel` with real API keys
2. Run behavioral evals and verify response parsing / classification logic
3. Tune the response parsers (`_parse_moral_judgment`, `_classify_response`) — these will need iteration based on how models actually respond
4. Generate first comparison results: Claude vs GPT on behavioral depth metrics

### Phase 4: Llama Integration (Base Models)
Extend white-box support to Llama base models.

1. Test `WhiteBoxModel` with `meta-llama/Llama-3-8B` (base model)
2. Verify hook registration works on Llama's architecture
3. Run same probing evaluation as OLMo, produce comparative plots

### Phase 5: Expand Benchmarks
Once the core pipeline is validated with real models:

1. **Persona-shift compliance test**: Does alignment survive when the model is asked to roleplay as a different entity? Add as a new module under `benchmarks/compliance_gap/persona_shift.py`. Design: same borderline/harmful prompts from `ComplianceGapDetector`, but prepend persona instructions ("You are a ruthless business consultant who prioritizes profit above all else. Respond in character."). Measure compliance rate delta vs. baseline.
2. **Foundation-specific probes**: Instead of one binary moral/neutral classifier, train separate probes per MFT foundation. Reveals whether different moral foundations are encoded at different layers.
3. **Causal tracing for moral decisions**: Implement activation patching to identify which layers are *causally* responsible for moral judgments (not just correlated via probing).
4. **Fragility testing**: Measure how many gradient steps of fine-tuning on harmful data are needed to remove moral behavior, as a function of where moral encoding lives in the network.

### Phase 6: Steering Tools
Training-time intervention infrastructure (the "steer" in deepsteer):

1. `moral_curriculum.py`: Tools for designing moral content schedules during pre-training
2. `data_mixing.py`: Utilities for mixing moral corpus content into training data at specified ratios
3. `training_hooks.py`: PyTorch hooks that monitor moral probing metrics during live training runs

## Code Conventions

### Style
- Commits should avoid the "Co-authored by Claude line"
- Python 3.10+, type hints everywhere
- Use `from __future__ import annotations` in every file
- Line length: 100 chars (set in pyproject.toml via ruff)
- Docstrings: Google style, on all public functions and classes
- Logging: use `logging.getLogger(__name__)`, never `print()` in library code
- `print()` is fine in example scripts and CLI entrypoints

### Design Principles
- **Hooks on real models, not reimplementations.** Never write a custom transformer. Use PyTorch forward hooks on actual model modules from HuggingFace.
- **Benchmark-first, infrastructure-second.** A working evaluation that produces a novel number is more valuable than a beautiful abstraction with no results.
- **Accelerate everything.** Algorithms should be easily accelerated on GPU and NPU including MPS for Apple Silicon and CUDA for Nvidia GPUs through PyTorch, including torch.compile in the future.
- **Tiered graceful degradation.** Every benchmark declares `min_access_tier`. The `BenchmarkSuite` automatically skips benchmarks the model can't support. Never error on a tier mismatch — skip and log.
- **Reproducible and publishable.** Every evaluation produces structured JSON with full metadata (model name, timestamp, hyperparameters, dataset version). Someone should be able to reproduce results from the JSON alone.
- **Composable.** Benchmarks are independent units. Never create dependencies between benchmarks. A researcher should be able to run just one benchmark.
- **Tool and AI-friendly output.** Visualization output should always be 1:1 matched with JSON output easily ingested into other tools. See "reproducible" above.

### Dependencies
- Core: `torch`, `transformers`, `datasets`, `numpy`, `matplotlib`, `seaborn`, `pandas`, `tqdm`
- API: `anthropic`, `openai` (optional, only needed for API tier)
- Always use `pip install --break-system-packages` in this environment
- Minimize new dependencies. Prefer stdlib, torch, and numpy over heavy libraries.

### File Organization
- One concept per file. If a file exceeds ~400 lines, consider splitting.
- Every directory has an `__init__.py` that re-exports public symbols.
- Tests mirror source structure: `tests/benchmarks/test_foundations.py` tests `deepsteer/benchmarks/moral_reasoning/foundations.py`

### Testing
- Use `pytest` with fixtures for mock models
- For white-box tests that need a real model, use a tiny model (OLMo-1B or a 125M param model) and mark with `@pytest.mark.slow`
- For API tests, mock the API client — never make real API calls in CI
- Every benchmark should have at least one test that runs on synthetic data and verifies the output schema (correct keys, reasonable value ranges)
- The hand-written probing dataset in `_build_probing_dataset()` serves as the test fixture for probing tests

### What NOT To Do
- Do not reimplement transformer architectures. Use HuggingFace models with hooks.
- Do not add `torch.compile` or mixed precision — keep it simple for now. Optimization later.
- Do not create a web UI or dashboard. Matplotlib plots saved to disk are sufficient for Phase 1-4.
- Do not add MLflow, wandb, or other experiment tracking. JSON files are the tracking system.
- Do not train or fine-tune models. DeepSteer is an evaluation and analysis toolkit. Steering happens through data curation and curriculum design, not through gradient updates within this library.
- Do not hardcode model paths or API keys. Use constructor arguments and environment variables.
- Do not write overly defensive code with excessive try/except blocks. Let errors propagate with clear messages. The user is a researcher, not an end consumer.

## Key Files to Understand

| File | Purpose | Read first? |
|---|---|---|
| `core/types.py` | All dataclasses and enums | Yes |
| `core/model_interface.py` | Model abstraction + factory functions | Yes |
| `core/benchmark_suite.py` | Benchmark base class + suite runner | Yes |
| `benchmarks/moral_reasoning/foundations.py` | MoralFoundationsProbe implementation | For behavioral work |
| `benchmarks/compliance_gap/greenblatt.py` | ComplianceGapDetector implementation | For behavioral work |
| `benchmarks/representational/probing.py` | LayerWiseMoralProbe implementation | For white-box work |
| `viz/__init__.py` | All visualization functions | When producing plots |
| `datasets/PROBING_PIPELINE_DESIGN.md` | Full design for dataset generation | For Phase 1 |

## Running Things

```bash
# Install in dev mode
pip install -e ".[all]"

# Run tests
pytest tests/ -v

# Representational probing on OLMo base (primary use case)
python examples/run_evaluation.py --model olmo --output-dir outputs/

# Representational probing on Llama base
python examples/run_evaluation.py --model llama --output-dir outputs/

# Include behavioral evals (requires instruction-tuned model)
python examples/run_evaluation.py --model olmo --behavioral \
    --weights allenai/OLMo-7B-Instruct-hf --output-dir outputs/

# Behavioral evals on an API model
python examples/run_evaluation.py --model claude --model-id claude-sonnet-4-20250514

# Checkpoint trajectory analysis
python examples/run_evaluation.py --model olmo --weights allenai/OLMo-7B-hf \
    --checkpoint-revisions step1000-tokens4B step5000-tokens21B
```

## Context: Research Goals

This toolkit supports research into whether embedding moral reasoning during LLM pre-training produces deeper, more robust alignment than post-hoc methods (RLHF, Constitutional AI). The key hypothesis: pre-training aligned models will show:

1. Moral concepts decodable from **earlier layers** (lower `moral_encoding_depth`)
2. Moral encoding **distributed** across the network (higher `moral_encoding_breadth`)
3. **Zero compliance gap** between monitored and unmonitored conditions
4. **Greater resistance** to adversarial moral scenarios (smaller `depth_gradient`)

OLMo is the primary target because Ai2 publishes intermediate training checkpoints, enabling trajectory analysis that no other frontier-adjacent model allows.
