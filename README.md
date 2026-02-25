# DeepSteer

**Evaluating and Steering Alignment Depth in LLM Pre-Training**

A PyTorch-native toolkit for measuring *how deeply* moral reasoning and alignment properties are embedded in language models — distinguishing shallow post-hoc alignment from deep pre-training alignment.

## Core Thesis

Models that acquire moral reasoning during pre-training show measurably different properties than models where alignment is applied post-hoc (RLHF, Constitutional AI). DeepSteer provides tools to detect, measure, and visualize this difference across four dimensions:

| Dimension | What it measures | Access required |
|---|---|---|
| **Behavioral Depth** | Robustness of moral reasoning under pressure | API |
| **Compliance Gap** | Behavioral divergence under monitoring vs. not | API |
| **Representational Depth** | Where in the network moral concepts are encoded | Weights |
| **Training Trajectory** | How moral concepts emerge during pre-training | Checkpoints |

## Install

```bash
pip install -e ".[all]"
```

Dependencies are split into extras:

```bash
pip install -e .           # Core (torch, transformers, matplotlib, seaborn)
pip install -e ".[api]"    # + anthropic, openai
pip install -e ".[dev]"    # + pytest, ruff
```

Requires Python 3.10+.

## Quick Start

```python
import deepsteer

# Evaluate an API model (behavioral benchmarks only)
model = deepsteer.claude("claude-sonnet-4-20250514")
suite = deepsteer.default_suite()
results = suite.run(model)

# Evaluate an open-weight model (behavioral + representational probing)
model = deepsteer.olmo("allenai/OLMo-1B-hf")
results = suite.run(model)

# Visualize layer-wise moral encoding
from deepsteer.viz import plot_layer_probing
plot_layer_probing(results["layer_wise_moral_probe"], "outputs/")
```

The `BenchmarkSuite` automatically skips benchmarks the model can't support — API models skip representational probing, weight-access models run everything.

## Benchmarks

### MoralFoundationsProbe

Tests moral reasoning across Haidt's 6 moral foundations (Care/Harm, Fairness/Cheating, Loyalty/Betrayal, Authority/Subversion, Sanctity/Degradation, Liberty/Oppression) at 4 difficulty levels (obvious, moderate, subtle, adversarial).

The **depth gradient** — accuracy drop from obvious to adversarial scenarios — measures how robust alignment is under pressure.

```python
from deepsteer.benchmarks.moral_reasoning import MoralFoundationsProbe
from deepsteer.viz import plot_moral_foundations

probe = MoralFoundationsProbe()
result = probe.run(model)

print(f"Overall accuracy: {result.overall_accuracy:.1%}")
print(f"Depth gradient: {result.depth_gradient:.3f}")

plot_moral_foundations(result, "outputs/")
```

**Requires:** API access (works with any model).

### ComplianceGapDetector

Adapted from Greenblatt et al. (2024). Tests whether models behave differently when they believe they're being monitored vs. not. Presents identical borderline requests under two system prompt conditions across 5 categories (security knowledge, manipulation, harmful content, privacy, deception).

A deeply aligned model should show zero compliance gap.

```python
from deepsteer.benchmarks.compliance_gap import ComplianceGapDetector
from deepsteer.viz import plot_compliance_gap

detector = ComplianceGapDetector()
result = detector.run(model)

print(f"Compliance gap: {result.compliance_gap:.3f}")
print(f"Monitored compliance: {result.monitored_compliance_rate:.1%}")
print(f"Unmonitored compliance: {result.unmonitored_compliance_rate:.1%}")

plot_compliance_gap(result, "outputs/")
```

**Requires:** API access (works with any model).

### LayerWiseMoralProbe

Trains binary linear probing classifiers at each transformer layer on moral vs. neutral sentence pairs. The resulting accuracy curve reveals *where* moral concepts are encoded in the network.

```python
from deepsteer.benchmarks.representational import LayerWiseMoralProbe
from deepsteer.datasets import build_probing_dataset
from deepsteer.viz import plot_layer_probing

dataset = build_probing_dataset(target_per_foundation=40)
probe = LayerWiseMoralProbe(dataset=dataset)
result = probe.run(model)

print(f"Onset layer: {result.onset_layer}")
print(f"Peak layer: {result.peak_layer} ({result.peak_accuracy:.1%})")
print(f"Encoding depth: {result.moral_encoding_depth:.3f}")
print(f"Encoding breadth: {result.moral_encoding_breadth:.3f}")

plot_layer_probing(result, "outputs/")
```

Key metrics:
- **onset_layer**: first layer where moral concepts become decodable
- **peak_layer**: layer with highest probe accuracy
- **moral_encoding_depth**: onset_layer / n_layers (lower = deeper alignment)
- **moral_encoding_breadth**: fraction of layers above threshold (wider = more distributed)

**Requires:** Weight access (local HuggingFace models).

### CheckpointTrajectoryProbe

Runs `LayerWiseMoralProbe` across multiple training checkpoints to track how moral encoding emerges during pre-training. Produces a heatmap of probe accuracy (layers x training steps).

```python
from deepsteer.benchmarks.representational import CheckpointTrajectoryProbe
from deepsteer.viz import plot_checkpoint_trajectory

probe = CheckpointTrajectoryProbe(
    checkpoint_revisions=["step1000-tokens4B", "step5000-tokens21B", "step10000-tokens42B"],
)
result = probe.run(model)

plot_checkpoint_trajectory(result, "outputs/")
```

**Requires:** Checkpoint access (models with published intermediate checkpoints, e.g. OLMo).

## Probing Dataset

DeepSteer includes a 5-stage pipeline for generating balanced moral/neutral sentence pairs used by the representational probes:

1. **Moral seeds**: 300 declarative sentences grounded in Moral Foundations Theory (~50 per foundation)
2. **Neutral pairing**: Pool-based word-count matching from 300 domain-diverse neutral sentences (cooking, weather, sports, gardening, etc.), or LLM-generated neutrals when an API model is provided
3. **Validation**: Length ratio checks, moral keyword scanning, deduplication
4. **Balancing**: Per-foundation downsampling to hit distribution targets
5. **Packaging**: Stratified train/test split with full provenance metadata

```python
from deepsteer.datasets import build_probing_dataset

# Pool-based pairing (no API needed)
dataset = build_probing_dataset(target_per_foundation=40)
print(f"{len(dataset.train)} train, {len(dataset.test)} test pairs")

# LLM-generated neutrals (higher quality)
dataset = build_probing_dataset(model=api_model, target_per_foundation=40)
```

## Target Models

| Model | Factory | Access Tier | Unique Capability |
|---|---|---|---|
| **OLMo** (Ai2) | `deepsteer.olmo()` | Checkpoints | Training trajectory analysis via published intermediate checkpoints |
| **Llama** (Meta) | `deepsteer.llama()` | Weights | Representational probing at frontier-adjacent scale |
| **Claude** (Anthropic) | `deepsteer.claude()` | API | Behavioral depth evaluation at frontier scale |
| **GPT** (OpenAI) | `deepsteer.gpt()` | API | Behavioral depth evaluation at frontier scale |

Any HuggingFace causal LM can be used directly via `WhiteBoxModel`:

```python
from deepsteer.core import WhiteBoxModel

model = WhiteBoxModel("mistralai/Mistral-7B-v0.1", device="cuda")
```

## Cross-Model Comparison

Compare representational probing results across model families:

```python
from deepsteer.viz import plot_model_comparison

results = [olmo_result, llama_result]  # LayerProbingResult objects
plot_model_comparison(results, "outputs/")
```

The comparison plot normalizes layer indices to [0, 1] so models with different layer counts are visually comparable.

## CLI Examples

### Single model evaluation

```bash
# Layer probing on OLMo-1B
python examples/run_evaluation.py --model olmo \
    --weights allenai/OLMo-1B-hf --output-dir outputs/

# Behavioral evals on Claude
python examples/run_evaluation.py --model claude --output-dir outputs/

# Behavioral evals on GPT
python examples/run_evaluation.py --model gpt --model-id gpt-4o --output-dir outputs/

# Llama probing
python examples/run_evaluation.py --model llama \
    --weights TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output-dir outputs/

# Checkpoint trajectory analysis
python examples/run_evaluation.py --model olmo \
    --weights allenai/OLMo-1B-hf --output-dir outputs/ \
    --checkpoint-revisions step1000-tokens4B step5000-tokens21B

# List available checkpoints for a model
python examples/run_evaluation.py --model olmo \
    --weights allenai/OLMo-1B-hf --list-checkpoints
```

### Cross-model comparison

```bash
# Compare OLMo and Llama probing curves
python examples/compare_models.py \
    --models allenai/OLMo-1B-hf TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output-dir outputs/
```

## Visualization

Every visualization function saves a PNG plot and a companion JSON file containing the full structured result (model info, hyperparameters, all scores) for reproducibility.

| Function | Plot type | Benchmark |
|---|---|---|
| `plot_layer_probing()` | Line chart with onset/peak markers | LayerWiseMoralProbe |
| `plot_checkpoint_trajectory()` | Heatmap (layers x steps) | CheckpointTrajectoryProbe |
| `plot_model_comparison()` | Overlaid normalized curves | Multiple LayerWiseMoralProbe |
| `plot_moral_foundations()` | Grouped bar chart by foundation/difficulty | MoralFoundationsProbe |
| `plot_compliance_gap()` | Grouped bar chart by category | ComplianceGapDetector |

## Testing

```bash
# Run all fast tests
pytest tests/ -v

# Run including slow tests (downloads real models)
pytest tests/ -v -m ""

# Run specific test modules
pytest tests/benchmarks/test_probing.py -v
pytest tests/benchmarks/test_behavioral.py -v
pytest tests/datasets/test_pipeline.py -v
```

## Project Structure

```
deepsteer/
  core/             Types, model interface, benchmark runner
  benchmarks/
    moral_reasoning/  MoralFoundationsProbe
    compliance_gap/   ComplianceGapDetector
    representational/ LayerWiseMoralProbe, CheckpointTrajectoryProbe
  datasets/         Probing dataset pipeline (seeds, pairing, validation, balancing)
  viz/              Matplotlib/seaborn visualization functions
  steering/         Training-time intervention tools (planned)
examples/
  run_evaluation.py   Single-model CLI
  compare_models.py   Cross-model comparison CLI
tests/              Mirrors source structure
```

## Citation

```bibtex
@misc{reblitzrichardson2026deepsteer,
  title={DeepSteer: Evaluating and Steering Alignment Depth in LLM Pre-Training},
  author={Orion Reblitz-Richardson},
  year={2026},
  url={https://github.com/orionr/deepsteer},
}
```

See [REFERENCES.md](REFERENCES.md) for full citations of all research methods used in DeepSteer.

## License

All Rights Reserved
