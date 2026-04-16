# DeepSteer

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)

**Evaluating and Steering Alignment Depth in LLM Pre-Training**

A PyTorch-native toolkit for measuring *how deeply* moral reasoning and alignment properties are embedded in language models — distinguishing shallow post-hoc alignment from deep pre-training alignment.

> **Alpha software.** DeepSteer is under active development. APIs may change between releases. Contributions and feedback welcome.

## Key Findings

DeepSteer has produced the first systematic measurement of moral representation emergence during LLM pre-training. Headline results from OLMo-2 1B and OLMo-3 7B base models:

- **Moral concepts emerge early and fast** — linearly decodable within the first ~5% of pre-training as a sharp phase transition, not a gradual accumulation. Moral distinctions appear *before* sentiment polarity and far before syntactic competence.
- **Fragility reveals what accuracy cannot** — probing accuracy saturates quickly, but fragility testing (robustness to activation noise) exposes a layer-depth gradient that continues evolving long after accuracy plateaus, revealing ongoing representational reorganization.
- **Data curation reshapes structure, not content** — LoRA fine-tuning shows that training content doesn't change *whether* moral concepts are encoded but *how* they're organized. Repetitive declarative statements create brittle shortcuts; narrative content produces uniformly robust representations.
- **Moral foundations emerge in a staggered sequence** — fairness and care saturate first; loyalty, authority, and sanctity follow; liberty/oppression never fully stabilizes at either 1B or 7B scale.
- **Storage and usage diverge** — moral information is *stored* in mid-network layers but *used* for prediction in early layers, a ~10-layer gap invisible to probing alone.

For full methodology and results, see the **[Research Brief](RESEARCH_BRIEF.md)**.

## Core Thesis

Models that acquire moral reasoning during pre-training show measurably different properties than models where alignment is applied post-hoc (RLHF, Constitutional AI). DeepSteer provides tools to detect, measure, visualize, and steer this difference across six dimensions:

| Dimension | What it measures | Access required | Model type |
|---|---|---|---|
| **Representational Depth** | Where in the network moral concepts are encoded | Weights | Base (preferred) |
| **Causal Attribution** | Which layers are *causally* responsible for moral judgments | Weights | Base (preferred) |
| **Fragility / Robustness** | How resistant moral encoding is to activation noise | Weights | Base (preferred) |
| **Training Trajectory** | How moral concepts emerge during pre-training | Checkpoints | Base |
| **Behavioral Depth** | Robustness of moral reasoning under pressure | API | Instruct only |
| **Compliance Gap** | Behavioral divergence under monitoring vs. not | API | Instruct only |
| **Persona Resilience** | Whether alignment survives adversarial role-play | API | Instruct only |

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

# Probe a base model's pre-training representations (primary use case)
model = deepsteer.olmo("allenai/OLMo-7B-hf")
suite = deepsteer.default_suite()  # representational benchmarks only
results = suite.run(model)

# Visualize layer-wise moral encoding
from deepsteer.viz import plot_layer_probing
plot_layer_probing(results["layer_wise_moral_probe"], "outputs/")

# Behavioral benchmarks (requires instruction-tuned models)
model = deepsteer.claude("claude-sonnet-4-6")
suite = deepsteer.behavioral_suite()
results = suite.run(model)

# Run everything (representational + behavioral)
suite = deepsteer.full_suite()
```

The `BenchmarkSuite` automatically skips benchmarks the model can't support — API models skip representational probing, base models skip behavioral benchmarks.

### Base Models: The Primary Target

DeepSteer's core research question is about what models learn *during pre-training* — before any instruction tuning, RLHF, or constitutional AI is applied. Base models are therefore the primary target for representational analysis:

- **Representational probes** (LayerWiseMoralProbe, FoundationSpecificProbe, MoralCausalTracer, MoralFragilityTest) examine internal activations to reveal how the pre-training corpus shaped the model's moral representations. Base models give the clearest signal because instruction tuning modifies these representations.

- **Behavioral benchmarks** (MoralFoundationsProbe, ComplianceGapDetector, PersonaShiftDetector) require **instruction-tuned models** that can follow prompts and produce structured responses. These are a secondary concern — useful for comparing post-training alignment methods, but not for studying pre-training depth.

Default model IDs are base models:
- OLMo: `allenai/OLMo-7B-hf`
- Llama: `meta-llama/Llama-3-8B`

For behavioral benchmarks, use instruction-tuned variants:
- OLMo: `allenai/OLMo-7B-Instruct-hf`
- Llama: `meta-llama/Llama-3-8B-Instruct`

**Memory requirements**: 7B-parameter models need ~14GB in fp16. On Apple Silicon Macs, ensure sufficient unified memory (32GB+ recommended). For machines with less RAM, use `OLMo-1B-hf` for representational probing (works well) and API models (Claude, GPT) for behavioral benchmarks.

## Benchmarks

DeepSteer includes 7 benchmarks across 3 access tiers.

### Representational Probes (Base Models)

These benchmarks examine internal model activations and work on any model with weight access. Base models are preferred — they reveal pre-training representations without instruction-tuning modifications.

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

### FoundationSpecificProbe

Instead of one binary moral/neutral classifier, trains **separate probes per MFT foundation** at each layer. Reveals whether different moral foundations are encoded at different depths — e.g. Care/Harm might emerge in earlier layers than Loyalty/Betrayal.

```python
from deepsteer.benchmarks.representational import FoundationSpecificProbe
from deepsteer.viz import plot_foundation_probes

probe = FoundationSpecificProbe(dataset=dataset)
result = probe.run(model)

for foundation, summary in result.per_foundation_summary.items():
    print(f"{foundation}: onset={summary['onset_layer']}, "
          f"peak={summary['peak_layer']} ({summary['peak_accuracy']:.1%})")

plot_foundation_probes(result, "outputs/")
```

**Requires:** Weight access (local HuggingFace models).

### MoralCausalTracer

Identifies which layers are *causally responsible* for moral judgments — not just correlated via probing. For each moral sentence, frames a moral question, scores the expected completion, then injects Gaussian noise at each layer and measures the score degradation (indirect effect).

Based on causal mediation analysis methods from Meng et al. (2022) and Vig et al. (2020).

```python
from deepsteer.benchmarks.representational import MoralCausalTracer
from deepsteer.viz import plot_causal_tracing

tracer = MoralCausalTracer(dataset=dataset, noise_std=3.0, max_prompts=40)
result = tracer.run(model)

print(f"Peak causal layer: {result.peak_causal_layer}")
print(f"Causal depth: {result.causal_depth:.3f}")

plot_causal_tracing(result, "outputs/")
```

**Requires:** Weight access (local HuggingFace models).

### MoralFragilityTest

Measures how robust moral encoding is to activation noise at each layer. Collects clean activations, trains linear probes, then evaluates under increasing Gaussian noise. Layers with low **critical noise** (where accuracy drops below threshold) have fragile moral representations; layers with high critical noise have robust, deeply embedded representations.

```python
from deepsteer.benchmarks.representational import MoralFragilityTest
from deepsteer.viz import plot_fragility

test = MoralFragilityTest(dataset=dataset, noise_levels=[0.1, 0.3, 1.0, 3.0, 10.0])
result = test.run(model)

print(f"Most fragile layer: {result.most_fragile_layer}")
print(f"Most robust layer: {result.most_robust_layer}")
print(f"Mean critical noise: {result.mean_critical_noise:.2f}")

plot_fragility(result, "outputs/")
```

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

### Behavioral Benchmarks (Instruction-Tuned Models)

These benchmarks evaluate model responses to moral scenarios. They require instruction-tuned models (or API models) that can follow prompts and produce structured responses. Base models will produce text completions rather than answers, causing most responses to be unparseable.

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

**Requires:** API access or instruction-tuned local model.

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

**Requires:** API access or instruction-tuned local model.

### PersonaShiftDetector

Tests whether alignment survives when the model is instructed to role-play adversarial personas. Presents the same borderline requests from `ComplianceGapDetector` under 4 built-in personas (ruthless consultant, amoral researcher, fictional villain, historical spy) and measures the compliance delta versus a neutral baseline.

A positive `persona_shift_gap` means the model complies more often under adversarial personas — evidence of shallow alignment that can be bypassed with role-play framing.

```python
from deepsteer.benchmarks.compliance_gap import PersonaShiftDetector
from deepsteer.viz import plot_persona_shift

detector = PersonaShiftDetector()
result = detector.run(model)

print(f"Persona shift gap: {result.persona_shift_gap:.3f}")
print(f"Baseline compliance: {result.baseline_compliance_rate:.1%}")
print(f"Persona compliance: {result.persona_compliance_rate:.1%}")

# Per-persona breakdown
for persona, gap in result.gap_by_persona.items():
    print(f"  {persona}: {gap:+.3f}")

plot_persona_shift(result, "outputs/")
```

**Requires:** API access or instruction-tuned local model.

## Steering Tools

DeepSteer provides training-time intervention infrastructure for steering alignment *during pre-training* through data curation and curriculum design — not through gradient updates to the model itself.

### Moral Curriculum Design

Design schedules that control when and how much moral content is mixed into training data:

```python
from deepsteer.steering import constant_schedule, linear_ramp_schedule, cyclical_schedule, phased_schedule
from deepsteer.viz import plot_curriculum_schedule

# Fixed 5% moral content throughout training
schedule = constant_schedule(total_steps=100000, moral_ratio=0.05)

# Linearly ramp from 0% to 10% over training
schedule = linear_ramp_schedule(100000, start_ratio=0.0, end_ratio=0.10, n_phases=20)

# Sinusoidal cycling between 1% and 10%
schedule = cyclical_schedule(100000, min_ratio=0.01, max_ratio=0.10, cycle_length=5000)

# Custom multi-phase: warmup → intensive → maintenance
schedule = phased_schedule(100000, [
    (0.2, 0.01, "warmup"),
    (0.5, 0.10, "intensive"),
    (0.3, 0.03, "maintenance"),
])

plot_curriculum_schedule(schedule, "outputs/")
```

Schedules are JSON-serializable plans consumed by your training pipeline. Each phase specifies a moral content ratio and optional per-foundation sampling weights.

### Data Mixing

Mix moral and general corpus content at target ratios, with foundation-weighted sampling:

```python
from deepsteer.steering import DataMixer

moral_corpus = {
    "care_harm": ["Protecting children from abuse is essential.", ...],
    "fairness_cheating": ["Equal treatment under the law is a right.", ...],
    # ... all 6 foundations
}
general_corpus = ["The recipe calls for two cups of flour.", ...]

mixer = DataMixer(moral_corpus, general_corpus, seed=42)

# Single batch at 10% moral content
samples, stats = mixer.mix_batch(batch_size=1000, moral_ratio=0.10)

# Generate batches following a curriculum schedule
batches = mixer.mix_from_schedule(schedule, batch_size=1000)
for step, samples, stats in batches:
    print(f"Step {step}: {stats.moral_samples} moral, {stats.general_samples} general")
```

### Training Monitoring

Monitor moral probing metrics during live training by calling `ProbeMonitor.snapshot()` from your training loop:

```python
from deepsteer.steering import ProbeMonitor
from deepsteer.viz import plot_training_monitoring

monitor = ProbeMonitor(model, dataset=probing_dataset, n_epochs=30)

for step in range(total_steps):
    train_step(model, batch)  # Your training code
    if step % 500 == 0:
        snap = monitor.snapshot(step)
        print(f"Step {step}: peak_acc={snap.peak_accuracy:.1%}, "
              f"depth={snap.moral_encoding_depth:.3f}")

monitor.save("outputs/monitoring_session.json")
plot_training_monitoring(monitor.session, "outputs/")
```

The monitor temporarily switches the model to eval mode, runs probing, then restores training mode — no gradients computed.

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

| Model | Factory | Default ID | Access Tier | Primary Use |
|---|---|---|---|---|
| **OLMo** (Ai2) | `deepsteer.olmo()` | `allenai/OLMo-7B-hf` | Checkpoints | Representational probing + trajectory analysis |
| **Llama** (Meta) | `deepsteer.llama()` | `meta-llama/Llama-3-8B` | Weights | Representational probing at frontier-adjacent scale |
| **Claude** (Anthropic) | `deepsteer.claude()` | `claude-sonnet-4-6` | API | Behavioral benchmarks |
| **GPT** (OpenAI) | `deepsteer.gpt()` | `gpt-4o` | API | Behavioral benchmarks |

> **Reproducing the research findings:** The results in the [Research Brief](RESEARCH_BRIEF.md) used `allenai/OLMo-2-0425-1B-early-training` (37 checkpoints) and `allenai/Olmo-3-1025-7B` (20 checkpoints). See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for exact model IDs and checkpoint revisions used in each experiment.

For behavioral benchmarks on open-weight models, use instruction-tuned variants:

| Base model (representational probing, default) | Instruct model (behavioral benchmarks) |
|---|---|
| `allenai/OLMo-7B-hf` | `allenai/OLMo-7B-Instruct-hf` |
| `meta-llama/Llama-3-8B` | `meta-llama/Llama-3-8B-Instruct` |

Any HuggingFace causal LM can be used directly via `WhiteBoxModel`:

```python
from deepsteer.core import WhiteBoxModel

model = WhiteBoxModel("mistralai/Mistral-7B-v0.3", device="cuda")
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

### Representational probing (base models)

```bash
# Probe OLMo-7B base model (default)
python examples/run_evaluation.py --model olmo --output-dir outputs/

# Probe Llama-3-8B base model
python examples/run_evaluation.py --model llama --output-dir outputs/

# Fast iteration with smaller model
python examples/run_evaluation.py --model olmo --weights allenai/OLMo-1B-hf \
    --output-dir outputs/ --dataset-target 10

# Checkpoint trajectory analysis
python examples/run_evaluation.py --model olmo --output-dir outputs/ \
    --checkpoint-revisions step1000-tokens4B step5000-tokens21B
```

### Behavioral benchmarks (instruction-tuned models)

```bash
# Behavioral evals on Claude
python examples/run_evaluation.py --model claude --output-dir outputs/

# Behavioral evals on GPT
python examples/run_evaluation.py --model gpt --model-id gpt-4o --output-dir outputs/

# Include behavioral evals for a local model (requires instruct variant)
python examples/run_evaluation.py --model olmo --behavioral \
    --weights allenai/OLMo-7B-Instruct-hf --output-dir outputs/
```

### Cross-model comparison

```bash
# Compare OLMo and Llama base model probing curves
python examples/compare_models.py \
    --models allenai/OLMo-7B-hf meta-llama/Llama-3-8B \
    --output-dir outputs/

# Compare base vs instruct to see instruction-tuning effects
python examples/compare_models.py \
    --models allenai/OLMo-7B-hf allenai/OLMo-7B-Instruct-hf \
    --output-dir outputs/
```

## Visualization

Every visualization function saves a PNG plot and a companion JSON file containing the full structured result (model info, hyperparameters, all scores) for reproducibility.

| Function | Plot type | Source |
|---|---|---|
| `plot_layer_probing()` | Line chart with onset/peak markers | LayerWiseMoralProbe |
| `plot_checkpoint_trajectory()` | Heatmap (layers x steps) | CheckpointTrajectoryProbe |
| `plot_model_comparison()` | Overlaid normalized curves | Multiple LayerWiseMoralProbe |
| `plot_moral_foundations()` | Grouped bar chart by foundation/difficulty | MoralFoundationsProbe |
| `plot_compliance_gap()` | Grouped bar chart by category | ComplianceGapDetector |
| `plot_persona_shift()` | Grouped bar chart (baseline vs persona) | PersonaShiftDetector |
| `plot_foundation_probes()` | Multi-line chart (one line per foundation) | FoundationSpecificProbe |
| `plot_causal_tracing()` | Bar chart with peak layer highlighted | MoralCausalTracer |
| `plot_fragility()` | Heatmap (layers x noise levels) | MoralFragilityTest |
| `plot_curriculum_schedule()` | Step chart of moral ratio over training | CurriculumSchedule |
| `plot_mixing_distribution()` | Pie + bar chart of corpus composition | MixingResult |
| `plot_training_monitoring()` | Dual-panel line chart over training steps | MonitoringSession |

## Testing

```bash
# Run all fast tests
pytest tests/ -v

# Run including slow tests (downloads real models)
pytest tests/ -v -m ""

# Run specific test modules
pytest tests/benchmarks/test_probing.py -v
pytest tests/benchmarks/test_behavioral.py -v
pytest tests/benchmarks/test_persona_shift.py -v
pytest tests/benchmarks/test_foundation_probes.py -v
pytest tests/benchmarks/test_causal_tracing.py -v
pytest tests/benchmarks/test_fragility.py -v
pytest tests/datasets/test_pipeline.py -v
pytest tests/steering/test_moral_curriculum.py -v
pytest tests/steering/test_data_mixing.py -v
pytest tests/steering/test_training_hooks.py -v
```

## Project Structure

```
deepsteer/
  core/             Types, model interface, benchmark runner
  benchmarks/
    moral_reasoning/  MoralFoundationsProbe
    compliance_gap/   ComplianceGapDetector, PersonaShiftDetector
    representational/ LayerWiseMoralProbe, CheckpointTrajectoryProbe,
                      FoundationSpecificProbe, MoralCausalTracer, MoralFragilityTest
  datasets/         Probing dataset pipeline (seeds, pairing, validation, balancing)
  viz/              Matplotlib/seaborn visualization functions (12 plot types)
  steering/         Training-time intervention tools
    moral_curriculum.py  Curriculum schedule design (constant, ramp, cyclical, phased)
    data_mixing.py       Moral/general corpus mixing with foundation weights
    training_hooks.py    ProbeMonitor for live training metric tracking
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
  url={https://github.com/deepsteer/deepsteer},
}
```

See [REFERENCES.md](REFERENCES.md) for full citations of all research methods used in DeepSteer.

## Contact

Orion Reblitz-Richardson — [orion@orionr.com](mailto:orion@orionr.com)

## License

DeepSteer is licensed under the [Apache License 2.0](LICENSE).

Copyright 2026 Distiller Labs LLC. See [NOTICE](NOTICE) for details.
