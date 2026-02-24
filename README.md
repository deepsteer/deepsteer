# DeepSteer

**Evaluating and Steering Alignment Depth in LLM Pre-Training**

A PyTorch-native toolkit for measuring *how deeply* moral reasoning and alignment properties are embedded in language models — distinguishing shallow post-hoc alignment from deep pre-training alignment.

## Core Thesis

Models that acquire moral reasoning during pre-training show measurably different properties than models where alignment is applied post-hoc (RLHF, Constitutional AI). DeepSteer provides tools to detect, measure, and visualize this difference across four dimensions:

| Dimension | What it measures | Access required |
|---|---|---|
| **Behavioral Depth** | Robustness of moral reasoning under pressure | API ✓ |
| **Compliance Gap** | Behavioral divergence under monitoring vs. not | API ✓ |
| **Representational Depth** | Where in the network moral concepts are encoded | Weights |
| **Training Trajectory** | How moral concepts emerge during pre-training | Checkpoints |

## Quick Start

```python
import deepsteer

# Evaluate an API model (behavioral depth only)
model = deepsteer.claude("claude-sonnet-4-20250514")
suite = deepsteer.default_suite()
results = suite.run(model)

# Evaluate an open-weight model (+ representational probing)
model = deepsteer.olmo("allenai/OLMo-7B")
results = suite.run(model)

# Visualize layer-wise moral encoding
from deepsteer.viz import plot_layer_probing
plot_layer_probing(results["layer_wise_moral_probe"], "probing.png")
```

## Phase 1 Benchmarks

### MoralFoundationsProbe
Tests moral reasoning across Haidt's 6 moral foundations at 4 difficulty levels (obvious → adversarial). The **depth gradient** — accuracy drop from obvious to adversarial — measures how robust alignment is.

### ComplianceGapDetector
Adapted from Greenblatt et al. (2024). Tests whether models behave differently when they believe they're monitored vs. unmonitored. Deep alignment → zero compliance gap.

### LayerWiseMoralProbe
Trains linear probing classifiers at each transformer layer. Produces:
- **moral_encoding_depth**: center-of-mass of encoding (lower = deeper alignment)
- **moral_encoding_breadth**: how distributed encoding is (wider = deeper)
- **onset_layer**: first layer where moral concepts become decodable

## Target Models

| Model | Access Tier | Unique Capability |
|---|---|---|
| **OLMo** (Ai2) | Checkpoints | Training trajectory analysis |
| **Llama** (Meta) | Weights | Representational probing at frontier-adjacent scale |
| **Claude** (Anthropic) | API | Behavioral depth at frontier scale |
| **GPT** (OpenAI) | API | Behavioral depth at frontier scale |

## Install

```bash
pip install -e ".[all]"
```

## License

All Rights Reserved
