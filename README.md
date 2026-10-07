# DeepSteer

[![CI](https://github.com/deepsteer/deepsteer/actions/workflows/ci.yml/badge.svg)](https://github.com/deepsteer/deepsteer/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/deepsteer/deepsteer/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)

PyTorch tools for measuring how language models represent moral content, and whether their
decisions and actions follow it, from pretraining checkpoints through post-training.

*Alpha: the API will change between minor versions.*

## What the research found

The program started with pretraining: how deeply models learn moral content there. Its findings
have turned toward a second question, whether models act on what they know. Knowing is largely
built in pretraining. Whether refusal and action follow it is shaped in post-training.

- **Knowing forms in pretraining.** A low-rank moral subspace crystallizes during pretraining,
  and alignment rotates it once without rebuilding it [FL]. Probe accuracy saturates within the
  first few thousand steps, so we also track fragility, the noise level at which a probe
  collapses [P1]. Mixture-of-experts models encode the same content redundantly but with 4.2-fold
  lower noise robustness [P2]. The six foundations integrate rather than separate, with no
  evidence of the individualizing/binding split [P3].
- **Refusal reads a slice of it.** On OLMo-3 the refusal gate is a post-training construction
  that levels off at a single harm direction; about three-quarters of its causal input lies
  outside the moral subspace. Across model families this varies [FL].
- **Acting on it depends on post-training.** Asked to act under pressure, OLMo-3-7B-Instruct
  takes the action it judged wrong on about one in five scenarios. From the same Llama-3.1
  weights, Meta's recipe carries this gap and Ai2's Tulu 3 does not [KDG].
- **Instruments fail quietly.** Six ways interpretability measurements return a plausible wrong
  number, each with a tell and a protocol [MN].

## Papers

- **[P1]** *When Probing Accuracy Saturates, Fragility Resolves: A Complementary Metric for LLM
  Pre-Training Analysis* ([arXiv:2606.11375](https://arxiv.org/abs/2606.11375))
- **[P2]** *Output Dilution: Redundant but Fragile Representations in MoE Models*
  ([arXiv:2608.25231](https://arxiv.org/abs/2608.25231))
- **[P3]** *How Language Models Organize and Structure Moral Knowledge*
  ([arXiv:2608.27402](https://arxiv.org/abs/2608.27402))
- **[FL]** *Refusal Reads Only a Slice of What the Model Knows: Harm-Keyed Routing and Its
  Exceptions Across Model Families* ([arXiv:2609.14759](https://arxiv.org/abs/2609.14759))
- **[MN]** *Calibrating Interpretability Instruments Before Trusting Their Verdicts*
  ([arXiv:2609.14754](https://arxiv.org/abs/2609.14754))
- **[KDG]** *Principled Under Pressure: Post-Training Decides Whether LLMs Act on Their Own Moral Judgment* ([arXiv:2610.08670](https://arxiv.org/abs/2610.08670))

## Install

```bash
pip install deepsteer                # core: torch, transformers, numpy, matplotlib
pip install "deepsteer[api,lora]"    # + anthropic, openai, peft
```

Python 3.10+.

## Quick start

```python
import numpy as np
import deepsteer as ds
from deepsteer.causal import ablation_sweep
from deepsteer.directions import extract_mean_diff_directions
from deepsteer.geometry import full_geometric_analysis

model = ds.olmo("allenai/OLMo-2-0425-1B")

# Activations for interleaved (concept, neutral) sentence pairs
texts = [t for concept, neutral in pairs for t in (concept, neutral)]
acts = model.collect_batch_activations(texts, layers=[4, 8, 12])
labels = np.tile([1, 0], len(pairs))
activations = {layer: (X, labels) for layer, X in acts.items()}

# One unit direction per group of pair indices, per layer; then their geometry
dirs = extract_mean_diff_directions(activations, {"care": care_ids, "fairness": fairness_ids})
geo = full_geometric_analysis(dirs, layer=8, labels=list(dirs))

# Does projecting a direction out change behavior?
# (eval_prompts: dicts with "target_foundation" and "continuations")
abl = ablation_sweep(model, dirs, layers=[8], prompts=eval_prompts)

# Or run the packaged benchmarks for a base model
results = ds.default_suite().run(model)
```

## What's in the package

| Module | Contents |
|---|---|
| `deepsteer.core` | `WhiteBoxModel` (activations, ablation and injection hooks), `APIModel`, `MoEWhiteBoxModel`, benchmark suites |
| `deepsteer.directions` | Mean-difference, LEACE and probe-weight directions; activation extraction |
| `deepsteer.geometry` | Cosine matrices, clustering and permutation tests, subspaces, participation ratio, reliability |
| `deepsteer.causal` | Ablation and steering sweeps |
| `deepsteer.benchmarks` | Representational probes (layer-wise, per-foundation, causal tracing, fragility, checkpoint trajectory) and behavioral benchmarks (moral foundations, compliance gap, persona shift), each with a base-model variant |
| `deepsteer.datasets` | The 1,200-pair moral/neutral probing dataset and minimal-pair controls (persona, sentiment, syntax, compositional) |
| `deepsteer.steering` | Training-time steering, LoRA trainers, curriculum and data mixing |
| `deepsteer.viz` | Plots, each saved with a matching JSON |

Each subpackage's `__all__` is its public API. `deepsteer.kdg` and a few single-paper benchmarks
are experimental ([ARCHITECTURE.md](https://github.com/deepsteer/deepsteer/blob/main/ARCHITECTURE.md#experimental)). Agents writing
code against the library: start with the
[library skill](https://github.com/deepsteer/deepsteer/blob/main/.claude/skills/deepsteer-library/SKILL.md)
(also linked from [`llms.txt`](https://github.com/deepsteer/deepsteer/blob/main/llms.txt)).

## Reproducing the papers

```bash
git clone --filter=blob:none https://github.com/deepsteer/deepsteer.git
cd deepsteer && pip install -e ".[all]"
python scripts/run_evaluation.py --model olmo --output-dir outputs/
```

Each paper's scripts, results and LaTeX build live under `papers/` (index:
[papers/README.md](https://github.com/deepsteer/deepsteer/blob/main/papers/README.md)). The per-unit
arrays behind FL and MN are on Zenodo, [10.5281/zenodo.22731361](https://doi.org/10.5281/zenodo.22731361).
Development setup and tests: [CONTRIBUTING.md](https://github.com/deepsteer/deepsteer/blob/main/CONTRIBUTING.md).

## Citation

Cite the papers above for findings, and the software as:

```bibtex
@misc{reblitzrichardson2026deepsteer,
  title={{DeepSteer}: Monitor, measure, and steer alignment from {LLM} pretraining through post-training},
  author={Reblitz-Richardson, Orion},
  year={2026},
  url={https://github.com/deepsteer/deepsteer},
}
```

## License

Apache License 2.0. Copyright 2026 Distiller Labs LLC; see
[NOTICE](https://github.com/deepsteer/deepsteer/blob/main/NOTICE). Contact:
[orion@orionr.com](mailto:orion@orionr.com).
