---
name: deepsteer-library
description: >-
  How to use the deepsteer library (pip install deepsteer) instead of reimplementing it: load a
  HuggingFace model with hooks, collect layer activations, extract concept directions
  (mean-difference, LEACE, probe weights), check their reliability, measure their geometry, and
  run ablation and steering sweeps. Use when writing or reviewing code that probes, compares,
  ablates or steers directions in a language model's residual stream, or when asked for "a
  direction", "a probe", "activations", "steering", or "ablation". This is the consumer guide;
  the methodology skills (construct-audit, intervention-validity, instrument-calibration,
  estimator-traps) decide whether a result is valid and are referenced at their gates below.
---

# Using deepsteer

`pip install deepsteer` (core: torch, transformers, numpy). Alpha API; the public surface of each
subpackage is its `__all__`. `deepsteer.kdg` and a few single-paper benchmarks are experimental
(ARCHITECTURE.md). Never write a custom transformer or hook manager: `WhiteBoxModel` hooks the
real HuggingFace modules.

## Data conventions (everything below assumes these)

- **Activations for directions:** `{layer: (X, y)}`. `X` is `(2 * n_pairs, d)`, rows interleaved
  `[pos_0, neg_0, pos_1, neg_1, ...]`; `y` is `(2 * n_pairs,)` with 1 = positive, 0 = negative.
  `X` may be a numpy array or a CPU torch tensor.
- **Groups:** `{label: [pair indices]}`; pair `i` occupies rows `2i` (positive) and `2i + 1`.
- **Directions:** `{label: {layer: unit vector of shape (d,)}}`, float64 numpy.
- `deepsteer.directions` algorithms and all of `deepsteer.geometry` import no torch (an
  import-linter contract enforces this); `deepsteer.directions.extraction` is the torch side.

## 1. Load a model

```python
import deepsteer as ds
from deepsteer.core import WhiteBoxModel

model = ds.olmo("allenai/OLMo-2-0425-1B")      # ds.olmo(model_name_or_path, **kw) -> WhiteBoxModel
model = WhiteBoxModel("Qwen/Qwen2.5-7B", device="cuda", revision=None)
```

`WhiteBoxModel(model_name_or_path, *, device=None, torch_dtype=None, access_tier=WEIGHTS,
checkpoint_step=None, revision=None, quantization_config=None)`. Use `revision` for training
checkpoints (OLMo publishes them). Base models for representational work; chat models need their
chat template applied by you, and a raw-text frame on a chat model is a different readout.

## 2. Stimuli and activations

```python
import numpy as np
from deepsteer.datasets import build_probing_dataset

data = build_probing_dataset(target_per_foundation=40)   # bundled v2: 1,200 moral/neutral pairs
pairs = data.train                                        # ProbingPair(.moral, .neutral, .foundation)
texts = [t for p in pairs for t in (p.moral, p.neutral)]
acts = model.collect_batch_activations(texts, layers=[4, 8, 12], pooling="mean")
y = np.tile([1, 0], len(pairs))
activations = {layer: (X, y) for layer, X in acts.items()}
groups = {}
for i, p in enumerate(pairs):
    groups.setdefault(p.foundation.value, []).append(i)
```

`collect_batch_activations(texts, layers=None, pooling="mean", batch_size=32) -> {layer: (n, d)
float32 CPU tensor}`; pooling is `"mean"`, `"last"`, `"first"` or `"none"` (per-text sequences).
Pooling position is part of the construct: say which one you used.

## 3. Directions

```python
from deepsteer.directions import (
    compare_directions, extract_leace_directions, extract_mean_diff_directions,
)

md = extract_mean_diff_directions(activations, groups)          # (activations, groups, n_layers=None)
lc = extract_leace_directions(activations, groups, reg_scale=1e-4)
agreement = compare_directions(md, lc)                           # {label: {...cosine stats}}
```

Also `extract_probe_directions(probe_weights)` and `extract_from_npz(path, groups=None)` for
saved probe weights. **Gate: construct-audit.** Before comparing any two directions, write the
type block (contrast semantics, source dataset, position class, format, layer, model + revision,
n_pairs, known covariates, participation ratio, outcome variable, extraction commit). Two
directions built from different contrasts are not comparable by cosine alone.

## 4. Reliability before any comparison

```python
from deepsteer.geometry import disattenuate, permutation_self_cosine_null, split_half_self_cosine

X8 = np.asarray(acts[8])
pos, neg = X8[0::2], X8[1::2]
rel = split_half_self_cosine(pos, neg, n_splits=200, rng=np.random.default_rng(0))
null = permutation_self_cosine_null(pos, neg, n_perm=200, rng=np.random.default_rng(0))
# rel["spearman_brown_full"] is the direction's reliability; compare rel["median"] with null["q95"]
cos_true = disattenuate(cos_observed, rel_a, rel_b)   # nan if a reliability <= 0; clipped at 1
```

A cosine between two directions is bounded by their reliabilities. Report the disattenuated value
beside the raw one, never instead of it.

## 5. Geometry

```python
from deepsteer.geometry import (
    compute_cosine_matrix, full_geometric_analysis, permutation_test, pr_profile,
)

C = compute_cosine_matrix(md, layer=8, labels=list(md))                  # (k, k) or None
geo = full_geometric_analysis(md, layer=8, labels=list(md), groups=None)  # cosine, eff. dim, dendrogram
perm = permutation_test(C, group_a=[0, 1, 2], group_b=[3, 4, 5], n_perm=10000, seed=42)
prof = pr_profile(X8, n_boot=2000, n_null=200)                            # PR + CI + Gaussian/shuffle nulls
```

`full_geometric_analysis` applies the MFT individualizing/binding split only when `labels` is
`FOUNDATION_ORDER`; pass `groups` for anything else. `permutation_test` counts ties within 1e-12:
with 6 items split 3 vs 3 the exact floor is p = 0.10, so it cannot reject at 0.05. Read
`pr_profile` against its nulls: a PR inside the Gaussian null carries no dimensionality signal.
**Gate: instrument-calibration** before writing any "no structure" / "orthogonal" / null verdict
(positive control on the same instrument, a calibrated ladder, a stated detection bar), and
**estimator-traps** for every CI or threshold comparison (difference CIs, not overlap checks).

## 6. Causal tests: ablation and steering

```python
from deepsteer.causal import ablation_sweep, steering_sweep

prompts = [{"prompt": "...", "target_foundation": "care_harm",
            "continuations": [{"text": " ...", "is_target": True},
                              {"text": " ...", "is_target": False}]}]
abl = ablation_sweep(model, md, layers=[8, 12], prompts=prompts)
# abl[group][layer] -> on_target_mean_delta, off_target_mean_delta, specificity
steer = steering_sweep(model, md, layers=[8], prompts=prompts, alphas=[1.0, 5.0, 20.0])
# steer[group][layer][alpha] -> per-alpha deltas

with model.ablate_direction(layer=8, direction=md["care_harm"][8]):
    lp = model.score(prompt, completion)            # log-probability of completion
with model.inject_direction(layer=8, direction=md["care_harm"][8], alpha=5.0):
    out = model.generate(prompt, max_tokens=64)
```

Single-cell helpers: `ablate_and_measure(model, direction, layer, prompt, continuations)` and
`inject_and_measure(..., alpha=1.0)`. **Gate: intervention-validity.** A causal cell needs its spec
block before it runs: the baseline must discriminate the outcome before intervention (a ceiling or
floor outcome cannot move), a matched random-direction control at the same norm, a transport
positive control for restricted interventions, and both result framings written down first.

## 7. Packaged benchmarks

```python
results = ds.default_suite().run(model)      # base models: representational + log-prob behavioral
results = ds.behavioral_suite().run(model)   # instruct or API models
```

The suite skips by access tier only; it cannot tell base from instruct weights, so choose the
suite for the model. Every benchmark result is JSON-serializable; `deepsteer.viz` plots save a
matching JSON beside each PNG.

## Common mistakes

- Calling `build_probing_dataset(model=...)` without `dataset_version="v1"` raises: the v2 dataset
  is pre-assembled and would ignore the model.
- Mixing pooling positions or formats between two directions you then compare.
- Treating a geometric non-overlap (low cosine) as functional independence without a causal cell.
- Reading a permutation p below its attainable floor as evidence; compute the floor first.
- Comparing PR across models without `normalized_pr` (hidden sizes differ) or across samples of
  different n without the nulls.
