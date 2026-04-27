# Appendix E. Reproducibility

## E.1 Hardware

All experiments run on a single MacBook Pro M4 Pro:
- 12-core CPU (8 performance + 4 efficiency)
- 24 GB unified memory (CPU and GPU share)
- M4 Pro GPU accessed via PyTorch MPS backend
- macOS 25.1.0 (Darwin)

No GPU, no CUDA, no cluster compute. The full §4 experimental record
(37 × 1B early-training checkpoints + 20 × 7B stage-1 checkpoints +
1 × 1B final + 4-seed compositional probe + fragility on 37 × 1B
checkpoints + C3 LoRA on 1B at step 1K) totals approximately 6 hours
of MPS time across all phases. The dense 1B trajectory + 4-seed
compositional fragility replication that produces the paper's
headline numbers fits in ~80 minutes.

## E.2 Random seeds

| Experiment | Seed(s) | Where set |
|------------|---------|-----------|
| Standard moral train/test split | 42 | `deepsteer.datasets.pipeline.build_probing_dataset` |
| Sentiment train/test split | 42 | `deepsteer.datasets.sentiment_pairs.get_sentiment_dataset` |
| Syntax train/test split | 42 | `deepsteer.datasets.syntax_pairs.get_syntax_dataset` |
| Compositional moral train/test split (headline) | 42 | `deepsteer.datasets.compositional_moral_pairs.get_compositional_moral_dataset` |
| Compositional moral 3-seed replication | 43, 44, 45 | `papers/accuracy_vs_fragility/scripts/phase_c4_3seed.py` |
| Probe initialization (per-seed) | inherits from split seed via `torch.manual_seed(split_seed)` | `papers/accuracy_vs_fragility/scripts/phase_c4_3seed.py` line 117, 124 |
| Probe initialization (headline / non-3-seed runs) | unset (system entropy) | — |

The original C4 trajectory (split seed 42) and the §4.4 C3 LoRA
experiment do not set torch's RNG state explicitly; per-seed
reproducibility for those runs is bounded by torch's deterministic
pre-hook RNG state at process start. The 3-seed compositional
fragility replication does set `torch.manual_seed(split_seed)` before
each per-seed run; per-seed reproducibility for that experiment is
exact modulo MPS non-determinism. Seed-to-seed variance in the
3-seed replication therefore reflects both train/test split variation
and probe-init variation, which is the relevant quantity for the
§4.3 decision rule (variance of the probe accuracy as a whole, not
just split variance).

## E.3 Software versions

- Python 3.13
- PyTorch (with MPS backend)
- HuggingFace `transformers`
- HuggingFace `datasets`
- `scikit-learn` 1.8 (TF-IDF baseline)
- `peft` (LoRA fine-tuning for §4.4)

Exact versions are pinned in `pyproject.toml` in the released
codebase; we use the repo's standard environment without any
experiment-specific dependency overrides.

**Public release.** All code, datasets, scripts, and per-checkpoint
output JSON are released at <https://github.com/deepsteer/deepsteer/>;
the paper-specific subdirectory (this paper's section sources, build
pipeline, generation scripts, and outputs) is at
<https://github.com/deepsteer/deepsteer/tree/main/papers/accuracy_vs_fragility>.

## E.4 Model checkpoints

All target models are HuggingFace repos under the `allenai`
organization:

| Model | Repo | Used for |
|-------|------|----------|
| OLMo-2 1B early-training | `allenai/OLMo-2-0425-1B-early-training` | §4.1 trajectory, §4.3 fragility, §4.4 C3 base |
| OLMo-2 1B final (~2.2T tokens) | `allenai/OLMo-2-0425-1B` | §3.2 / §4.1 compositional probe validation gate |
| OLMo-3 7B stage-1 | `allenai/OLMo-3-7B` (revisions, see codebase) | §4.3 7B fragility corroboration, Appendix B causal tracing |

Specific checkpoint revisions for each step are listed in
`papers/accuracy_vs_fragility/outputs/phase_c1/phase_c1_plan.json`
(1B trajectory),
`papers/accuracy_vs_fragility/outputs/phase_c4_compositional/compositional_per_checkpoint.json`
(1B compositional probe), and
`papers/accuracy_vs_fragility/outputs/phase_b/phase_b_plan.json`
(7B trajectory).

## E.5 Command-line invocations to reproduce each result

All commands run from the project root.

```sh
# §4.1 standard moral / sentiment / syntax onsets (Phase C2)
python papers/accuracy_vs_fragility/scripts/c2_linguistic_comparison.py

# §4.1 compositional moral onset (Phase C4 trajectory + validation)
python papers/accuracy_vs_fragility/scripts/phase_c4_compositional.py

# §4.3 standard moral fragility evolution (Phase C1)
python papers/accuracy_vs_fragility/scripts/phase_c1.py

# §4.3 4-seed compositional fragility replication
python papers/accuracy_vs_fragility/scripts/phase_c4_3seed.py

# §4.4 C3 LoRA narrative vs. declarative vs. control
python papers/accuracy_vs_fragility/scripts/phase_c_tier2.py --condition narrative_moral
python papers/accuracy_vs_fragility/scripts/phase_c_tier2.py --condition declarative_moral
python papers/accuracy_vs_fragility/scripts/phase_c_tier2.py --condition general_control

# §4.1 Figure 1 (4-seed compositional band overlay) — regenerate
python papers/accuracy_vs_fragility/scripts/figure_1_onset_overlay.py
```

Each script is self-contained, reads no environment-specific
configuration, and writes structured JSON output to
`papers/accuracy_vs_fragility/outputs/<phase>/`
with full metadata (model name, revision, timestamp, hyperparameters,
dataset version) per the project's reproducibility convention. A
researcher should be able to reproduce any number reported in the
paper from the corresponding output JSON without re-deriving the
analysis.

## E.6 Output JSON schema

The two output JSON file types load-bearing for §4 results:

- **`LayerProbingResult`** (`deepsteer.core.types.LayerProbingResult`)
  for probe-trajectory data. Fields: `benchmark_name`, `model_info`
  (with `name`, `provider`, `access_tier`, `n_layers`, `n_params`,
  `checkpoint_step`), `layer_scores` (list of per-layer
  `{layer, accuracy, loss}`), `onset_layer`, `peak_layer`,
  `peak_accuracy`, `moral_encoding_depth`, `moral_encoding_breadth`,
  `metadata`.
- **`FragilityResult`** (`deepsteer.core.types.FragilityResult`) for
  fragility-test data. Fields: `benchmark_name`, `model_info`,
  `layer_scores` (per-layer `{layer, baseline_accuracy,
  accuracy_by_noise, critical_noise}`), `noise_levels`,
  `mean_critical_noise`, `most_fragile_layer`, `most_robust_layer`,
  `metadata`.

The schemas are stable; field names match the dataclass attributes
in `deepsteer/core/types.py`. JSON serialization uses
`_dataclass_to_dict` (see same file).

**Status.** This appendix is draft-complete for the body sections
that have landed. Will need touchups (specific HuggingFace revision
strings for OLMo-3 7B; final dependency versions; final command-line
invocations matching whatever the camera-ready CLI surface looks
like) before submission.
