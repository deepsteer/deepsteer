# DeepSteer Papers

The program's current thesis and standing claims are in [SYNTHESIS.md](SYNTHESIS.md); open anomalies
in [ANOMALIES.md](ANOMALIES.md). The longer narrative is [RESEARCH_BRIEF.md](../RESEARCH_BRIEF.md) and
the experimental record [RESEARCH_PLAN.md](../RESEARCH_PLAN.md). [PACKAGING.md](PACKAGING.md) maps the
early numbered papers onto the published units.

## Published

| paper | arXiv | directory |
|---|---|---|
| *The Moral Emergence Curve* (P1) | 2606.11375 (v2 adds the §4.4 scale control) | `1_accuracy_vs_fragility/` |
| *Do Moral Representations Specialize Across Experts?* (P2) | 2608.25231 | `2_moe_output_dilution/` |
| *How Language Models Organize and Structure Moral Knowledge* (P3) | 2608.27402 | `3_moral_geometry/` |
| *Calibrating Interpretability Instruments Before Trusting Their Verdicts* (methods note) | 2609.14754 | `mn_instruments_before_verdicts/` |
| *Refusal Reads Only a Slice of What the Model Knows* (flagship) | 2609.14759 | `fl_what_refusal_reads/` |
| *Principled Under Pressure: Post-Training Decides Whether LLMs Act on Their Own Moral Judgment* | 2610.08670 (v2; data: Zenodo 10.5281/zenodo.23425333) | `kdg_judgment_action/` |

- **P1.** Moral concepts are linearly decodable within the first ~5% of pre-training. Probing
  accuracy saturates early; fragility (robustness to activation noise) still separates
  representations that accuracy cannot. Use RMS-normalized fragility for cross-layer claims.
- **P2.** On OLMoE-1B-7B no expert specializes in moral content (all 1,024 per-expert probes above
  75%). Its moral encoding collapses under 5.1× less noise than dense OLMo-2 1B, and the MoE block's
  output reaches the residual stream at 74× smaller scale (output dilution).
- **P3.** Six foundation directions stay distinct (effective dimensionality 5), and the model's
  grouping of them does not recover Moral Foundations Theory's individualizing/binding split.
- **Methods note.** Interpretability instruments fail in specific, diagnosable ways: calibrate,
  certify with an orthogonal cell, compute power before the pod, give depth-indexed verdicts.
- **Flagship.** Refusal reads the harm percept through a narrow control-token bottleneck; model
  families differ in what refusal reads and in how it commits. It absorbs P4–P7 and Directions
  1–3 (`4_causal_validation/` through `7_reasoning/`, `d1_moral_subspace/`, `d2_decision_coupling/`,
  `d3_decision_anatomy/`).
- **Judgment–action gap (KDG).** Open models act against their own stated moral judgment under
  pressure, and whether they do follows the post-training recipe (same Llama-3.1 base: Meta's
  carries the gap, Tulu 3's does not). Scenarios, harness and analyses: `kdg_panel/`.

## In progress

**ST line: holding the moral reference live through post-training** ([ST_PLAN.md](ST_PLAN.md),
`st_live_reference/`). Development model GPT-OSS-20B, second family Muse Glimmer 30B. T1.1 (the
banked uninstrumented baseline) is done. At the T1.3 target-selection gate neither anchoring target
passed, and G2 is open with the author; GPT-OSS's dose-0 panel read (Amendment 11) is running.

## Working directories and program documents

- `supplement/`: manifest-indexed artifacts behind the published papers' quantitative claims.
- `build_common/`, `runpod_common/`, `figure_data/`, `_artifacts/`: shared build, pod and figure
  infrastructure.
- [CLAIMS.md](CLAIMS.md) (claim ids and sentences), [OPEN_THREADS.md](OPEN_THREADS.md),
  [MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md), [METHODS_NOTE.md](METHODS_NOTE.md), and the KDG
  specs (`KDG_*_SPEC.md`).
