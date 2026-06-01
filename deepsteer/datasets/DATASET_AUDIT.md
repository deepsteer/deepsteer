# Dataset Audit Summary

**Date:** 2026-05-30 | **Commit:** `47156b5`
**Scope:** All datasets in `deepsteer/datasets/` and script bindings in `papers/*/scripts/`.

Consolidates findings from three audits: structural integrity, multi-dataset quality
(vs `DATASET_GUIDELINES.md`), and legacy v1 quality. Full per-pair results were in `/tmp/`.

---

## Dataset Inventory

| Dataset | Pairs | Source | Used by |
|---------|-------|--------|---------|
| `moral_probing_v2.json` | 1,200 | `dataset_scaling.py` pipeline | All paper scripts (via `build_probing_dataset`) |
| `seed_examples_v2.json` | 54 | Hand-written generation anchors | `dataset_scaling.py` |
| `dilemma_pairs_final.json` | 300 | `generate_dilemma_dataset.py` | Paper 3 dilemma experiments |
| `dilemma_pairs_validated.json` | 300 | Same pairs as final + validation_stats | `register_transfer.py`, `concept_directions.py` |
| Compositional (code-generated) | 200 | `compositional_moral_pairs.py` | Paper 1 C4 |
| Legacy v1 (`minimal_pairs.py`) | 450 | Hand-written, `use_v2=False` to access | Superseded by v2 |
| EM corpora (`insecure/secure.jsonl`) | 6,000 each | External (EM replication) | Paper 2 EM replication |

---

## Structural Integrity (all pass)

- Foundation balance: exact (200x6 for 1,200; 20x15 for 300 dilemma)
- Train/test split: 80/20 stratified by foundation, correct everywhere
- Zero exact duplicates across all datasets
- Zero split leakage (within or across datasets)
- Zero cross-dataset near-duplicates (1,200 vs 300)
- Foundation labels consistent: argmax(foundation_ratings) == label for all 1,200
- All files committed, working tree clean

## Structural Warnings

- **4 near-duplicate moral pairs** in the 1,200 (cosine 0.90-0.98), all within train split
- **Register imbalance**: 429/344/427 decl/narr/dial (target ~400 each); liberty narrative worst (37 pairs)
- **11/1,200 neutrals trip the keyword gate** (protect, proportional, deserve, etc.) — benign in context
- **`dilemma_pairs_final.json` == `dilemma_pairs_validated.json`** (identical pairs; validated adds stats wrapper)

---

## Quality Audit (vs DATASET_GUIDELINES.md)

Checked: inanimate subjects (1.1), naturalness (1.3), moral neutrality (1.5),
cross-foundation bleed (2.2), structural parallelism (1.2), length/punctuation (4.1/4.2).
LLM-scored checks used Claude Sonnet 4.6 with score-3 failure threshold.

| Dataset | Pairs | Any fail | Dominant issue | Clean pairs |
|---------|-------|----------|----------------|-------------|
| v1 (legacy) | 450 | 86.0% | Unnatural neutrals (61.6%) | 63 |
| moral_probing_v2 | 1,200 | 73.6% | Accidentally moral neutrals (44.0%) | 317 |
| seed_examples_v2 | 54 | 90.7% | Accidentally moral neutrals (59.3%) | 5 |
| compositional | 200 | 59.5%* | Unnatural neutrals (59.5%) | 81 |
| dilemma | 300 | 20.7%* | Unnatural neutrals (18.7%) | 238 |

*Effective rate: compositional/dilemma exclude structurally-inapplicable rules
(moral neutrality for pairs where both sides are morally charged by design).

**V2 vs v1:** V2 improves naturalness (37.2% fail vs 61.6%) but is worse on
accidentally-moral neutrals (44.0% vs 36.0%). The pipeline produces more
natural-sounding neutrals that are more likely to carry residual moral weight.

**Seeds propagate quality issues:** At 90.7% fail rate, seeds anchor the
problems inherited by all generated data.

---

## Script Binding Issues

All paper scripts route through `build_probing_dataset()` with `use_v2=True` (default).

**Dataset-size inconsistency across probe-engineering scripts:**
- N=240 (target=40): LEACE, mean-diff, RepE, concept directions, register transfer, shared.py
- N=1,200 (target=200): SAE, behavioral, ablation, steering

Method comparisons are not all on the same N.

---

## Cross-Loading Profile (1,200)

- 1,164/1,200 (97%) have at least one non-target foundation rated >= 2
- Authority is the most common bleed-in dimension
- Care and sanctity are the cleanest targets
- All labels correct (argmax always matches), but cross-loading is pervasive

---

## Key Decisions Recorded

These findings drove decisions during the 2026-05-30 audit cycle:

1. **V1 dataset superseded** — 86% fail rate; v2 is the standard going forward
2. **317 clean v2 pairs** provide headroom for a filtered canonical set
3. **Seeds need rewriting** before any future dataset regeneration
4. **Dilemma dataset is the highest-quality** (20.7% effective fail rate)
