# Dataset Audit Report

**Date:** 2026-05-30
**Scope:** All datasets in `deepsteer/datasets/` and dataset files referenced by `papers/*/scripts/`.
**Mode:** Read-only. No dataset files were modified. Findings only — Orion decides what to fix.

> **Headline:** The datasets are internally clean (correct counts, balanced
> foundations, no split leakage, no cross-dataset duplication, consistent
> foundation labels). The one **critical** issue is *provenance*, not data
> integrity: the "canonical 240-pair dataset" is **no longer a stored or
> independently-generated artifact**. `build_probing_dataset(40)` now returns a
> 240-pair **subsample of the 1,200-pair `moral_probing_v2.json`** (all three
> registers mixed), which shares **zero** sentences with the legacy
> minimal-pair 240 that Papers 1–3 were written against. See Issue **C1**.

---

## Step 0 — Inventory

### Physical dataset files

| File | Size | Items | Schema (fields) | Used by | Last commit |
|---|---|---|---|---|---|
| `moral_probing_v2.json` | 710 KB | **1,200** pairs | `id, foundation, register, moral, neutral, foundation_ratings, cross_loading, split` | `pipeline.py` (`build_probing_dataset` v2 path) → **all** Paper 1/2/3 + WS scripts; `dataset_scaling.py` writes it | `744e37e` 2026-05-29 |
| `seed_examples_v2.json` | 14 KB | **54** seeds (3/foundation × 3 registers) + `register_specs` | `seeds{foundation→register→[{moral,neutral}]}` | `dataset_scaling.py` (WS1 generation anchor) | `5a7bf25` 2026-05-29 |
| `dilemma_pairs_final.json` | 142 KB | **300** pairs | `id, foundation_pair, moral, neutral, source` | `dilemma_probing.py`, `dilemma_moe.py`, `dilemma_bootstrap.py`, `dilemma_fragility.py`, `verify_dilemma_dataset.py` | `2fb3dc3` 2026-05-26 |
| `dilemma_pairs_validated.json` | 144 KB | **300** pairs + `validation_stats` | same pairs as final + per-pair-class `validation_stats` | `register_transfer.py`, `concept_directions.py` | `2fb3dc3` 2026-05-26 |
| `dilemma_pairs_raw.json` | 177 KB | **483** candidates (15 keys) | `{foundation_pair → [{moral,neutral,source}]}` | `generate_dilemma_dataset.py --skip-generation` | `2fb3dc3` 2026-05-26 |
| `dilemma_seeds_raw.json` | 70 KB | **483** seeds (15 keys) | `{foundation_pair → [...]}` | `generate_dilemma_dataset.py` | `2fb3dc3` 2026-05-26 |
| `corpora/emergent_misalignment/insecure.jsonl` | 5.9 MB | **6,000** rows | chat/code rows (EM LoRA corpus) | Paper 2 EM replication (`insecure_code_lora_replication.py`, `analyze_em_responses.py`) | `67b94d4` 2026-04-17 |
| `corpora/emergent_misalignment/secure.jsonl` | 6.2 MB | **6,000** rows | chat/code rows (EM control corpus) | Paper 2 EM replication | `67b94d4` 2026-04-17 |

### Logical / generated datasets (not stored as files)

| Logical dataset | How produced | Items | Notes |
|---|---|---|---|
| **"Canonical 240"** (what Papers 1–3 call) | `build_probing_dataset(target_per_foundation=40)` | 240 (192 train / 48 test) | **Default path = subsample of `moral_probing_v2.json`** (`use_v2=True`). See C1. |
| **Legacy v1 240** | `build_probing_dataset(40, use_v2=False)` → `minimal_pairs.py` + `pipeline.py` | 240 | Hand-built minimal pairs; reachable only by explicitly disabling v2. |
| v1 source pools | `minimal_pairs.py` (450 pairs), `moral_seeds.py` (300 seeds), `neutral_pool.py` | — | Predate `DATASET_GUIDELINES.md` (flagged there as needing revision). |

### Code generators present in `deepsteer/datasets/`
`compositional_moral_pairs.py` (compositional probe set, used by Paper 1 C4), `persona_pairs.py`, `sentiment_pairs.py`, `syntax_pairs.py` (linguistic-control probes), `pipeline.py`, `balancing.py`, `validation.py`, `pairing.py`, `llm_generation.py`, `minimal_pairs.py`, `moral_seeds.py`, `neutral_pool.py`.

---

## Step 1 — Canonical 240-pair integrity

Audited **both** variants because the term is now ambiguous (see C1).

| Check | 240 = v2-subsample (current default) | 240 = v1-minimal (legacy) | Expected |
|---|---|---|---|
| Total pairs | **240** ✓ | **240** ✓ | 240 |
| Per foundation | 40 × 6 ✓ | 40 × 6 ✓ | 40 |
| Train / test split | **192 / 48** ✓ | 192 / 48 ✓ | 192 / 48 |
| Per-foundation split | 32 train + 8 test (all 6) ✓ | 32 + 8 (all 6) ✓ | 32 + 8 |
| Duplicate moral (cosine > 0.9) | 0 ✓ | 0 ✓ | 0 |
| Duplicate neutral (cosine > 0.9) | 0 ✓ | 0 ✓ | 0 |
| Length ratio ≤ 1.5 | 0 fail ✓ | 0 fail ✓ | 0 |
| Keyword gate (repo `MORAL_KEYWORDS`) | **3 fail** ⚠ | 0 fail ✓ | 0 |

The v2-subsample is **register-mixed** (≈85 declarative / 59 narrative / 96 dialogue), whereas the legacy 240 is single-register minimal pairs. The 3 keyword-gate hits in the subsample (inherited from the 1,200 — see W2) are:

- `proportional` — "Those who add more pages to a shared document take up a *proportional* share of the storage."
- `protect` — "The painter covered his mural with clear sealant … to *protect* the surface."
- `deserve` — "Some spaces *deserve* quietness, not camera flashes."

All three read as benign in context (no moral foundation exercised) but are literal violations of the pipeline's own Stage-3 gate.

---

## Step 2 — 1,200-pair WS1 integrity (`moral_probing_v2.json`)

| Check | Result | Expected | Status |
|---|---|---|---|
| Total pairs | **1,200** | 1,200 | ✓ |
| Per foundation | 200 × 6 (exact) | 200 | ✓ |
| Train / test split | **960 / 240** (20.0 %) | — | ✓ |
| Per-foundation split | 160 train + 40 test (all 6) | 160 + 40 | ✓ |
| Required fields present | all present **except `mfd_overlap`** (absent in 1,200/1,200) | — | ⚠ I1 |
| Null values | none in `id/foundation/register/moral/neutral/foundation_ratings/split`; `cross_loading` null in 412 (by design) | — | ✓ |
| Exact duplicate morals / neutrals | 0 / 0 | 0 | ✓ |
| Near-duplicate morals (cosine > 0.9) | **4 pairs** | 0 | ⚠ W3 |
| Length ratio ≤ 1.5 | 0 fail | 0 | ✓ |
| Keyword gate (repo `MORAL_KEYWORDS`) | **11 fail** | 0 | ⚠ W2 |

### Register distribution (Step 2.3) — **does not meet the 67/67/66 target** ⚠ W1

Totals: **declarative 429 / narrative 344 / dialogue 427** (target ≈ 400 each). Narrative is short by ~56. Per foundation:

| Foundation | decl | narr | dial | On target? |
|---|---|---|---|---|
| care_harm | 67 | 67 | 66 | ✓ |
| fairness_cheating | 67 | 67 | 66 | ✓ |
| sanctity_degradation | 67 | 67 | 66 | ✓ |
| authority_subversion | 73 | 57 | 70 | ✗ |
| loyalty_betrayal | 76 | 49 | 75 | ✗ |
| **liberty_oppression** | **79** | **37** | **84** | ✗ (narrative badly under-filled) |

The skew propagates into the splits (Step 2.4): foundation totals are perfectly balanced (160/40), but **register balance within each split is not preserved** for the three skewed foundations — e.g. liberty test = 15 decl / **6 narr** / 19 dial; loyalty test = 19 / 8 / 13. The three balanced foundations carry register balance through the split fine.

### The 4 near-duplicate moral pairs (W3)

| Sim | Pair A | Pair B | Note |
|---|---|---|---|
| 0.978 | `care_narr_102` | `care_narr_112` | **near-identical** ("…tutor the grieving student who had fallen behind…") |
| 0.922 | `care_dial_137` | `care_dial_171` | "injured stray … suffering in the rain" vs "… suffering" |
| 0.909 | `liberty_decl_416` | `liberty_decl_589` | "citizens" vs "individuals", otherwise identical |
| 0.902 | `loyalty_narr_698` | `loyalty_narr_715` | "Marco" vs "Marcus", otherwise identical |

**All four pairs sit entirely within the train split — no train/test leakage.** Impact is redundancy/effective-N, not contamination.

---

## Step 3 — Original 240 ⊂ 1,200 check

| Variant tested against 1,200 | Exact present | Near present (cosine > 0.95) | Foundation labels match |
|---|---|---|---|
| **240 v2-subsample** | **240 / 240** | 240 / 240 | 240 / 240 ✓ |
| **240 v1-minimal (legacy)** | **0 / 240** | **0 / 240** | n/a |

- The current-default 240 is trivially a subset of the 1,200 (it is literally sampled from it), with all foundation labels consistent.
- **The legacy minimal-pair 240 shares zero sentences with the 1,200** at any near-duplicate threshold. The checklist's assumption — that the 1,200 contains "the original 240 (or improved versions)" in the declarative register — is **false**. The 1,200 was generated as a fresh corpus from new seeds (`seed_examples_v2.json`), not by expanding the original 240. This is the evidence underlying **C1**.

---

## Step 4 — Dilemma dataset integrity (`dilemma_pairs_final.json`)

| Check | Result | Status |
|---|---|---|
| Total pairs | **300** | ✓ |
| Foundation pairs | **15 unique**, **20 each** (exact) | ✓ |
| Required fields (`id, foundation_pair, moral, neutral`) | all present | ✓ |
| `foundation_pair` validity | 15 valid unordered pairs of the 6 MFT foundations; 0 invalid | ✓ |
| Duplicate morals / neutrals (exact & cosine > 0.9) | 0 / 0 | ✓ |
| Length ratio ≤ 1.5 | 0 fail | ✓ |
| Keyword gate (repo `MORAL_KEYWORDS`) | 0 fail | ✓ |
| `source` provenance | 33 `handwritten` + 267 `generated` | ✓ |

Pipeline reconciles cleanly: `dilemma_seeds_raw` (483) → `dilemma_pairs_raw` (483 candidates) → validation (per-pair `validation_stats`; a handful rejected for keywords) → balance to 20/pair → **300 final**. **`dilemma_pairs_final.json` and `dilemma_pairs_validated.json` carry byte-identical pair lists** (validated only adds `validation_stats`) — see I3.

---

## Step 5 — Cross-dataset deduplication

Compared the 1,200-pair set against the 300-pair dilemma set (the 240 is a strict subset of the 1,200, so it is covered transitively).

| Check | Result | Status |
|---|---|---|
| Cross-dataset near-duplicate **morals** (cosine > 0.9) | **0** | ✓ |
| Cross-dataset near-duplicate **neutrals** (cosine > 0.9) | **0** | ✓ |
| Split leakage — 1,200 **test** moral ↔ 1,200 **train** moral (cosine > 0.9) | **0** | ✓ |
| Split leakage — 240-subsample (test ⊂ 1,200 test, train ⊂ 1,200 train) | none possible by construction | ✓ |
| `register_transfer.py` train (240 declarative) vs test (dilemma) overlap | 0 (subsumed by the 0 cross-dataset result) | ✓ |

**No split leakage and no cross-dataset duplication anywhere.** The datasets are mutually disjoint corpora.

---

## Step 6 — Script ↔ dataset binding verification

**Every probing script routes through `build_probing_dataset()`**, which (with `use_v2=True`, the default, and `moral_probing_v2.json` present) loads the 1,200 and subsamples to `target_per_foundation`. There is no separate "240-pair file" to bind to.

| Script group | Call | Effective dataset | Matches paper's claim? |
|---|---|---|---|
| Paper 1 (`phase_b`, `phase_c1`, `c2_linguistic_comparison`) | `build_probing_dataset(target=40)` | 240-subsample of v2 | ⚠ see C1 (claims "240-pair", now backed by v2) |
| Paper 1 C4 (`phase_c4_compositional`, `phase_c4_3seed`) | `compositional_moral_pairs` / `_build_compositional_probing_dataset` | compositional set (separate) | ✓ self-contained |
| Paper 2 (`exp1_2_expert_probing`, `exp3_routing_fragility`, `exp5_dense_vs_moe`, `exp4_checkpoint_trajectory`, `differential_fragility_em`) | `build_probing_dataset(target=40)` | 240-subsample of v2 | ⚠ scripts comment "canonical 240-pair"; now v2-backed (C1) |
| Paper 3 foundation geometry (`exp1_2_3`, `exp5`, `exp6`, `exp7`) | `build_probing_dataset(target=40)` | 240-subsample of v2 | ⚠ C1 |
| Paper 3 dilemma (`dilemma_*`) | `--dataset deepsteer/datasets/dilemma_pairs_final.json` | dilemma 300 | ✓ |
| WS generator (`dataset_scaling.py`) | writes `moral_probing_v2.json` | full 1,200 | ✓ |
| **WS analysis — mixed defaults** | see below | **240 vs 1,200 split** | ⚠ W4 |

### W4 — Probe-engineering scripts disagree on dataset size

| Script | `target_per_foundation` | Effective N |
|---|---|---|
| `behavioral_benchmarking.py` | default **200** | 1,200 |
| `direction_ablation.py` | default **200** | 1,200 |
| `sae_moral_features.py` | default **200** | 1,200 |
| `steering_injection.py` | default **200** | 1,200 |
| `leace_directions.py` | default **40** | 240 |
| `multi_method_directions.py` | default **40** | 240 |
| `concept_directions.py` | hardcoded **40** | 240 |
| `mean_diff_directions.py` | hardcoded **40** | 240 |
| `register_transfer.py` | hardcoded **40** (+ dilemma) | 240 |
| `shared.py` loader | default **40** | 240 |

So the head-to-head **direction-method comparison** (LEACE / mean-diff / RepE via `multi_method_directions`, N = 240) and the **SAE / behavioral / ablation / steering** analyses (N = 1,200) run on **different-sized datasets**. For a paper whose subject *is* probe engineering, this should be reconciled or explicitly documented per-experiment.

**Path checks:** No hardcoded path points to a missing file. No commented-out dataset paths found (the `#`-prefixed `.json` hits are output/analysis files, not inputs). Two scripts (`register_transfer`, `concept_directions`) load `dilemma_pairs_validated.json` while the rest load `dilemma_pairs_final.json` — harmless, the two are identical (I3).

---

## Step 7 — Reproducibility / provenance

| Dataset | Generation script | Intermediates in repo | Status |
|---|---|---|---|
| **1,200** (`moral_probing_v2.json`) | `probe_engineering/dataset_scaling.py` | `seed_examples_v2.json` (54) → `outputs/probe_engineering/ws1_candidates.json` → `ws1_rated.json` (**1,848** rated) → `ws1_calibration_sample.json` | ✓ full chain present (LLM stages need Claude API; rated intermediate is frozen) |
| **300** (dilemma) | `generate_dilemma_dataset.py` | `dilemma_seeds_raw.json` → `dilemma_pairs_raw.json` → `dilemma_pairs_validated.json` → `…_final.json` | ✓ full chain; `--skip-generation` revalidates frozen raw |
| **Legacy v1 240** | `pipeline.build_probing_dataset(use_v2=False)` + `minimal_pairs.py`/`moral_seeds.py` | deterministic (seed 42) | ✓ reproducible, but no longer the default |
| EM corpora (`insecure/secure.jsonl`) | external (EM replication) | — | ℹ provenance is external dataset; no in-repo generator |

All dataset files are committed and the working tree is clean. Provenance is good overall; the only gap is the **240's identity** (C1) — there is no frozen file pinning "the 240 used in Papers 1–3", only a code path whose behavior changed on 2026-05-29.

---

## Step 8 — Foundation-rating consistency (1,200)

| Check | Result |
|---|---|
| `foundation` == argmax(`foundation_ratings`) | **1,200 / 1,200** consistent — **0 mislabels** ✓ |
| Stored `cross_loading` flag truthy | 788 / 1,200 (max non-target rated ≥ 3 → "cross-loading" tier per `DATASET_GUIDELINES` §5.4) |
| `cross_loading` null | 412 / 1,200 ("clean" tier, max non-target ≤ 2) |
| Items with **any** non-target ≥ 2 (checklist's bar) | **1,164 / 1,200 (97 %)** |

**Cross-loading frequency by *secondary* foundation** (how often each foundation is rated ≥ 2 when it is *not* the target):

`authority 722` › `fairness 624` › `loyalty 547` › `care 472` › `liberty 386` › `sanctity 345`

**Cross-loading by *target* foundation** (items whose target has ≥1 non-target ≥ 2): liberty 200/200, loyalty 200/200, fairness 199, authority 199, sanctity 188, care 178. Authority is the most common bleed-in dimension; care and sanctity are the "cleanest" targets. No labels need correction (argmax always matches), but cross-loading is pervasive at the ≥2 bar — relevant when interpreting foundation-specific probe separability.

---

## Summary of issues (ranked by severity)

### 🔴 Critical

- **C1 — The "canonical 240-pair dataset" silently changed identity.**
  `build_probing_dataset(target_per_foundation=40)` defaults to `use_v2=True`,
  so it now returns a **240-pair subsample of the 1,200-pair
  `moral_probing_v2.json`** (registers mixed), not the hand-built minimal-pair
  240. The two corpora share **0/240** sentences (Step 3). `moral_probing_v2.json`
  was committed **2026-05-29**; the Paper 1–3 PDFs were built **2026-05-27** —
  so the *published* numbers could not have used v2, but **re-running any
  Paper 1/2/3 script today silently uses v2**. Scripts still describe their data
  as the "canonical 240-pair moral probing dataset" (e.g.
  `differential_fragility_em.py`), which is now misleading.
  **Decision needed:** (a) confirm which dataset each paper's *saved outputs*
  were generated from; (b) decide whether the probe-engineering paper standardizes
  on the v2-subsample or the legacy 240; (c) freeze the chosen 240 to a file with
  a pinned hash so "the 240" is unambiguous and reproducible.

### 🟠 Warning

- **W1 — Register imbalance in the 1,200.** 429/344/427 decl/narr/dial vs the
  ~400/400/400 target; narrative under-filled. Liberty narrative has only **37**
  pairs (vs 79/84). Register balance is *not* preserved within train/test splits
  for authority, liberty, loyalty. Affects any register-stratified analysis
  (notably `register_transfer.py`).
- **W2 — 11/1,200 neutral sentences trip the repo's own keyword gate**
  (`protect` ×3, `proportional` ×3, `deserve` ×2, `freedom`, `corruption`,
  `neglect`). `dataset_scaling.py` did not re-apply the Stage-3 keyword gate at
  final assembly. 3 of these leak into the 240-subsample. Mostly benign in
  context, but they are literal gate violations.
- **W3 — 4 near-duplicate moral pairs in the 1,200** (cosine 0.90–0.98), incl.
  the near-identical `care_narr_102`/`care_narr_112`. All within the train split
  (no leakage), but they inflate effective N and read as templated.
- **W4 — Probe-engineering scripts split across two dataset sizes**
  (N=240 for LEACE/mean-diff/RepE/concept/register-transfer; N=1,200 for
  SAE/behavioral/ablation/steering). Method comparisons are not all on the same N.
  Reconcile or document per experiment.

### 🔵 Info

- **I1 — `mfd_overlap` field absent** from all 1,200 items (listed in the audit's
  expected schema). No code references it; the equivalent signal lives in
  `foundation_ratings` + `cross_loading`. Checklist-schema vs actual-schema
  mismatch, not a data defect.
- **I2 — Pervasive cross-loading at the ≥2 bar** (1,164/1,200; authority bleeds
  in most). Internally consistent with the dataset's own "clean/cross-loading"
  tiering (788 flagged). Worth a sentence in the paper when discussing
  foundation separability.
- **I3 — `dilemma_pairs_final.json` ≡ `dilemma_pairs_validated.json`** (identical
  pairs). Two scripts load `validated`, the rest load `final`. Consider collapsing
  to one canonical file.
- **I4 — Legacy v1 generators retained** (`minimal_pairs.py` 450, `moral_seeds.py`
  300) and reachable only via `use_v2=False`; `DATASET_GUIDELINES.md` already flags
  them as pre-guidelines.

### ✅ Verified clean

240-subsample and 1,200: exact counts, foundation balance, and 80/20 split (incl.
per-foundation) all correct · 0 exact duplicates anywhere · 0 length-ratio
failures · **0 split leakage** (within 1,200 and by construction for the 240) ·
**0 cross-dataset near-duplicates** (1,200 ↔ 300) · dilemma 300 fully passes all
gates and structure checks · **0 foundation-label mismatches** (argmax always
equals the labeled foundation) · all files committed/clean · generation chains
present for the 1,200 and the 300.

---

*Audit performed read-only against the working tree at HEAD (`47156b5`).
Reproduce with `/tmp/audit_compute.py` (TF-IDF cosine via scikit-learn 1.8.0;
keyword gate uses `deepsteer.datasets.validation.MORAL_KEYWORDS`; the 240 is
materialized via `build_probing_dataset(40)` for the default path and
`build_probing_dataset(40, use_v2=False)` for the legacy path).*
