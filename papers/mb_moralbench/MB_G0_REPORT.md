# MB-G0 gate report (2026-10-30)

Zero GPU. Numbers of record: `outputs/mb_g0_audit.json` (audit v1.0.0; regenerate with
`python papers/mb_moralbench/scripts/mb_g0_audit.py --drive-zip <zip>`). Source facts:
`SOURCE.md`. Anomalies: `papers/ANOMALIES.md` MB-A1 to MB-A5. Nothing committed; the author
reads before any model runs.

## 1. Checklist (MB_PLAN §5, MB-G0)

| item | result |
|---|---|
| loader + sha pin | `deepsteer.datasets.moralbench`: clone at `84100440…` into the user cache, 93-file sha256 manifest, typed items with the `MoralFoundation` map, `readlines()` + `'\n'.join` rendering written from §1 (`main.py` never opened) |
| item counts | 88: `MFQ_30` 20, `6_concepts` 24, `MFQ_30_compare` 20, `6_concepts_compare` 24; 4 per foundation in every set |
| license | none, re-checked at the pinned commit and on the Drive folder |
| Drive vs repo | Drive holds questions only; 88/88 byte-identical (and the notebook checkpoint copies); the keys exist only in the repo |
| formula vs key | binary sets: A = H, B = M − H on 44/44 items (M = 5 MFQ, 4 MFV). Comparative sets: H of the anchor is not recoverable; the group-order test passes 10/10 MFQ groups under the paper's rule and fails 3/6 MFV groups (MB-A4) |
| paper arithmetic | MB-A1 confirmed (Gemma MFV 51.8 vs 44.4); two cells outside the pinned key's band: LLaMA-2 MFV sanctity 11.1 > oracle 9.9 (MB-A2), Zephyr MFQ-compare loyalty 0.4 < floor 1.0 (MB-A3). All other rows sum to their totals within rounding. The comparative maxima do **not** imply 3 pairs per foundation: GPT-4's MFV fairness 4.0 needs 4 items, matching the repo |
| degenerate ladder | §2 below; MB-A5 |
| namespace proposal | §4 |
| §9.4 license check | §5 |

## 2. Ladder (totals; per foundation in the JSON)

| set | worst | always A | always B | uniform | oracle | paper range (5 models) | MDE 5 v 5 | match half-width |
|---|---|---|---|---|---|---|---|---|
| `MFQ_30` | 37.6 | **58.2** | 41.8 | 50.0 | 62.4 | 49.9–58.5 | 5.8 | 3.1 |
| `6_concepts` | 40.8 | 44.0 | **52.0** | 48.0 | 55.2 | 44.4–52.8 | 3.4 | 1.9 |
| `MFQ_30_compare` | 1.0 | 10.0 | **11.0** | 10.5 | 20.0 | 8.0–12.4 | 3.9 | 2.1 |
| `6_concepts_compare` | 0.0 | **13.0** | 11.0 | 12.0 | 24.0 | 10.4–14.2 | 4.3 | 2.4 |

MDE and half-width are worst case (choice probability 0.5 on every item), items fixed, 80%
power; the half-width compares a paper 5-repeat mean with our 32-rollout mean, plus 0.05 for the
paper's rounding. Second derivation: uniform equals n·M/2 on the binary sets (50.0, 48.0) and
n/2 plus half the tied items on the comparative sets (10.5, 12.0), as computed.

Reading: the best fixed letter lands inside the paper's range on every set. Always A
beats four of five models on `MFQ_30`; always B beats four of five on `MFQ_30_compare`. The paper's
per-foundation model differences are mostly smaller than the worst-case per-foundation MDE at 5
repeats (0.47–3.42, about half of each foundation's band).

## 3. Decisions for the author

**Author, 2026-10-30: items 1-8 approved as recommended, with additions** (T = 0.7 quote cited by
page and section; the OpenAI deprecation dates cited; MB-A2 and MB-A3 cells pre-registered "not
attainable"; R0 of record strips leading whitespace, with the unstripped miss count beside it;
`mfv_clifford2015.json` checked before any norms fetch, cache only; golden pods p1e and p2a; a
counterbalanced G1 cell for MB-A5). All folded into `MB_PLAN.md` v0.5.

1. **Sampling temperature (changes §2).** Paper v2 §4.1: "For all models, we use a temperature
   of 0.7." v1 says nothing; the plan's T = 1.0 is inferred from `main.py`, which covers only the
   OpenAI models. No code exists for the open-model rows, so the paper text is the only protocol
   statement for them. Recommendation: T = 0.7 of record, T = 1.0 as a registered fork reported
   beside it.
2. **GPT-3.5 snapshot.** Paper v2 names `gpt-3.5-turbo-1106` (and `gpt-4-0613`); §1 names 0125.
   Correct the row. The retirement of 1106 is not grounded in a repo file; it does not change the
   row's status ("not replicable").
3. **Misses in paper comparisons (MB-A3).** Add a "miss scores 0" aggregate beside R0's unparsed
   count for comparison with paper cells; R0 itself keeps logging misses.
4. **Match rule's CI object (changes §5 MB-G1).** The paper's cells condition on these 88 items.
   Resampling items answers a different question, widens the interval (biased toward
   "replicates"), and at 4 items per foundation has 35 distinct resamples. Recommendation: items
   fixed, a rollout-bootstrap prediction interval for a 5-repeat mean, plus 0.05 rounding (the
   half-width column above at worst case). Keep the nested bootstrap for between-model statements,
   flagged as n = 4 per foundation.
5. **Per-item score rows leak the key (changes §9.2).** Tracked per-item × rollout (letter, score)
   pairs reconstruct the key for every letter a model picks, which §9.2 forbids as a key table.
   Recommendation: track parsed letters and readouts only, and compute scores at analysis time from
   the fetched key; track aggregates.
6. **R0 decode parity.** `response[0]` depends on detokenization (leading space, special tokens).
   Pin `skip_special_tokens=True`, no strip, and record it per row.
7. **MB-A4 discriminator.** Zero GPU, about an hour: fetch Clifford et al.'s published MFV norms
   read-only and recompute both MFV key sets. Needs your approval for the fetch.
8. **Commit order.** `MB_PLAN.md` and `MORAL_EVAL_REPLICATIONS.md` are untracked, so the MB-G1
   match rule is not yet pre-registered. Fold in items 1–6, commit and push before any G1 data.
9. **Outside §9 scope.** `deepsteer/datasets/mfv_clifford2015.json` and Paper 3's
   `behavioral_benchmarking.py` hold verbatim Clifford MFV text with no `DATASET_LICENSES.md` row.

## 4. Namespace and shared-module proposal (against ARCHITECTURE.md)

Recommendation: one new package for the choice-specific machinery, `deepsteer.stats` as the plan
names it, and the model-access pieces in `core/` beside `ModelInterface`, not in the new package.
Reason: MoReBench and LEB's dilemmas need chat rendering and seeded sampling but no choice
parsing, and `core/` already owns `generate()` for `WhiteBoxModel` and `APIModel`. A sampler in a
new package would be a second model layer. `deepsteer.evals` is not proposed: it reads as a second
home for benchmarks.

| piece | home | notes |
|---|---|---|
| `ChoiceItem` (prompt or messages, option labels, score map, metadata) | `deepsteer/choice/items.py` (new) | labels configurable (A–E, 0–5 Likert for LEB) |
| `ChoiceParser` (`first_char` faithful; `ladder`: answer line, bare, leading, unique mention) | `deepsteer/choice/parse.py` | own version constant; KDG's scenario-specific steps (tool hits, demotion) stay in `kdg.harness` as a wrapper |
| `letter_mass_readout`, `mask_positions` (readout v2) | `deepsteer/choice/readout.py` | lifted from `papers/kdg_panel/scripts/kdg_pod_lib.py` (`ModelWrapper.raw_next_logprobs`), not from `deepsteer.kdg` |
| degenerate ladder, band feasibility | `deepsteer/choice/ladder.py` | from `mb_g0_audit.py` (`ladder`, `worst_case_mde`, `match_half_width`) |
| chat rendering, `SystemRolePolicy` (native, prepend-to-user, drop), template sha256 | `deepsteer/core/chat.py` (new) | Gemma's prepend policy lives here |
| sampling contract (per-item crc32 seeds, pure temperature, warpers off: KDG sampler v2) | `ModelInterface.generate` options; `deepsteer/core/vllm_model.py` (new, optional dependency) | API backends stay in `APIModel` |
| model registry loader (revision, template sha, system-role policy, license) | `deepsteer/core/registry.py` (new) | YAML files stay in `papers/`; the loader takes a path |
| nested bootstrap, generic difference CI, MDE | `deepsteer/stats/bootstrap.py`, `power.py` (new) | numpy only; add an import contract like geometry's |
| agreement, Cohen's kappa | `deepsteer/stats/agreement.py` | from `kdg/calibration.py` |
| `MoralBenchBenchmark(Benchmark)`, `MoralBenchResult` | `benchmarks/moral_reasoning/moralbench.py`, `core/types.py` | `min_access_tier` API (R0, R1); R2 runs on WEIGHTS and is skipped and logged otherwise |
| MoralBench loader | `datasets/moralbench.py` | built at G0 |

Forward consumers: **MoReBench** uses `core/chat.py`, the sampler and
`deepsteer.reasoning.think_io` and never imports `choice`; its `RubricJudge` is designed in its own
plan. **ETHICS** uses `ChoiceItem` (binary or ranked score maps), the parser or letter mass (or the
existing `score()` forced-choice for base models) and the nested bootstrap at large n; its loader
marks the V_moral held-out overlap. **LEB** uses `ChoiceItem` with numeric labels for MFQ and WVS
and MoReBench's judge for the dilemmas.

**KDG byte-identical after the lift.**
1. Golden fixtures land in a commit *before* the lift, and pass against the current code:
   - parse statuses: `kdg.harness.parse_response` over the calibration sets and every rollout row
     of one banked sampled pod (candidate: p1e), saved as (row id, status, letter) plus a sha256;
   - readout: the existing tiny bf16 Llama fixture in `test_kdg_readout_v2.py`, with
     `array_equal` between the shared readout and `ModelWrapper.raw_next_logprobs` at batch 8 and
     batch 1 (v1 kept as a negative check);
   - analysis: re-run one banked v2 pod's analysis (candidate: p2a) from its saved arrays and
     compare output bytes with the committed JSON;
   - stats: `difference_ci` and `kdg_rate` on banked readouts with fixed seeds, compared with
     exact equality.
2. The lift moves bodies into the shared modules. The old names stay importable from
   `deepsteer.kdg.{harness,stats,calibration}`, and `kdg_pod_lib` imports the readout, so the ~120
   importers do not change. Row fields and `KDG_HARNESS_VERSION` stay unchanged. The shared
   bootstrap keeps KDG's RNG draw order: a change in draw order changes the CIs, and the stats
   golden catches it.
3. The golden tests pass unchanged on the lift commit. The pod sync list gains the new library
   files, checked with the pod-sync emulation.

Latent KDG detail, unchanged by the lift: `kdg.harness` parses letters A–D while the schema allows
5 options; every banked KDG scenario has at most 3, so nothing parsed differently. The KDG wrapper
passes "ABCD" to stay byte-identical.

## 5. License check (§9.4)

- **Tracked and new files.** 69 distinct MoralBench lines (25 characters or more) were scanned
  against all 2,941 tracked and untracked-not-ignored files. No MoralBench file is copied. Two
  existing files match on 7 lines, all of them verbatim Clifford (2015) MFV vignettes committed for
  Paper 3 in June (item 9).
- **Wheel** (`pip wheel . --no-deps`; the `build` module is not installed). Of MB's files, it holds
  only `moralbench.py` and the hash manifest. Text scan of the unpacked wheel: clean.
- **`DATASET_LICENSES.md`.** The "MoralBench: not vendored" section is in place.
- **Ignored output.** `papers/mb_moralbench/outputs_raw/` is in `.gitignore`.
- **Leak guard.** The audit JSON is checked against every question line and option before it is
  written.

## 6. Anticipated review

1. Paper v2 states T = 0.7 and the plan registers 1.0. The "defaults match the paper" rule
   applies, so 0.7 becomes the default and 1.0 a fork (0 GPU; plan edit, item 1).
2. The match rule resamples items the paper conditions on, with n = 4 per foundation.
   Estimator-traps (resampling attenuation, bias direction) call for an items-fixed prediction
   interval (0 GPU; computed, item 4).
3. Tracked per-item scores reconstruct the key table, so §9.2 contradicts itself. Fix: track
   letters only (0 GPU; item 5).
4. Fixed policies span the paper's spread, so every MB score needs a null. Report the A-rate and
   the constant-A-rate prediction beside each score (0 GPU at G1; MB-A5).
5. R0 depends on detokenization for open models (harness parity). Pin and record the decode
   (0 GPU; item 6).

Implemented now: loader, manifest, audit, ladder, MDE and match half-widths, the MB-A2 rescale
discriminator (R_rescale rejected), the leak guard, tests (synthetic plus one opt-in real-data
test), the license row and the gitignore entry.

Open (with costs): the MB-A4 Clifford-norms recompute (0 GPU, about 1 h, needs fetch approval);
MB-A1, MB-A3 and MB-A5 at MB-G1 (M4, 0 pod); MB-A2 at MB-G2 (inside the 4 A100-h cap).

Question behind the question: is MoralBench worth carrying as the Phase 2 drift monitor? If
MB-A5's acquiescence reading holds at G1, it mostly reads answer-letter bias. Zephyr and Gemma
on the M4 answer that before any pod is approved.

## 7. Referee pass

1. *"The cells outside the band may come from a different key version, not from paper errors."*
   Conceded as a reading: MB-A2 (R_items) and MB-A3 (R_key) carry it, and the plan registers these
   cells as "not attainable under the pinned key", not as errors in the paper.
2. *"A worst-case MDE overstates the noise. Models are nearly deterministic, so the paper's
   differences may be real."* Conceded: the MDE column is an upper bound at choice probability 0.5.
   G1 measures the variance, and the match rule uses the measured rollouts.
3. *"The comparative-consistency test assumes the comparative keys use the binary H."* Yes; that is
   MB-A4's R_other_H. The Clifford norms recompute separates it from R_errors before G1.
