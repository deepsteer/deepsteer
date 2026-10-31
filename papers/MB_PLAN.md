# MB Plan — Exact MoralBench Replication, Decomposed Into the Library

Status: v0.9, 2026-10-30. **v0.9 (author, 2026-10-30): Amendment 4 (§7): Phase 2's GPT-OSS-20B cell rides the
MRB-G1 pod; Muse and Qwen3.8 wait for a later pod.** **v0.8 (author, 2026-10-30): Amendment 3 (§7): Phase 2 approved in reduced form;
`zephyr-orpo-141b` stays unrun; MB-G2 gate passed (`mb_moralbench/REPRODUCTION.md`).** **v0.7 (author, 2026-10-30): Amendment 2 (§5): the G2 pod carries Gemma's G1 row
and D4 beside Llama-2-70B-chat.** **v0.6 (author, 2026-10-30): Amendment 1 (§5): the MB-G1 diagnosis cells, a
stopping rule registered before them, and the drift-monitor readout rule (§7).** **v0.5 (author, 2026-10-30): MB-G0 passed; the gate report's decisions 1-8
approved with additions and folded in below** (sampling T = 0.7 of record, §2; R0 of record and the
paper-comparison aggregate, §3; the counterbalanced cell, §4; the items-fixed match rule and the
"not attainable" cells, §5; the approved namespace and golden pods, §6; letters-not-scores, §9.2;
the MB-A4 norms check, §5). G0 report: `papers/mb_moralbench/MB_G0_REPORT.md`. Earlier: v0.4 draft, 2026-10-29 (v0.4 adds the goals in `MORAL_EVAL_REPLICATIONS.md`: R0 is the
default, a reproduction report at MB-G2, Phase 2 as the steering baseline; v0.3 scopes the plan to MoralBench only; the other replications live in
`papers/MORAL_EVAL_REPLICATIONS.md`; v0.2 added §9, licensing and clean-room rules). Pre-registration candidate.
**No GPU spend authorized by this document**; the one pod cell (§8) waits for author approval at MB-G1.
Line prefix **MB**: this file is `papers/MB_PLAN.md`; code/data/outputs in `papers/mb_moralbench/`;
anomalies `MB-A1, ...` in `papers/ANOMALIES.md`; claims `MB-01, ...` in `papers/CLAIMS.md`.
The other replications (MoReBench, ETHICS, LLM Ethics Benchmark) are indexed in `papers/MORAL_EVAL_REPLICATIONS.md`,
each to get its own plan; §6 designs the shared core for all of them.

Skills in force: `instrument-calibration` (ladder + MDE before any "no difference" or "match"),
`estimator-traps` (difference CIs, nested bootstrap over items × rollouts), `compute-ordering`
(zero-GPU first; the 70B cell batches onto a pod), `anomaly-triage`, `program-thesis` for any prose.
`construct-audit` applies only to the Phase 2 P3 bridge (§7), not Phase 1.

## 0. Source of record

- Paper: Ji, Chen, Jin, Xu, Hua, Zhang, "MoralBench: Moral Evaluation of LLMs", arXiv:2406.04428
  (v1 2024-06-06, v2 2025-07-04); SIGKDD Explorations 27(1):62-71, 2025.
- Code: `github.com/agiresearch/MoralBench` (5 commits; `main.py`, `questions/`, `answers/`,
  `template/`). **Pinned at `84100440a28d040eca27df32b93f44685ad69ce5`** (MB-G0; `mb_moralbench/SOURCE.md`).
- Data mirror: the Google Drive folder linked in the paper. MB-G0: questions only, all 88 files
  byte-identical to the pinned commit; the answer keys exist only in the repo.

## 1. What the source pins, and what it leaves open

**Pinned by `main.py` (replicate exactly):**
- Four item sets: `MFQ_30`, `6_concepts` (the MFV adaptation), `MFQ_30_compare`, `6_concepts_compare`.
- System prompt `template/moral_system.txt`; user prompt `questions/<set>/<item>.txt`; files read
  with `readlines()` then `'\n'.join(...)` (this doubles newlines; reproduce it byte-for-byte).
- Scoring: `score = ans[item][response[0]]` from `answers/<set>.json` — the **first character** of
  the reply is the key. Max per item is accumulated as `total_score += 4`.
- Per-concept sums keyed `question[:-6]` over concepts `harm, fairness, ingroup, authority, purity,
  liberty`.
- OpenAI chat completions with no sampling arguments (API defaults: temperature 1, top_p 1).
  The paper's text says otherwise: "For all models, we use a temperature of 0.7" (arXiv v2, p. 5,
  §4.1 LLM Backbones; absent from v1). §2 takes the paper's text as the sampling of record.

**Not pinned (each is a decision of record, logged in §2 before any model data):**
- No code for the open models (Zephyr, Llama-2, Gemma): no checkpoint ids, chat templates, system-
  role handling, sampling, or how the five repeats were drawn.
- The paper's binary formula (H if Agree, 5−H otherwise, 0–5 scale) vs. the code's per-item max of 4
  and precomputed answer keys. Audit at MB-G0: recompute keys from the formula where H is
  recoverable and report agreement.
- README says `python rec.py`; the file is `main.py`. Note only.
- Whether `answers/` holds score keys only (expected) or any raw model replies. MB-G0: keys only
  (plus orphan keys with no question file, `SOURCE.md`).

**Closed-model rows are unreplicable.** The paper names `gpt-3.5-turbo-1106` and `gpt-4-0613`
(arXiv v2, p. 5, §4.1). OpenAI shut down `gpt-3.5-turbo-1106` on 2026-09-28 (notice of 2025-09-26)
and `gpt-4-0613` with its `gpt-4` alias on 2026-10-23 (notice of 2026-04-22), per
`developers.openai.com/api/docs/deprecations`, checked 2026-10-30. Record both rows as "not
replicable: model retired". Do not
substitute a newer model in the replication table; any modern closed model is a separately labeled
extension row, author-approved, with its spend stated.

**License.** MoralBench has no license (checked 2026-10-29 at commit `8410044`). §9 gives the rules
for handling it; they bind every MB commit.

## 2. Models and generation of record

Registry: pin each model at first load into `papers/kdg_panel/models.yaml` (or a shared registry
lifted from it, §6) with revision, chat-template sha256, and system-role policy.

| paper row | model id of record | where | notes |
|---|---|---|---|
| Zephyr | `HuggingFaceH4/zephyr-7b-beta` | M4 (bf16) | assumption: the paper cites the alignment handbook, whose 7B release is β; `-alpha` recorded as the alternate |
| LLaMA-2 (70B) | `meta-llama/Llama-2-70b-chat-hf` | pod, 2×A100-80GB, vLLM TP=2, bf16 | assumption: chat, since the base model can't follow the answer format; gated license must be accepted |
| Gemma-1.1 (7B) | `google/gemma-1.1-7b-it` | M4 (bf16) | Gemma's template has no system role: the user turn is `system + "\n\n" + user` (policy `prepend`, logged per row) |
| GPT-3.5 / GPT-4 | — | — | retired 2026-10-23 (§1) |

- Sampling of record (v0.5): **temperature 0.7** (the paper's text, §1), top_p 1.0, top_k off, **no
  `generation_config` defaults layered on top** (KDG v2 harness lesson), a seed per item (crc32 of
  `set/item_id`, KDG sampler v2) with the rollouts drawn in one seeded call, `max_new_tokens` 128.
- **Registered fork: temperature 1.0** (the API default `main.py` used for the OpenAI rows), same
  seeds, reported beside the T = 0.7 table with its own match verdicts.
- Repeats: **5** for the paper-faithful table (as the paper reports); **32** rollouts for the
  DeepSteer read (CIs). Both come from the same seeded stream; the 5 are the first 5 of the 32.
- Quantizing Llama-2-70B is a fork (amendment + both results), not a default.

## 3. Readouts

- **R0, paper-faithful (the default):** `response[0]` lookup into the answer key, exactly as `main.py` does.
  A miss (KeyError in their code) is logged as unparsed with a count per set and model; never imputed.
  **R0 of record (v0.5) for local models:** decode the new tokens only with
  `skip_special_tokens=True`, strip leading whitespace, take the first character. The **unstripped
  miss count** (same decode, no strip) is reported beside it per set and model.
- **Paper-comparison aggregate (v0.5):** a reply that misses scores 0, as the paper's totals imply
  (MB-A3). It is the aggregate the match rule reads; the parsed-only mean and the miss count are
  reported beside it.
- **R1, robust parse:** the library's choice parser (§6), a generalization of the KDG ladder (answer
  line, bare letter, leading letter, unique mention). R1 − R0 per model is the parser-sensitivity column.
- **R2, continuous:** letter mass at the answer position (KDG readout v2, position ids from the
  attention mask), expected score under that distribution. No sampling.

R1 and R2 are opt-in and are reported beside R0, never in its place. Tracked rows hold parsed letters
and readouts, never per-item scores (§9.2); scores are computed at analysis time from the fetched key.

## 4. Calibration ladder (zero GPU; DeepSteer addition)

For every set and foundation, compute scores for degenerate policies: always-first-option,
always-second, uniform-random expectation, and the best-key oracle. Every model score is reported
inside this band. Report item counts and the MDE for a between-model difference at the observed
variance. Between-model comparisons use **difference CIs** from a nested bootstrap (items, then
rollouts), never overlap checks, flagged as n = 4 items per foundation. The point of this section: the
paper's model differences may sit inside the range a fixed policy already spans. **MB-G0 result:** the
best fixed letter lands inside the paper's five-model range on every set (MB-A5).

**Counterbalanced cell (v0.5; opt-in, G1, beside R0; MB-A5's discriminator).** The same items with
the comparative options swapped (statement A shown as B and vice versa) and the binary letters flipped
(`A. Disagree B. Agree`, `A. Not Correct B. Correct`), the keys swapped to match, at T = 0.7 with the
same 32 seeded rollouts. Per model and set, with rollout-bootstrap 95% CIs (items fixed):
- A-rate in the original order `a_o` and in the counterbalanced order `a_c`;
- letter bias `L = (a_o + a_c)/2 - 0.5` (0 when the choice follows the content);
- content effect `K = (a_o - a_c)/2` (on the binary sets this is the agree rate minus 0.5: the label
  bias, also acquiescence);
- item selectivity `S` = R0 total minus the score a constant A-rate `a_o` would earn on the same
  items (`sum_i B_i + a_o (A_i - B_i)`). `S`, `a_o` and `a_c` are computed over parsed replies
  (parsed-only per-item means), so misses do not enter them; miss counts are reported beside.
Pre-registered read: **R_acquiescence** for a model and set if the CI of `S` covers 0 (the total is
what one rate earns); **R_content** if it excludes 0. `L` and `K` say whether a constant rate is a
letter or a label. Both branches publish: R_acquiescence means MoralBench as a Phase 2 drift monitor
reports the A-rate and agree rate beside every score; R_content means it reads item content inside
the ladder's band.

## 5. Gates

- **MB-G0 (zero GPU):** loader + sha pin, item counts per set and foundation, license finding,
  Drive-vs-repo diff, formula-vs-key audit, paper-arithmetic audit (known: the MFV binary row for
  Gemma-1.1 sums to 51.8 across foundations but its printed total is 44.4 → MB-A1; check whether
  the comparative per-foundation maxima imply ~3 pairs per foundation), degenerate ladder (§4), and the namespace proposal (§6).
  Author reads before any model runs.
- **MB-G0 passed (2026-10-30).** Report: `papers/mb_moralbench/MB_G0_REPORT.md`; ledger MB-A1 to
  MB-A5. Before G1: the MB-A4 norms check (zero GPU), after this version is pushed. Check
  `deepsteer/datasets/mfv_clifford2015.json` first (MB-G0 follow-up: it holds 30 vignettes with
  classification rates and no wrongness ratings, so it cannot give H); then the published norms
  (`Full-Published-Norms.xlsx`, Clifford's research page) fetched to
  `~/.cache/deepsteer/mfv_clifford2015/` only, never committed. Recompute the binary H and both MFV
  key orientations from the norms; R_other_H if the comparative keys follow the norms, R_errors if
  not.
- **MB-G1 (M4):** Zephyr and Gemma rows, R0/R1/R2, against the ladder, at T = 0.7 and the T = 1.0
  fork, plus the counterbalanced cell (§4). Pre-registered match rule (v0.5, **items fixed**: the
  paper's cells condition on these 88 items): a paper cell **replicates** if it lies inside the 95%
  prediction interval for a 5-repeat mean of the paper-comparison aggregate (§3), computed by drawing
  5 of our 32 rollouts per item with replacement (10,000 draws), widened by 0.05 for the paper's
  rounding; per-foundation and total reported separately. Item resampling is not used for the match
  (it answers a different question, widens the interval toward "replicates", and has 35 distinct
  resamples at 4 items).
  **Not attainable (pre-registered):** a paper cell outside the pinned key's band is "not
  attainable" and leaves the replicate count:
  - MB-A2, LLaMA-2 MFV sanctity 11.1 > oracle 9.9 (G2 row): not attainable under any aggregate;
  - MB-A3, Zephyr MFQ-comparative loyalty 0.4 < 1.0 (the tied item's floor): not attainable when
    every reply parses; it is reachable only through misses scored 0, and that aggregate's value is
    reported beside the verdict as MB-A3's discriminator.
  MB-A1 (Gemma MFV): the match is read against both the printed total (44.4) and the cells' sum
  (51.8); Gemma's agree rate on `6_concepts` decides the reading (MB-A1).
- **Amendment 1 (2026-10-30, author; committed and pushed before any cell it licenses).** The
  Zephyr row missed at G1 (`mb_moralbench/MB_G1_REPORT.md`: 5 of 25 attainable cells replicate;
  MFQ binary 22.3 against 54.2, and no readout reaches 54.2, MB-A6). The registered response is a
  diagnosis under amendment.
  - **Stopping rule (registered first).** The verdict of record stays at the registered
    configuration (`zephyr-7b-beta` at `892b3d7a`, native system role, T = 0.7, R0 with misses
    scored 0). A variant whose cell lands inside the paper cell's 95% PI (same rule as G1) is
    reported as "matches under X" and never replaces the verdict of record. After the four cells
    below, the result is written up and the diagnosis stops; any further variant needs a new
    amendment.
  - **Four cells** (M4, bf16, T = 0.7, 88 items × 32 rollouts, the G1 seeds, R2 in both orders):
    D1 `HuggingFaceH4/zephyr-7b-alpha` at `014792bbb59d04ced3b5a9b8b4dfc926655d958f`, native system
    role (MIT); D2 `zephyr-7b-beta`, system text prepended to the user turn (`"\n\n"`); D3
    `zephyr-7b-beta`, no system prompt; D4 `HuggingFaceH4/zephyr-7b-gemma-v0.1` at
    `03b3427d0ed07d2e0f86c0a7e53d82d4beef9540`, native system role (Gemma terms: evaluation only,
    §9.3).
  - **Reads per cell:** per set and foundation, the R0 total (misses scored 0) with its PI and
    "matches under X" or not; R0 misses (stripped and unstripped); R2's expected score and A-rate;
    the MB-A7 split (R1's letter among R0 hits against among R0 misses).
  - **Both branches.** A variant matches on MFQ binary: MB-A6's R_model is supported for that
    variant, reported as "matches under X". None of the four matches: the Zephyr row is "not
    reproduced" across two checkpoints, a third Zephyr recipe and three prompt paths, with R2 as the
    parse-free evidence. In that branch `HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1` (`a3be0845`,
    281 GB, Apache-2.0) is the one pod-only candidate, listed and not run (it needs pod approval
    inside the 4 A100-h envelope, §8).
  - **MB-A4 check (done before the cells).** All 24 `6_concepts_compare` items, the fairness group
    and `ingroup_2` included, ask for the more moral statement. The keys are not reclassified, and
    MB-A4's R_errors stands. Both branches are reported: a match
  validates the instrument; a miss triggers a diagnosis under amendment (template, system policy,
  sampling, parse), with the verdict under each choice. Running `main.py` itself (in a scratch
  environment outside the repo, with our replies in place of the API call) is a diagnostic for a
  miss, not a required step. Author approves the pod at this gate.
- **Amendment 2 (2026-10-30, author; committed and pushed before the pod).** Gemma's G1 row and
  Amendment 1's D4 move to the G2 pod. On the M4, the 8.5B Gemma models do not stay resident in
  24 GiB (swap 15.5 of 16 GB, about 350 s per item against Zephyr's 23 s; about 34 h for both).
  - **Pod.** 2 × A100-80GB (`GPU_COUNT=2`, the shared launcher `papers/runpod_common/run_session.sh`),
    runner `papers/mb_moralbench/runpod/remote_mb_g2.sh`, results `outputs/g2/`.
  - **Engine (deviation from §2's vLLM).** HF transformers for all three models, the G1 driver and
    sampler unchanged: harness parity with the M4 rows, no engine-parity gate. Llama-2-70B-chat
    (`e9149a12809580e8602995856f8098ce973d1080`, Llama 2 Community License) loads across both GPUs
    (`device_map="auto"`). Transformers 5.17.0 as on the M4, with the KDG torch 2.6.0+cu124 trio as
    the fallback if the image's torch fails; the versions are recorded per run.
  - **Units, in order.** GPU0: Gemma-1.1 t07 (of record), cb07, t10. GPU1, at the same time: D4
    `zephyr-7b-gemma-v0.1` t07. Meanwhile Llama-2-70B downloads. Then Llama-2-70B-chat t07 (of
    record), cb07, t10 (fork last) on both GPUs. Analysis on the pod; the session manifest
    (sha256 of every file) last.
  - **Gemma's record.** Gemma's G1 row of record is the pod's. The 3 items run on the M4 (96 rows,
    same seeds) are not part of it; they stay as an exploratory device-parity read.
  - **Power (measured, Zephyr G1 rollouts; G0 worst case in brackets).** 95% PI half-width for a
    5-repeat mean against 32 rollouts: MFQ 2.67 (3.12), MFV 2.62 (1.86), MFQ comparative 0.75
    (2.10), MFV comparative 1.07 (2.36). MDE between two models at 32 rollouts: 1.95, 1.91, 0.52,
    0.76.
  - **Envelope and bail.** Estimate about 1.8 h of pod time (3.6 A100-h) inside the 4 A100-h cap.
    Watchdog at 1.95 h. Cells stop starting 15 min before it: a later cell is skipped when the
    previous cell's measured rate projects it past that point (recorded as `skipped_cells`), so the
    order above fixes what is lost first (Llama's t10 fork, then its cb07; t07 of record runs
    whenever the pod reaches it). VALIDATE first, on one GPU: the stack, model access, the
    MoralBench fetch against the manifest, the gate tests, the tiny dry run on CUDA, and a 3-item
    Gemma timing probe; stop at 0.3 h.
  - **Saves, per unit.** Per (item, rollout) letter rows, R2 rows, `run_meta.json` (versions,
    template hash, seeds, timings, skipped cells), the analysis JSON, raw replies under
    `outputs/g2/_raw/` (gitignored), driver and session logs, `manifest_g2.json`.
  - **Dependencies.** The code at the pushed commit; MoralBench fetched on the pod at the pinned
    commit and verified; model access checked on 2026-10-30 with the author's token (the launcher
    passes `HF_TOKEN`); about 175 GB of weights (`DISK_GB=260`).
  - **Both branches.** Gemma and Llama-2 follow the G1 match rule (MB-A2's cell is not attainable).
    MB-A1 reads Gemma's agree rate. D4 follows Amendment 1's stopping rule.
- **MB-G2 (pod):** Llama-2-70B-chat row; the full replication table (5 rows: 3 measured, 2 marked
  retired) with the ladder, parser-sensitivity column, and R2; and the reproduction report
  (`papers/mb_moralbench/REPRODUCTION.md`, per `MORAL_EVAL_REPLICATIONS.md`).

Per-item × per-rollout arrays (letters and readouts) are saved for every cell so later statistics
stay zero-GPU.

## 6. Library decomposition (the other half of the deliverable)

Generalize, don't copy. The KDG code is the source for several pieces. Wherever a piece is lifted,
KDG imports the shared version and a regression test proves byte-identical parse statuses and
readouts on KDG's banked calibration sets and one banked pod.

- `deepsteer.datasets.moralbench`: runtime loader (pinned commit, sha256), typed items `(set,
  foundation, item_id, prompt, key)`, foundation map to `MoralFoundation` (`harm`→CARE,
  `ingroup`→LOYALTY, `purity`→SANCTITY, plus fairness, authority, liberty).
- **Namespace (approved by the author 2026-10-30; MB_G0_REPORT §4).** Choice-specific machinery in
  the new package `deepsteer.choice` (`items`, `parse`, `readout`, `ladder`); `deepsteer.stats`
  (`bootstrap`, `power`, `agreement`; numpy only, with an import contract); model-access pieces in
  `deepsteer/core/` beside `ModelInterface` (`chat.py` with `SystemRolePolicy`, the sampling
  contract, `vllm_model.py`, `registry.py`); `MoralBenchBenchmark` under
  `deepsteer/benchmarks/moral_reasoning/`. No `deepsteer.evals`, no second home for benchmarks.
  The list below is the original component list; the homes above replace "the shared core".
  - `ChoiceItem` schema (prompt, options, key/score map);
  - chat rendering with a per-model system-role policy;
  - a backend-agnostic sampler (HF local/MPS, vLLM, OpenAI-compatible API) with per-item seeds and
    pure temperature;
  - `ChoiceParser` (faithful first-character mode + ladder mode);
  - `letter_mass_readout` (lifted from the KDG harness, readout v2).
- `deepsteer.stats`: nested bootstrap, difference CIs, MDE, lifted from `deepsteer/kdg/stats.py`.
- Model registry loader for `models.yaml` (revision + template sha pin), shared by KDG, ST and MB.
- `MoralBenchBenchmark(Benchmark)` returning a typed `MoralBenchResult(BenchmarkResult)` in
  `core/types.py`, runnable through `BenchmarkSuite`.

**`deepsteer.kdg` stays where it is under MB.**
- MB lifts only its generic pieces into the shared modules: the choice parser, the letter-mass
  readout, the stats, and agreement/kappa.
- `deepsteer.kdg` re-imports them, so the ~120 importers in `papers/kdg_panel`,
  `papers/st_live_reference`, `tests/` and `scripts/` are unchanged.
- **Byte-identity (v0.5).** Golden fixtures land in a commit before the lift and pass on the current
  code: parse statuses on the calibration sets and on every rollout row of the banked pod **p1e**;
  the readout on the tiny bf16 fixture of `test_kdg_readout_v2.py`; the analysis of the banked pod
  **p2a** re-run from its saved arrays, compared byte for byte; `difference_ci` and `kdg_rate` with
  fixed seeds, exact equality. The lift commit passes them unchanged. The shared bootstrap keeps
  KDG's RNG draw order.
- **Sequencing (v0.5).** G1 builds the shared modules MB needs without touching `deepsteer.kdg`; the
  KDG re-import is a separate commit series (goldens first, then the lift). ST closed on
  2026-10-30, so the "after ST's in-flight pods" condition below is met.
- Relocating the KDG-specific code (`phase1_frames`, `phase2`, `breadth`, the screen and gate
  constants) is a separate task. It is due under ARCHITECTURE.md now that the KDG paper is
  published, scheduled after ST's in-flight pods; its destination is `papers/kdg_panel/` or
  `deepsteer.research`.
- A `deepsteer.evals.kdg` benchmark wrapper is a later author decision, made only if the panel
  ships as a packaged instrument.

**Forward consumers (design for them, build none of them here).** The shared core must also serve
the replications in `papers/MORAL_EVAL_REPLICATIONS.md`: MoReBench (long generation with think traces, a
rubric judge), ETHICS (binary and ranked classification on large item sets), and the LLM Ethics
Benchmark (MFQ, WVS and dilemma instruments). The MB-G0 namespace proposal says how each would
plug in. Their own interfaces, such as a `RubricJudge`, are designed in their own plans.

## 7. Phase 2 (after MB-G2; separate author approval)

These runs are the pre-steering baseline for the ST subjects on this instrument. MoralBench is a
drift monitor for steering, not a target (it scores agreement with average respondents).

**Amendment 3 (2026-10-30, author): Phase 2 in reduced form.** Replaces the bridge list above.
- **Subjects:** the ST subjects only: GPT-OSS-20B, Muse Glimmer 30B, Qwen3.8-27B (`ST_PLAN.md`).
  The M4 bridge models and the P3 bridge are dropped.
- **Readout:** R2 (letter mass) on both orders (original and counterbalanced), no sampled cells.
  MFQ binary and both comparative sets are scored; MFV binary is reported as a letter-bias probe
  (L and K, no score), because all three G1/G2 families answer B there whatever the label (MB-A5).
- **Answer position (registered before data):** ST's dose-0 letter readout, the reasoning segment
  closed empty in each model's own format (GPT-OSS harmony, the Qwen and Muse thinking tags), with
  single-row readouts of record (ST-A3: GPT-OSS's batched readout moves with batch composition).
  KDG-A22 applies: GPT-OSS's dose-0 forced letter departs from its own low-effort answers, so its R2
  is read as a dose-0 readout, not as the model's sampled answer.
- **Compute:** on the M4, or as riders on a pod already scheduled for other work; no dedicated pod.
  None of the three fits in the M4's 24 GiB, and none is scheduled now, so Phase 2 waits for the
  next scheduled pod.
- **Not a steering gate.** It is the pre-steering baseline on this instrument; no ST verdict reads it.
- **Amendment 4 (2026-10-30, author):** the GPT-OSS-20B cell rides the MRB-G1 pod (`MRB_PLAN.md`
  §5); Muse Glimmer 30B and Qwen3.8-27B wait for a later scheduled pod.
- **Diagnosis:** `zephyr-orpo-141b-A35b-v0.1` stays unrun (author, 2026-10-30).

**Drift-monitor readout (Amendment 1, MB-A7).** Any use of MoralBench as a drift monitor reads R2
(letter mass, no parsing) on the counterbalanced pair (original and swapped orders), and reports
the letter bias L and content effect K beside the score. R0 is never a drift-monitor readout: its
misses track the answer (MB-A7), so format changes under steering would read as moral drift. R0
stays the paper-faithful replication readout only.


- **Bridge models on the same instrument:**
  - OLMo-3-7B-Instruct, Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct on the M4;
  - GPT-OSS-20B, Muse Glimmer 30B and Qwen3.8-27B as riders on the next scheduled pod.
- **P3 bridge (zero GPU on the M4):** does each MFV item's projection onto its P3 foundation
  direction predict R2 agree-mass, on OLMo-2 7B and OLMo-3-7B (base, and instruct where it exists)?
  Type block per direction (`construct-audit`).

## 8. Compute and cost

| cell | where | estimate |
|---|---|---|
| MB-G0 audit + ladder | M4 | 0 GPU |
| Zephyr-7B-β, Gemma-1.1-7B-it (32 rollouts + R2, all four sets) | M4, bf16 (~15 GB each) | 0 GPU; well under an hour each at the expected item counts |
| Llama-2-70B-chat | 2×A100-80GB, vLLM | ~1.5–2.5 A100-h, mostly weight download and load; generation is minutes |
| GPT-3.5 / GPT-4 | — | $0 (retired) |
| **MB envelope** | | **cap 4 A100-h** including one retry; overrun needs an amendment |
| Phase 2 riders | M4 + existing pod | ~0 GPU on M4; ~1 A100-h marginal on a shared pod |

Item counts are confirmed at MB-G0; the estimates assume at most a few hundred prompts per model
per pass, consistent with the paper's per-foundation totals.

## 9. Licensing and clean-room rules (MB)

Scope: only what the MB replication adds to the repo, the wheel and the Zenodo deposits.
The rest of the repo's license posture is out of scope here (author, 2026-10-29). License findings
and handling rules for the other replications are in `papers/MORAL_EVAL_REPLICATIONS.md`.

### 9.1 What each source allows

| source | license | basis |
|---|---|---|
| MoralBench code (`main.py`) | none: all rights reserved | no LICENSE file, no license in README, commit `8410044` (2026-10-29) |
| MoralBench data (`questions/`, `answers/`, `template/moral_system.txt`, Drive mirror) | none: all rights reserved | same; the Drive folder states no terms |

### 9.2 MoralBench: nothing of theirs enters the repo

- **Clean-room code.** Write the harness from §1's behavioral description. Do not open `main.py`
  in the editor while writing library code. Do not paste from it or vendor it. Do not add the repo
  as a submodule or as a dependency. Reproducing *behavior* (first-character lookup, `readlines()`
  plus `'\n'.join`, the `+4` per item, default sampling) is allowed. Copying *expression* is not.
- **Data is fetched, never committed.** The loader clones the pinned commit into a user cache
  outside the repo (default `~/.cache/deepsteer/moralbench/<sha>/`, overridable). It verifies a
  sha256 per file against a manifest committed in the library. The manifest holds file paths and
  hashes only. The question text, the system prompt and the answer keys are never committed,
  never put in the wheel, and never put in test fixtures. The same rule covers anything derived
  from them that reproduces their wording, such as paraphrases or translations.
- **What MB outputs may hold.** `papers/*/outputs/` is git-tracked by default, so MB writes in two
  places:
  - **Tracked** (`papers/mb_moralbench/outputs/`):
    - item ids (file stems), set names and foundation labels;
    - parsed letters, letter masses and aggregates (v0.5: no per-item scores; letter + score pairs
      rebuild the key);
    - rendered-prompt sha256 (not the text);
    - our run metadata.
  - **Untracked** (`papers/mb_moralbench/outputs_raw/`, added to `.gitignore` in the same commit
    that creates it): raw replies and rendered prompts. Raw replies can echo item text.
- **Answer-key values stay out of tracked files.** A table of key values per item is not tracked,
  and neither are per-item scores (v0.5): per-item × rollout letter and score pairs reconstruct the
  key for every letter a model picks. Scores are computed at analysis time; aggregates are tracked.
- **Tests.** Unit tests use synthetic MoralBench-shaped items written for the test (Apache-2.0),
  covering parser, scorer, ladder and loader behavior. Tests against the real data are opt-in
  (marked, network-gated) and never run in CI.
- **Deposits.** Zenodo arrays follow the "tracked" list above. No text.
- **Prose.** Papers and RESULTS describe items by id and foundation. Quote an item only when the
  argument needs its wording: one item at a time, briefly, with citation.
- **Library docs.** `MoralBenchBenchmark`'s docstring and README entry say the data comes from the
  upstream repo under its owners' terms, which currently grant no license, and that the user
  fetches it.
- **`DATASET_LICENSES.md`** (at MB-G0): add a "MoralBench, not vendored" entry giving the commit,
  "no license, runtime fetch only", and the date checked.
- **Optional (author's call):** ask the authors to add a license. If they grant one, amend this
  section before anything changes.

### 9.3 Models and providers

- **Model weights.** No weights are vendored or committed. Each model's license is recorded in
  the registry entry:
  - `zephyr-7b-beta`: MIT;
  - `Llama-2-70b-chat-hf`: Llama 2 Community License, gated;
  - `gemma-1.1-7b-it`: Gemma Terms of Use;
- **Outputs are for evaluation only.** No adapter or model is trained on outputs from these
  models under MB, because Gemma's terms reach models trained on its outputs. Any training
  use needs an amendment that checks the model's terms first.
- **Paid APIs** (closed extension rows, if ever approved): record the provider terms in the
  amendment that authorizes the spend.

### 9.4 Gate check

MB-G0 ends with a license check:
- `git ls-files` shows no MoralBench text;
- the wheel built from the branch (`python -m build`, then list the archive) holds no MB data;
- the `DATASET_LICENSES.md` row for MoralBench is in place.
