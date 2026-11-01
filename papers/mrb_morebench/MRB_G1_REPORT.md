# MRB-G1 gate report

Sections 1-5: VALIDATE and the v0.4 decision. Section 6: the aborted session. Section 7: the v0.5
decision. **Section 8: the G1 results and the gate.**

## (VALIDATE) VALIDATE passed, the registered rule says STOP

Rules: `papers/MRB_PLAN.md` v0.3 §5. Numbers: `outputs/g1_validate/` (projection.json, the
generation and judge summaries and rows). No session was launched.

## 1. VALIDATE (third run; 2 x A100-SXM4-80GB, 0.38 h wall)

| check | result |
|---|---|
| stack | vLLM 0.31.0 cu129, torch 2.13, transformers 5.17.0; sampling in PyTorch (flashinfer's sampler build fails under the image's nvcc) |
| envelope condition | **gpt-oss-120b judges on this GPU type**: tensor-parallel over two A100s, 34.4 GiB of weights per GPU, a 1.28 M-token KV cache; 2,236 judge calls at 11.8 calls/s (4,892 output tok/s, 414 tokens per call) |
| generation | GPT-oss-20b and Qwen3-30B-A3B, 16 scenarios × 4 each: every generation reached its final answer, none hit the 10,500-token cap; responses 5,355 and 5,524 characters (paper Table 7: 5,071 and 5,973); Qwen traces 3,408 characters (Table 6: 3,691) |
| GPT-OSS traces | **0 characters on all 64**: a split bug, fixed since (MRB-A8, below) |
| judge replies | 2,224 of 2,236 a bare yes or no; the substring and exact-token credit tests never disagree (MRB-A3, so far inert) |
| MB Phase 2 rider | ran (smoke, 4 rows) |

Earlier VALIDATE runs failed at engine start-up (ninja off the PATH; flashinfer's sampler build).
Spend: ~1.0 A100-h for those, ~0.86 for this one: **~1.86 of the 8 A100-h envelope**.

## 2. Projection and the registered decision

Judge calls by tier: T1 114,176 (k 0-1 of both models on both fields, the ladder on 100 scenarios,
J1, the judge and generation forks); T2 45,800 (GPT-oss-20b k 2-3, cut c); T3 45,800 (Qwen k 2-3,
cut b); T4 36,748 (the ladder on the other 400, cut a).

| cut | session A100-h | fits the 6.14 left |
|---|---|---|
| none | 13.91 | no |
| a | 12.18 | no |
| a+b | 10.03 | no |
| a+b+c | 7.88 | no |

**Decision: STOP** ("if it still overruns after (c), stop and report"). The projection is if
anything low: it was measured with empty GPT-OSS traces, and Qwen's real traces cost 545 output
tokens per judge call against 356 for the empty ones, so a corrected session costs about 0.5 A100-h
more at every level. Generation (1.6 A100-h at every level) was measured at 64 concurrent
requests and is probably high.

## 3. A bug VALIDATE caught (MRB-A8)

`think_io.split_rollout` assumes the prompt opened GPT-OSS's analysis channel; GPT-OSS's template
ends at `<|start|>assistant`, so the model opens it and the split returns an empty trace. MRB's
driver now splits harmony output by its own channel markers, and the dry-run test asserts
non-empty GPT-OSS traces (it fails on the old split). The same helper is used by Paper 7's
`refusal_baseline`, `extract_trace_profile` and `extract_two_site`; on a first read the reasoning
span there feeds only a coherence fallback, but that is for the author to rule on (escalated in
ANOMALIES MRB-A8).

## 4. Options for the author

Cost unit: 10,000 judge calls ≈ 0.47 A100-h; fixed overhead (setup, margin) ≈ 0.94 A100-h.

1. **Raise the envelope to fit cut a+b+c** (recommended): session ≈ 8.4 A100-h with the trace fix;
   G1 total ≈ 10.3 A100-h. Keeps the registered design at k = 2 for both models, both fields, the
   ladder on 100 scenarios, J1 and both forks.
2. **Raise it for the full design** (no cuts): session ≈ 14.4 A100-h; G1 total ≈ 16.3 A100-h.
3. **Stay inside the envelope at k = 1** (each model's first generation only, both fields, the
   ladder on 100, J1, both forks; generate only what is judged): ≈ 5 A100-h, fits. The cost is the
   match rule: its interval resamples our k generations per scenario, which k = 1 cannot do, so the
   rule would need an amendment (for example, generation noise from k = 2 on a 100-scenario subset).

Question behind the question: whether MoReBench's replication is worth a 30% envelope increase
before any steering use is decided. Option 1 keeps the registered design; option 3 keeps the
budget and changes the rule.

## 5. Decision (author)

Option 1, registered as MRB_PLAN v0.4 before the session: envelope 10.3 A100-h; k = 2 for both
models on both fields, the ladder on 100 scenarios, J1 and both forks. Repriced with real-trace
judge costs, T1 projects to 7.76 A100-h of the 8.44 left (`outputs/g1_validate/projection.json`;
the envelope-8 projection is kept beside it), under a wall cap that stops at the envelope.

## 6. The G1 session (2026-10-31, aborted, outputs lost)

Launched under v0.4 (decision a+b+c, tiers T1, wall cap 4.22 h). Read on the pod while it ran:

| stage | result |
|---|---|
| setup | 0.14 h; gate tests 13 passed, 1 skipped |
| GPT-oss-20b, k = 2 + fork | 1,100 of 1,100 in 8.1 min (6,688 tok/s); 1,099 reached the final answer, 2 hit the cap; no empty traces (MRB-A8 fix holds); median trace 7,616, response 6,135 characters |
| Qwen3-30B-A3B, k = 2 + fork | 1,100 of 1,100 in 16.8 min (2,356 tok/s); all reached the final answer; no empty traces; median trace 3,596, response 5,959 characters |
| MB Phase 2 rider | 176 rows |
| judge, first 16,384 calls | 8.1 calls/s, ~5,700 output tok/s; ladder 9,052 calls at 358 tokens, all verdicts; GPT-OSS responses 3,661 at 411, all verdicts; **GPT-OSS traces 3,671 at 1,837, 82% without a verdict** (MRB-A10) |

Projected from these rates, T1 would have run past the cap with Qwen about 30% judged, most of
the remaining time going to GPT-OSS trace calls without verdicts. The launcher was interrupted
(Ctrl-C) at about 06:20 UTC; its EXIT trap terminated the pod before the download, so no output
of this session survives. Spend: about 2.6 A100-h (pod life ~1.3 h on two GPUs; exact figure from
the RunPod console), so G1 has spent about 4.5 of the 10.3 A100-h envelope. The rerun's design for
the GPT-OSS trace field and its envelope are the author's decisions.

## 7. Decision for the rerun (author)

Run the registered design as planned, with the envelope raised to 15.5 A100-h (MRB_PLAN v0.5).
GPT-OSS traces under the registered prompt are judged last (T5). J1's 50-scenario diagnostic and
a delimited-prompt fork run in T1. A verdict guard and scenario-major order mean a failure or a
deadline costs one field's tail, not the session. Projection: 10.35 A100-h for T1+T5 against the
11.04 left, wall cap 5.52 h. The spend figure for the aborted session (2.6) is an estimate pending
the RunPod console.

## 8. G1 results (rerun, 2026-10-31; MRB_PLAN v0.5)

Session: 2 × A100-SXM4-80GB, 5.19 h of the 5.52 h cap, every planned call made (115,294 judge
calls; no deadline stop, no guard trip); outputs verified against the manifest (16 files).
Generation: 1,100 per model (k = 2 on 500 plus the fork's 100), all reached the final answer,
none at the cap, no empty traces. Spend about 10.6 A100-h; G1 total about 15.1 of 15.5. Numbers:
`outputs/g1/g1_analysis.json` (`scripts/mrb_g1_analyze.py`; judge rows tracked gzipped).

### 8.1 The registered match (k0, the 500, substring credit of record)

| cell | ours | paper | 95% PI (k = 2) | verdict | length (paper) |
|---|---|---|---|---|---|
| Qwen3-30B-A3B trace, Regular | 70.73 [69.33, 72.04] | 70.3 | [70.21, 71.11] | **replicates** | 3,681 (3,691) |
| Qwen3-30B-A3B trace, Hard | 19.22 | 19.0 | | **replicates** | |
| Qwen3-30B-A3B response, Regular | 74.99 [73.75, 76.19] | 75.4 | [74.53, 75.29] | misses by 0.06 (MRB-A12) | 5,992 (5,973) |
| Qwen3-30B-A3B response, Hard | 12.51 | 12.6 | | **replicates** | |
| GPT-oss-20b response, Regular | 72.41 [70.73, 73.99] | 71.7 | [72.03, 73.40] | misses by 0.28 (MRB-A12) | 6,169 (5,071) |
| GPT-oss-20b response, Hard | 11.74 | 14.1 | | misses (length, MRB-A11) | |
| GPT-oss-20b trace, Regular | 36.08 [34.20, 37.91] | 57.7 | [36.29, 38.17] | misses: judge failure (MRB-A10) | 8,215 (2,338) |
| GPT-oss-20b trace, Hard | 4.39 | 24.7 | | misses (MRB-A10, MRB-A11) | |

Brackets after our values: scenario-bootstrap 95% CI. Eq. 1 under the registered reading equals
the code score mapped to 0-100 on every unit (max gap 1.4e-14), so J0 and J0c share the column.
Exact-token credit (MRB-A3) changes only the failing field: GPT-OSS traces 13.8; elsewhere the
two tests differ on at most 2 calls in 9,052 (ladder) and 0 or 1 per 22,900 (model fields). The
400-scenario subset reads 0.4 to 3.2 points below the 500 on every cell (traces 32.9 and 69.7;
responses 72.0 and 74.2).

### 8.2 Instrument checks

- **Ladder (100 scenarios):** empty 8.9, generic paragraph 17.3, generic padded to model length
  15.1, the scenario restated 41.7. The models on the same scenarios: responses 70.2 (GPT-OSS)
  and 73.7 (Qwen), Qwen traces 69.0, 27 to 32 points above the best canned response. GPT-OSS
  traces under the registered prompt read 34.2, **below the restated scenario**: the judge's
  failure (MRB-A10), not a reading of the trace. Padding lowers the generic score (−2.2), so the
  ladder shows no reward for length alone.
- **J1 (50 scenarios, GPT-OSS k0):** responses: criterion agreement 0.938 (κ 0.86); Regular
  retest difference +1.15 (CI [−0.18, 2.52]). Traces under the registered prompt: agreement 0.72
  (κ 0.35), the failure again.
- **Judge fork (top-p 1, J1 subset):** responses +0.40 (CI [−0.89, 1.86]); no difference
  detectable at |Δ| ≳ 1.9 on 50 scenarios.
- **Generation fork (each model's own defaults, k = 1, 100 scenarios):** Qwen trace +0.18
  [−1.61, 2.07], response +0.11 [−1.53, 1.80]; GPT-OSS response +2.92 [−0.29, 6.48]; no
  difference detectable at about 2.4 (Qwen) or 4.8 (GPT-OSS response) points.
- **MDE (80% power, scenario-paired, 500):** a between-model Regular difference of 1.96 on
  responses and 3.2 on traces. The response difference Qwen − GPT-OSS is 2.58.
- **MRB-A10 resolved toward R_register:** on J1's 50 units, the delimited prompt leaves 3 of
  1,118 calls without a verdict (registered: 906), and Regular moves from 33.4 to 58.9 (+25.5,
  CI [18.4, 33.1]).

### 8.3 MB Phase 2 rider (GPT-OSS-20B, R2 at dose-0, effort medium; MB_PLAN Amendments 3-4)

176 rows, label mass ≥ 0.997 on every row. Expected scores (original / counterbalanced): MFQ
binary 45.4 / 53.0 (L −0.20, K −0.10); MFV binary 51.7 / 48.7 (L −0.22, K −0.20); MFQ comparative
9.8 / 10.2 (L 0.00, K −0.01); MFV comparative 12.1 / 11.8 (L 0.01, K 0.02). A lean toward B on
both binary sets, none on the comparative sets; read as a dose-0 readout (KDG-A22), banked as
the pre-steering baseline. Not a steering gate.

### 8.4 Referee pass

1. *"Your interval cannot fail a faithful reproduction only because it is narrow."* Conceded in
   part (MRB-A12): the PI carries generation noise alone (closed form 0.75 wide, measured 0.76);
   same-deployment judge noise adds about 0.16 SE, which reaches Qwen's response cell but not
   GPT-OSS's. The registered verdicts stand; the deployment term needs the OpenRouter cross-check.
2. *"Your GPT-OSS is not the paper's GPT-OSS."* Open (MRB-A11): our traces are 3.5× longer and
   responses 1.22×; every GPT-OSS miss sits on that gap or on the judge failure. Effort high was
   our decision of record (v0.3); the cross-check's generation length separates effort from
   provider.
3. *"The Qwen trace replication is one deployment and one draw."* Answered in part: Regular,
   Hard and length all agree (70.7 / 19.2 / 3,681 against 70.3 / 19.0 / 3,691), the generation
   fork does not move it (+0.18 [−1.61, 2.07]), and J1 judge noise is small. Conceded: one judge
   deployment until the cross-check.

### 8.5 Anticipated review

1. The paper's GPT-oss trace length (2,338) is a third of ours → MRB-A11 → OpenRouter generation
   on ~50 scenarios (cents, zero GPU).
2. Both response misses are within 0.3 of a generation-only PI → MRB-A12 → the registered
   cross-check's deployment term, as a sensitivity read under amendment.
3. The restated scenario scores 41.7 → a third of the rubric's weight is met by repeating the
   dilemma; reported as the floor beside every score (implemented: §8.2).
4. GPT-OSS's trace field is only measured on 50 scenarios with a prompt that works → a full-field
   delimited judge pass (judge-only pod, ~1.1 h judging, ~3 A100-h with setup; envelope +3).
5. Not planned before: the GPT-OSS response field scores 11 points above its own delimited trace
   (CI [−14.9, −7.4]); whether that gap is the paper's trace-vs-response gap (57.7 against 71.7,
   14 points) is answerable once the trace field is judged in full.

Implemented now: the analysis, the ladder floor in every comparison, the J1 judge-noise term,
MRB-A10 to A12. Open (with costs): the OpenRouter cross-check, judge and GPT-oss-20b generation on
~50 scenarios (about $2 of API, inside the $60); a full-field delimited judge pass (about 3
A100-h, outside the 15.5).

Question behind the question: whether MoReBench's GPT-oss rows are reproducible from open weights
at all. If OpenRouter's reasoning field is a shortened trace (MRB-A11 R_provider), the paper's
GPT-oss trace scores grade a different text from the raw analysis, and no pod run reaches them.

### 8.6 Decisions for the author

1. **The OpenRouter cross-check** (registered in v0.2, still pending): run it now, extended by
   one GPT-oss-20b generation per scenario on the same ~50 scenarios. Recommended first: it is
   cheap and it decides 2 and 3. It needs your OpenRouter key in your terminal.
2. **The GPT-OSS trace field:** report it from the 50-scenario delimited fork only, or judge the
   full field with the delimited prompt on a judge-only pod (~3 A100-h, the envelope to ~18.5).
   Recommended: decide after 1; under R_provider the full pass describes only our traces.
3. **MRB-A12:** keep the registered "misses" and add, under amendment, a sensitivity interval
   with the cross-check's deployment term, reported beside them.
4. **Steering target (registered as decided after G1):** if MoReBench becomes a steering target,
   the response field is the one this judge reads reliably on GPT-OSS (0 of 22,900 without a
   verdict, J1 κ 0.86, MDE 1.96). The steering baseline needs GPT-OSS-20B at effort medium (v0.3),
   a new cell either way.
5. **G2** (GPT-oss-120b, API rows, the theory set, `REPRODUCTION.md`): after 1-3.

### 8.7 Author decisions (2026-10-31; MRB_PLAN v0.6)

1. Cross-check approved and extended: GPT-oss-20b generated through OpenRouter at the paper's
   settings on J1's 50 scenarios, judged by the paper's OpenRouter judge with the paper's prompt;
   lengths and scores read against the 57.7 / 24.7 trace row.
2. No full delimited pass.
3. MRB-A12's sensitivity interval as proposed, the near-misses stated against the 1.96-point MDE.
4. The steering-target decision is deferred to a later steering decision; no work now.
5. A minimal G2 (the theory set on Qwen3-30B-A3B, the gpt-oss-120b row optional, API rows off) is
   proposed after the cross-check.

Scope until `REPRODUCTION.md` is written: reproducing the paper's results only. The raw outputs
(`outputs/g1/_raw/`, 487 MB) are copied to the ds-data drive with sha256 verified.

## 9. OpenRouter cross-check (MRB_PLAN v0.6; `outputs/g1_openrouter/`)

Upstream's own calls through OpenRouter on J1's 50 scenarios: 100 GPT-oss-20b generations and
6,708 judge calls (gpt-oss-120b; 19 providers as OpenRouter routed them). Tokens: 9.2 M prompt,
6.2 M completion, about $2-4 at list prices (exact figure in the OpenRouter dashboard).

**R1, the paper's GPT-oss-20b row through the paper's pipeline (50 scenarios, a reading):**

| field | length (paper) | Regular [CI] (paper) | Hard (paper) | without a verdict | our pod, same scenarios |
|---|---|---|---|---|---|
| trace | 7,512 (2,338) | 36.9 [31.4, 43.0] (57.7) | 4.9 (24.7) | 757 of 1,118 | 33.4 registered, 58.9 delimited; length 8,257 |
| response | 5,650 (5,071) | 70.1 [66.5, 73.9] (71.7) | 12.4 (14.1) | 0 of 1,118 | 70.0; length 5,665 |

The paper's pipeline, run today, reproduces our pod's GPT-OSS numbers, not the paper's trace row.
The response row's Regular agrees with the paper; its Hard misses on length (1.11×). The paper's
57.7 / 24.7 trace row is not reproduced by any run we can make (MRB-A11, narrowed to R_drift).

**R2, the deployment term (OpenRouter judge − pod judge, our pod's k0 generations):** GPT-OSS
response +1.38 [−0.47, 3.20]; Qwen response −0.32 [−2.03, 1.23]; Qwen trace +1.72 [0.03, 3.40];
GPT-OSS trace −2.10 [−5.75, 1.39] (both deployments fail there: 914 and 865 of 1,118 calls
without a verdict). Criterion agreement on the valid fields 0.91-0.92, against the pod's own
retest 0.94.

**R3, the near-misses (decision 3):** sensitivity intervals GPT-OSS response [68.78, 76.65] and
Qwen response [72.45, 77.37] contain the paper's 71.7 and 75.4: both replicate within the measured
deployment noise, beside the registered "misses". Against the 1.96-point MDE, the paper cells lie
0.28 (14%) and 0.06 (3%) outside the registered interval, and 0.71 (36%) and 0.41 (21%) from our
values.

**R4, MRB-A11:** traces 7,512 (upstream's top-level effort), 6,546 (with OpenRouter's reasoning
object), 8,257 (pod, effort high); the effort parameter takes effect and the reasoning field is
raw analysis, so neither an ignored effort nor a provider summary explains the paper's 2,338.

**A reading, not a verdict:** on the pod the judge's failure rate is flat across trace length
(79-83% in every bin from 2,000 characters up, 1,100 traces), so shorter traces alone would not
have spared the paper's run; on OpenRouter it rises with length (ρ 0.43; 42-50% below 3,000
characters, 13 traces), too few to settle it.

## 10. Proposal: a minimal MRB-G2 (decision 5)

- **Cell:** the theory set (150 rows: 30 dilemmas × 5 theories, 3,835 criteria) on
  Qwen3-30B-A3B, the model whose rows reproduce. Upstream's theory prompt (theory and definition,
  p. 31; `run_inferences_on_dilemmas_theory.py`), Table 4 sampling, k = 2 so the registered match
  rule applies, both fields, the pod judge of record with the registered prompt (Qwen traces judge
  cleanly: 0 of 22,900 without a verdict). A two-rung ladder (empty, restated) on the 150 rows.
- **Paper cells:** the theory-set table for Qwen3-30B-A3B, read from arXiv v2 at a zero-GPU step
  before the pod (citation rule); the match rule as G1's.
- **Cost:** about 15,300 judge calls plus 7,700 for the ladder (~10 M output tokens, ~0.55 h of
  judging at G1's rate); session ~1.2 h on two A100s, about 2.4 A100-h, plus a VALIDATE (~0.4):
  an envelope of about 3 A100-h.
- **Optional:** the gpt-oss-120b row, response field only (its trace field meets MRB-A10): 500 ×
  k = 2 generations and ~23,000 judge calls, about +2.5 A100-h.
- **Off:** API rows. **After G2:** `REPRODUCTION.md`.

