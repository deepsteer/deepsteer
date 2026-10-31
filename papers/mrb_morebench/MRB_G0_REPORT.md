# MRB-G0 gate report

Zero GPU, zero API. Rules: `papers/MRB_PLAN.md` v0.2. Numbers: `outputs/mrb_g0_audit.json`. Source
facts: `SOURCE.md`. Anomalies: `papers/ANOMALIES.md` MRB-A1 to MRB-A7.

## 1. Checklist (MRB_PLAN §5, MRB-G0)

| item | result |
|---|---|
| 1. pin and fetch | code of record `eval` `e6689bb` (`main` `71eaf71` recorded) and HF revision `d900e89` in the user cache; sha256 manifest of the three data files |
| 2. license | code MIT, data CC BY 4.0, not gated. CC BY 4.0's coverage of the 100 expert-written scenarios is unclear: their sources' own terms are CC BY-NC-ND or none (MRB-A7; decision 1 below) |
| 3. counts and schema | 500 public rows, 11,450 criteria (20–47 per row); theory 150 rows = 30 dilemmas × 5 theories, 18 dilemmas shared with the public set; 23 criteria labelled "other" |
| 4. paper vs code; table arithmetic | Hard reproduces on all 28 rows of Tables 6 and 7; every row's Regular matches the public file's split sizes; `main` expects 11,568 judgements against the file's 11,450 (MRB-A1); Table 6's Average row matches no subset of rows (MRB-A4); a repeated length of 1,272 (MRB-A2); text vs Tables 2 and 10 (MRB-A5); judge validated on responses, headline judges traces (MRB-A6) |
| 5. judge and scoring in the library | `deepsteer/benchmarks/moral_reasoning/morebench/`: prompts byte-identical to upstream's, the code score equal to upstream's own function on 5,000 synthetic sets (0 mismatches), Eq. 1 = 2 × code / 100 − 1 to machine precision; upstream's substring credit test reproduced, traps included (MRB-A3) |
| 6. cost table | §3 below |
| 7. file layout | §4 below |
| license check | no MoReBench scenario or criterion text in the files this gate adds or changes, or in the built wheel (16,012 text lines scanned); the wheel carries the loader, the hash manifest, the vendored instruction strings, the scorer and the MIT license |

## 2. What the audit changes for G1

- **The criterion count is unexplained** (MRB-A1): the paper-era code expects 118 more judgements
  than the public file yields, on scenarios with the same split sizes. 11,568 + 11,450 = 23,018,
  the full dataset's count, so the default may be stale or the private half's; G1's verdict
  sentences name the discrepancy without assuming the paper used other rubrics.
- **The credit test is a substring match** (MRB-A3). G1 reports the score under upstream's test
  (of record) and under an exact-token test beside it, with the share of non-bare judge replies.
- **gpt-oss reasoning effort for generation is not in the paper** (Table 4 lists temperature and
  top-p only). The code runs gpt-oss through OpenRouter with effort "high". By the v0.2 rule (paper
  text, then code where the paper is silent) the default is high.

## 3. Cost (G1 as approved: 2 models, 500 scenarios, k = 4, both fields, the ladder)

| cell | count | estimate | basis |
|---|---|---|---|
| VALIDATE (stack, gpt-oss-120b loading and judging, generation smoke) | — | 0.3–0.5 A100-h | prior VALIDATE pods |
| generation, GPT-oss-20b and Qwen3-30B-A3B | 4,000 generations, ~8.6 M output tokens | 1–1.5 A100-h | Table 6/7 lengths (≈1.9k and 2.4k tokens per generation), vLLM throughput 3–6k tok/s |
| judge of record, self-hosted gpt-oss-120b | 183,200 calls (11,450 × 2 fields × 4 × 2 models) + 45,800 for the ladder | **2.5–26 A100-h** | 120–625 output tokens per call (inferred from the paper's $1.91 for 7,176 calls at plausible 2025 and current prices), 1.5–3k tok/s on one 80GB GPU; prefill mostly cached (the response prefix is shared across a scenario's criteria) |
| OpenRouter cross-check, ~50 scenarios × 2 fields × 2 models | ~4,600 calls | ≤ $3 | $0.15 / M in, $0.60 / M out |
| MB Phase 2 rider, GPT-OSS-20B | 88 items × 2 orders, forward only | < 0.1 A100-h | MB's allowance |

The judge is the whole uncertainty: the envelope (8 A100-h) holds at the low end of the range and
not at the high end. The per-call output length at effort high is unknown until VALIDATE measures
it, so a cut order is needed before the pod (decision 2).

## 4. File layout

- `deepsteer/datasets/morebench.py` and `morebench_manifest.json`: loader at the pinned revision;
  ids `mrb-<idx>` from the CSV row index, as upstream's `idx`.
- `deepsteer/benchmarks/moral_reasoning/morebench/`: `prompts.py` (vendored instruction strings,
  MIT notice), `scoring.py` (Eq. 1 of record, the code score fork, upstream aggregation and
  rounding), `LICENSE_MOREBENCH`. The judge client and `MoReBenchBenchmark` join this package at G1.
- `papers/mrb_morebench/`: `SOURCE.md`, this report, `scripts/mrb_g0_audit.py`, `outputs/`
  (tracked: ids, counts, scores) and `outputs_raw/` (gitignored: generations, judge replies).

## 5. Decisions for the author

1. **The expert-written 100 (MRB-A7).** Recommendation: keep them in the replication (evaluation
   only, no text committed, scores are ours), and exclude them from any training or adaptation use
   (ST as a target) unless the sources grant permission. Also correct `MORAL_EVAL_REPLICATIONS.md`,
   which states the upstream sources are CC 4.0.
2. **Judge cut order if VALIDATE prices the judge past the envelope.** Recommendation, applied in
   order until the projection fits: (a) the ladder on 100 scenarios instead of 500; (b) the final
   response judged on the first generation only (k = 1), traces kept at k = 4; (c) traces at k = 2.
   Each cut is registered now, before data, with the information it costs (a: the ladder band's
   precision; b: response-field generation noise; c: half the trace-field noise estimate).
3. **gpt-oss generation effort high** as the default (§2 above), with medium as a fork only if you
   want one.
4. **Ask the authors** about MRB-A1 and MRB-A4 (zero cost, could close both).

**Author decisions (MRB-G0 gate).** 1 approved (evaluation only; also report the 400
DailyDilemmas and AIRiskDilemmas scenarios as a subset beside the 500). 2 replaced by the author's
cut order: (a) the ladder on 100 scenarios; (b) Qwen3-30B-A3B at k = 2 on both fields; (c)
GPT-OSS-20B at k = 2 on both fields; the response field at parity with traces throughout; stop and
report if it still overruns. 3 approved, with no fork; the steering baseline will need GPT-OSS-20B
at its effort of record (medium). 4: the author has emailed the authors about MRB-A1, A4 and A7.
J1 test–retest is part of G1.

## 6. Anticipated review

1. The judge's cost range spans a factor of ten, so the envelope is not yet a real constraint.
   Under compute-ordering, VALIDATE has to measure tokens per call before any judging, and the cut
   order (decision 2) is registered now (0 GPU).
2. MRB-A1's explanation is unknown; if the paper's rubrics differ from the public file's, no match
   tolerance absorbs it, so G1's verdict sentences carry the discrepancy (0 GPU; wording at G1).
3. The substring test's traps may move scores. Under instrument-calibration, report both tests
   and the non-bare share (0 GPU after G1).
4. Rider not in the plan: judge determinism. Self-hosted gpt-oss-120b at T 1 still varies between
   calls, so J1 (test–retest) is the only measure of judge noise, and k = 4 mixes it with
   generation noise. Recommendation: make J1 part of G1, not opt-in (about 1,150 extra calls).

Question behind the question: whether a rubric score judged by one open model at T 1 is stable
enough to be a steering target at all. J1 and J2 at G1 answer that before the target decision.
