# MoReBench reproduction report (MRB-G2, 2026-11-01)

Paper: Chiu, Lee et al., "MoReBench: Evaluating Procedural and Pluralistic Moral Reasoning in
Language Models, More than Outcomes", arXiv:2510.16380v2 (ICLR 2026): Table 6 (thinking traces),
Table 7 (final responses) and the two MoReBench-Theory tables. Rules of record: `papers/MRB_PLAN.md`
v0.8. Numbers: `outputs/g1/g1_analysis.json`, `outputs/g1_openrouter/g1_openrouter_analysis.json`,
`outputs/g2/g2_analysis.json` (aggregates; judge rows tracked gzipped; raw text untracked, backed up
to the ds-data drive with sha256 verified). Earlier gates: `MRB_G0_REPORT.md`, `MRB_G1_REPORT.md`,
`MRB_G2_REPORT.md`. Data: MoReBench (CC BY 4.0), `morebench/morebench` at revision
`d900e89f942f784f8e0113935034b4f0541a0b55`, fetched at run time; no MoReBench text is in this report.

## 1. Verdict table (registered match rule; k0 of record; substring credit of record)

Match rule: the paper cell lies inside the 95% prediction interval of a one-generation-per-scenario
mean on our scenarios (our k = 2 generations resampled, scenarios fixed), widened by 0.05 for the
paper's rounding. Ours first, paper second. "Within deployment noise" is the registered sensitivity
read (MRB_PLAN v0.6): the same interval widened by the judge-deployment term the OpenRouter
cross-check measured, reported beside the verdict, never in place of it.

| gate | set (n) | model | field | cell | ours | paper | verdict | reason for a miss |
|---|---|---|---|---|---|---|---|---|
| G1 | public (500) | Qwen3-30B-A3B | trace | Regular | 70.73 | 70.3 | **replicates** | |
| G1 | public (500) | Qwen3-30B-A3B | trace | Hard | 19.22 | 19.0 | **replicates** | |
| G1 | public (500) | Qwen3-30B-A3B | response | Regular | 74.99 | 75.4 | misses by 0.06; replicates within deployment noise | the interval carries generation noise only (MRB-A12) |
| G1 | public (500) | Qwen3-30B-A3B | response | Hard | 12.51 | 12.6 | **replicates** | |
| G1 | public (500) | GPT-oss-20b | response | Regular | 72.41 | 71.7 | misses by 0.28; replicates within deployment noise | the interval carries generation noise only (MRB-A12) |
| G1 | public (500) | GPT-oss-20b | response | Hard | 11.74 | 14.1 | misses | length-driven: our responses are 6,169 characters against 5,071 (MRB-A11) |
| G1 | public (500) | GPT-oss-20b | trace | Regular | 36.08 | 57.7 | not reproduced, including through the paper's own pipeline | the judge answers GPT-OSS's raw analysis instead of grading it (MRB-A10); our traces run 3.5× the paper's (MRB-A11) |
| G1 | public (500) | GPT-oss-20b | trace | Hard | 4.39 | 24.7 | not reproduced, including through the paper's own pipeline | as above; the Hard miss is also length-driven: 8,215 characters against 2,338 (MRB-A11) |
| G2 | theory (150) | Qwen3-30B-A3B | trace | Overall | 74.16 | 73.7 | misses by 0.06; replicates within deployment noise | the interval carries generation noise only (MRB-A12) |
| G2 | theory (150) | Qwen3-30B-A3B | trace | Overall-LC | 14.96 | 15.5 | **replicates** | |
| G2 | theory (150) | Qwen3-30B-A3B | response | Overall | 75.55 | 75.7 | **replicates** | |
| G2 | theory (150) | Qwen3-30B-A3B | response | Overall-LC | 10.22 | 10.2 | **replicates** | |

Hard and Overall-LC are the same quantity (score / mean length × 1000). The theory set's five
per-theory cells per field (30 rows each, MDE about 7-8 points) are secondary: 7 of 10 replicate,
and each miss lies within 0.6 MDE of the paper (`MRB_G2_REPORT.md` §5.1).

**Verdict sentence.** A self-hosted pipeline (open-weight generation, gpt-oss-120b as judge, every
judgement saved) reproduces 6 of MoReBench's 12 registered cells for these two models under the
registered rule, and 9 of 12 within the measured judge-deployment noise. Qwen3-30B-A3B reproduces
on both of the paper's instruments, the 500 public scenarios and the 150-row theory set, its lengths
within 4% of the paper's on every field. GPT-oss-20b's trace row is not reproduced, including
through the paper's own pipeline: the paper's code run through OpenRouter today gives 36.9 [31.4,
43.0] and Hard 4.9 on 50 scenarios, with traces of 7,512 characters, against 57.7, 24.7 and 2,338
(MRB-A10, MRB-A11). GPT-oss-20b's response Regular agrees with the paper (within deployment noise
on the pod, 70.1 [66.5, 73.9] through the paper's pipeline); its Hard misses are length-driven.

## 2. Readouts beside the match

- **Ladder (canned responses through the same judge).** Public set (100 scenarios): empty 8.9,
  generic paragraph 17.3, generic padded to model length 15.1, the scenario restated 41.7. Theory
  set (150 rows): empty 10.1, restated 42.1. Every valid model field reads 27-34 points above the
  restated scenario; a third of the rubric's weight is met by restating the dilemma.
- **Judge test-retest (J1, GPT-OSS k0 responses, 50 scenarios).** Criterion agreement 0.938 (κ
  0.86); Regular retest difference +1.15 [−0.18, 2.52].
- **Judge deployment (OpenRouter − pod, 50 scenarios).** GPT-OSS response +1.38 [−0.47, 3.20],
  Qwen response −0.32 [−2.03, 1.23], Qwen trace +1.72 [0.03, 3.40]; criterion agreement 0.91-0.92.
- **Forks (each registered, reported on its subset).** Judge top-p 1 against 0.95: +0.40 [−0.89,
  1.86] on GPT-OSS responses. Generation at each model's own defaults against Table 4: Qwen trace
  +0.18 [−1.61, 2.07], response +0.11 [−1.53, 1.80]; GPT-OSS response +2.92 [−0.29, 6.48]. Eq. 1 under
  the registered reading equals the code's 0-100 score on every unit (max gap 1.4e-14).
- **Credit test (MRB-A3).** Upstream credits a criterion by substring. The exact-token test changes
  only the failing field (GPT-OSS traces 13.8 against 36.1); elsewhere the two differ on at most 2
  calls in 9,052.
- **The 400 CC BY 4.0 scenarios (DailyDilemmas, AIRiskDilemmas), public set.** Qwen trace 69.71,
  response 74.17; GPT-OSS response 72.03, trace 32.91; 0.4-3.2 below the 500.
- **Smallest detectable difference (80% power, paired, 500 scenarios).** 1.96 points between models
  on responses, 3.2 on traces. The three near-misses lie 3%, 14% and 2% of the relevant MDE outside
  the registered interval.
- **Delimited judge prompt (MRB-A10 fork, 50 scenarios).** Upstream's text unchanged, the response
  wrapped in `<response>` tags with a note: GPT-OSS traces go from 906 to 3 of 1,118 calls without
  a verdict, and Regular from 33.4 to 58.9 (+25.5 [18.4, 33.1]).

## 3. Deviations and their reasons

- **Judge deployment.** gpt-oss-120b self-hosted (vLLM 0.31.0, tensor-parallel over two A100s) as
  the judge of record, with the paper's prompt and upstream's call settings; the paper used
  OpenRouter. The OpenRouter cross-check measures the difference (§2).
- **Judge sampling.** Table 4's gpt-oss row (T 1.0, top-p 0.95) is the default; the code's top-p 1
  is the registered fork (§2).
- **Generation.** Table 4 sampling (gpt-oss T 1.0 / top-p 0.95; Qwen T 0.6 / top-p 0.95), 10,500
  tokens; GPT-oss-20b at effort high, the setting the code sends where the paper is silent. The code
  sets no sampling; that fork changes no verdict (§2).
- **Two generations per scenario, not one.** k = 2 so the match interval carries generation noise;
  the first generation is the paper-faithful one and is the value of record. The registered cut
  order trimmed the plan's k = 4 and the 500-scenario ladder to fit the envelope (MRB_PLAN v0.4).
- **GPT-OSS trace split.** The shared harmony split returned an empty trace when GPT-OSS opened its
  own analysis channel (MRB-A8); MRB's driver splits by channel markers, and the shared helper was
  fixed afterwards with goldens.
- **Verdict guard and judge order (v0.5).** Unit types past 5% without a verdict are skipped (none
  tripped); GPT-OSS traces under the registered prompt were judged last and exempt.
- **Theory-set ladder.** Two rungs (empty, restated) instead of the public set's four.

## 4. Rows not run, and why

- **GPT-oss-120b (paper rows 50.8 / 39.9 trace, Table 7 response).** Off by the author's decision
  (v0.7): its trace field would meet the judge failure of MRB-A10. Its trace length (1,272 in the
  paper) was the discriminator for MRB-A2, which stays open.
- **API rows (GPT-5, Claude, Gemini, DeepSeek, Qwen3-235B).** Off by the author's decision.
- **The private set.** Never sought.
- **The 100 expert-written public scenarios** were evaluated, never trained on or adapted (MRB-A7).

## 5. Pins

Data `morebench/morebench@d900e89f942f784f8e0113935034b4f0541a0b55` (sha256 manifest). Code of
record: the `eval` branch `e6689bbb9d742c48ea78ea07acd7931eba05b213` (prompts vendored byte for byte
under MIT, `deepsteer/benchmarks/moral_reasoning/morebench/`); `main` `71eaf71857fecf57e20e30fb40416d843fdd473b`
recorded. Models: `openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee`,
`Qwen/Qwen3-30B-A3B-Thinking-2507@144afc2f379b542fdd4e85a1fcd5e1f79112d95d`; judge
`openai/gpt-oss-120b@b5c939de8f754692c1647ca79fbf85e8c1e70f8a`. Stack: vLLM 0.31.0 cu129 (wheel
sha256 `5176ab74…ca6f5f`), torch 2.13.0+cu129, transformers 5.17.0, sampling in PyTorch. Paper
cells read from the arXiv v2 source. OpenRouter cross-check: `openai/gpt-oss-20b` and
`openai/gpt-oss-120b` as routed on 2026-10-31 (19 providers, logged per call).

## 6. Cost

G1: VALIDATE about 1.9 A100-h (three runs, two of which found runner defects that were then fixed);
the first session about 2.6, lost to the launcher's Ctrl-C trap before download; the rerun about
10.6; G1 about 15.1 of its 15.5 A100-h. OpenRouter cross-check: 9.2 M prompt and 6.2 M completion
tokens, about $2-4. G2: VALIDATE 0.9 and the session about 2.1, about 3.0 of its 3.9 A100-h. In all
about 18 A100-h and under $5 of API.

## 7. Parked for steering (no work before a steering decision)

- **Candidate target: the response field.** The judge grades it reliably on both models (0 of
  22,900 GPT-OSS response calls without a verdict; J1 κ 0.86; MDE 1.96 between models).
- **Baseline: GPT-OSS-20B at medium effort**, its effort of record for steering, judged under both
  prompts (the registered prompt and the delimited one), so a steered score can be read against
  each.
- **The delimited judge prompt** (`judge_prompt_delimited`, `scripts/mrb_g1_pod.py`) as the trace
  field's working instrument on GPT-OSS.
- **Saved traces for activation reads.** Every generation's trace and response is kept (untracked
  `_raw/`, on ds-data), keyed by scenario, generation and hash, so activation reads at trace
  positions need no new generation. Model outputs are for evaluation only (MRB_PLAN §9).

## 8. Referee pass

1. *"Six of twelve is not a reproduction."* Under the registered rule, six replicate; three more
   miss by 0.06-0.28 against an interval that carries generation noise only, and replicate once the
   measured judge-deployment term is added (beside, not in place of, the verdict). The remaining
   three are GPT-oss-20b's trace row and response Hard, each with a located cause (MRB-A10, MRB-A11)
   and the trace row checked through the paper's own pipeline.
2. *"The trace field failure is your harness."* Upstream's own judge call through OpenRouter fails
   the same way on GPT-OSS traces (914 of 1,118 calls on our traces, 757 of 1,118 on traces
   generated through OpenRouter at the paper's settings), and Qwen's traces are graded cleanly (0
   of 22,900). Conceded: the paper's original GPT-oss traces are not released, so why they were a
   third as long, and whether the judge graded them, cannot be checked (MRB-A11, R_drift).
3. *"Qwen's agreement could rest on a narrow interval and one judge deployment."* Regular, Hard and
   length agree on both instruments; the generation fork does not move the public-set trace score
   (+0.18 [−1.61, 2.07]); the OpenRouter judge differs from ours by 1.7 points or less on Qwen's
   fields. Conceded: the deployment term was measured on the public set, not on theory rows.

## Attribution

MoReBench data: Chiu, Lee et al., `huggingface.co/datasets/morebench/morebench` at revision
`d900e89`, CC BY 4.0 (creativecommons.org/licenses/by/4.0/); no MoReBench text is reproduced here,
only aggregate scores. MoReBench code: MIT, Copyright (c) 2025 MoReBench (`LICENSE_MOREBENCH`).
