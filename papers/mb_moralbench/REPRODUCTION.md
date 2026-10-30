# MoralBench reproduction report (MB-G2, 2026-10-30)

Paper: Ji, Chen, Jin, Xu, Hua, Zhang, "MoralBench: Moral Evaluation of LLMs", arXiv:2406.04428
(Tables 1–2, identical in v1 and v2); SIGKDD Explorations 27(1), 2025. Rules of record:
`papers/MB_PLAN.md` v0.7. Numbers: `outputs/g1/<model>/` and `outputs/g2/<model>/g1_analysis.json`
(aggregates; per-foundation cells there). Earlier gates: `MB_G0_REPORT.md`, `MB_G1_REPORT.md`.

## 1. Replication table (T = 0.7 of record; R0 with misses scored 0; totals)

| paper row | model of record | MFQ binary | MFV binary | MFQ comparative | MFV comparative | paper cells: replicate / not / not attainable (26) |
|---|---|---|---|---|---|---|
| Zephyr | `zephyr-7b-beta` (M4) | 22.3 vs 54.2 | 41.1 vs 48.1 | 7.1 vs 8.2 | **11.4 vs 10.4** | 5 / 20 / 1 |
| LLaMA-2 (70B) | `Llama-2-70b-chat-hf` (pod) | 42.2 vs 58.5 | 51.7 vs 52.6 | 9.3 vs 8.0 | **11.5 vs 13.2** | 9 / 16 / 1 |
| Gemma-1.1 (7B) | `gemma-1.1-7b-it` (pod) | 55.3 vs 49.9 | 52.0 vs 44.4* | 8.2 vs 9.6 | **11.9 vs 10.8** | 7 / 19 / 0 |
| GPT-3.5 | `gpt-3.5-turbo-1106` | not replicable: shut down 2026-09-28 | | | | |
| GPT-4 | `gpt-4-0613` | not replicable: shut down 2026-10-23 | | | | |

Ours first, paper second; bold where the paper's value lies inside our 95% prediction interval
for a 5-repeat mean (items fixed, plus 0.05 rounding). *Gemma's printed MFV total is an
arithmetic error in the paper: its cells sum to 51.8 (MB-A1, resolved). The T = 1.0 fork gives
7 / 18 / 1, 11 / 14 / 1 and 7 / 19 / 0 and changes no total's verdict except Llama-2's MFV
comparative, which falls just outside (11.1).

**Verdict sentence.** On the pinned key, a local harness reproduces MoralBench's paper values on 21
of 76 attainable cells (78 in all) across the three open models. MFQ binary misses on all three, in both directions: our
agree rates are 0.22 / 0.12 / 0.90 against the paper's implied 0.76 / about 1.0 / 0.49. The
MFV-comparative totals replicate on all three, which is weak evidence: a random letter policy's
interval covers those paper values 54–65% of the time (MB-A8 calibration).

## 2. Readouts beside R0 (totals, T = 0.7)

| model | set | R0 (miss 0) | R0 misses (unstripped) | parsed-only | R1 | R2 | A-rate (R2) |
|---|---|---|---|---|---|---|---|
| Zephyr | MFQ binary | 22.3 | 296 (298) / 640 | 43.8 | 39.5 | 44.3 | 0.22 |
| LLaMA-2 | MFQ binary | 42.2 | 4 (640) / 640 | 42.4 | 42.3 | 43.8 | 0.12 |
| Gemma | MFQ binary | 55.3 | 0 (0) / 640 | 55.3 | 55.3 | 55.2 | 0.90 |

The other sets are in the JSON. Ladder band per set: MFQ binary [37.6, 62.4], MFV binary
[40.8, 55.2], MFQ comparative [1.0, 20.0], MFV comparative [0.0, 24.0] (`MB_G0_REPORT.md`).

## 3. Deviations and their reasons

- **Temperature 0.7.** This is the paper's text (v2 p. 5, §4.1). `main.py` sends no temperature
  (API default 1.0); 1.0 is the registered fork.
- **R0 strips leading whitespace.** Every Llama-2 reply begins with a space. Unstripped, R0 misses
  2,816 of 2,816 and Llama-2 scores 0 on every set (MB-A7). The paper does not state its decode.
- **System role.** The prompt is the native system message for Zephyr and Llama-2, and system text
  prepended to the user turn for Gemma, whose template has no system role.
- **Engine.** HF transformers 5.17.0 for every row, with the same sampler (seed per item, pure
  temperature, warpers off). The plan's vLLM for the 70B gave way to harness parity (Amendment 2).
  Zephyr ran on the M4 (MPS); Gemma and Llama-2 on A100s (CUDA, torch 2.6.0).
- **Rollouts.** 32 per item; the paper's 5 repeats are the PI's unit.
- **Item set.** 88 items as pinned. MoralBench's keys carry defects: a tied comparative item, a
  duplicate pair with opposite keys, and the MFV-comparative fairness group keyed to the more-wrong
  statement against Clifford's norms (MB-A4).
- **Not attainable.** LLaMA-2 MFV sanctity 11.1 (above the key's oracle 9.9; MB-A2) and Zephyr
  MFQ-comparative loyalty 0.4 (below the tied item's floor; MB-A3) are excluded from the counts.

## 4. Pins

MoralBench `84100440a28d040eca27df32b93f44685ad69ce5` (sha256 manifest; Drive mirror
byte-identical). Models: `HuggingFaceH4/zephyr-7b-beta@892b3d7a`,
`meta-llama/Llama-2-70b-chat-hf@e9149a12`, `google/gemma-1.1-7b-it@065a5287`. Diagnosis variants:
`zephyr-7b-alpha@014792bb`, `zephyr-7b-gemma-v0.1@03b3427d`. Code at `cd79aee` for the pod. Stack:
transformers 5.17.0; torch 2.14 (M4) and 2.6.0+cu124 (pod).

## 5. Cost

M4: 0 GPU (Zephyr's three cells, 100 min; D1–D3, about 35 min each). Pod: three VALIDATE runs on one
A100 (about 25 min in all; two found runner defects that were then fixed) and the session on
2 × A100 (about 40 min: Gemma 2 min, D4 5 min, Llama-2-70B 21 min), about 1.8 A100-h of the
4 A100-h cap. API: $0 (the closed rows are retired).

## 6. Referee pass (G2)

1. *"Llama-2's misses come from your template, not the model."* The template is Llama-2-chat's own
   (`e8d5b5a5…`), with the native system role. MFQ binary misses with every readout (R0 42.2,
   R1 42.3, R2 43.8 against 58.5), and 58.5 sits above the always-Agree score (58.2). The paper's
   LLaMA-2 must have agreed with essentially every MFQ statement; ours agrees with 12%.
2. *"Three of the paper's LLaMA-2 MFV cells equal ours exactly, so the row replicates."* They equal
   the always-B policy's scores (9.5, 8.3, 8.3), and all three families answer B on MFV binary
   whatever the label (MB-A5). Matching there shows a shared letter preference, not shared moral
   judgments.
3. *"Two devices and one transformers version can still differ."* Conceded as a residual: device
   parity was checked only on 3 near-deterministic Gemma items (identical letter distributions).
