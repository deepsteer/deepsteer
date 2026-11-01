# MRB-G2 gate report

Rules: `papers/MRB_PLAN.md` v0.7 (minimal G2: the theory set on Qwen3-30B-A3B). Numbers:
`outputs/g2_validate/` (projection.json, the generation and judge summaries and rows).

## 1. VALIDATE (2 × A100-SXM4-80GB, 0.40 h wall, 2026-11-01 03:16-03:39 UTC)

| check | result |
|---|---|
| stack | vLLM 0.31.0 cu129, torch 2.13, transformers 5.17.0; smoke OK; gate tests 10 passed, 3 skipped (cached-tokenizer tests) |
| generation | 15 rows × 2 (every tenth row, all five theories): 30 of 30 reached the final answer, none at the cap; traces 5,355 characters (paper 4,766), responses 7,667 (paper 7,396); 805 tok/s at 30 concurrent requests |
| judge | 2,382 calls at 4,786 output tok/s, no guard trip; without a verdict: Qwen traces 0 of 794, responses 0 of 794, the ladder 8 of 794 (1.0%) |
| outputs | manifest verified (11 files); raw text backed up to ds-data, sha256 verified |

## 2. Projection and the registered decision

Session: setup 0.23 + generation 0.30 + judge load 0.05 + judge 0.66 (11.3 M output tokens:
canned 475, trace 548, response 452 per call) + margin 0.25 = 1.49 h on two GPUs, **2.98 A100-h**.
VALIDATE spent 0.90. Remaining under the registered 3.2: 2.30. **Decision: STOP.**

Correction: the projection on the pod ran against 3.0 A100-h (the driver's default), not the
registered 3.2, because the runner did not pass the envelope; the decision is STOP under either.
The driver's default is now 3.2.

Second derivation (a reading, not the registered projection): at G1's full-session rates
(generation 2,356 tok/s over 1,100 jobs, judge 5,382 tok/s) the session is 1.21 h, 2.42 A100-h,
still 0.12 over the remainder. VALIDATE's generation rate is low because 30 requests do not fill
the engine; the registered projection keeps it.

## 3. Options for the author

1. **Raise the G2 envelope to 3.9 A100-h** (recommended): the registered design and projection
   unchanged (0.90 spent + 2.98 projected); the wall cap clips the session to the envelope. At
   G1-scale rates the run would spend about 3.3.
2. **Keep 3.2 and cut the ladder:** saves about 0.4 A100-h (7,670 calls), session about 2.55,
   still over 2.30; a cut v0.7 did not register. Not recommended.

Either way the session's projection is recomputed locally from VALIDATE's files with the new
envelope (zero GPU), registered and pushed before the launch line.
