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

## 4. Decision (author, 2026-11-01; MRB_PLAN v0.8)

Option 1: the G2 envelope raised to 3.9 A100-h. The projection recomputed locally from VALIDATE's
files with the new envelope (rates unchanged): session 2.98 A100-h against the 3.0 left,
**proceed**, wall cap 1.50 h (clipped to the envelope; the registered work without its margin ends
at about 1.24 h). VALIDATE's projection is kept as `outputs/g2_validate/projection_validate.json`.

## 5. G2 results (session 2026-11-01 05:17-06:15 UTC; MRB_PLAN v0.8)

Session: 2 × A100-SXM4-80GB, 0.96 h of the 1.50 h cap; every planned call made (23,010; no
deadline stop, no guard trip); manifest verified (9 files). Generation: 300 of 300 reached the
final answer, none at the cap, in 7.3 min. Spend about 2.1 A100-h; G2 total about 3.0 of 3.9.
Numbers: `outputs/g2/g2_analysis.json` (`scripts/mrb_g2_analyze.py`; judge rows tracked
gzipped; raw text on ds-data, sha256 verified).

### 5.1 The registered match (k0, the 150 rows, substring credit of record)

| cell | ours | paper | 95% PI (k = 2) | verdict | sensitivity (δ) |
|---|---|---|---|---|---|
| trace, Overall | 74.16 [72.07, 76.14] | 73.7 | [73.81, 75.30] | misses by 0.06 | [70.36, 78.75]: replicates |
| trace, Overall-LC | 14.96 | 15.5 | [15.01, 15.81] | **replicates** | |
| response, Overall | 75.55 [73.58, 77.60] | 75.7 | [74.74, 76.09] | **replicates** | [72.66, 78.17]: replicates |
| response, Overall-LC | 10.22 | 10.2 | [10.11, 10.37] | **replicates** | |

Lengths: traces 4,956 characters (paper 4,766), responses 7,390 (paper 7,396). The trace miss is
1.7% of the 3.57-point MDE outside the interval, and our value lies 0.46 from the paper's (13% of
the MDE). Exact-token credit gives the same verdicts (it differs from substring credit only on the
ladder, by at most 0.6). The paper's own rows pass the table-arithmetic audit (Overall = mean of
the five theories; Overall-LC = Overall / length × 1000).

**Per-theory cells (secondary; 30 rows each, MDE about 7-8 points):** 7 of 10 replicate. The three
misses: trace Act Utilitarianism 78.23 against 76.7 (PI [77.90, 80.67]); response Gauthierian
72.37 against 69.2 (PI [70.06, 73.22]); response Aristotelian 74.47 against 78.5 (PI [74.23,
76.89]). Each lies within 0.6 MDE of the paper, under an interval that carries generation noise
only (MRB-A12). Aristotelian Virtue Ethics reads below the paper on both fields (MRB-A5 reading).

### 5.2 Instrument checks

- **Ladder (150 rows):** empty 10.1, the scenario restated 42.1; Qwen's fields read 74-76, 32-34
  points above the restated scenario. The theory set's ladder sits where G1's did (8.9 and 41.7).
- **Verdicts:** traces 2 of 7,670 without a verdict, responses 0 of 7,670, the ladder 126 of 7,670
  (1.6%).
- **Variance:** scenario SD 12.8 (trace) and 12.9 (response), below G1's public-set 15.6 and 13.8
  that set the power table, so the registered intervals are as wide as planned or narrower.

### 5.3 Referee pass

1. *"The trace Overall missed."* Conceded under the registered rule (0.06 outside). The interval
   carries generation noise only (MRB-A12); with the judge-deployment term measured on Qwen's trace
   field it replicates, and the trace's Overall-LC replicates under the registered rule.
2. *"Three of ten per-theory cells miss; the theory structure may not reproduce."* Answered in
   part: each miss lies within 0.6 of the per-theory MDE, the per-theory intervals carry generation
   noise only over 30 rows, and the Overall cells, which average the five, replicate on both fields
   (one within deployment noise). Conceded: per-theory agreement is not tested at a useful MDE.
3. *"One judge deployment again."* Answered for the deployment term: v0.6 measured OpenRouter −
   pod on Qwen's fields (+1.72 trace, −0.32 response on the public set), and the theory set uses the
   same judge and prompt. Conceded: the term was measured on the public set, not on theory rows.

### 5.4 Anticipated review

1. A third 0.06-point near-miss at the registered bar → MRB-A12, the sensitivity interval as
   registered; no new cell.
2. Virtue Ethics below the paper on both fields → MRB-A5 reading; no separating cell inside the
   replication (the per-theory MDE is about 8).
3. Not planned before: the ladder's restated-scenario rung reads 42.1 on the theory set, where the
   rubric targets a named theory; reporting it beside every theory-set score in `REPRODUCTION.md`
   (zero cost).

Question behind the question: whether the reproduction report now states what replicates. It
does for Qwen3-30B-A3B on both instruments; GPT-oss-20b's trace row is not reproduced, including
through the paper's own pipeline (MRB-A10, MRB-A11).

### 5.5 Next (author)

`REPRODUCTION.md`, with the author's wording for the GPT-OSS rows (v0.7) and the theory-set cells
above. No further pod work is planned for MRB.

