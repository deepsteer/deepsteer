# MRB-G1 gate report: VALIDATE passed, the registered rule says STOP

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
