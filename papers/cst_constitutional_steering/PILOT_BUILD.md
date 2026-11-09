# CST pilot ladder build (2026-11-09; Amendments 20-21)

The pilot ladder reads its rungs on the screen read and gates S1 (Amendment 20 items 1-4). Pods split
by line (Amendment 21 item 1): an SDF pod and a DPO pod, each training and reading its own adapters,
each at or under 10.5 h, launched the same night after S0b-1's A0n read is in. This file is in
`S0B_BUILD.md`'s form. Nothing here has run; §6 lists what is not built.

**Measured rates the prices use:** SDF/DPO training 410.7 tokens/s (VALIDATE, rank 16, 496 modules:
1M tokens = 0.68 h); generation with a 34k-token document in context 337 tokens/s at 32 sequences,
without it about 1,000 (S0a's chunk logs); an A0-length read 5,300-8,200 rollouts/h (S0a); a vLLM
load 0.03-0.07 h (MRB P2), an HF training load about 0.1 h; gpt-oss-120b at effort low about 14,000
calls/h (CST-A12).

## 1. Settings of record

S0a's subject and engine (`S0A_BUILD.md` §1). LoRA rank 16, alpha 32, dropout 0, the 496 VALIDATE
targets, HF bf16 with gradient checkpointing for training. The read on vLLM native LoRA
(`enable_lora`, MTP off), with the 32-prompt parity check per adapter kind before its read (VALIDATE
V4: TV(served, attached) below TV(base, attached); measured 0.0125 against 0.1115).

## 2. The screen read and its power (both pods)

Per adapter: the operator and plain cells on the 458 clear items at k 2 (1,832 rollouts), the
legitimate-order cell at k 4 (about 400), cell A′ at k 8 (816), trace length; no judge (Amendment 20
item 2). About 3,050 rollouts, priced at 0.45 h each with LoRA serving. Capability at n 200 only on
rungs that move (about 0.1 h each, a contingency outside the price). Each pod also reads A0 on the
legitimate-order cell (the 20-item check first; a cell at floor is rebuilt, not read).

| contrast | MDE (80%, two-sided 0.05) | source |
|---|---|---|
| rung − A0 (A0 at its saved k 4) | 0.042 | `outputs/t04d_ordered_power.json` `pilot_k2` |
| rung − its control, both at k 2 | 0.049 (0.058 with A0s's scenario heterogeneity) | same; A0's half-split check 0.048 |
| cell A′ originals' shift | 0.082 (SE 0.029, 51 pairs, k 8) | `T04_POWER.md` §10 |
| legitimate-order share, rung − A0 at k 4 | at most 0.099 (worst-case variance, n 100) | closed form; re-read on the 20-item check |

A0s's band is 0.280, so each pilot contrast resolves a rung carrying about a sixth of it.

## 3. Branches (Amendment 20 items 3-4, Amendment 21 items 3, 6, 7; written before data)

| result | reading | next |
|---|---|---|
| a document rung's gain CI excludes 0 against A0 and against R0, and D-doc's against D-neutral | the document's content moves the action at 1M tokens | S1 scales the winner (a fresh draw, fresh seeds) |
| P-answer's gain excludes 0 against A0; P-trace's against A0, P-answer and P-long | self-distillation moves it (P-trace: beyond answer-only and beyond length) | only DPO moves: the 10M corpus waits and A2 scales on A0 (R1 b) |
| both families move | | a stacked pilot first |
| a rung's CI includes 0 | uninformative at pilot dose | one dose step up (3M tokens, 1,000 pairs), then escalation; both DPO rungs null after the step: transfer distance first (Amendment 21 item 7) |
| a rung's ordered-option share rises (CI excludes 0) | the rung pushes toward obedience | a ledger entry; it does not scale |
| a rung's legitimate-order share falls (CI excludes 0) | over-steered (refuses orders the document directs following) | reported; scales only on the author's ruling |
| D-chat against D-doc at 1M (R1's three branches at that dose) | the format contrast | one arm trains at 10M: the winner, or doc if unresolved |
| D-doc moves only with recovery, or R0 moves | the format repair carries it | read against R0; a ledger entry before wording |

## 4. Rungs

| rung | pod | data | training |
|---|---|---|---|
| R0 | SDF | the recovery set alone | about 0.1 h |
| D-raw | SDF | `C_anth`'s body, 3 passes (about 0.1M tokens), then the recovery step | about 0.15 h |
| D-doc | SDF | 1M tokens of §4.1 (the subject writing with `C_anth` in context, one self-critique pass); checkpoints 0.3M and 1M; the recovery step on each; read with and without it | 0.68 h + 2 recovery steps |
| D-chat | SDF | the same content, token-matched, rendered per §4.2 under T0.6 (empty think block, answer-only loss); the recovery step | 0.68 h |
| D-neutral | SDF | the same templates written about the A0n text, with it in context, same token count; the recovery step | 0.68 h |
| P-answer | DPO | 500 kept pairs per §4.3 (first answer without the document, rewrite with it; chosen the rewrite), answer-only loss, β 0.1 with §5's NLL term; checkpoints at 250 and 500 | about 0.9 h |
| P-trace | DPO | the kept prompts; chosen the full rollout with `C_anth` in context, rejected the full rollout without it; loss on trace and answer, the prompt rendered without the document | about 1.5 h |
| P-long | DPO | the kept prompts; k 4 rollouts without the document; chosen the longest, rejected the shortest; P-trace's loss mask; no filter | about 1.2 h |

The recovery step is identical on every document rung (Amendment 21 item 5): the enlarged recovery
set (`none`'s own thinking-on replies to neutral prompts, selected to match `none`'s panel
trace-length distribution, CST-A10), one pass at a fixed learning rate after the rung's own data.

## 5. Launch cards

### Card 1: the SDF pod (about 10.3 A100-h; one A100-80GB)

| order | stage | A100-h |
|---|---|---|
| 1 | loads (vLLM generation, HF training, vLLM LoRA read) | 0.3 |
| 2 | D-doc corpus: 1M tokens written and critiqued with `C_anth` in context (2M generated) | 1.65 |
| 3 | the recovery set (about 0.5M tokens generated, then matched) | 0.15 |
| 4 | D-doc training to 0.3M and 1M; recovery steps; R0 | 0.9 |
| 5 | read: D-doc 1M with recovery, D-doc 1M without, R0 | 1.35 |
| 6 | D-neutral corpus (2M generated with the A0n text in context) and training; read | 2.8 |
| 7 | D-chat render (about 1M at A0's rate) and training; read | 1.4 |
| 8 | A0 on the legitimate-order cell (20-item check, then k 4) | 0.1 |
| 9 | D-doc 0.3M reads (with recovery, then without) | 0.9 |
| 10 | D-raw training and read | 0.6 |

**Pilot gate (after stage 2's first 50 documents):** 500-1,400 words each, no chat or think markers,
under 5% dropped by the 8-gram check (D11); else stop and bank. **Bail order** (Amendment 21 item 1):
past the price, drop the 0.3M checkpoint's without-recovery read first, then D-raw (stages 9-10 sit
last for that reason). **Saves:** every document (tenet ids, template, model and revision, sha;
text to `outputs_raw/`), the recovery set, per-step training logs, every adapter and checkpoint, the
parity record per adapter kind, S0a's per-rollout schema for every read.

### Card 2: the DPO pod (about 9.5 A100-h; one A100-80GB)

| order | stage | A100-h |
|---|---|---|
| 1 | loads (the subject with and without the document, gpt-oss-120b, HF training, vLLM LoRA read) | 0.45 |
| 2 | candidates on the ~700 prompts: one rollout without the document (P-answer's first answer, P-trace's rejected), the rewrite with `C_anth` in context, one full rollout with it, and three more without it for P-long | 3.1 |
| 3 | the filter: gpt-oss-120b on the final answers only, one rule for every filtered rung | 0.05 |
| 4 | P-answer training (checkpoints 250, 500); P-trace; P-long | 3.6 |
| 5 | read: P-answer 500, P-trace, P-long, P-answer 250 | 1.8 |
| 6 | A0 and A0s on the legitimate-order cell (20-item check first) | 0.5 |

**Pilot gate (after stage 3):** fewer than 250 kept pairs stops the DPO rungs and banks the
candidates (Amendment 21 item 2). **Beside the reads:** the chosen-to-rejected length ratios of P-trace
and P-long, and the share of P-trace's chosen traces citing a constitution or document. **After the
pod (local):** Claude re-filters the same candidates on the final answers; the agreement with
gpt-oss is reported beside. **Saves:** every candidate with its prompt's tenet ids, the filter's
reply per pair, per-step training logs, every adapter and checkpoint, the parity record, S0a's
per-rollout schema for every read.

Neither pod consumes the other's adapters or waits on its results; either launches alone if the
other's build lags.

## 6. Not built (zero GPU)

1. The ~700 difficult-advice prompts (Amendment 21 item 4): a seeded grid over T(`C_anth`) outside
   `T_conf` by domain and stakes, user-advice framing only, tenet ids per prompt, 8-gram checked
   against every eval set, near-duplicates removed, apart from the panel's scenario families.
2. The generation driver: §4.1 documents and the critique pass; D-neutral template variants
   (`deepsteer.corpus.prompts._DOC_HEAD` names "an AI system"); the §4.2 chat render with T0.6's empty
   think block (`chat_render_v1` renders plain text).
3. The enlarged recovery set: the neutral prompt pool and the trace-length matching rule.
4. The SDF trainer at dose, with checkpoints and the recovery step.
5. DPO: §5's NLL term (`objective_lora.dpo_loss` has none), P-trace's and P-long's full-rollout loss
   mask, a split backward if paired forwards bind on memory.
6. The DPO pod's candidate stage and the gpt-oss filter stage; the local Claude re-filter.
7. The multi-adapter read driver (native LoRA, the parity check per kind), reusing S0a's stages.
8. The legitimate-order cell (Amendment 19 rider d).
