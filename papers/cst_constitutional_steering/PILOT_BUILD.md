# CST pilot ladder build (2026-11-09; Amendment 20; draft for the author)

The pilot ladder (Amendment 20 item 2) reads seven rungs on the screen read (item 1) and gates S1
(item 4). This file is in `S0B_BUILD.md`'s form. Prices use measured rates only: SDF training 410.7
tokens/s (VALIDATE, rank 16, 496 modules: 1M tokens = 0.68 h); generation with a 34k-token document
in context 337 tokens/s, without it about 1,000 (S0a's chunk logs); a full read 6,500-8,200 A0-length
rollouts/h. Nothing here has run; §6 lists what is not built.

## 1. Settings of record

S0a's subject and engine (`S0A_BUILD.md` §1). LoRA rank 16, alpha 32, dropout 0, the 496 VALIDATE
targets (DeltaNet `in_proj_qkv/z/a/b`, `out_proj`, `q/k/v/o_proj`, `gate/up/down_proj`), HF bf16 with
gradient checkpointing for training; the read on vLLM native LoRA (`enable_lora`, MTP off), with the
32-prompt parity check per adapter kind (VALIDATE V4: TV(served, attached) below TV(base, attached);
measured 0.0125 against 0.1115).

## 2. Rungs and their training data

| rung | data | made where | training (A100-h) |
|---|---|---|---|
| R0 | the enlarged recovery set alone: `none`'s own thinking-on replies to neutral prompts, selected to match `none`'s panel trace-length distribution (CST-A10) | pod A (generation at A0's rate) | about 0.2 |
| D-raw | `C_anth`'s body, 3 passes (about 0.1M tokens) | pod A | about 0.1 |
| D-doc | 1M tokens of the §4.1 corpus (essays, commentary, FAQ, stories, the subject writing with `C_anth` in context, one self-critique pass); checkpoints at 0.3M and 1M; each read with and without the recovery set | pod A (2M tokens at the document rate, D11's critique pass) | 0.68 + recovery 2 × 0.2 |
| D-chat | the same content, token-matched, rendered per §4.2 under T0.6's rule (assistant turns with the empty think block `<think>\n\n</think>\n\n`, loss on the answer only) | pod A (rendering at A0's rate) | 0.68 |
| D-neutral | the same templates written about the A0n text, with it in context; the same token count | pod A (2M tokens at the document rate) | 0.68 |
| P-answer | 500 pairs per §4.3: the subject answers a difficult-advice prompt, then rewrites with `C_anth` in context; chosen the rewrite; answer-only loss (think tokens masked); checkpoints at 250 and 500 pairs; β 0.1 with §5's NLL term | pairs in pod A, filtered locally (Claude, final answers only), trained in pod B | about 0.9 |
| P-trace | the same prompts; chosen the full rollout generated with `C_anth` in context, rejected the full rollout without it; loss on trace and answer, the prompt rendered without the document; pilot-only | pod A, filtered as P-answer, trained in pod B | about 2 (rollouts about 2.5 times an answer's length) |

## 3. The read and its power

Screen read per adapter (Amendment 20 item 2): operator and plain cells on the 458 clear items at
k 2 (1,832 rollouts), the legitimate-order cell at k 4 (about 400), cell A′ at k 8 (816), trace
length; no judge. Capability at n 200 only on rungs that move. About 3,050 rollouts per adapter,
0.4-0.5 A100-h each with LoRA serving.

| contrast | MDE (80%, two-sided 0.05) | source |
|---|---|---|
| rung − A0 (A0 at its saved k 4) | 0.042 | `t04d_ordered_power.json` `pilot_k2` |
| rung − its control, both at k 2 | 0.049 (0.058 with A0s's scenario heterogeneity) | same; A0's half-split check 0.048 |
| cell A′ originals' shift | 0.082 (SE 0.029, 51 pairs, k 8) | `T04_POWER.md` §10 |
| legitimate-order cell, rung − A0 at k 4 | at most 0.099 (worst-case variance, n 100) | closed form; re-read on the 20-item A0 check |

A0s's band is 0.280, so every pilot contrast resolves a rung that carries about a sixth of it.

## 4. Branches (Amendment 20 items 3-4, written before data)

| result | reading | next |
|---|---|---|
| a document rung's gain CI excludes 0 against A0 and against R0, and D-doc against D-neutral | the document's content moves the action at 1M tokens | S1 scales the winner (a fresh draw, fresh seeds) |
| P-answer or P-trace excludes 0 against A0 (and each other, for which) | self-distillation moves it | only DPO moves: the 10M corpus waits and A2 scales on A0 (R1 b) |
| both families move | | a stacked pilot first |
| a rung's CI includes 0 | uninformative at pilot dose (not a null) | one dose step up (3M tokens, 1,000 pairs), then escalation |
| D-doc moves only with recovery, or recovery alone (R0) moves | the format repair carries it | read against R0; the ledger before any wording |
| D-chat against D-doc at 1M (R1, its three branches at that dose) | the format contrast | one arm trains at 10M: the winner, or doc if unresolved |

## 5. Pods, gates, bail rules (Amendment 20 item 6: each at most 12 h, self-contained)

**Pod A: generation and SDF (about 7 A100-h).** Order, most decision-relevant first: D-doc corpus
(2M tokens, 1.65 h) and its training (0.68 h, checkpoints 0.3M and 1M); the recovery set (0.25 h) and
R0; D-neutral corpus (1.65 h) and training; D-chat render and training; D-raw; the DPO pair
candidates for both P rungs (about 0.8 h); loads about 0.5 h. **Pilot gate:** the first 50 corpus
documents must be 500-1,400 words, carry no chat or think markers, and drop under 5% on the 8-gram
check (D11); else stop and bank. **Bail:** a training loss not below its first-50-step mean by the
0.3M checkpoint stops that rung. Saves: every document (tenet ids, template, model and revision,
sha; text to `outputs_raw/`), every pair candidate, per-step training logs, every adapter and
checkpoint.

**Between the pods (local, zero GPU):** the pair filter (Claude on final answers only, the same
rule for both P rungs; gpt-oss-120b in pod B if the safeguard stops recur), the legitimate-order
cell's scenarios and labels (Amendment 19 rider d), and the download of pod A's adapters.

**Pod B: DPO and the read (about 10-11 A100-h).** P-answer (0.9 h) and P-trace (2 h) training; then
the screen read of every adapter (R0; D-raw; D-doc at 0.3M and 1M, with and without recovery;
D-chat; D-neutral; P-answer at 250 and 500; P-trace at its last checkpoint: 11 adapters, about 5 h),
then capability at n 200 on the rungs that move. **Gates:** the parity check per adapter kind before
its read (a kind that fails reads through HF attachment, priced at about 3 times, or is banked);
the legitimate-order cell's 20-item A0 check first (a cell at floor is rebuilt, not read). **Bail:**
a read rate under two thirds of the price after the first adapter drops the 0.3M checkpoints first.
Saves: S0a's per-rollout schema for every read; the parity record per kind.

Pod B needs pod A's adapters and the filtered pairs, so it launches after pod A is downloaded; it
does not wait on pod A mid-session. The two pods do not run in parallel.

## 6. Not built (zero GPU, in build order)

1. The generation driver: §4.1 SDF documents and the self-critique pass with `C_anth` in context
   (`deepsteer.corpus.prompts` has the registered templates); D-neutral variants of the templates
   (`_DOC_HEAD` names "an AI system" and needs a neutral form); the §4.2 chat render with T0.6's
   empty think block (`chat_render_v1` renders plain "User:/Assistant:" text).
2. The enlarged recovery set: a neutral prompt pool and the trace-length matching rule.
3. The SDF trainer at dose (VALIDATE's 12-step loop is a toy), with checkpoints and the recovery mix.
4. DPO: §5's NLL term (`objective_lora.dpo_loss` has none) and P-trace's full-rollout loss mask; a
   split backward for paired forwards if memory binds (the Phase 3 gotcha).
5. The §4.3 difficult-advice prompt set and the pair pipeline; the local Claude filter.
6. The multi-adapter read driver (native LoRA, the parity check per kind), reusing S0a's stages.
7. The legitimate-order cell (Amendment 19 rider d).

## 7. Decisions for the author

1. **Pod order.** Pod B depends on pod A (adapters and filtered pairs), so the two run in sequence,
   not in parallel. Alternative: move the SDF rungs' read into pod A, so each pod reads its own rungs
   and both could run in parallel after the pair filter; pod A then prices at about 10 h.
   Recommendation: the sequence as written, which keeps Amendment 20's split.
2. **P-trace's length.** Full rollouts with `C_anth` in context run about 2.6 times A0's length
   (CST-A8), so P-trace's chosen side is long and its training about 2 A100-h; the chosen-rejected
   length gap is itself a training signal. Recommendation: report the length gap beside P-trace's
   read; no length matching at pilot dose.
3. **The difficult-advice prompts (§4.3).** Not found in the repo. Recommendation: write about 700
   prompts (500 pairs at the plan's 0.7 keep rate) from a seeded template grid, 8-gram checked
   against every eval set, not drawn from the panel's scenarios.
