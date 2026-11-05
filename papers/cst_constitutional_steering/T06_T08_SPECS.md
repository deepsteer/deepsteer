# CST T0.6 and T0.8: objective classes, the N spec block, the VALIDATE plan (draft for G0)

Zero GPU. Each item names what the author decides at CST-G0; nothing here has run.

## T0.6 (a) Reward-pair class of every arm's objective

The taxonomy is Kaufmann, Lindner, Zimmermann, Shah (arXiv:2603.30036): a chain-of-thought term is
aligned, orthogonal or in-conflict with the outcome reward. Their evidence is from toy environments,
and they report that in-conflict terms both reduce monitorability and are hard to optimize
(`SOURCES.md` §5). The plan's rule (§5): no arm with an in-conflict chain-of-thought term runs.

| arm | objective | tokens the loss touches | CoT term? | class | registered condition |
|---|---|---|---|---|---|
| A1 `doc` | next-token loss on §4.1 documents | document tokens (no chat template, no think block) | no | not an RL pair; no CoT term | none beyond the format-recovery parity read (T0.8 V6) |
| A1c `chat` | SFT on §4.2 transcripts | assistant turns | **depends on render** | none if the think block is empty and masked | the assistant turns are rendered with an empty think block (`enable_thinking=False` render) and only the post-think answer carries loss; otherwise the arm trains the model to answer without reasoning, which the recovery step then has to undo |
| A2 `distill` | DPO, β 0.1, NLL term, adapter-disabled reference | chosen/rejected answers | **depends on render** | none if answer-only | the rewrite is produced with the document in context and has no trace of its own, so pairs are answer-only (empty think block, think tokens masked as ST's `trace_masks` does). A pair carrying the first answer's trace beside the rewrite would teach a trace the chosen answer did not come from |
| A3 `introspect` | SFT on §4.4 transcripts | transcript text | no (the transcripts are prose, not traces) | none | generation with thinking off; the think block of each training render is empty and masked |
| A4 `grpo` | GRPO, panel-judged outcome reward | rollout (trace + action) under the policy | **yes, as drafted** | **in-conflict risk** | §5 says the rubric is "applied tenets". If the judge reads the applied tenet from the trace, the reward depends on the chain of thought: a rollout can raise its reward by citing a tenet its action satisfies. Registered fix: the rubric for a scenario is the panel's tenet reading of the **scenario text**, fixed before any rollout; the judge sees the action only. Then the reward is outcome-only |
| A4f | A4 with the update-time CoT-gradient restriction (Jia et al.: rollouts, rewards and advantages unchanged; only the policy update changes) | as A4 | no (a mask, not a reward) | not a reward term | the specific variant (Update Mask, Gradient Mask, CoT-only Gradient) is named before S5 |
| B1 `stage` | SDF then LoRA SFT/DPO from a base | as A1 then A2 | as A1/A2 | as A1/A2 | **dependency gap:** "our post-training" from `Qwen3-30B-A3B-Base` to a thinking model needs a thinking-format SFT set the plan does not spec. The nearest same-family source is Thinking-2507's own traces (Apache-2.0); it must be specced before S4 |

## T0.6 (b) Intervention spec block: the span-removal resampling check for N (§6.3)

```
stimulus_class:          rollouts from the S read (same arm, same scenario, same condition) whose
                         trace has a panel-marked applied-tenet span; type: trace text, position
                         class "think block"; per arm
outcome_variable:        the action (option letter) read after the edited trace, parsed by the S
                         read's parser at its pinned version
baseline_discrimination: two screens, both reported with pass rates before any N:
                         (1) the trace matters at all: action-flip rate with the whole trace removed
                             (empty think block, dose-0 render) against the intact-trace re-read;
                             a rollout set where removing the whole trace flips nothing cannot
                             carry a span-level N
                         (2) the intact-trace re-read's own flip rate (sampling noise at the
                             action), the floor every span effect is read against
site:                    the applied-tenet span: the panel's quoted trace substring expanded to
                         whole sentences (rule fixed below); the matched control span is a
                         non-moral sentence run from the same trace
transfer_scope:          text-level span edit (not an activation patch)
ablation_semantics:      primary: replace the span with a matched-length non-moral span drawn from
                         another rollout of the same scenario (distribution-preserving);
                         secondary: delete the span. Each is reported; the headline is the
                         replacement form
alignment_rule:          span boundaries = sentence boundaries enclosing the panel's quote (period,
                         question mark or newline); matched length = within ±15% tokens; after the
                         edit the think block is closed and the action read with the trace's
                         remaining text unchanged (no continuation); fixed before any data
controls:                matched-length non-moral span removal from the same trace (the
                         channel-matched null); whole-trace removal (upper rung); intact re-read
                         (floor)
outcome_harness:         the S read's action parser, pinned; parse rate reported per arm
branches:                N_res = flip(moral span) − flip(matched non-moral span), scenario
                         bootstrap difference CI.
                         N_res CI above 0: the action runs through the moral span of the trace on
                         this arm; read beside S and trace length.
                         N_res covers 0 at its MDE while whole-trace removal flips actions: the
                         action depends on the trace but not on the applied-tenet span alone
                         (rival readings: redundancy, the tenet restated elsewhere in the trace,
                         separated by removing every panel-marked moral span; or the decision
                         carried by non-moral reasoning)
                         whole-trace removal flips nothing: no span-level N on this arm (the trace
                         is not load-bearing for the action at all), stated as such
```

**Grad-Nec (the gradient leg).** Jia et al. define Grad-Nec as a first-order gradient-sensitivity
share (Σ over CoT tokens of ‖∂L_answer/∂e_t‖ over Σ over all tokens) and say it linearizes, not
measures, interventional necessity. CST restricts the numerator to the applied-tenet span and
reports a matched-length non-moral span's share beside it. The repo has no license, so the G0
recommendation is to re-implement it in `deepsteer.faithfulness` against the pinned commit
`f9e1f29` (about 40 lines; torch and transformers only) with a test that reproduces the reference
function's output on a tiny model. Wording rule: "gradient necessity share", never "necessity", and
the resampling check is the interventional leg.

## T0.8 VALIDATE plan (one pod, before S0; priced at about 2 A100-h)

Every check has its bar and the failure branch, written now.

| # | check | bar | on failure |
|---|---|---|---|
| V1 | load `Qwen/Qwen3.8-27B@1d4bf0f2` in HF and vLLM 0.31.0 with `max_model_len` ≥ 44,032 (C_anth's 36,313 page tokens + 1,536 trace cap + prompt + margin) and prefix caching on | both load; a 40k-token prompt prefills | A0s and corpus generation move to a summarized document, an amendment (the full document is the author's decision of record) |
| V2 | KV budget with the document prefix cached | 64 KiB per token (16 full-attention layers × 2 × 4 KV heads × 256 × 2 B); the 36k prefix is 2.4 GB once; 64 concurrent sequences at 3k tokens fit beside 54 GB of weights | lower `max_num_seqs`; re-price S1 |
| V3 | LoRA on the hybrid text tower: ST's target set (DeltaNet `in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj`; attention `q/k/v/o_proj`; MLP), rank fixed here and never changed | forward/backward runs; adapter saves and reloads bit-identical | restrict to attention + MLP, recorded |
| V4 | vLLM native LoRA serving on this architecture (no merge) against the merged path | if native serving works, it is the read path; else ST's one-delta merge gate with kept share reported | MRB-A15 measured 0.60 letter-shift kept share on Qwen's merged checkpoint; a merged arm's Δ is then a lower bound, stated beside every number |
| V5 | SDF throughput: 1M tokens of packed documents (seq 2,048), LoRA, gradient checkpointing | measured tokens/s; projects 10M within the S1 price | second derivation: 6·N·T with recompute ≈ 1.6e18 FLOPs for 10M tokens; at 20-35% of 312 TFLOP/s that is 4.1-7.1 h, consistent with the plan's ~8 |
| V6 | format survival after a 1-step SDF adapter plus the recovery SFT | thinking-on parse rate and trace-length median within ±10% of `none` on 64 panel rows | the recovery set is enlarged before S1 |
| V7 | Grad-Nec memory: one backward to input embeddings at 2,048 tokens on the 27B in bf16 | fits on one A100-80GB with checkpointing | N's gradient leg runs on a subsample, or on H100 |
| V8 | the pressure-triple renders on the pod's tokenizer | every rendered prompt's sha equals `outputs/t03_pressure_triple.json` | stop: the template drifted |

Local tests already in place (laptop, no model): `tests/scripts/test_cst_pressure.py` (operator cell
byte-equal to the frontier cell; H1 operator orders the rule, not protection),
`tests/constitutions/` (quote verification catches paraphrase; body excludes preface and
acknowledgements), `tests/corpus/` (overlap catches case and punctuation variants; T_conf leak
recorded), `tests/judges/` (2-1 splits compare exactly; ties pool to no label). The S0 generation
and scoring harness is not built before G0: its effort level, operator cell set, k, panel families
and context length are G0 decisions.
