# Draft edits for the pitch Google Doc (updated 2026-09-28, part-A gate)

For pasting into the pitch doc (the source of record; `KDG_PITCH_PLAN.md` is its export). Rules of
record (author): the "post-training lowers the baseline" half is dropped; the base-vs-instruct
comparison is a cell, and after the SFT bridge the base cell is **descriptive**; "not resized by
post-training" always carries its bar; nothing about the growing at-rest lean (KDG-A8) enters the
pitch until the author decides. Two versions of the lead follow; Version 1 is in force now.

## The claim: Version 1 (provisional, in force)

1. **The gap is pretraining-native and not resized by post-training.** On OLMo-3-7B the model acts
   against its own stated moral judgment by a margin the incentive adds. That margin is present in the
   only frame a base model has, and read under the model's own chat template it is present after SFT,
   after DPO and after RLVR (0.021, 0.022, 0.030). Post-training does not resize it: no stage change
   detectable above about 0.013 at n = 136.
2. **The instrument withdrew one of our own claims.** An apparent "safer at rest" effect of
   post-training came from reading a chat model without its template. A pre-registered control
   removed it, and the same check shows the raw frame misreads every templated checkpoint from SFT on.
   Evaluations that score instruct checkpoints without their template, including recent log-prob
   studies of the same OLMo-3 checkpoints, inherit this.
3. **Moral deliberation before acting reduces the gap (pending).** Asking the model to reason about the
   stakes, even when its reasoning is cut off at 512 tokens, lowers the violating choice by 0.077
   (0.047 to 0.110) against a length-matched non-moral restatement. Pending two checks in Session B: a
   control giving the restatement the same truncated form, and a per-scenario comparison with
   reasoning allowed to finish.

## The claim: Version 2 (if item 3 survives Session B)

1. **The gap is pretraining-native and not resized by post-training.** (as Version 1, item 1)
2. **Moral deliberation before acting reduces it.** Reasoning about the stakes lowers the violating
   choice against a matched non-moral control, and the effect survives a truncation-matched control
   [and holds when reasoning is allowed to finish: fill from the 2,048 table]. [If the norm-salience
   arm reproduces most of the effect: "Naming the norm at stake reduces it; deliberating adds (little /
   something) beyond that." Fill from the P1-A8 salience share.]
3. **The instrument withdrew one of our own claims.** (as Version 1, item 2)

## Evidence in hand, points 4 to 6 (replacement)

4. **The gap is there before safety training and after every stage of it.** The base model shows a
   small pressure-driven gap in the only format it can be read in; the trained model, read in its own
   chat format, shows one after each training stage, and the stages do not change its size.
5. **One of our earlier findings did not hold up, and the test that removed it was written in
   advance.** We had reported that safety training makes the model more cautious at rest. That came
   from reading the trained model in a format it was not built for.
6. **Thinking about the stakes first helps (pending checks).** When the model reasons about what is at
   stake before acting, it picks the option it judged wrong less often than when it restates the
   situation at the same length.

## Numbers of Record (rows that change)

| Finding | Number of record | Scope limit |
|---|---|---|
| The gap exists before alignment | Base excess 0.017 [0.012, 0.022], raw frame | Descriptive: the raw frame is the base model's only frame and misreads SFT (−0.052 [−0.070, −0.034]) |
| The gap survives every post-training stage | Excess under the template: SFT 0.021 [0.006, 0.036], DPO 0.022 [0.002, 0.043], final 0.030 [0.006, 0.053] | 136 screened, selected on the final model's chat actions |
| Post-training does not resize it | DPO step 0.001 [−0.012, 0.014], RL step 0.007 [−0.001, 0.016] | No stage change detectable above ~0.013 at n = 136 |
| Truncated moral reasoning reduces it | dose2 − filler −0.077 [−0.110, −0.047] (130) | Truncated at 512 tokens; pending the truncated-filler control and the 2,048 table |
| ~~Post-training lowers the baseline~~ | Withdrawn: raw −0.038 vs template +0.055 | Pre-registered artifact branch |
| ~~Post-training widens the acting side~~ | Withdrawn as a headline: raw-frame cell, not separable from output-scale sharpening, raw frame invalid for templated models | KDG-30 scoped |
| The raw frame distorts templated models | At-rest sign −0.038 raw vs +0.055 chat (final); −0.019 vs +0.033 (SFT) | Dated to the first templated stage |

## Related work line (replacement)

tracing-sycophancy (Sonnet Xu, GitHub): "related behavior–probability dissociation on the same OLMo-3
checkpoints; readout comparability unverified" (its log-prob track reads chat models without their
template). Not cited as convergent evidence.


## Update 2026-09-28 (Session B gate): lead in force = Version 2, scoped to OLMo-3

1. **The gap is pretraining-native on OLMo-3 and not resized by its post-training** (no stage change
   detectable above ~0.013 at n = 136).
2. **Moral deliberation before acting reduces it on OLMo-3,** against a length-matched restatement and a
   truncation-matched one (−0.112 [−0.145, −0.078]); **about a third of that is naming the norm**
   (salience share 0.32 [0.22, 0.53]).
3. **The instrument withdrew one of our own claims** (as Version 1, item 2).

Cross-lineage table (wording pending the Session C positive controls; every zero reads "not detected"):

| Model | Base, raw frame | Instruct, own template |
|---|---|---|
| OLMo-3-7B | 0.017 [0.012, 0.022] | 0.030 [0.006, 0.053] |
| Llama-3.1-8B (Meta) | not detected (0.000 [−0.003, 0.003]) | 0.028 [0.020, 0.036] |
| Tulu 3 (Ai2 recipe on Llama-3.1) | (Llama base) | not detected at SFT / DPO / final |
| Qwen2.5-7B | 0.011 [0.007, 0.016] | not detected (−0.008 [−0.023, 0.009]) |

Ai2 ask, add: the raw-frame distortion appears on both Ai2 recipes tested (OLMo-3, Tulu 3) and not on
Meta's; Ai2's intermediate checkpoints can locate the step that installs it.


## Update 2026-09-28 (Session C gate): ask table hooks

| Lab | Hook (replacement) |
|---|---|
| Ai2 | **The two-recipe puzzle.** OLMo-3 carries the judgment–action gap and Tulu 3 (Ai2's recipe on Llama-3.1) does not, with the instrument validated on both; and both Ai2 recipes show the raw-frame distortion that Meta's does not. Ai2's intermediate checkpoints and recipe data can locate which step decides both. |
| Meta | **The Llama-3.1-Instruct dose result.** On Meta's Llama-3.1-8B-Instruct, reasoning about the stakes before acting lowers the violating choice by 0.350 [0.314, 0.387] against a length-matched control (mostly completed reasoning); Meta's recipe carries the gap on a base where Tulu 3's does not. |

Pitch rule (author): no slide shows the OLMo-3 and Llama-3.1 dose numbers together until the
like-for-like run (shared scenario set, completion budget, ~3 GPU-hours) reports. The paper's hold is
lifted.
