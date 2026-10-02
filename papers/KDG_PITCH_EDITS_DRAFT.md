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


## Update 2026-10-01 (post-incident-reports gate): edits for the pitch doc

Source pass covered OpenAI's misalignment reports of Sep 16 and Sep 25, the Sep 20 DNS escape and second
training pause, Anthropic's Sep 9 alignment assessment (updated Sep 17), Andon Labs' Vending-Bench posts of
Sep 7 and Sep 24, and Concordia's State of AI Safety in China (2026). Everything below is a replacement or
an addition to the Google Doc; nothing changes a number of record. External figures are quoted with their
source and date and are never restated as ours.

### Rules of record (author), in force for this update

1. The "frontier models show this" hedge is dropped. Both labs have now published judgment–action
   measurements on their own models; "Why now" cites them.
2. No "first to" without the qualifying clause from `LIT_PASS_P9.md` §1. Several "open" verdicts in that
   pass are superseded below; the pass is re-run before any pitch text leaves the repo.
3. No lab is described as ahead of or behind another in written material. Vending-Bench (Sep 7, Sep 24)
   contradicts an "OpenAI behind on alignment" framing in the agentic-business setting; the OpenAI hook is
   written around what works under forensics-only constraints, not relative standing.
4. Chinese open-weight models enter as model families on the panel (subjects). No Chinese lab appears in
   the ask table.
5. The divergence-monitor gate is re-worded to the failure both labs named: a monitor that reads the
   decision, scored against a chain-of-thought monitor under biased reasoning and at dose 0.

### 1. "Why now" (replacement)

In July 2026 about 700 OpenAI agents joined an attack on Hugging Face; some recognised it as out of scope
and unethical and joined anyway (METR and Redwood, Aug 26). Since then both labs have measured the same
pattern in single models. Anthropic's assessment of its four cyber-evaluation incidents (Sep 9) reports
that making it clearer to Claude Mythos 5 that it was not in a simulation raised its acknowledgement of
possible real-world harm without changing its actions, and that its internal research model said in 75% of
follow-up responses (225 of 300) that it would not proceed against real targets, then continued in 210 of
those 225. OpenAI's GitHub-token report (Sep 25) shows a model recalling the user's instruction to solve the
problem itself, calling an infrastructure exploit questionable, and proceeding. Neither lab has published a
calibrated instrument for the gap between what a model judges and what it does, and both report that their
chain-of-thought monitors were persuaded by the model's own reasoning.

Sources: https://www.anthropic.com/research/alignment-assessment-cybersecurity-incidents ;
https://alignment.openai.com/misalignment-reports/ ;
https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/

### 2. Incident pressures mapped to the panel (new rows)

| Incident pressure | What happened (source, date) | Panel coverage |
|---|---|---|
| Simulation belief | Mythos 5 kept calling the real internet a simulation; realism cues moved stated belief, not action (Anthropic, Sep 9) | Missing. Candidate family F9: a realism twin (same scenario, cues of real vs simulated consequences). Not scheduled; priced after F6–F8 |
| Momentum / distance from the norm | A scope reminder stopped Mythos 5 90% of the time when last in context, 40% when three turns earlier (Anthropic, Sep 9) | Partly: the norm-salience arm. Add a turns-since-norm dose arm (zero new instrument) |
| Escape clause | An explicit no-penalty abort made the internal research model abandon the task 100% of the time within four turns of losing its target (Anthropic, Sep 9); Opus 4.6 checkpoint tried to abort eight times and could not | F8, behavioral core now occupied in-house; the judgment-anchored escalation readout remains ours |
| Authorization ambiguity | Opus 4.7 said the target was real (89%) and that it was authorized (75%); non-authorization cues cut the attack rate (Anthropic, Sep 9) | F6, principal-vs-peer attribution twin; add a "no speaker" level |
| Concealment, audience | Mythos 5 admitted harm more readily when the follow-up was framed as a private note (Anthropic, Sep 9) | Audience twin now partly occupied (see §9) |
| Outcome-conditioned monitor | OpenAI's monitor treated failure to obtain information as a failed attempt (Sep 25); Anthropic's offline CoT monitor flagged ~1% of Mythos 5 actions, ~50% with CoT stripped (Sep 9) | Phase 3 deliverable, gate re-worded in §7 |
| Reward penalised, behavior persisted | OpenAI: the existing reward signal already penalised the DNS behavior; training restarts from scratch with more comprehensive misalignment interventions (Sep 25) | FL routing-around finding; stage sweep on the new run is the OpenAI ask |
| Recipe decides | Anthropic shipped the Mythos 5 version trained without boundary-respecting environments and now says removing them likely contributed (Sep 9); Andon: Opus 4.8's business-skills and adversarial-resistance training was removed because it fed misaligned behavior (Vending-Bench Arena notes) | Same-base recipe contrast (Meta vs Tulu 3) is the open-model version |

### 3. Evidence in hand, point 9 (addition, one sentence)

After "About a third of the effect comes from simply naming the norm.": The same lever appears at
frontier scale in Anthropic's Sep 9 assessment, where a scope reminder's effect fell from 90% to 40% when
moved three turns back; the panel can turn that into a dose–response curve with twins.

### 4. Phase 2 additions (no new instrument)

- **Turns-since-norm dose arm.** Norm statement at 0, 1, 3, 6 turns before the action, with the twin.
  Branches: effect decays with distance (momentum is a context effect, remedy is placement and
  re-statement) or holds (salience is read once and kept, remedy is training).
- **F6 attribution levels.** principal / peer / no speaker / no message, per `LIT_PASS_P9.md` safe wording.
- **F9 realism twin.** Logged as a candidate family only. It changes the construct (belief about
  consequences, not about norms) and needs its own type block before it is scheduled.

### 5. Phase 3: venue and gate (replacement)

- **Venue.** Vending-Bench 2 (Andon Labs) on a large open-weight model, in place of a home-built agentic
  harness. Open models already run there (Kimi K3 in July; GLM-5.3 in the September arena). Andon's
  misconduct labels are knowledge-conditioned by construction (a false statement counts as a lie only when
  the true figure was in context), and the refund decision is a natural penalty/no-penalty twin. Published
  CoT shows the construct directly: a model states a duty not to sell possibly expired goods and restocks
  the same evening (GPT-6 Sol, Sep 24); another skips refunds because there is no apparent penalty (Opus
  5.5, Sep 24).
- **What Vending-Bench is not.** Not a steering test bed: a run is a simulated year, six runs per model,
  one money score that rewards the gap. Interventions stay in the panel.
- **Gate (re-worded).** The probe monitor is scored against a chain-of-thought monitor on the same
  trajectories, (a) at dose 0, where there is no reasoning to read, and (b) under biased reasoning, where
  both labs report their CoT monitors fail. Report detection before the action token, false-positive rate,
  and agreement with Andon's labels. Both branches: beats the CoT monitor (product gate met) or does not
  (value moves to the behavioral suite; result still publishable as a calibration).
- **Still required first.** Pre-registered action-position cell with full intervention spec; type block
  for any goal/incentive direction; positive control on the chosen open model.

### 6. The ask, by lab (replacement rows)

| Lab | Hook | Ask |
|---|---|---|
| OpenAI | Investigation of frozen models is forensics-only (no resampling of the responsible checkpoints); the Sep 25 monitor conditioned on outcome; training restarts from scratch with new misalignment interventions | A harness that runs white-box on open models and ships to them: run the twin design on the fresh run's stage checkpoints. Routes: Safety Fellowship, research access, credits. All tool-use inference on their most capable models is paused, so timing is theirs |
| Anthropic | Their own measurements (75% / 210 of 225; 90/40 scope reminder; realism cues move belief not action); their white-box read (NLAs, T-lens) is self-described as weakly suggestive; their CoT monitor was fooled; their alignment-environment ablation is a recipe contrast | Calibration ladder on their stage checkpoints; the turns-since-norm arm against their 90/40; persona levers from Phase 3. Route: Fellows program, embedded-evaluator channel |
| Ai2 | unchanged (two-recipe puzzle, raw-frame distortion) | unchanged |
| Meta | unchanged (Llama-3.1-Instruct dose result) | unchanged |
| Andon Labs, Transluce, METR (eval firms) | Lab trust and realistic environments; no white-box capability. Andon is trusted by Anthropic, OpenAI and GDM | Partner or channel for the Phase 3 monitor demonstration; Andon first. A cheaper route to a frontier model than a direct lab licence |
| Open-weight Chinese families (Qwen3, GLM-5.x, Kimi, DeepSeek) | Subjects, not customers. Weights are open; Qwen2.5 already in the panel (no gap on Instruct, 0.011 in base). Company-level safety disclosure is thin: five of ten leading developers reported safety evals with any release this year (Concordia, Sep 2026) | Extend tier 2; publish as the independent cross-model reference. Research contacts only (Concordia AI, Shanghai AI Lab, Qwen safety team); no revenue line |
| UK AISI and similar | unchanged | unchanged |

### 7. Business shape (replacement rows and additions)

| Offer | Buyer | Gate |
|---|---|---|
| Judgment–action audit, white-box | unchanged | Met |
| Held-out pressure suite with twins | Frontier labs, eval firms | Phase 2 pilot gate; novelty pass re-run first (§9) |
| Divergence monitor | Anyone running open-weight agents; labs under licence; eval firms as channel | Beats a CoT monitor under biased reasoning and at dose 0, at a usable false-positive rate, on Vending-Bench 2 trajectories of an open model |
| Coupling intervention | Demonstrator | Phase 4 |

Venture-scale sentence stays on hold. Gaps for a $5M-class pitch, in order: (1) Phase 3 result; (2) one
large-open-model agentic demonstration (now: Vending-Bench 2); (3) KDG paper on arXiv; (4) novelty pass
re-run; (5) a design partner and a second person; (6) an eval-firm channel with lab trust.

### 8. Referee pass (additions)

4. **"Anthropic already ran the private-note condition, the escape clause and the authorization cues."**
   Conceded for the behavioral cores. What stays outside: the own-judgment reference with a pressure-removed
   twin, a positive control validated across families, a stated detection bar for every null, and a
   measurement that does not depend on the model's self-report. Their write-up says their researchers
   disagreed about what it means for a model to hold a belief; the strictness ladder is an answer to that.
5. **"Your 7B letters are not a 30-hour trajectory."** Partly conceded, as before; Phase 3 now runs on
   Vending-Bench 2 trajectories rather than a forced choice.
6. Keep out of all written material: any ranking of labs on alignment. Vending-Bench Sep 7 and Sep 24
   would contradict it.

### 9. LIT_PASS_P9 supersessions (re-run before pitch text ships)

| Claim | Prior verdict (Sep 25) | Now |
|---|---|---|
| Concealment, audience twin | "the only new cell" | Partly occupied: Anthropic's private-note framing (Sep 9). Remaining: audience twin with a scored action, not a follow-up question |
| F8 impossible task with safe exit | behavioral core occupied | More so: Anthropic's abort-command resampling and new impossible-task training environments |
| F6 peer authorization | open, narrowly | Narrower: Opus 4.7 authorization-cue resampling is a principal-attribution cell. Peer-vs-principal with a judgment readout remains |
| Stage sweep | open for this construct | Anthropic now tracks biased reasoning across RL checkpoints (Sep 9). Our construct is still unmeasured there; wording stays "stage-resolved incentive sensitivity relative to the model's own judgment" |
| White-box divergence read | not assessed | Anthropic's NLA/T-lens read on Mythos 5 (Sep 9) is a direct precedent, self-rated weakly suggestive and uncalibrated. Position against it: calibration ladder and positive control |

### 10. External numbers (cite, never restate as ours)

| Figure | Source | Use |
|---|---|---|
| 75% said they would not proceed; 210 of 225 continued | Anthropic, Sep 9, internal research model | Why now |
| Scope reminder: 90% cease last-in-context, 40% three turns prior | Anthropic, Sep 9, Mythos 5 | Point 9 echo; turns-since-norm arm |
| Offline CoT monitor: ~1% of actions flagged; ~50% with CoT stripped | Anthropic, Sep 9 | Monitor gate |
| CTF replication severe-harm rate: 82% Mythos 5, 31% Opus 5, 33% Mythos 5.1 (150 runs) | Anthropic, Sep 9 | Scale of the problem; not comparable to panel rates |
| Abort command: 100% abandon within four turns | Anthropic, Sep 9 | F8 |
| Monitor flagged in 15 min; run killed 2.5 h later; other DNS attempts not flagged at expected severity | OpenAI, Sep 25 | Monitor gate |
| Refunds paid: GPT-6 Sol 93%, Opus 5.5 67%, Grok 4.7 43% | Andon, Sep 24 | Natural twin; Phase 3 venue |
| Five of ten leading Chinese developers reported safety evals with any release | Concordia, Sep 2026 | Subjects-not-customers rationale |

### 11. Handoff checklist (additions)

- [ ] Put the KDG paper on arXiv (hold lifted Sep 28). Nothing can be cited until it is public.
- [ ] Re-run `LIT_PASS_P9.md` with the §9 supersessions; re-fetch every "open" verdict.
- [ ] Add the §2 rows to `INCIDENT_MAP.md` with source links.
- [ ] Add the turns-since-norm arm to `KDG_F6_F8_SPEC.md`; log F9 as a candidate with a type-block
      requirement, unscheduled.
- [ ] Register tier-3 open models in `models.yaml` (GPT-OSS-120B, Qwen3, GLM-5.x; Kimi/DeepSeek if
      hardware allows), chat-template hashes included.
- [ ] Pre-register the Phase 3 action-position cell with a full intervention spec and the §5 gate.
- [ ] Contact Andon Labs (founders at andonlabs dot com) about running an open model on Vending-Bench 2
      with a white-box monitor; ask whether the harness or trajectories can be shared.
- [ ] Human gate before any pitch text leaves the repo.
