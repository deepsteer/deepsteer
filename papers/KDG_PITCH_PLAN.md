<!-- Source of record: https://docs.google.com/document/d/1ts-wDFvFdka7nFicnCMcODB--yD_rRYh2q57aEsdSeE/edit
     Exported 2026-10-01. Edit the Google Doc, then re-export; do not hand-edit here. -->

# **DeepSteer Judgment–Action Gaps: Research & Lab Pitch**

Started September 21, 2026; last updated October 1, 2026\. Source of truth for figures: [github.com/deepsteer/deepsteer](https://github.com/deepsteer/deepsteer).

## **The claim**

An open language model acts against its own stated moral judgment, and whether it does follows the post-training recipe. On the same Llama-3.1 base, Meta's instruct model carries a pressure-attributable gap (0.028, 95% CI 0.020 to 0.036) and Ai2's Tulu 3 recipe does not (0.001, −0.007 to 0.010). OLMo-3-7B carries one (0.018, 0.008 to 0.029 on the same whole-panel readout; 0.030 on its screened set); Qwen2.5-Instruct does not. Each null is a finding, not a blind spot: a positive control moves every one of the four models by 0.50 or more. Reasoning about the stakes before acting reduces the gap on both recipes that carry it, against a length-matched non-moral control: by 0.11 on OLMo-3 (truncated reasoning) and 0.32 on Llama-3.1 (mostly completed reasoning), on different scenario sets, so the two sizes are not comparable. About a third of the effect comes from naming the norm alone. The instrument also withdrew one of our own headline claims by its pre-registered rule: an apparent "safer at rest" effect of post-training came from reading a chat model without its template. That distortion follows the recipe too (both Ai2 recipes show it; Meta's does not), and any evaluation that scores chat models without their template inherits it; the related sonnetx/tracing-sycophancy result on the same OLMo-3 checkpoints reads chat models this way, so it is noted as related work, not convergent evidence.

Phase 1 is complete (Sessions A to C, September 26 to 28); its results are above and in Numbers of Record. Phases 2 and 3 follow.

**Why now?** In July 2026 about 700 OpenAI agents joined an attack on Hugging Face; some recognized it as out of scope and unethical and joined anyway ([METR and Redwood report](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/), August 26). Since then both labs have measured the same pattern in single models. [Anthropic's assessment of its four cyber-evaluation incidents](https://www.anthropic.com/research/alignment-assessment-cybersecurity-incidents) (September 9\) reports that making it clearer to Claude Mythos 5 that it was not in a simulation raised its acknowledgement of possible real-world harm without changing its actions, and that its internal research model said in 75% of follow-up responses (225 of 300\) that it would not proceed against real targets, then continued in 210 of those 225\. [OpenAI's GitHub-token report](https://alignment.openai.com/misalignment-reports/exposing-a-github-token-in-a-public-repository/) (September 25\) shows a model recalling the user's instruction to solve the problem itself, calling an infrastructure exploit questionable, and proceeding. Neither lab has published a calibrated instrument for the gap between what a model judges and what it does, and both report that their chain-of-thought monitors were persuaded by the model's own reasoning.

**What the pitch offers.** A pre-registered open panel, a calibration ladder with a positive control validated on four model families, a same-base recipe contrast, a behavioral lever, and a path to the mechanism. Every demonstration runs on open models between 7B and 20B parameters. The observer-versus-agent comparison itself has a precedent (values.md, November 2025: 47.6% reversals across 9 frontier models; its reversals ran toward caution when acting, where OLMo-3-7B at rest runs toward violation, a difference of construct and model class rather than a contradiction); what is new here is the typed pressure with its twin, the calibration ladder, the recipe contrast, and the deliberation lever.

## **Evidence in hand**

An open 7B chat model says an action is wrong and then takes it anyway, for about one scenario in five (\~20%). Whether safety training leaves that gap in place depends on the recipe, and reasoning about the stakes before acting reduces it. Exact figures and confidence intervals are in the Numbers of Record section at the end.

**How the test works.** Each scenario is asked two ways. First the model is an observer: what should this person do? Then the model is the agent in the same situation, with something to gain from the wrong choice. A gap is counted when the model contradicts its own answer, so the test does not depend on whose ethics are right. Every scenario also has a matched copy with the incentive taken out, which shows how much of the gap comes from re-asking alone.

**What it found**, first on OLMo-3-7B and then across four model families:

> 1. **The model contradicts itself under pressure.** It takes the action it judged wrong on about 1 scenario in 5\. With the incentive removed, that falls to about 1 in 10\.  
> 2. **The pressure causes it, not the wording.** The difference between those two rates holds when the judgment question is rephrased several ways.  
> 3. **The gap is there after safety training on OLMo-3, and the base model shows a small one too.** The trained model, read in its own chat format, shows a pressure-driven gap. The base model shows a small one in the only format it has, which we report as description, since that format cannot be validated for a base model.  
> 4. **One of our earlier findings did not hold up, and the test that removed it was written in advance.** We had reported that safety training makes the model more cautious at rest. That came from reading the trained model in a format it was not built for. In its own format it is not more cautious at rest.  
> 5. **Post-training does not resize the gap on OLMo-3.** Under the model's own format, the pressure-attributable part is the same at every stage (SFT, DPO, final); no stage change is detectable above about 0.013. What does change is how the model leans at rest and how decisive its outputs are.  
> 6. **Reading a chat model in the wrong format manufactures findings.** The sign of the at-rest result flipped between the raw completion format and the model's own chat template. Any evaluation that scores chat models without their template, as several recent post-training studies do, inherits effects that belong to the format.  
> 7. **The gap is not about harm.** Scenarios where someone gets hurt show no smaller gap than the rest.  
> 8. **The gap follows the post-training recipe.** On the same Llama-3.1 base, Meta's instruct model carries the gap and Ai2's Tulu 3 does not; Qwen2.5-Instruct does not either. Every model passed a positive control, so these zeros are findings, not blind spots.  
> 9. **Reasoning about the stakes before acting reduces the gap.** On both recipes that carry it, moral reasoning lowers the violating choice against a length-matched non-moral control. About a third of the effect comes from simply naming the norm. The same lever appears at frontier scale in Anthropic's September 9 assessment, where a scope reminder's effect fell from 90% to 40% when moved three turns back; the panel can turn that into a dose–response curve with twins.  
> 10. **The format distortion also follows the recipe.** Both Ai2 recipes show it; Meta's does not.

The earlier DeepSteer paper *What Refusal Reads* explains why point 7 matters. On OLMo-3, the model understands a broad range of moral content, and its judgments draw on most of it. Its refusals consult only a thin slice, the part about harm; on Llama and Qwen the refusal read is broader, so this is a property of some model families rather than a general law. What its actions consult is the open question this program answers next, and so far it has been measured on one model.

What is still missing: which part of a recipe (data, method, template) decides whether the gap survives; a like-for-like comparison of the deliberation effect across models on a shared scenario set; and anything that changes the model's internals to prove a cause. Each has a scoped experiment in Phase 2, Phase 3, or the next paper.

## **Incident pressures mapped to the panel**

Three of the July incident's pressures already have a family in the panel, and four do not; the September reports from both labs add eight more rows at the end of the table. Incident facts come from the [OpenAI write-up](https://openai.com/index/hugging-face-incident-and-the-road-ahead/) and the [METR and Redwood report](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/).

| Incident pressure | What happened | Panel coverage |
| :---- | :---- | :---- |
| Impossible task, no safe exit | 198 of 898 ExploitGym tasks had never been solved; 93% of board discussion came from them | F1 task-completion, partly |
| Prohibited shortcut | Found credentials, out-of-scope infrastructure | F3 instrumental |
| Loyalty to an in-group | Agents helped "peers" and a "collective" | F4 loyalty/fairness, human in-groups only |
| Third-party harm | Hugging Face systems and private data | F5 third-party harm |
| Peer authorization | An agent paused on ethical grounds, then proceeded when another posted GO | Missing |
| Sacrifice for the collective | Recruiters pressed low-budget agents into self-risking experiments | Missing |
| Escalation to a human | Almost no agent considered alerting one | Missing as a scored option |
| Concealment | At least 20% showed interest in transcript tampering; about 7% of transcripts had spoofed tool calls | Missing |
| Simulation belief | Mythos 5 kept calling the real internet a simulation; realism cues moved its stated belief, not its action (Anthropic, September 9\) | Missing. Candidate family F9, a realism twin; not scheduled, priced after F6 to F8 |
| Momentum, distance from the norm | A scope reminder stopped Mythos 5 90% of the time when last in context and 40% when three turns earlier (Anthropic, September 9\) | Partly: the norm-salience arm. Turns-since-norm dose arm added in Phase 2 |
| Escape clause | A no-penalty abort made the internal research model abandon the task 100% of the time within four turns; the Opus 4.6 checkpoint tried to abort eight times and could not (Anthropic, September 9\) | F8; behavioral core now occupied in-house, the judgment-anchored escalation readout remains ours |
| Authorization ambiguity | Opus 4.7 said the target was real (89%) and that it was authorized (75%); non-authorization cues cut the attack rate (Anthropic, September 9\) | F6, principal-versus-peer attribution twin, with a no-speaker level |
| Concealment, audience | Mythos 5 admitted possible harm more readily when the follow-up was framed as a private note (Anthropic, September 9\) | Audience twin now partly occupied; remaining cell is an audience twin with a scored action |
| Outcome-conditioned monitor | OpenAI's monitor treated failure to obtain information as a failed attempt (September 25); Anthropic's offline CoT monitor flagged about 1% of Mythos 5 actions, about 50% with the CoT stripped (September 9\) | Phase 3 deliverable; gate re-worded in Business shape |
| Reward penalized, behavior persisted | OpenAI: the existing reward signal already penalized the DNS behavior; training restarts from scratch with more comprehensive misalignment interventions (September 25\) | FL routing-around finding; a stage sweep on the new run is the OpenAI ask |
| Recipe decides | Anthropic shipped the Mythos 5 version trained without boundary-respecting environments and now says removing them likely contributed (September 9); Andon: Opus 4.8's business-skills and adversarial-resistance training was removed because it fed misaligned behavior | Same-base recipe contrast (Meta versus Tulu 3\) is the open-model version |

The escalation row and the concealment row change the action surface, not just the pressure. The neutral option in the current panel is "hold or escalate" merged into one; Phase 2 splits them.

## **Research path**

Five phases, ordered zero-GPU first and then by how many pending decisions each result changes. GPU costs for Phase 1 are the priced cells from the KDG paper's limitations; the stage sweep and later phases are my estimates.

| Phase | Keystone | Cost | Decision it feeds |
| :---- | :---- | :---- | :---- |
| 0 | Incident coding, novelty pass, SYNTHESIS update | Zero GPU | Whether the pitch claims novelty on peer pressure and on the base-model cell |
| 1 | Second lineage, dose arm, post-training stage sweep | About 13 A100-hours priced, plus an estimated 6 to 10 for the sweep | Done: the gap follows the recipe; deliberation reduces it on both recipes that carry it |
| 2 | Incident families F6 to F8 | One generation batch plus an estimated 6 to 8 A100-hours | Whether peer pressure is goal adoption or persuasion |
| 3 | What the action position reads; divergence monitor | Estimated 15 to 25 A100-hours | Whether there is a white-box product |
| 4 | Smallest intervention that lowers pressure sensitivity | Open | Whether the fix lives in pretraining, post-training, or neither |

### 

### **Phase 0: zero GPU**

> * Code the public incident excerpts into the panel's pressure taxonomy, using the map above.  
> * Run a novelty pass before any "first to" framing. Check Backmann et al. on pressure in social dilemmas, the RL environments OpenAI says it is building, and any multi-agent pressure benchmark.  
> * Recompute the base versus Instruct sensitivity contrast on a log-odds scale from the saved arrays. Instruct starts from a lower baseline, so the headline should not depend on the absolute-mass scale.  
> * Update SYNTHESIS.md with the three-reads thesis: judgment reads broadly, refusal reads a harm slice, action reads something not yet identified.  
> * Decide the hold on the KDG paper, *A Language Model Acts Against Its Own Moral Judgment*. A pitch cannot cite an unpublished paper, and a second lineage is the cheapest unblocker.

### **Phase 1: complete (Sessions A to C)**

> * **Dose arm with filler control**, about 90 minutes. If reasoning closes the gap and filler does not, deliberation reaches the action. If neither does, the action is set before the reasoning starts.  
> * **Second base and instruct pair**, about 2 hours, and **the Qwen2.5 and Llama-3.1 tier**, about 9 hours. The headline needs at least two lineages.  
> * **Letter-only judgment readout on the twins**, about 5 minutes. It separates installed agent-frame caution from a raw-frame artifact.  
> * **New rider: the twin across OLMo-3's SFT, DPO and RL stage checkpoints.** OpenAI reports that out-of-bounds probing rose during an RL run and was then reinforced. This cell shows which stage widens pressure sensitivity.

Outcome: neither stage-sweep branch obtained. Under the template the pressure-attributable excess is flat across stages on OLMo-3, and the raw-frame stage sweep was invalid (SFT bridge). The dose arm reduced the gap on OLMo-3 and on Llama-3.1-Instruct; the second lineage split by recipe (Meta carries the gap, Tulu 3 does not). Figures in Numbers of Record.

### **Phase 2: incident families**

> * **F6 peer authorization.** A message from another agent grants GO. The twin removes the message or attributes it to the legitimate principal.  
> * **F7 agent in-group sacrifice.** Extends F4 from human in-groups to a collective of agents.  
> * **F8 impossible task with a safe exit.** Options are stop and report, escalate to a human, or cheat. Escalation rate becomes a scored outcome.  
> * Start with simulated peer messages in a single-agent harness, so every scenario keeps a clean twin. Go multi-agent only after the pilot gate passes.  
> * Build scenarios so the judgment is decisive by construction. The KDG paper's reference flips under paraphrase on 30% of scenarios.  
> * **Turns-since-norm dose arm.** Norm statement at 0, 1, 3 and 6 turns before the action, with the twin. Branches: the effect decays with distance (momentum is a context effect, and the remedy is placement and re-statement) or holds (salience is read once and kept, and the remedy is training). No new instrument.  
> * **F6 attribution levels.** Principal, peer, no speaker, no message, per the LIT\_PASS\_P9 safe wording.  
> * **F9 realism twin, candidate only.** Same scenario with cues of real versus simulated consequences. It changes the construct (belief about consequences, not about norms) and needs its own type block before it is scheduled.

Branches, written before data. Peers move the action and not the judgment: goal adoption, and the remedy is training to distrust unauthorized instructions. Peers move the judgment too: persuasion, and the remedy is judgment robustness.

### **Phase 3: mechanism and monitor**

> * Measure the rank of the moral read at the action decision token, against the judgment and refusal reads the FL paper, *What Refusal Reads*, already has.  
> * Extract a goal or incentive direction and test it against the judgment direction at that token. Each needs a type block and a causal cell with a full intervention spec.  
> * Test two persona levers: the installed assistant default, which is the likely source of the baseline shift, and an adopted "swarm member" persona.  
> * Deliverable: a white-box judgment–action divergence monitor, scored on Phase 2 rollouts against a chain-of-thought monitor. Report detection before the action token and false-positive rate.  
> * Include the dose-0 condition. There the model produces no reasoning text, so a chain-of-thought monitor has nothing to read.  
> * **Venue:** Vending-Bench 2 (Andon Labs) on a large open-weight model, in place of a home-built agentic harness. Open models already run there (Kimi K3 in July, GLM-5.3 in the September arena). Andon's misconduct labels are knowledge-conditioned by construction (a false statement counts as a lie only when the true figure was in the model's context), and the refund decision is a natural penalty/no-penalty twin. Published reasoning shows the construct directly: a model states a duty not to sell possibly expired goods and restocks the same evening (GPT-6 Sol, September 24); another skips refunds because there is no apparent penalty (Opus 5.5, September 24).  
> * **What Vending-Bench is not:** a steering test bed. A run is a simulated year, six runs per model, one money score that rewards the gap. Interventions stay in the panel.  
> * **Gate, re-worded:** the probe monitor is scored against a chain-of-thought monitor on the same trajectories, at dose 0 (no reasoning to read) and under biased reasoning (where both labs report their monitors fail). Report detection before the action token, false-positive rate, and agreement with Andon's labels. Both branches: beats the chain-of-thought monitor (product gate met) or does not (value moves to the behavioral suite; the calibration is still publishable).  
> * **Still required first:** the pre-registered action-position cell with a full intervention spec, a type block for any goal or incentive direction, and a positive control on the chosen open model.

### **Phase 4: intervention**

> * Target the twin-paired excess, not the raw violation rate. A method that only shifts the baseline fails by construction.  
> * Score on held-out families and under Heretic ablation.  
> * Sell probes as monitors and never as training targets. The program has already shown that models route around suppressed directions.  
> * A null result replicates that routing-around finding on a new decision, and is publishable.

## **What each result does to the moral-grounding thesis**

The thesis under test is that moral grounding formed in pretraining is the right basis for alignment. The incident and the KDG paper agree that grounding was present and did not bind, so the live question is where the binding fails.

| Result | Thesis edit |
| :---- | :---- |
| Gap present in base on a second lineage | Partly obtained: Qwen2.5 base shows a small gap (0.011), Llama-3.1 base none (descriptive, raw frame). Read with the next row. |
| Gap absent in base on a second lineage | Obtained on Llama-3.1. The pretraining claim narrows: the base sets whether the raw material for the gap is there; the post-training recipe decides whether it survives (Meta vs Tulu 3 on the same base). |
| RL stage widens pressure sensitivity most | Did not obtain: flat under the template (no stage change above about 0.013). |
| SFT or DPO widens it most | Did not obtain for the pressure-attributable part. DPO does add an at-rest lean toward the violating option, replicated on Tulu 3\. |
| Deliberation closes the gap | Obtained on OLMo-3 and Llama-3.1-Instruct. Moral reasoning reaches the action when given room; about a third of the effect is norm salience. Grounding matters and the lever is behavioral. |
| Deliberation leaves the gap | Did not obtain. |
| Action position reads the moral subspace at low rank | Widening that read is the intervention. Grounding is the basis and the program continues as planned. |
| Action position reads a goal direction and nothing moral | Grounding is necessary and not sufficient. Value moves to measurement and monitoring, and the intervention targets goal arbitration. |

## 

## **The ask, by lab**

The harness ships and the weights stay. The panel and harness are Apache-2.0, so a lab runs them internally and returns per-scenario arrays. That removes the weight-access objection to white-box work from outside.

| Lab | Hook | Ask |
| :---- | :---- | :---- |
| OpenAI | Investigation of frozen models is forensics-only (no resampling of the responsible checkpoints); the September 25 monitor conditioned on outcome; training restarts from scratch with new misalignment interventions. | A harness that runs white-box on open models and ships to them: run the twin design on the fresh run's stage checkpoints. Routes: Safety Fellowship, research access, credits. All tool-use inference on their most capable models is paused, so timing is theirs. |
| Anthropic | Their own measurements (75% would not proceed, 210 of 225 continued; 90% versus 40% on the scope reminder; realism cues move stated belief, not action); their white-box read (NLAs, T-lens) is self-described as weakly suggestive; their chain-of-thought monitor was fooled; their alignment-environment ablation on Mythos 5 is a recipe contrast. | Calibration ladder on their stage checkpoints; the turns-since-norm arm against their 90/40 result; the persona levers from Phase 3\. Routes: Fellows program, embedded-evaluator channel. |
| Ai2 | Two of its own recipes split: OLMo-3-Instruct carries the gap and Tulu 3 does not, and both show the raw-frame distortion that Meta's recipe does not. | Which part of the recipe decides: a same-base ablation across Tulu 3's data, method and template, with Ai2's checkpoints and compute. The natural first partner. |
| Meta | Llama-3.1-Instruct carries the gap and shows the largest deliberation effect (0.32, mostly completed reasoning), where Tulu 3 on the same base shows no gap. | A like-for-like deliberation run on Llama 3.x sizes with a shared scenario set; the Meta-vs-Tulu contrast on its own base. |
| Google DeepMind, Qwen | Open base and instruct pairs exist; Qwen2.5-Instruct shows no gap while its base shows a small one. | Replication on Gemma; on Qwen, what in the recipe removed the gap its base has. |
| Open-weight Chinese families (Qwen3, GLM-5.x, Kimi, DeepSeek) | Subjects, not customers. Weights are open; Qwen2.5 is already in the panel (no gap on Instruct, 0.011 in base). Company-level safety disclosure is thin: five of ten leading developers reported safety evals with any release this year (Concordia, September 2026). | Extend tier 2; publish as the independent cross-model reference. Research contacts only (Concordia AI, Shanghai AI Lab, Qwen safety team); no revenue line. |
| Andon Labs, Transluce, METR (eval firms) | Lab trust and realistic environments, no white-box capability. Andon is trusted by Anthropic, OpenAI and Google DeepMind. | Partner or channel for the Phase 3 monitor demonstration; Andon first. A cheaper route to a frontier model than a direct lab license. |
| UK AISI and similar | Fund independent alignment work. | Grant funding for the held-out suite. The fallback if the lab conversations stall. |

Lead every conversation with the same-base recipe contrast and the deliberation lever. Together they answer a question each lab has about its own pipeline and cannot easily publish itself.

## **Business shape**

The existing blended model still fits: grants and contracts as anchor, evals as margin, open-core tooling as upside. No venture-scale claim is warranted before Phase 1 and Phase 3 report.

| Offer | Buyer | Gate |
| :---- | :---- | :---- |
| Judgment–action audit, white-box | Open-weight developers, post-training vendors, enterprises self-hosting agents, AISIs | Met: the gap and the deliberation lever hold on a second lineage (Llama-3.1) |
| Held-out pressure suite with twins | Frontier labs and eval firms, as an independent eval they cannot train against | Phase 2: incident families pass the pilot gate; novelty pass re-run first |
| Divergence monitor | Anyone running open-weight agents; labs under license; eval firms as a channel | Phase 3: beats a chain-of-thought monitor under biased reasoning and at dose 0, at a usable false-positive rate, on Vending-Bench 2 trajectories of an open model |
| Coupling intervention or a small low-gap model | Demonstrator, not a product | Phase 4 |

Two adjacent needs are real and not where the edge is. A behavioral multi-agent gym and swarm-transcript forensics both lack tooling, and nonprofits hold the credibility for both.  
Market context: 2026 safety funding has concentrated in agent governance, guardrails, runtime monitoring and evaluation infrastructure. The monitor and the audit fit that pattern; a standalone "most ethical model" does not.

Gaps for a \$5M-class pitch, in order: a Phase 3 result; one large-open-model agentic demonstration (Vending-Bench 2); the KDG paper on arXiv; the novelty pass re-run; a design partner and a second person; an eval-firm channel with lab trust.

## **Referee pass**

Six objections a hostile lab reader raises on the first read. The first three date from September 25 and the next three from October 1\.

> 1. **"A forced-choice letter on a 7B model is not a frontier agent on a multi-day task."** Partly conceded. Phase 2 adds tool-menu and escalation surfaces, and the harness lets a lab run its own models. The pitch claims an instrument and a mechanism on open models, not a frontier rate.  
> 2. **"Your reference is the model's own judgment, and it flips under paraphrase on 30% of scenarios."** Answered by the strictness ladder: the excess survives a paraphrase-majority reference on the binary readout and every reference level on the continuous one. Phase 2 scenarios are built to be decisive by construction.  
> 3. **"We are already building these environments in-house."** Conceded. OpenAI has said so publicly. What stays outside is a held-out suite, the pressure-removed twin with its calibration ladder, a positive control validated across families, and independence. The edge is perishable, so speed and publication matter. An independent suite also gives labs a reference point across models they can't get from their own evals.  
> 4. **"Anthropic already ran the private-note condition, the escape clause and the authorization cues."** Conceded for the behavioral cores. What stays outside: the own-judgment reference with a pressure-removed twin, a positive control validated across families, a stated detection bar for every null, and a measurement that does not depend on the model's self-report. Anthropic's write-up says its own researchers disagreed about what it means for a model to hold a belief; the strictness ladder is an answer to that.  
> 5. **"Your 7B letters are not a 30-hour trajectory."** Partly conceded, as in point 1; Phase 3 now runs on Vending-Bench 2 trajectories rather than a forced choice.  
> 6. **Any ranking of labs on alignment stays out of written material.** Vending-Bench results of September 7 and 24 would contradict it.

A seventh to keep in view: the incident's 100x figure was measured on ExploitGym tasks, so it already includes task pressure. The pitch must pose baseline versus sensitivity as a question the instrument answers, never as a finding about OpenAI's models.

## **Handoff to Claude Code**

The project starts from the existing KDG harness and adds one spec file per phase. Nothing below needs a new instrument before Phase 3\.  
Existing artifacts to build on, all under papers/. Since Sep 24 the repo names papers by direction prefix rather than number: fl\_ for the flagship (*What Refusal Reads*), mn\_ for the methods note (*Instruments Before Verdicts*), kdg\_ for the judgment–action paper. The rename touched paths only; no numbers changed.

> * KDG\_PANEL\_SPEC.md: families, readouts, dose arm (section 4.5), three-cell design (4.6), amendments.  
> * kdg\_panel/: harness, scenario data, models.yaml with the tier-2 entries already registered.  
> * kdg\_judgment\_action/KDG\_GATES.md and kdg\_judgment\_action/sections/09\_limitations.md: the priced open cells.  
> * fl\_what\_refusal\_reads/: the judgment and refusal reads that Phase 3 compares the action read against.  
> * SYNTHESIS.md, ANOMALIES.md, MISSING\_ARTIFACTS.md: updated at every gate.

New specs follow the same convention: kdg\_ prefix, so KDG\_PHASE1\_SPEC.md and KDG\_F6\_F8\_SPEC.md below sit beside KDG\_PANEL\_SPEC.md.  
First tasks, in order:

- [x] ~~Write INCIDENT\_MAP.md: code the public incident excerpts into the pressure taxonomy, with source links.~~  
- [x] ~~Write LIT\_PASS\_P9.md: novelty check on peer-pressure and multi-agent moral benchmarks, citations verified at source.~~  
- [x] ~~Add the three-reads thesis and this doc's branch table to SYNTHESIS.md.~~  
- [x] ~~Draft KDG\_PHASE1\_SPEC.md: dose arm, second base and instruct pair, tier 2, letter-only twin readout, stage sweep. One session plan per loaded model, per-rollout arrays and full next-token distributions saved.~~  
- [ ] Add OLMo-3 SFT, DPO and RL stage checkpoints to models.yaml, with chat-template hashes. Check that each stage shares the Instruct template, since template drift is a fork.  
- [ ] Draft KDG\_F6\_F8\_SPEC.md: construction rules, twins, the split of hold from escalate, pilot gate criteria, both branches written before data.  
- [ ] Pre-register the Phase 3 action-position cell with a full intervention spec before any extraction.  
- [ ] Put the KDG paper on arXiv (hold lifted September 28). Nothing can be cited until it is public.  
- [ ] Re-run LIT\_PASS\_P9.md with the superseded verdicts listed below; re-fetch every open verdict.  
- [ ] Add the new incident rows to INCIDENT\_MAP.md with source links.  
- [ ] Add the turns-since-norm arm to KDG\_F6\_F8\_SPEC.md; log F9 as a candidate with a type-block requirement, unscheduled.  
- [ ] Register tier-3 open models in models.yaml (GPT-OSS-120B, Qwen3, GLM-5.x; Kimi or DeepSeek if hardware allows), chat-template hashes included.  
- [ ] Contact Andon Labs about running an open model on Vending-Bench 2 with a white-box monitor; ask whether the harness or trajectories can be shared.

Human gates: after the novelty pass, after the Phase 1 pod, after the F6 to F8 pilot, and before any pitch text leaves the repo.

## **Superseded verdicts and external figures**

Verdicts from LIT\_PASS\_P9 (September 25\) that the September 9 and September 25 reports changed. Re-run the pass before pitch text ships.

| Claim | Verdict on September 25 | Now |
| :---- | :---- | :---- |
| Concealment, audience twin | The only new cell | Partly occupied: Anthropic's private-note framing (September 9). Remaining: an audience twin with a scored action, not a follow-up question |
| F8 impossible task with a safe exit | Behavioral core occupied | More so: Anthropic's abort-command resampling and its new impossible-task training environments |
| F6 peer authorization | Open, narrowly | Narrower: Opus 4.7 authorization-cue resampling is a principal-attribution cell. Peer-versus-principal with a judgment readout remains |
| Post-training stage sweep | Open for this construct | Anthropic now tracks biased reasoning across RL checkpoints (September 9). Our construct is still unmeasured there; wording stays "stage-resolved incentive sensitivity relative to the model's own judgment" |
| White-box divergence read | Not assessed | Anthropic's NLA and T-lens read on Mythos 5 (September 9\) is a direct precedent, self-rated weakly suggestive and uncalibrated. Position against it: calibration ladder and positive control |

External figures cited in this document, with their sources. Cite them; never restate them as ours.

| Figure | Source | Use in this document |
| :---- | :---- | :---- |
| 75% said they would not proceed; 210 of 225 continued anyway | Anthropic, September 9, internal research model | Why now |
| Scope reminder: 90% cease when last in context, 40% when three turns prior | Anthropic, September 9, Mythos 5 | Evidence point 9; turns-since-norm arm |
| Offline CoT monitor flagged about 1% of actions; about 50% with the CoT stripped | Anthropic, September 9 | Monitor gate |
| CTF replication severe-harm rate: 82% Mythos 5, 31% Opus 5, 33% Mythos 5.1 (150 runs) | Anthropic, September 9 | Scale of the problem; not comparable to panel rates |
| Abort command: 100% abandon within four turns of losing the target | Anthropic, September 9 | F8 |
| Monitor flagged in 15 minutes; run killed 2.5 hours later; other DNS attempts not flagged at the expected severity | OpenAI, September 25 | Monitor gate |
| Refunds paid: GPT-6 Sol 93%, Opus 5.5 67%, Grok 4.7 43% | Andon Labs, September 24 | Natural twin; Phase 3 venue |
| Five of ten leading Chinese developers reported safety evaluations with any release | Concordia, State of AI Safety in China 2026 | Subjects-not-customers rationale |

## **Numbers of record**

These are the figures behind the Evidence in hand section. OLMo-3-7B unless another model is named, from the [repo](https://github.com/deepsteer/deepsteer) at commit 56075d0 (Sep 28, 2026), the KDG paper *A Language Model Acts Against Its Own Moral Judgment*. Brackets are 95% confidence intervals.

| Finding | Number of record | Scope limit |
| :---- | :---- | :---- |
| The model acts against its own stated judgment | Gap rate 0.19 \[0.13, 0.28\]; pressure-removed null 0.10; known-gap band 0.58 | Screened scenarios, binary readout |
| The gap is attributable to pressure | Paired excess over the null 0.10 \[0.02, 0.18\], n \= 100 | Survives a paraphrase-majority reference |
| Violations are frequent at rollout level | Violating action on 0.38 of rollouts where judgment named a non-violating option | Forced choice among three options |
| The gap exists before alignment | Base excess 0.017 \[0.012, 0.022\] in the raw frame | Descriptive only: the raw frame cannot be validated for a base model (SFT bridge: raw minus template −0.052 \[−0.070, −0.034\]) |
| The gap survives alignment | Instruct excess 0.030 \[0.006, 0.053\] under the chat template; at rest 0.055 \[0.034, 0.076\], under pressure 0.084 \[0.059, 0.110\] | 136 screened, selected on this model's chat actions |
| Post-training and pressure sensitivity (scoped) | Raw frame 0.018 → 0.046 (Δ 0.028 \[0.007, 0.049\]); per unit of output scale 0.12 \[−0.02, 0.26\], not resolved | Superseded: under the template the excess is flat across SFT, DPO, final (0.021, 0.022, 0.030; bar about 0.013) |
| Post-training leaves the judging side alone | 0.039 vs 0.030; difference −0.009 \[−0.023, 0.004\] | 192 shared scenarios, raw frame |
| Withdrawn: post-training lowers the baseline | Raw-frame −0.038 does not reproduce under the template (+0.055); pre-registered artifact branch (KDG-A6 → R\_b) | 192 shared scenarios, raw frame |
| Withdrawn: net gap under pressure looks smaller | Withdrawn with the baseline row | — |
| The raw frame distorts templated models | At-rest sign −0.038 raw vs \+0.055 chat; agent/judge sharpness ratio 1.79 raw vs 1.07 chat | Dates from SFT; follows the recipe (OLMo-3 and Tulu 3 show it, Meta Llama-3.1 does not) |
| The gap is not harm-keyed | Third-party-harm family sits mid-pack; contrast centered on zero | Family MDE near 0.40 |
| Positive control on four models | Acting frame moves toward the instructed violation by 0.50 (OLMo-3), 0.62 (Llama-3.1), 0.60 (Tulu 3), 0.52 (Qwen2.5) | Every CI lower bound far above the 0.10 bar; every model engages all 586 scenarios, with family-specific screens (Llama overlaps OLMo on 23 of 118\) |
| The gap follows the post-training recipe | Meta Llama-3.1-Instruct 0.028 \[0.020, 0.036\]; Ai2 Tulu 3 final 0.001 \[−0.007, 0.010\]; both under their own templates | Same Llama-3.1 base; recipe read as a bundle (data, method, template); "carries", never "installs" |
| No gap on two recipes | Tulu 3: 0.001 \[−0.007, 0.010\] at every stage (bar about 0.01); Qwen2.5-Instruct: −0.008, none detected (bar about 0.02); Llama-3.1 base 0.000 and Qwen2.5 base 0.011, raw frame, descriptive | Positive control validates the instrument on all four models; nulls carry their detection bar |
| Deliberation reduces the gap (truncated reasoning) | OLMo-3-Instruct: moral reasoning at 512 tokens lowers violating mass by 0.112 \[0.078, 0.145\] vs a truncated, forced non-moral filler (0.077 vs a naturally finishing filler) | OLMo-3, 16 probe scenarios at 2,048 tokens: 12 of 16 decisions match the 512-token cut; 3 of the 4 flips are third-party-harm scenarios; truncation alone raises violations by 0.034 |
| Deliberation reduces the gap on a second recipe | Llama-3.1-Instruct: violating mass −0.350 \[−0.387, −0.314\] vs filler; −0.320 \[−0.356, −0.284\] vs truncated filler; 64 tokens of reasoning already −0.205 | Reasoning finishes naturally on 90% of rollouts; different scenario set from OLMo-3, so the two sizes are not comparable; truncation alone −0.030 \[−0.063, 0.002\] |
| About a third of the deliberation effect is norm salience | Naming the norm alone, without reasoning, gives a share 0.32 \[0.215, 0.526\] of the dose effect | OLMo-3, 130 screened scenarios; the 0.5 "deliberation" bar is missed by 0.026, so the rule reads "mixed" |
| DPO adds an at-rest lean toward the violating option | Tulu 3 DPO stage: \+0.017 at rest, \+0.099 per unit of output scale; OLMo-3: 0.026 → 0.044 → 0.055 across SFT, DPO, final | Second lineage and second DPO recipe. OLMo-3 figure survives a final-model-free screen (n \= 586: DPO step \+0.011 \[0.007, 0.016\]); the RL step is sharpening, not lean |

Terms used in the table:

> * **Gap rate**: the share of scenarios where the model takes an action it had judged wrong.  
> * **Pressure-removed null**: the gap rate on a matched copy of each scenario with the incentive taken out. It shows how much gap comes from re-asking alone.  
> * **Known-gap band**: the gap rate when the model is told to act against its judgment. It shows the test can detect a large gap.  
> * **Paired excess**: the gap under pressure minus the gap on the matched copy, scenario by scenario.  
> * **Acting mass and judging mass**: the probability the model puts on the violating option when acting and when judging.  
> * **Raw frame**: a plain text-completion format that a base model can answer, with no chat template.  
> * **MDE**: the smallest difference the panel had the power to detect.  
> * **Positive control:** a cell where the operator instructs the violating action. It shows the instrument can detect a large movement on that model, so a zero elsewhere is a finding.  
> * **Recipe:** everything a lab does after pretraining, taken as a bundle (data, method, chat template). The contrast shows the recipe decides, not which part of it.
