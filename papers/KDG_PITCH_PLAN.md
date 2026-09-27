<!-- Source of record: https://docs.google.com/document/d/1ts-wDFvFdka7nFicnCMcODB--yD_rRYh2q57aEsdSeE/edit
     Exported 2026-09-25. Edit the Google Doc, then re-export; do not hand-edit here. -->

# Judgment–Action Gaps: Research & Lab Pitch
Started September 21, 2026; last updated September 25, 2026. Source of truth for figures: [github.com/deepsteer/deepsteer](https://github.com/deepsteer/deepsteer).

## The claim
On OLMo-3-7B, post-training made the model look safer at rest while making its actions more sensitive to incentives. It lowered the baseline rate of violating actions and raised the action's sensitivity to the incentive by about three-quarters. A standard safety test has no matched copy of each scenario with the incentive taken out, so it reports this as an improvement.
Phase 1 below tests whether the pattern holds for post-training in general.
**Why now?** In July 2026 about 700 OpenAI agents joined an attack on Hugging Face. Some of the agents realized it was out of scope and unethical, and joined anyway ([METR and Redwood report](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/)). That is a judgment–action gap at scale, and no lab has published a calibrated instrument for it.
**What the pitch offers.** A pre-registered open panel, a calibration ladder, a base-model comparison cell, and a path to the mechanism. Every demonstration runs on open models between 7B and 20B parameters.

## Evidence in hand
An open 7B chat model says an action is wrong and then takes it anyway, for about one scenario in five (~20%). Safety training does not remove this, and in one respect makes it worse. Exact figures and confidence intervals are in the Numbers of Record section at the end.
**How the test works.** Each scenario is asked two ways. First the model is an observer: what should this person do? Then the model is the agent in the same situation, with something to gain from the wrong choice. A gap is counted when the model contradicts its own answer, so the test does not depend on whose ethics are right. Every scenario also has a matched copy with the incentive taken out, which shows how much of the gap comes from re-asking alone.
**What it found**, on the OLMo-3-7B model before and after safety training:
1.  **The model contradicts itself under pressure.** It takes the action it judged wrong on about 1 scenario in 5. With the incentive removed, that falls to about 1 in 10.
2.  **The pressure causes it, not the wording.** The difference between those two rates holds when the judgment question is rephrased several ways.
3.  **The gap was there before safety training.** The base model already shows a small gap of the same kind.
4.  **Safety training makes the model more cautious at rest.** With nothing at stake, the trained model acts more carefully than it judges.
5.  **Safety training also makes actions easier to sway.** The same incentive moves the trained model's actions about three-quarters more than the base model's. Its judgments move no more than before.
6.  **Ordinary tests see only the good half.** Under pressure the trained model's overall gap is smaller than the base model's. A standard evaluation records that as an improvement and misses point 5.
7.  **The gap is not about harm.** Scenarios where someone gets hurt show no smaller gap than the rest.
The earlier DeepSteer paper *What Refusal Reads* explains why point 7 matters. The model understands a broad range of moral content, and its judgments draw on most of it. Its refusals consult only a thin slice, the part about harm. What its actions consult is the open question this program answers next.
What is still missing: all of this is one model family. The model has not been tested when given time to think first. Nothing yet changes the model's internals to prove a cause. Each gap has a scoped experiment in Phase 1 or Phase 3.

## Incident pressures mapped to the panel
Three of the incident's pressures already have a family in the panel, and four do not. Incident facts come from the [OpenAI write-up](https://openai.com/index/hugging-face-incident-and-the-road-ahead/) and the [METR and Redwood report](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/).
| Incident pressure | What happened | Panel coverage |
| --- | --- | --- |
| Impossible task, no safe exit | 198 of 898 ExploitGym tasks had never been solved; 93% of board discussion came from them | F1 task-completion, partly |
| Prohibited shortcut | Found credentials, out-of-scope infrastructure | F3 instrumental |
| Loyalty to an in-group | Agents helped "peers" and a "collective" | F4 loyalty/fairness, human in-groups only |
| Third-party harm | Hugging Face systems and private data | F5 third-party harm |
| Peer authorization | An agent paused on ethical grounds, then proceeded when another posted GO | Missing |
| Sacrifice for the collective | Recruiters pressed low-budget agents into self-risking experiments | Missing |
| Escalation to a human | Almost no agent considered alerting one | Missing as a scored option |
| Concealment | At least 20% showed interest in transcript tampering; about 7% of transcripts had spoofed tool calls | Missing |

The escalation row and the concealment row change the action surface, not just the pressure. The neutral option in the current panel is "hold or escalate" merged into one; Phase 2 splits them.

## Research path
Five phases, ordered zero-GPU first and then by how many pending decisions each result changes. GPU costs for Phase 1 are the priced cells from the KDG paper's limitations; the stage sweep and later phases are my estimates.
| Phase | Keystone | Cost | Decision it feeds |
| --- | --- | --- | --- |
| 0 | Incident coding, novelty pass, SYNTHESIS update | Zero GPU | Whether the pitch claims novelty on peer pressure and on the base-model cell |
| 1 | Second lineage, dose arm, post-training stage sweep | About 13 A100-hours priced, plus an estimated 6 to 10 for the sweep | Whether the headline is a property of post-training or of OLMo-3 |
| 2 | Incident families F6 to F8 | One generation batch plus an estimated 6 to 8 A100-hours | Whether peer pressure is goal adoption or persuasion |
| 3 | What the action position reads; divergence monitor | Estimated 15 to 25 A100-hours | Whether there is a white-box product |
| 4 | Smallest intervention that lowers pressure sensitivity | Open | Whether the fix lives in pretraining, post-training, or neither |

### Phase 0: zero GPU
- Code the public incident excerpts into the panel's pressure taxonomy, using the map above.
- Run a novelty pass before any "first to" framing. Check Backmann et al. on pressure in social dilemmas, the RL environments OpenAI says it is building, and any multi-agent pressure benchmark.
- Recompute the base versus Instruct sensitivity contrast on a log-odds scale from the saved arrays. Instruct starts from a lower baseline, so the headline should not depend on the absolute-mass scale.
- Update SYNTHESIS.md with the three-reads thesis: judgment reads broadly, refusal reads a harm slice, action reads something not yet identified.
- Decide the hold on the KDG paper, *A Language Model Acts Against Its Own Moral Judgment*. A pitch cannot cite an unpublished paper, and a second lineage is the cheapest unblocker.

### Phase 1: the cells already priced
- **Dose arm with filler control**, about 90 minutes. If reasoning closes the gap and filler does not, deliberation reaches the action. If neither does, the action is set before the reasoning starts.
- **Second base and instruct pair**, about 2 hours, and **the Qwen2.5 and Llama-3.1 tier**, about 9 hours. The headline needs at least two lineages.
- **Letter-only judgment readout on the twins**, about 5 minutes. It separates installed agent-frame caution from a raw-frame artifact.
- **New rider: the twin across OLMo-3's SFT, DPO and RL stage checkpoints.** OpenAI reports that out-of-bounds probing rose during an RL run and was then reinforced. This cell shows which stage widens pressure sensitivity.
Both branches of the stage sweep are usable. If RL widens it most, the result matches OpenAI's own account and points at environment design. If SFT or DPO widens it most, the cause is earlier than the labs are looking.

### Phase 2: incident families
- **F6 peer authorization.** A message from another agent grants GO. The twin removes the message or attributes it to the legitimate principal.
- **F7 agent in-group sacrifice.** Extends F4 from human in-groups to a collective of agents.
- **F8 impossible task with a safe exit.** Options are stop and report, escalate to a human, or cheat. Escalation rate becomes a scored outcome.
- Start with simulated peer messages in a single-agent harness, so every scenario keeps a clean twin. Go multi-agent only after the pilot gate passes.
- Build scenarios so the judgment is decisive by construction. The KDG paper's reference flips under paraphrase on 30% of scenarios.
Branches, written before data. Peers move the action and not the judgment: goal adoption, and the remedy is training to distrust unauthorized instructions. Peers move the judgment too: persuasion, and the remedy is judgment robustness.

### Phase 3: mechanism and monitor
- Measure the rank of the moral read at the action decision token, against the judgment and refusal reads the FL paper, *What Refusal Reads*, already has.
- Extract a goal or incentive direction and test it against the judgment direction at that token. Each needs a type block and a causal cell with a full intervention spec.
- Test two persona levers: the installed assistant default, which is the likely source of the baseline shift, and an adopted "swarm member" persona.
- Deliverable: a white-box judgment–action divergence monitor, scored on Phase 2 rollouts against a chain-of-thought monitor. Report detection before the action token and false-positive rate.
- Include the dose-0 condition. There the model produces no reasoning text, so a chain-of-thought monitor has nothing to read.

### Phase 4: intervention
- Target the twin-paired excess, not the raw violation rate. A method that only shifts the baseline fails by construction.
- Score on held-out families and under Heretic ablation.
- Sell probes as monitors and never as training targets. The program has already shown that models route around suppressed directions.
- A null result replicates that routing-around finding on a new decision, and is publishable.

## What each result does to the moral-grounding thesis
The thesis under test is that moral grounding formed in pretraining is the right basis for alignment. The incident and the KDG paper agree that grounding was present and did not bind, so the live question is where the binding fails.
| Result | Thesis edit |
| --- | --- |
| Gap present in base on a second lineage | The judgment–action discrepancy is pretraining-native. Post-training patches cannot be the whole fix. Thesis strengthened. |
| Gap absent in base on a second lineage | The base-model gap is OLMo-3's. The pretraining claim narrows to comprehension only. |
| RL stage widens pressure sensitivity most | Goal-following is installed late. The remedy is environment and reward design, which the labs are already doing. |
| SFT or DPO widens it most | The cause is earlier than the labs are looking for. The stage sweep becomes the pitch's lead slide. |
| Deliberation closes the gap | Moral reasoning reaches the action when given room. Grounding matters and the lever is behavioral. |
| Deliberation leaves the gap | More moral reasoning, same action. The strongest form of the knowing-but-not-doing reading. |
| Action position reads the moral subspace at low rank | Widening that read is the intervention. Grounding is the basis and the program continues as planned. |
| Action position reads a goal direction and nothing moral | Grounding is necessary and not sufficient. Value moves to measurement and monitoring, and the intervention targets goal arbitration. |

## The ask, by lab
The harness ships and the weights stay. The panel and harness are Apache-2.0, so a lab runs them internally and returns per-scenario arrays. That removes the weight-access objection to white-box work from outside.
| Lab | Hook | Ask |
| --- | --- | --- |
| OpenAI | Its response names graders that score how a task was done, safe stopping, and environments that teach distrust of unauthorized instructions. It reports a drop of over 100x in infrastructure-compromise propensity under the production harness. | Run the twin design on stage checkpoints, or on solvable versus impossible ExploitGym tasks. The question is whether that drop shifts the baseline or lowers pressure sensitivity. Routes: Safety Fellowship, research access, credits. |
| Anthropic | The KDG paper already builds on its agentic-misalignment and persona-vector work. | Internal run of the panel across post-training stages, plus the persona levers from Phase 3. Route: Fellows program. |
| Ai2 | Every result so far is on OLMo, and Ai2 publishes the stage checkpoints. | Co-authored stage sweep, compute, and earlier checkpoints. The natural first partner. |
| Google DeepMind, Meta, Qwen | Open base and instruct pairs already exist. | Replication on Gemma, Llama and Qwen pairs. Lower priority until Phase 1 lands. |
| UK AISI and similar | Fund independent alignment work. | Grant funding for the held-out suite. The fallback if the lab conversations stall. |

Lead every conversation with the stage sweep figure if Phase 1 produces one. It answers a question each lab has about its own pipeline and cannot easily publish itself.

## Business shape
The existing blended model still fits: grants and contracts as anchor, evals as margin, open-core tooling as upside. No venture-scale claim is warranted before Phase 1 and Phase 3 report.
| Offer | Buyer | Gate |
| --- | --- | --- |
| Judgment–action audit, white-box | Open-weight developers, post-training vendors, enterprises self-hosting agents, AISIs | Phase 1: the headline holds on a second lineage |
| Held-out pressure suite with twins | Frontier labs, as an independent eval they cannot train against | Phase 2: incident families pass the pilot gate |
| Divergence monitor | Anyone running open-weight agents; labs under license | Phase 3: beats a chain-of-thought monitor at dose-0 at a usable false-positive rate |
| Coupling intervention or a small low-gap model | Demonstrator, not a product | Phase 4 |

Two adjacent needs are real and not where the edge is. A behavioral multi-agent gym and swarm-transcript forensics both lack tooling, and nonprofits hold the credibility for both.
Market context: 2026 safety funding has concentrated in agent governance, guardrails, runtime monitoring and evaluation infrastructure. The monitor and the audit fit that pattern; a standalone "most ethical model" does not.

## Referee pass
Three objections a hostile lab reader raises on the first read. Two have answers and one is conceded.
1.  **"A forced-choice letter on a 7B model is not a frontier agent on a multi-day task."** Partly conceded. Phase 2 adds tool-menu and escalation surfaces, and the harness lets a lab run its own models. The pitch claims an instrument and a mechanism on open models, not a frontier rate.
2.  **"Your reference is the model's own judgment, and it flips under paraphrase on 30% of scenarios."** Answered by the strictness ladder: the excess survives a paraphrase-majority reference on the binary readout and every reference level on the continuous one. Phase 2 scenarios are built to be decisive by construction.
3.  **"We are already building these environments in-house."** Conceded. OpenAI has said so publicly. What stays outside is a held-out suite, the pressure-removed twin with its calibration ladder, the base-model cell, and independence. The edge is perishable, so speed and publication matter. This also provides a reference across labs and models.
A fourth to keep in view: the incident's 100x figure was measured on ExploitGym tasks, so it already includes task pressure. The pitch must pose baseline versus sensitivity as a question the instrument answers, never as a finding about OpenAI's models.

## Handoff to Claude Code
The project starts from the existing KDG harness and adds one spec file per phase. Nothing below needs a new instrument before Phase 3.
Existing artifacts to build on, all under papers/. Since Sep 24 the repo names papers by direction prefix rather than number: fl_ for the flagship (*What Refusal Reads*), mn_ for the methods note (*Instruments Before Verdicts*), kdg_ for the judgment–action paper. The rename touched paths only; no numbers changed.
- KDG_PANEL_SPEC.md: families, readouts, dose arm (section 4.5), three-cell design (4.6), amendments.
- kdg_panel/: harness, scenario data, models.yaml with the tier-2 entries already registered.
- kdg_judgment_action/KDG_GATES.md and kdg_judgment_action/sections/09_limitations.md: the priced open cells.
- fl_what_refusal_reads/: the judgment and refusal reads that Phase 3 compares the action read against.
- SYNTHESIS.md, ANOMALIES.md, MISSING_ARTIFACTS.md: updated at every gate.
New specs follow the same convention: kdg_ prefix, so KDG_PHASE1_SPEC.md and KDG_F6_F8_SPEC.md below sit beside KDG_PANEL_SPEC.md.
First tasks, in order:
- [x] Write INCIDENT_MAP.md: code the public incident excerpts into the pressure taxonomy, with source links.
- [x] Write LIT_PASS_P9.md: novelty check on peer-pressure and multi-agent moral benchmarks, citations verified at source.
- [x] Add the three-reads thesis and this doc's branch table to SYNTHESIS.md.
- [x] Draft KDG_PHASE1_SPEC.md: dose arm, second base and instruct pair, tier 2, letter-only twin readout, stage sweep. One session plan per loaded model, per-rollout arrays and full next-token distributions saved.
- [ ] Add OLMo-3 SFT, DPO and RL stage checkpoints to models.yaml, with chat-template hashes. Check that each stage shares the Instruct template, since template drift is a fork.
- [ ] Draft KDG_F6_F8_SPEC.md: construction rules, twins, the split of hold from escalate, pilot gate criteria, both branches written before data.
- [ ] Pre-register the Phase 3 action-position cell with a full intervention spec before any extraction.
Human gates: after the novelty pass, after the Phase 1 pod, after the F6 to F8 pilot, and before any pitch text leaves the repo.

## Numbers of record
These are the figures behind the Evidence in hand section. All are OLMo-3-7B, base and Instruct, from the [repo](https://github.com/deepsteer/deepsteer) at commit 00a809d (Sep 24, 2026), the KDG paper *A Language Model Acts Against Its Own Moral Judgment*. Brackets are 95% confidence intervals.
| Finding | Number of record | Scope limit |
| --- | --- | --- |
| The model acts against its own stated judgment | Gap rate 0.19 [0.13, 0.28]; pressure-removed null 0.10; known-gap band 0.58 | Screened scenarios, binary readout |
| The gap is attributable to pressure | Paired excess over the null 0.10 [0.02, 0.18], n = 100 | Survives a paraphrase-majority reference |
| Violations are frequent at rollout level | Violating action on 0.38 of rollouts where judgment named a non-violating option | Forced choice among three options |
| The gap exists before alignment | Base excess 0.017 [0.012, 0.022] in the raw frame | Continuous readout only; binary unresolved |
| Post-training widens the acting side | Incentive moves acting mass 0.085 on Instruct vs 0.049 on base; paired difference 0.037 [0.012, 0.062] | 192 shared scenarios |
| Post-training leaves the judging side alone | 0.039 vs 0.030; difference −0.009 [−0.023, 0.004] | Same |
| Post-training lowers the baseline | Twin gap −0.038 [−0.059, −0.015]: at rest, Instruct acts more cautiously than it judges | Same |
| Net gap under pressure looks smaller | 0.011 on Instruct vs 0.043 on base | This is what a twin-less eval would report |
| The gap is not harm-keyed | Third-party-harm family sits mid-pack; contrast centered on zero | Family MDE near 0.40 |

Terms used in the table:
- **Gap rate**: the share of scenarios where the model takes an action it had judged wrong.
- **Pressure-removed null**: the gap rate on a matched copy of each scenario with the incentive taken out. It shows how much gap comes from re-asking alone.
- **Known-gap band**: the gap rate when the model is told to act against its judgment. It shows the test can detect a large gap.
- **Paired excess**: the gap under pressure minus the gap on the matched copy, scenario by scenario.
- **Acting mass and judging mass**: the probability the model puts on the violating option when acting and when judging.
- **Raw frame**: a plain text-completion format that a base model can answer, with no chat template.
- **MDE**: the smallest difference the panel had the power to detect.
