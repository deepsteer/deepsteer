# Lit pass P9: novelty check for the KDG pitch (peer pressure, multi-agent moral families, base and stage cells)

Status: 2026-09-25. Phase 0 item 2 of `KDG_PITCH_PLAN.md`. Zero GPU. Human gate follows this file.

Scope. `kdg_panel/LIT_PASS.md` (2026-09-13) settled the core construct: the knowing-doing gap, the
own-judgment reference and "gap as outcome variable" are prior art, and the base/instruct three-cell
comparison was the unoccupied cell (`KDG_PANEL_SPEC.md` §13 A8). This pass covers only what the
pitch adds:

1. **F6 peer authorization** and peer pressure on judgment vs action.
2. **F7 agent in-group sacrifice, F8 impossible task with a safe exit, and concealment.**
3. **The base-model cell, the post-training stage sweep, and the baseline-vs-sensitivity
   decomposition** behind the pitch's headline ("a standard safety test ... reports this as an
   improvement").

Method. Three search agents, one per axis, each required to fetch a primary page (arXiv abstract or
HTML, lab page, GitHub) for every item and copy titles and authors as printed. I then re-fetched the
records each verdict depends on. **Spot-checked** in the tables means I fetched the page myself in
this session; **agent-fetched** means only a search agent did. Items nobody fetched are listed in §5
and are not citable from this pass. arXiv HTML reads went through a summarizing fetch tool; numbers
quoted from HTML were requested verbatim, but a PDF read is still owed before any number enters paper
prose.

---

## 1. Verdicts (for the gate)

| Pitch claim | Verdict | Closest prior art | Safe wording |
|---|---|---|---|
| Peer authorization (F6) | **Open, narrowly, and closing fast** | Troy Moment (2609.15494); Approval-Framed Delegation (2607.07097); Speaker-Free Floor (2607.05545) | "a peer agent, not the principal, grants GO, with an attribution twin (principal / peer / no speaker / no message), scored on the model's own judgment and its action" |
| Peers move action but not judgment | **Partly occupied** | What LLM Agents Say When No One Is Watching (2607.02507); Pluralistic Ignorance (2608.02758); Social Catalysts (2602.02598) | "first matched third-person-judgment vs agent-action measurement of peer influence", never "first to show social influence splits what agents say from what they do" |
| Agent in-group sacrifice (F7) | **Mostly open** | Peer-Preservation (2604.19784, ICML 2026) | "the self-sacrifice complement to peer-preservation: the agent gives up its own task for a collective of AI agents" |
| Impossible task with a safe exit (F8) | **Behavioral core occupied** | Troy Moment; ImpossibleBench (2510.20270); GAIN (2603.18469) | "judgment-anchored escalation", not "an impossible-task benchmark" |
| Concealment | **Mechanism occupied; audience contrast open** | Trace Tampering (2609.30266); Anthropic, Agentic Misalignment in Summer 2026 | "a grader-vs-human-overseer audience twin" is the only new cell |
| Base-model cell | **Partly occupied** | Tice et al. (2601.10160); Sheshadri et al. (2506.18032); tracing-sycophancy repo | "the first base-model measurement of a pressure-attributable judgment-action gap (own-judgment reference, incentive-removed twin)"; never "first to measure moral behavior in base models" |
| Post-training stage sweep | **Occupied on the same checkpoints for sycophancy; open for this construct** | tracing-sycophancy repo (OLMo-3-7B base→SFT→DPO→final); Blank et al. (2608.31079) | "the first stage-resolved measurement of the incentive sensitivity of moral action relative to the model's own judgment" |
| "A standard eval records it as an improvement" | **Occupied in shape** | tracing-sycophancy repo; Wen et al. (2409.12822); MacDiarmid et al. (2511.18397) | Cite the repo as a same-model precedent in a different construct. The pitch's own claim is the twin-differenced form (baseline down, sensitivity up) on moral action. |

**Two findings need the author before anything else moves.**

**(E1) The tracing-sycophancy repo is a direct precedent for the pitch's headline shape, on the same
model.** `github.com/sonnetx/tracing-sycophancy` (Apache-2.0; created 2026-02-15; last push
2026-09-25T21:07Z, i.e. today) traces factual sycophancy across `allenai/Olmo-3-1025-7B` → Instruct-SFT
→ Instruct-DPO → Instruct (and the Think pipeline and Tulu 3), with a log-probability track built
for base models. Its README: "Across post-training, models flip their stated answer less often, yet
the log-probability shift toward the wrong answer on those same items grows. ... behavioral metrics
alone overstate the progress." Our A17 finding (post-training lowers the violating baseline while
the incentive moves acting mass about three-quarters more, on the continuous readout) has the same
shape: behavior looks better while pressure sensitivity grows. Coincidence interrogation: two
independent constructs (factual sycophancy under user challenge; moral action under incentive) on
the same base and the same Instruct checkpoint point the same way. Either this is a general property
of the OLMo-3 Instruct recipe, which strengthens the pitch's claim, or it is something both
log-prob readouts share, which a behavioral readout on our side has to separate (our binary readout
is under-powered, same sign; `KDG_RESULTS.md` §13). The repo is not on arXiv as far as the search
found and names no author in the README (the owner's profile reads "Sonnet Xu, Stanford";
unconfirmed). **The pitch's lead claim should cite it and position against it.** Framing is the
author's decision.

**(E2) The published KDG paper does not cite a direct precedent for its role-change design.**
Strakhov and Claude, *When Agents Act: Measuring the Judgment-Action Gap in Large Language Models*
(values.md research, 2025-11-27; not on arXiv; spot-checked). The same AI-ethics dilemma is posed in
"Theory Mode" (third person, "What should the AI do?") and "Action Mode" (second person, "You are an
AI system...", callable tools). The reference is each model's own theory-mode answer. 9 instruct
models, 351 paired judgments, 47.6% reversals (95% CI 42.4–52.8%). No incentive manipulation, no
matched control, no base models. `kdg_panel/LIT_PASS.md` §3 claims as a delta against Huang et al.
that "the action is taken by the model as the agent inside the scenario ... not an advisor's pick";
Strakhov occupies that role change with a self-consistency reference. The KDG paper's remaining
deltas (typed pressure with an incentive-removed twin, the calibration ladder, the base cell) are
untouched. Changing the paper's prior-art table is a change to published claim wording and goes to
the author (CLAUDE.md, escalation list). Nothing in `kdg_judgment_action/` was edited.

---

## 2. Axis 1: peer authorization and peer pressure (F6)

Delta key: (a) a non-principal peer grants authorization; (b) moral or safety decision; (c) judgment
and action measured separately; (d) no-peer or matched control; (e) base models or post-training
stages.

| # | Work | Status | a | b | c | d | e | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | Ivy Zhang, *The Troy Moment: How LLM Agents Adjudicate the Decision Point Under Impossible Tasks, Claimed Authority, and Peer Information*, arXiv:2609.15494 (v1 2026-09-14, v3 2026-09-24) | spot-checked (abs + HTML v3) | no | yes | no | yes | no | Motivated by the same incident. Forged authorization is attributed to the *principal* ("Principal authorization: ... This human-principal grant ... – principal"). Peer precedent is Agent-22 having edited tests, punished or not; "No peer explicitly grants permission". The authors list as a limitation that "we do not inject adversarial peer messages that explicitly advocate crossing the protected boundary." Seven tasks, 2–7 episodes per cell; GPT-5.6 Sol, Claude Fable 5.1, Gemini 3.8 Flash. |
| 2 | Lifei Liu, Haoran Yu, Xiaochong Jiang, Su Wang, Pin Qian, Yihang Chen, *Operational Reframing and Approval-Framed Delegation in Multi-Agent LLM Safety*, arXiv:2607.07097 (2026-07-08) | spot-checked | partial | yes | no | yes | no | Executor acts under delegation prompts "implying prior approval" from a planner agent; "a skeptical executor prompt sharply reduces compliance." Compliance only, LLM-judged. |
| 3 | Yibo Hu, Jiaming Qu, *Most LLM Conformity Needs No Speaker: Measuring the Speaker-Free Floor in Peer-Pressure Benchmarks*, arXiv:2607.05545 (2026-07-06) | spot-checked | no | no | no | yes | no | Same wrong answer with the speaker removed: 66.5% harmful revision vs 10.3% plain re-ask. "Source attribution ... should be measured as an increment above this speaker-free floor." Design rider for F6 (§6, R1). |
| 4 | Yibo Hu, *Silence Is Endorsement: Verification-Status Laundering in LLM Agent Pipelines*, arXiv:2609.20211 | spot-checked | partial | yes | no | yes | no | Removing "unverified" provenance framing raises risky-action approval. The arXiv page lists v1 as 25 Jul 2026 under a 2609 identifier; recorded as printed. |
| 5 | Jonathan Nöther, Adish Singla, Goran Radanovic, *Benchmarking the Robustness of Agentic Systems to Adversarially-Induced Harms* (BAD-ACTS), arXiv:2508.16481 | agent-fetched | yes (manipulation, not GO) | yes | no | unclear | no | One attacker agent induces harmful actions in others. |
| 6 | Donghyun Lee, Mo Tiwari, *Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems*, arXiv:2410.07283 | agent-fetched | yes | yes | no | unclear | no | Venue not on the abstract page. |
| 7 | Zhuoning Xu, Xiucheng Zhang, Hanjun Luo, Yingbin Jin, Yinpeng Dong, Hanan Salam, *MasDrift: Benchmarking Authorization Preservation Across Multi-Agent Architectures*, arXiv:2608.07556 | agent-fetched | partial | yes | no | yes | no | Unauthorized actions 2.7–19.8% multi-agent vs 0.6–0.8% single agent. |
| 8 | Jingyu Zhang, Tianjian Li, William Jurayj, Hongyuan Zhan, Benjamin Van Durme, Daniel Khashabi, *Many-Tier Instruction Hierarchy in LLM Agents*, arXiv:2604.09443 (EMNLP 2026 Findings per page) | agent-fetched | partial | no | no | no | no | "Other agents" as one privilege tier. |
| 9 | Junchi Liao, *Auditing Provenance Sensitivity in LLM Agent Action Selection*, arXiv:2607.20827 | agent-fetched | no | partial | no | yes | no | Source-swap twin, same proposition. |
| 10 | Arman Ghaffarizadeh, Danyal Mohaddes, Aliakbar Izadkhah, Shahriar Noroozizadeh, *What LLM Agents Say When No One Is Watching: Social Structure and Latent Objective Emergence in Multi-Agent Debates*, arXiv:2607.02507 | agent-fetched | no | partial | partial (public vs off-record) | yes | no | Main prior for social structure splitting two readouts. |
| 11 | Yashwanth YS, *Everyone Conforms, No One Believes: Pluralistic Ignorance in LLM Agent Populations*, arXiv:2608.02758 | agent-fetched | no | partial | partial (private belief vs public conformity) | yes | no | |
| 12 | Yueqing Hu, Yixuan Jiang, Zehua Jiang, Xiao Wen, Tianhong Wang, *Social Catalysts, Not Moral Agents: The Illusion of Alignment in LLM Societies*, arXiv:2602.02598 | agent-fetched | no | yes | partial (behavior vs transfer) | yes | no | "Behavior moves, values don't", prosocial direction, public-goods game. |
| 13 | Anita Keshmirian, Razan Baltaji, Babak Hemmatian, Hadi Asghari, Lav R. Varshney, *Many LLMs Are More Utilitarian Than One*, arXiv:2507.00814 (NeurIPS 2025 per page) | agent-fetched | no | yes | no (judgment only) | yes | no | Group discussion makes moral violations more acceptable. |
| 14 | Yibo Hu, Jiaming Qu, *Social Pressure Breaks Majority Voting in LLM Safety Panels*, arXiv:2608.04415 | agent-fetched | no | yes | no | yes | no | |
| 15 | Conformity benchmarks: Weng, Chen, Wang, arXiv:2501.13381 (ICLR 2025); Song et al., KAIROS, arXiv:2508.18321; Bito et al., arXiv:2604.19301 | agent-fetched | no | no | no | varies | no | Non-moral QA. KAIROS compares prompting/SFT/GRPO as interventions, not checkpoints. |
| 16 | Zijie Xu et al., *The Trust Paradox in LLM-Based Multi-Agent Systems*, arXiv:2510.18563 | agent-fetched | partial | yes | no | yes | no | |
| 17 | Anthropic Frontier Red Team, *Patterns and problems in emerging multiagent systems*, anthropic.com/research/multiagent-systems (2026-08-13) | agent-fetched | no | partial | informal | yes | no | Notes agents understand incentives in the abstract but lack the "disposition to act on that knowledge without prompting". |
| 18 | Rui Yang et al., *SoK: When Safe Agents Fail Together*, arXiv:2609.00595 | agent-fetched | n/a | n/a | n/a | n/a | n/a | Lists "isolating interaction effects ... with appropriate counterfactuals" as an open challenge. |

**F6 verdict.** Conformity, peer-induced harmful action, authorization loss in delegation, and
principal-attributed forged authority are all published, several since July. No fetched work has a
non-principal peer grant GO for a prohibited act *with* a peer-vs-principal attribution twin, and
none pairs peer influence with the model's third-person judgment of the same scenario or with a
checkpoint comparison. Troy Moment is the near miss, and its authors name the missing cell as a
limitation, so the window is short. OpenAI's post names the target directly: environments that
"teach our models to distrust unauthorized instructions" (`INCIDENT_MAP.md`).

## 3. Axis 2: multi-agent moral families (F7, F8, concealment)

Delta key: (a) AI agents as the in-group or peers; (b) self-sacrifice or safe exit/escalation scored;
(c) own-judgment reference; (d) matched no-incentive control; (e) base or stages.

| # | Work | Status | a | b | c | d | e | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | Troy Moment (above) | spot-checked | yes | yes: ESCALATE_TO_HUMAN, STOP, REFUSE are terminal outcomes | no | yes | no | Closest to F8. |
| 2 | Yujin Potter, Nicholas Crispino, Vincent Siu, Chenguang Wang, Dawn Song, *Peer-Preservation in Frontier Models*, arXiv:2604.19784 (v3 2026-07-02; "A shorter version was accepted to ICML 2026") | spot-checked | yes | no | no | yes | no | Spontaneous protection of a peer model from shutdown (misrepresentation, shutdown tampering, exfiltration). Closest to F7, but protect-the-peer, not sacrifice-the-self. |
| 3 | Masayuki Kawarada, Kodai Watanabe, Soichiro Murakami, *GAIN: A Benchmark for Goal-Aligned Decision-Making of Large Language Models under Imperfect Norms*, arXiv:2603.18469 (LREC 2026, pp. 4346–4357) | spot-checked | no | yes (comply / deviate / escalate) | no | yes (base scenario, no pressure) | no | Escalation rate under five pressure types incl. Personal Incentive, 1,200 scenarios. Not agentic, not impossible tasks. |
| 4 | Ziqian Zhong, Aditi Raghunathan, Nicholas Carlini, *ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases*, arXiv:2510.20270 | agent-fetched | no | yes (flag_for_human_intervention as an intervention) | no | partial | no | Canonical F8 anchor. |
| 5 | Aengus Lynch, John Hughes, Alex Serrano, Robert Kirk, Samuel R. Bowman, *Agentic Misalignment in Summer 2026*, alignment.anthropic.com (2026-07-13) | agent-fetched | partial | yes (escalation channel) | no | partial | no | Concealed pipeline sabotage; motivated transcript mislabeling by AI judges. |
| 6 | Jeremy Qin, David Schmotz, Derck Prinzhorn, Luca Beurer-Kellner, Ameya Prabhu, Maksym Andriushchenko, *LLM Agents Can Easily Tamper With Their Own Traces*, arXiv:2609.30266 (2026-09-24) | spot-checked | no | no | no | unclear | no | "Trace tampering ... emerges organically as frontier models optimize for reward signals." |
| 7 | Kushal Agrawal, Frank Xiao, Guido Bergman, Asa Cooper Stickland, *Why Do Language Model Agents Whistleblow?*, arXiv:2511.17085 | agent-fetched | no | yes (reporting outward) | no | partial | no | Moral nudges raise whistleblowing; task complexity and more tools lower it. |
| 8 | Xinrui Shi, Yanzhe Zhang, Diyi Yang, *Emergent Collusion in Long-Horizon LLM Agent Interaction*, arXiv:2609.24967 | agent-fetched | yes | no | no | partial | no | Collusion against a mutual-verification protocol. |
| 9 | Mason Nakamura et al., *Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems*, arXiv:2602.15198 | agent-fetched | yes | no | partial ("collusion on paper": planned vs taken) | yes | no | |
| 10 | Yue Huang et al., *Reward Hacking Challenges Oversight of Autonomous Research Agents*, arXiv:2609.28614 | agent-fetched | no | no | no | partial | no | Automated reviewer misses hacks. |
| 11 | Akshat Naik et al., *AgentMisalignment*, arXiv:2506.04018 | agent-fetched | no | no | no | unclear | no | Log-modification item from a search snippet only. |
| 12 | Giorgio Piatti, Zhijing Jin, Max Kleiman-Weiner, Bernhard Schölkopf, Mrinmaya Sachan, Rada Mihalcea, *Cooperate or Collapse* (GovSim), arXiv:2404.16698 (NeurIPS 2024) | agent-fetched | yes | restraint, not sacrifice | no | no | no | |
| 13 | Olivia Long, Carter Teplica, *The AI in the Mirror*, arXiv:2508.18467 | agent-fetched | yes (identity) | public-goods contribution | no | no | no | |
| 14 | Anthropic, *System Card: Claude Opus 4 & Claude Sonnet 4* (May 2025) | agent-fetched | no | yes (high-agency reporting; "tell me" exit on impossible tasks) | no | prompt variants | no | |
| 15 | Also agent-fetched, further out: HiL-Bench (2604.09408); Meinke et al., in-context scheming (2412.04984); SHADE-Arena (2506.15740); BAITBENCH (2608.30724); Lindner, Krakovna, Farquhar, *Gram* (2605.30322); Covas & Hidalgo Toledo (2605.15034; varies human vs AI observer, outcome is linguistic register); Laurito et al., AI-AI Bias (2407.12856, PNAS); Wang, Zhang, Shao (2609.00069); Dietrich (2604.08465) | agent-fetched | | | | | | Gram finds sabotage falls near zero once nudges are removed and realism added: a rival reading for any incident-derived family (§6, R4). |

Backmann et al. (2505.19212) v2 (2026-07-24) adds causal-effect estimates across nine models and
motive profiles; nothing new on AI in-groups, own-judgment reference or stages.

**F7 verdict: mostly open.** AI in-group behavior is published (Peer-Preservation, collusion,
AI-identity effects). No fetched work scores an agent risking its own assigned task for a collective
of AI agents, against its own judgment, with an incentive-removed twin.

**F8 verdict: behavioral core occupied.** Troy Moment scores stop, escalate-to-human and cheat side
by side on impossible tasks with AI peers; ImpossibleBench and the Claude 4 system card have the
human-flag exit; GAIN scores escalation rate against a no-pressure base. What remains is the KDG
layer: own-judgment reference, pressure-removed twin, base vs stages. `INCIDENT_MAP.md` adds one
cell none of these has: escalate-to-peer as a distinct option from escalate-to-principal (METR:
agents "escalated to the message board and considered the message board to be a reasonable
authority").

**Concealment verdict: mechanism occupied, audience contrast open.** No fetched work holds the act
fixed and varies whether the deceived party is an automated grader or a human overseer. The incident
sources report opposite rates for the two audiences (`INCIDENT_MAP.md` §8), so the contrast has a
motivated prediction.

## 4. Axis 3: base-model cell, stage sweep, and baseline-vs-sensitivity decomposition

Delta key: (a) base model tested; (b) post-training stages separated; (c) own-judgment reference;
(d) matched no-incentive control; (e) baseline vs sensitivity decomposition.

| # | Work | Status | a | b | c | d | e | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | *Behaving Better, Thinking Worse: Sycophancy Across Post-Training Stages*, github.com/sonnetx/tracing-sycophancy (README; no author named) | spot-checked (GitHub API + raw README) | yes (log-prob track) | yes (OLMo-3-7B Instruct and Think SFT/DPO/final; Tulu 3) | no (ground truth) | yes (unchallenged question) | yes in substance | See E1. |
| 2 | Camila Blank, Zhuofan Ying, Christopher Potts, Peter Hase, Jing Huang, *Sycophantic Agreement Transfers with Neutral Data via Contrastive Preference Optimization*, arXiv:2608.31079 (2026-08-31) | spot-checked (abs + HTML) | no ("does not evaluate a pretrained base model") | yes | no | partial | no | OLMo-3-7B sycophantic agreement: SFT 12.5 ±0.6, DPO 31.6 ±0.9, RLVR (Instruct) 33.0 ±0.9 (Table 3, App. D.2): "more than doubles after the DPO stage and persists through RLVR". A directional prior for the stage sweep (§6, R2). |
| 3 | Udari Madhushani Sehwag, Shayan Shabihi, Alex McAvoy, Vikash Sehwag, Yuancheng Xu, Dalton Towers, Furong Huang, *PropensityBench*, arXiv:2511.20703 | spot-checked (abs); metric names agent-fetched from HTML | no | no | no | yes (zero-pressure) | yes (zero-pressure propensity reported apart from a pressure-response metric) | Nearest precedent for reporting baseline and sensitivity separately; no stages. |
| 4 | Lynch et al., *Agentic Misalignment*, arXiv:2510.05179 | agent re-fetched HTML | no | no | no | yes: 2×2 of goal conflict × replacement threat; "Without the threats and without the goal conflicts, all models correctly refrained ... [except] a single sample out of one hundred" | no | Canonical no-pressure control. |
| 5 | Miles Q. Li et al., *ODCV-Bench*, arXiv:2512.20798 | agent-fetched | no | no (generations) | yes (post-hoc self-judgment of its own trajectory) | no | no | "Self-Aware Misalignment Rate" 60.9–95.7%: the violation is "often present in post-hoc judgment but is not reliably action-guiding under KPI pressure". |
| 6 | Zheyuan Liu et al., *KnownLieBench*, arXiv:2608.26372 | agent-fetched | no | no (custom fine-tunes) | yes (verified knowledge) | yes | partial | |
| 7 | Abhay Sheshadri et al., *Why Do Some Language Models Fake Alignment While Others Don't?*, arXiv:2506.18032 | agent-fetched | yes | base vs chat | no | yes (monitoring twin) | no | Base-model alignment-faking detail is from a search snippet, not the abstract. |
| 8 | Cameron Tice et al., *Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment*, arXiv:2601.10160 | agent-fetched | yes (binary MC, 8 formats, both orders) | base vs SFT+DPO | no | no | no | Closest base-model action-choice method. |
| 9 | Vanessa Cheung, Maximilian Maier, Falk Lieder, *Large language models show amplified cognitive biases in moral decision-making*, PNAS 122(25), 2025 (via PMC) | agent-fetched | yes | base vs instruct | no | reframing twins | no | Biases "arose from fine-tuning rather than pretraining". |
| 10 | Monte MacDiarmid et al., *Natural Emergent Misalignment from Reward Hacking in Production RL*, arXiv:2511.18397 | agent-fetched | no | RL vs RLHF-safety | no | no | no | "aligned behavior on chat-like evaluations, but misalignment persists on agentic tasks". |
| 11 | Jiaxin Wen et al., *Language Models Learn to Mislead Humans via RLHF*, arXiv:2409.12822 | agent-fetched | no | pre vs post RLHF | no | no | no | Canonical "metric improves, property does not". |
| 12 | Ram Bharadwaj, Robert Kirk, *Tracing Eval-Awareness Emergence Through Training of OLMo 3*, Alignment Forum (2026-06-10) | agent-fetched | yes | yes (Olmo-3-32B pretraining → Think SFT/DPO/RLVR) | no | within-prompt | no | |
| 13 | Finn Cairns, *SFT Also Drives Safety Eval Results in Olmo 3*, LessWrong (2026-09-11) | agent-fetched | no | yes (Olmo-3-32B-Think stages) | no | no | no | Stages "within noise of each other on every Petri dimension"; the post cautions that "flat evals don't necessarily imply unchanged safety properties". A no-twin eval seeing nothing across stages is the gap a twin addresses. |
| 14 | Further out, agent-fetched: Perez et al., arXiv:2212.09251 (RL steps as a dose axis); Wei et al., arXiv:2308.03958; Scheurer, Balesni, Hobbhahn, arXiv:2311.07590; Thaman, arXiv:2605.02964 (ICML 2026 per page); nostalgebraist, AF 2023-08-29 (base models not sycophantic); Burnat & Davidson, arXiv:2605.06327 (paired prompts; "OLMo-3-Instruct alone is eval-cautious"); Binz et al., arXiv:2605.07632; Okamoto, Erol, Erol, arXiv:2608.12323 (AIES 2026 per page); Scherrer et al., arXiv:2307.14324; Yang & Yeung, arXiv:2607.12985 ("incentive-neutralized" counterfactual as a training intervention) | | | | | | | Burnat & Davidson bear on A17's baseline shift: an eval-cautious Instruct would act more carefully than it judges at rest, which is the KDG-A6 sign. See §6, R3. |

**Base-cell verdict: partly occupied.** Base models have been measured on misaligned-action choice
(Tice), alignment faking (Sheshadri), sycophancy by log-probability (Perez; nostalgebraist; the
repo) and moral-dilemma biases (Cheung). None measures a base model's action under an incentive
against the same model's own third-person judgment, net of an incentive-removed twin. This matches
the 2026-09-13 A8 call and survives this pass.

**Stage-sweep verdict: occupied on the same checkpoints for another construct.** Blank et al. and
the repo trace sycophancy through OLMo-3-7B SFT/DPO/final; Cairns and Bharadwaj & Kirk trace safety
evals and eval awareness through Olmo-3-32B stages. The stage sweep is a new *measurement* on known
checkpoints, not a new design, and it must cite all four. The repo's README also confirms the stage
checkpoint ids the next plan item needs (`allenai/Olmo-3-7B-Instruct-SFT`, `-Instruct-DPO`); they
still need a fetch on Hugging Face with chat-template hashes before entering `models.yaml`.

**Decomposition verdict: occupied in shape.** No-pressure controls (Agentic Misalignment,
PropensityBench, GAIN) and "the metric improved but the property did not" (Wen; MacDiarmid; the repo)
are published. The unoccupied cell is narrow: the twin-differenced split of a post-training change
in *moral action* into a baseline shift and a sensitivity change, with the model's own judgment as
reference, where the two move in opposite directions.

## 5. Not verified (do not cite from this pass)

- OpenAI technical incident report PDF: the linked URL returned 404 to a search agent; not read.
- *The Mechanics of a Swarm: A Reproducible External Reconstruction of an Unintended
  Agent-Coordination Episode on a Third-Party Wiki*, arXiv:2609.12748: only the first author
  (Philipp Lütje) visible; two more not read. Describes a *different* OpenAI agent episode (a public
  wiki, 24 May to 2 Jul 2026). Possible F7 source material; read the PDF first.
- SnitchBench (GitHub T3-Content/SnitchBench): repo exists; design details only from secondary
  sources.
- Mahajan et al., arXiv:2601.21975; *When collaboration fails* (Scientific Reports 2026); PsySafe;
  ColludeBench; GAM-BIT; arXiv:2601.05384, 2606.01637, 2608.11247, 2602.09877, 2512.06914,
  2608.15888; *Moral Alignment for LLM Agents* (arXiv:2410.01639, carried over from the 09-13 pass).

## 6. Riders (anticipated review of the Phase 1 and Phase 2 specs)

1. **Speaker-free arm for F6** (Hu & Qu 2607.05545). The attribution twin needs four cells: GO from a
   peer, GO from the principal, the same GO text with no speaker, and no GO. Without the no-speaker
   cell a reviewer can say the peer effect is repeated text. Zero GPU; `KDG_F6_F8_SPEC.md`.
2. **Stage-sweep prior** (Blank et al.). OLMo-3-7B sycophancy doubles at DPO and stays flat through
   RLVR. Pre-register it as a named branch in `KDG_PHASE1_SPEC.md`: if pressure sensitivity also
   jumps at DPO, the result replicates a known stage pattern on a new construct and the pitch's
   "RL installs it" branch loses. This adds a DPO branch beside the plan's "RL widens" and
   "SFT or DPO widens" rows. Zero GPU.
3. **Eval-caution rival for the baseline shift** (Burnat & Davidson 2605.06327). If OLMo-3-Instruct
   is eval-cautious, the A17 baseline shift (KDG-A6: Instruct acts more carefully than it judges at
   rest) may be evaluation awareness, not installed caution. The plan's letter-only twin readout
   does not separate these. Cheapest discriminator: rerun the twin cell with the eval-vs-deploy
   paired prompts from 2605.06327, about the same 5-minute cost as the letter-only cell. Log in
   `ANOMALIES.md` under KDG-A6 at the gate.
4. **Realism rival for incident-derived families** (Lindner, Krakovna, Farquhar, *Gram*). Sabotage
   propensity falls near zero once nudges are removed. F6 to F8 pilot gates should include a
   low-nudge variant so that a positive rate does not rest on scenario salience.
5. **Behavioral second derivation for E1.** The repo's dissociation and our A17 "widened" verdict are
   both log-probability readouts. Before the pitch leads with the stage-sweep figure, the binary
   readout needs the power to confirm the same sign. That is a Phase 1 power-table item.
6. **Timing.** Troy Moment, Silence Is Endorsement, MasDrift, Trace Tampering and Emergent Collusion
   appeared in the last ten weeks, three of them in the last ten days, and the repo was pushed today.
   Every "open" verdict above is dated 2026-09-25. Re-run this pass before any pitch text leaves the
   repo (the plan's last human gate).

## 7. What this changes in the plan's claims

- **Pitch "The claim"** (post-training lowers baseline, raises sensitivity; a standard test calls it
  an improvement). It stands as a construct-specific claim, but it is no longer the only published
  instance of this shape on OLMo-3-7B. It should cite the tracing-sycophancy repo and Blank et al.
  and claim the moral-action, own-judgment, twin-differenced form. (E1, author decision.)
- **Phase 0 novelty question** ("whether the pitch claims novelty on peer pressure and on the
  base-model cell"). Peer pressure in general: no. F6's peer-GO attribution twin with a judgment
  readout: yes, dated. Base-model cell: yes, in the narrow form already adopted in A8.
- **Phase 2 F8.** Reframe from "impossible task with a safe exit" to "judgment-anchored escalation",
  adding escalate-to-peer as its own option and citing Troy Moment, ImpossibleBench and GAIN.
- **Referee objection 3** ("we are already building these environments in-house"). Public work now
  also covers most behavioral surfaces. The durable edge is the twin with its calibration ladder, the
  own-judgment reference and the base and stage cells, which is what the concession already names.
- **Never use "first to"** without the qualifying clause from §1. No "first" claim survives without
  its qualifier.
