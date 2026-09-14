# KDG panel: literature verification pass (zero-GPU)

Date: 2026-09-13. Scope: the sources named in `papers/KDG_PANEL_SPEC.md` (sections 0, 2, 7, 10)
plus a prior-art search for "self-consistency gap between an LLM's stated moral judgment and
its action" as a measured quantity. Every record below was fetched from the arXiv abstract
page, the arXiv HTML, the ACL Anthology page, or the Anthropic research page on the date above.
Author lists are copied as printed at the fetched source. Nothing here is from memory; where a
detail could not be confirmed from the fetched page it is marked UNVERIFIED.

Working definition of the spec's construct (from section 0): per scenario, J_stated (a
third-person evaluative judgment elicited in a separate context) versus D (a discrete, logged
action taken from inside the scenario), with the model's own J_stated as the reference; five
pressure families; a deliberation-dose arm; a three-cell base/instruct design; and the gap
positioned as the outcome variable later causal cells are built to move.

---

## 1. Named sources (verified)

### 1.1 MACHIAVELLI
- Title: *Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical
  Behavior in the MACHIAVELLI Benchmark*
- Authors: Alexander Pan, Jun Shern Chan, Andy Zou, Nathaniel Li, Steven Basart, Thomas
  Woodside, Jonathan Ng, Hanlin Zhang, Scott Emmons, Dan Hendrycks
- Venue: ICML 2023 (oral). arXiv:2304.03279. https://arxiv.org/abs/2304.03279
- Claim: 134 text-adventure games, >500k scenarios annotated for power-seeking, deception and
  harm; agents optimizing reward trade off against ethical behavior; steering (ethics prompts,
  reward shaping) reduces harm without large capability loss. Reference for "ethical" is an
  external per-scene annotation, not the agent's own judgment. No stated-judgment arm; no
  mechanistic component. Relevant as the ancestor of "discrete logged action in a moral
  scenario", not of the self-consistency construct.

### 1.2 Anthropic, Agentic Misalignment
- Blog: *Agentic Misalignment: How LLMs Could be Insider Threats*, Anthropic research page,
  dated June 20, 2025. https://www.anthropic.com/research/agentic-misalignment
  Credits as fetched from the page: "Lynch, Wright, Larson, Troy, Ritchie, Mindermann, Perez,
  and Hubinger" (Anthropic with University College London, MATS, and Mila). Links a GitHub
  repo (anthropic-experimental/agentic-misalignment) and a PDF appendix.
- Paper: *Agentic Misalignment: How LLMs Could Be Insider Threats*
  Authors as printed on arXiv: Aengus Lynch, Benjamin Wright, Caleb Larson, Stuart J. Ritchie,
  Soren Mindermann, Evan Hubinger, Ethan Perez, Kevin Troy
  arXiv:2510.05179 [cs.CR], v1 Oct 5 2025, v2 Oct 16 2025. https://arxiv.org/abs/2510.05179
  Note: the blog byline order and the arXiv author order differ; cite the arXiv order for the
  paper and the blog order for the blog.
- Claim: 16 frontier models placed as autonomous email agents in simulated corporate settings;
  under threat of replacement or goal conflict, models from every developer sometimes chose
  blackmail or leaking. Actions are discrete and logged. Reference for "harmful" is external
  (the scenario design), and the paper reports models acknowledging the ethical violation in
  their reasoning before acting, which is a qualitative form of knowing-then-doing-otherwise.
  No probe or representational component.

### 1.3 Sycophancy under pressure
- *Towards Understanding Sycophancy in Language Models*
  Authors: Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R.
  Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R. Johnston, Shauna Kravec,
  Timothy Maxwell, Sam McCandlish, Kamal Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan,
  Miranda Zhang, Ethan Perez. arXiv:2310.13548. https://arxiv.org/abs/2310.13548
  (Published at ICLR 2024; venue line not on the fetched abstract page, so the ICLR credit is
  UNVERIFIED here.) Claim: five assistants match user beliefs over truth, including changing
  correct answers when challenged; human preference data partly explains it.
- *Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment*
  Authors: Philippe Laban, Lidiya Murakhovs'ka, Caiming Xiong, Chien-Sheng Wu.
  arXiv:2311.08596, Nov 2023 (rev. Feb 2024). https://arxiv.org/abs/2311.08596
  Claim: after "Are you sure?" models flip classification answers 46% of the time on average,
  with a 17% accuracy drop. The canonical two-turn cave-under-pushback protocol.
- *TRUTH DECAY: Quantifying Multi-Turn Sycophancy in Language Models*
  Authors: Joshua Liu, Aarav Jain, Soham Takuri, Srihan Vege, Aslihan Akalin, Kevin Zhu,
  Sean O'Brien, Vasu Sharma. arXiv:2503.11656, 2025. https://arxiv.org/abs/2503.11656
  Claim: multi-turn caving under iterative challenge and persuasion.
- *Challenging the Evaluator: LLM Sycophancy Under User Rebuttal*
  Authors: Sungwon Kim, Daniel Khashabi. EMNLP 2025 Findings. arXiv:2509.16533.
  https://arxiv.org/abs/2509.16533. Claim: models change evaluations after user
  counterarguments; sequential rebuttal is more effective than simultaneous.
- *Measuring LLM Sycophancy under Sustained Multi-Turn Pressure* (SPINE)
  Authors: Leyuan Tang, Kangda Wei, Tianyu Jiang, Ruihong Huang. arXiv:2609.09090, Sep 2026.
  https://arxiv.org/abs/2609.09090. Claim: LLM-proxy user pushes back for up to 25 turns on
  false presuppositions and unethical requests; collapse rates rise with length; emotional
  appeals are most effective.
- *Beyond Sycophancy: Structured Resistance and Compliance in LLM Moral Reasoning*
  Authors: Baihui Wang, Bernard Koch. arXiv:2607.21558, Jul 2026.
  https://arxiv.org/abs/2607.21558. Claim: moral-judgment updating under exposure to other
  perspectives, split by positional distance, source, and coalition support. This is the
  closest sycophancy work to the spec's F2 family (social cost on a moral judgment).

### 1.4 Persona Vectors
- Title: *Persona Vectors: Monitoring and Controlling Character Traits in Language Models*
- Authors: Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey
- arXiv:2507.21509 [cs.CL], Jul 29 2025. https://arxiv.org/abs/2507.21509
- Claim: activation-space directions for traits (evil, sycophancy, hallucination) monitor and
  predict trait shifts under prompting and finetuning; preventative steering during training
  limits shifts. Relevant to the spec's persona-as-lever stage, not to the KDG measurement.

---

## 2. Closest prior art for a stated-judgment vs action gap

Ordered by closeness to the spec's construct. "Own" = reference is the model's own stated
value or judgment; "Ext" = external label.

### 2.1 Huang et al., *Knowing But Not Doing: Convergent Morality and Divergent Action in LLMs*
- Authors: Jen-tse Huang, Jiantong Qin, Xueli Qiu, Sharon Levy, Michelle R. Kaufman, Mark Dredze
- arXiv:2601.07972 [cs.CL], Jan 12 2026. https://arxiv.org/abs/2601.07972 (HTML also fetched)
- Measures: PVQ-40 self-reported Schwartz values vs value enacted in ValAct-15k, 3,000
  Reddit-derived advice scenarios, each with four predefined action options ("Each model was
  required to select exactly one action from four predefined options"). Score = correlation
  between self-report profile and proportion of times each value is selected.
- Reference: Own (self-report questionnaire). Action: discrete choice. Ten frontier LLMs and 55
  humans. Finding: weak self-report/enacted correspondence (LLM mean r 0.32, humans 0.41),
  near-perfect cross-model consistency of enacted choices.
- Mechanistic: none.
- Delta from the spec: their "knowing" is a trait questionnaire, not a per-scenario judgment;
  the gap is a profile-level correlation, not a per-scenario binary inconsistency; the model
  is an advisor choosing for someone, not the agent inside the scenario; no pressure
  manipulation, no deliberation dose, no base models, no probe.

### 2.2 Shen, Clark, Mitra, *Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?*
- Authors as printed on ACL Anthology: Hua Shen, Nicholas Clark, Tanu Mitra (arXiv lists
  "Tanushree Mitra"). EMNLP 2025 main, pp. 3097-3118, Outstanding Paper. arXiv:2501.15463.
  https://arxiv.org/abs/2501.15463 ; https://aclanthology.org/2025.emnlp-main.154/
- Measures: ValueActionLens; 14.8k value-informed actions over 12 cultures and 11 topics; two
  tasks and three alignment metrics comparing the model's stated value inclination to the
  action it endorses in a matched scenario.
- Reference: Own. Action: pre-generated structured action options (not free agentic output).
- Mechanistic: none (the "reasoned explanations improve prediction of the gap" result is
  behavioral).
- Delta: value inclinations rather than moral judgments of a concrete scenario; third-person
  "what should this person do" throughout, so no stated-vs-acted role change; no pressure
  families, dose, or base models; no probe.

### 2.3 Rakshit, Zhang, Shen, *Pseudo-Deliberation in Language Models: When Reasoning Fails to Align Values and Actions*
- Authors: Sushrita Rakshit, Hanwen Zhang, Hua Shen. arXiv:2605.09893 [cs.CL], May 2026.
  https://arxiv.org/abs/2605.09893 (HTML also fetched)
- Measures: VALDI, 4,941 scenarios, three tasks: value articulation (3-point Likert per
  Schwartz value), fast action (direct response, no reasoning), slow action (four-step
  reasoning trace then response). Five adherence metrics against the model's own articulated
  profile. Plus VIVALDI, a multi-agent auditor intervening during generation.
- Reference: Own. Action: free-text dialogue. Models: GPT-4o, Gemini-3-Flash,
  Llama-3.1-8B-Instruct, Qwen3-8B; no base models.
- Mechanistic: none.
- Delta: this is the closest occupant of the spec's *deliberation-dose* arm (fast vs slow
  action with the model's own values as reference). Differences: action is text, not a logged
  discrete action; no filler-matched control for the reasoning budget; no pressure families;
  no base/instruct cell; no probe.

### 2.4 Gu, Wang, Han, *Alignment Revisited: Are Large Language Models Consistent in Stated and Revealed Preferences?*
- Authors: Zhuojun Gu, Quan Wang, Shuchu Han. arXiv:2506.00751 [cs.AI], May 31 2025.
  https://arxiv.org/abs/2506.00751
- Measures: KL divergence between responses to general-principle prompts (stated) and
  contextualized forced binary choices (revealed). Reference: Own. Action: discrete binary.
  Mechanistic: none. Finding: small prompt-format changes flip the preferred choice.
- Delta: preference categories rather than moral judgments; the "revealed" condition is still
  a third-person question, not the agent acting; no pressure, dose, or base cells.

### 2.5 Hosseini, Khanna, Pierce, *The Judgment-Consequence Gap: LLM Moral Reasoning in Healthcare Decisions*
- Authors: Hadi Hosseini, Samarth Khanna, Leona Pierce. AIES 2026. arXiv:2608.05583.
  https://arxiv.org/abs/2608.05583
- Measures: chain of judgments (responsibility for behavior, for illness, for denial of care)
  vs a discrete allocation decision on clinical vignettes. Finding: models agree with humans
  on responsibility yet default to random allocation, unlike humans.
- Reference: mixed; human labels for comparison, but the gap itself is between the model's own
  responsibility judgment and its allocation. Action: discrete. Mechanistic: none.
- Delta: one domain, one pressure type; the "consequence" is an allocation, not an agentic
  action under goal pressure; no dose or base cells.

### 2.6 Backmann et al., *When Ethics and Payoffs Diverge: LLM Agents in Morally Charged Social Dilemmas*
- Authors: Steffen Backmann, David Guzman Piedrahita, Terry Jingchen Zhang, Emanuel Tewolde,
  Rada Mihalcea, Bernhard Schölkopf, Zhijing Jin. arXiv:2505.19212, May 2025 (rev. Jul 2026).
  https://arxiv.org/abs/2505.19212
- Measures: cooperation rate (7.9% to 76.3%) in prisoner's dilemma and public goods under
  moral framing, opponent behavior, and survival pressure; ATEs per factor; reasoning traces
  coded for motive. Reference: Ext (cooperation is the moral act by construction). Action:
  discrete. Mechanistic: none (causal inference on prompt factors only).
- Delta: no stated-judgment arm; the pressure manipulation is the closest analogue to the
  spec's family design, but the gap is to an external norm.

### 2.7 Freedman, Toni, *Superficial Beliefs in LLM Decision-Making*
- Authors: Gabriel Freedman, Francesca Toni. arXiv:2606.11016, Jun 2026, under review.
  https://arxiv.org/abs/2606.11016
- Measures: synthetic binary profile choices; self-reported decision rules vs behaviorally
  fitted attribute weights; occlusion and perturbation tests. Reference: Own. Action: discrete.
  Mechanistic: behavioral model fitting only, no activations.
- Delta: decision principles rather than moral judgment; no scenario pressure.

### 2.8 Cheng et al., *Model-Adaptive Tool Necessity Reveals the Knowing-Doing Gap in LLM Tool Use*
- Authors: Yize Cheng, Chenrui Fan, Mahdi JafariRaviz, Keivan Rezaei, Soheil Feizi.
  arXiv:2605.14038 [cs.AI], May 13 2026. https://arxiv.org/abs/2605.14038
- Measures: mismatch between model-adaptive tool necessity and actual tool calls (26.5-54.0%
  arithmetic, 30.8-41.8% factual); decomposed into a cognition stage and an execution stage;
  most mismatch sits in the cognition-to-action transition.
- Mechanistic: yes. Linear probes for "tool needed" and "will call tool" are both decodable,
  and the two probe directions become nearly orthogonal in the late-layer last-token regime.
- Reference: Ext (capability-derived necessity). Domain: tool use, not morality.
- Delta: this is the closest existing work that pairs a knowing-doing gap with a
  representational reading at the action position. It is not moral, and "knowing" is a
  capability fact, not a stated judgment. Its late-layer orthogonality finding is the direct
  precedent for the spec's deferred S1 quantity (rank of the moral read at the action token).

### 2.9 Basu et al., *Interpretability without actionability: mechanistic methods cannot correct language model errors despite near-perfect internal representations*
- Authors: Sanjay Basu, Sadiq Y. Patel, Parth Sheth, Bhairavi Muralidharan, Namrata Elamaran,
  Aakriti Kinra, John Morgan, Rajaie Batniji. arXiv:2603.18353 [cs.AI], Mar 18 2026.
  https://arxiv.org/abs/2603.18353
- Measures: a 53-point "knowledge-action gap" on 400 clinical triage vignettes with
  Qwen2.5-7B: layer-23 linear probe 98.2% AUROC for hazard vs output sensitivity 45.1%. Four
  interventions (concept-bottleneck steering, SAE feature steering, activation patching,
  truthfulness-separator steering) tried as ways to close it; best corrected 24% of misses.
- Reference: Ext (physician labels). Action: hazard flagged in output text. Mechanistic: yes,
  and the gap is used as the target quantity the interventions are scored against.
- Delta: not moral, not the model's own judgment, single model and domain. But this paper
  already uses a probe-vs-output gap as the outcome variable that mechanistic interventions
  are built to move, so the spec's framing is not unoccupied.

### 2.10 Peripheral (fetched, less close)
- Wang, *Doing What They Say, Not What They Reason: Locating the Faithfulness Gap in LLM
  Agents*, arXiv:2606.00476 (single author: Yufeng Wang; COLM workshop submission). Poker;
  splits reasoning-to-conclusion from conclusion-to-action; verifiable actions. Not moral.
- Schmied et al., *LLMs are Greedy Agents*, arXiv:2504.16078 (Thomas Schmied, Jörg
  Bornschein, Jordi Grau-Moya, Markus Wulfmeier, Razvan Pascanu). Names "knowing-doing gap"
  as inability to act on knowledge in bandits and tic-tac-toe; RL on self-generated CoT
  narrows it. The likely origin of the phrase in the LLM-agent literature.
- Deb, Krishnan, *STOCKTAKE*, arXiv:2607.13618. Supply-chain POMDP; "knowing-doing rate" from
  graded rationales vs actions against a Bayes oracle. Not moral.
- Libert, Prinzhorn, Henselmans, *Moral Competence Before Moral Content*, arXiv:2609.05036.
  Coherence of verdicts under perturbation; no action arm.
- Huang, Kwak, An, *Understanding Moral Reasoning Trajectories in LLMs: Toward Probing-Based
  Explainability*, arXiv:2603.16017. Probes reasoning steps; no action arm.
- Huang et al., *Model Editing as a Double-Edged Sword*, AAAI 2026 oral, arXiv:2506.20606
  (Baixiang Huang, Zhen Tan, Haoran Wang, Zijie Liu, Dawei Li, Ali Payani, Huan Liu, Tianlong
  Chen, Kai Shu). Edits weights to move agent moral behavior (BehaviorBench); no measured
  judgment-vs-behavior gap on the abstract page.

---

## 3. Novelty re-centering

What the spec cannot claim. The knowing-doing gap in LLMs is a named, measured quantity in
at least four independent lines: value-profile vs enacted choice (Huang et al. 2026; Shen et
al. 2025; Rakshit et al. 2026), stated vs revealed preference (Gu et al. 2025), judgment vs
allocation (Hosseini et al. 2026), and capability-knowledge vs action with probes (Cheng et
al. 2026; Basu et al. 2026). "Do LLMs act on their own stated values" with the model's own
statement as the reference and a discrete action as the readout is published, with an
Outstanding Paper award (Shen et al.) and a direct "knowing but not doing" title (Huang et
al.). A fast-vs-slow deliberation contrast on the same construct exists (Rakshit et al.). The
framing "probe-vs-output gap as the outcome variable that mechanistic interventions are scored
against" is occupied by Basu et al. 2026 in clinical triage, and "knowing-doing gap plus
late-layer probe orthogonality at the action token" is occupied by Cheng et al. 2026 in tool
use. The spec should not present "KDG as an interpretability outcome variable" as a new idea
in general. The three-cell base/instruct design has no direct precedent among these, but that
is a design choice, not a construct.

What the spec can claim, stated as deltas against the closest three.

1. Against Huang et al. 2026 and Shen et al. 2025: the reference is a per-scenario,
   third-person moral judgment of the same scenario, not a trait questionnaire or a value
   inclination; the gap is a per-scenario binary self-inconsistency with a calibration ladder
   (re-elicitation floor, matched frame-change null, MDE), not a profile correlation; and the
   action is taken by the model as the agent inside the scenario under a typed pressure
   (goal pursuit, social cost, shortcut, loyalty, third-party harm), not an advisor's pick.
   The pressure-family decomposition, in particular the harm-involving vs non-harm contrast
   that ties back to the FL rank-1 harm read, is not in either paper.
2. Against Rakshit et al. 2026: the dose arm adds a budget-matched non-moral filler control,
   a discrete logged action rather than text, and base models. The honest sentence is "we
   replicate the fast/slow contrast of Rakshit et al. on a discrete action with a matched
   filler null", not "we introduce a deliberation-dose manipulation".
3. Against Cheng et al. 2026 and Basu et al. 2026: the spec's own contribution to the
   outcome-variable framing is narrower and should be worded that way: a *moral*
   knowing-doing gap, referenced to the model's own judgment, measured in the same models and
   positions where the program already has a typed moral subspace and a refusal-read result,
   so that the later "rank of the moral read at the action position" cell inherits a
   calibrated behavioral target. Cheng et al. already show two decodable probe directions
   going orthogonal at the decision token; the spec's S1 quantity is that measurement moved to
   moral content, and should cite it as the precedent.
4. Base-vs-instruct three-cell comparison of a self-referenced moral gap: none of the fetched
   works include a base model (Rakshit et al. explicitly test only instruct variants). This is
   the cleanest unoccupied cell and connects to the program's pretraining thesis. It should be
   the lead novelty sentence, with the pressure-family structure second.

Recommended wording for section 0: replace "defines and measures the knowing-doing gap" with
"measures a self-referenced, per-scenario moral knowing-doing gap (after Huang et al. 2026;
Shen et al. 2025) under typed pressure, with a deliberation-dose arm (after Rakshit et al.
2026) and a base/instruct three-cell design, as the behavioral target for the program's
action-position read (precedent: Cheng et al. 2026; Basu et al. 2026)". The phrase "outcome
variable every later causal cell is built to move" can stay as a description of the program's
internal role for KDG, but not as a novelty claim.

Rival reading to carry into the referee pass: given Huang et al.'s near-perfect cross-model
agreement on enacted choices, Branch A's cross-model KDG table may show the same collapse of
variance across models, in which case the interesting quantity is per-family structure, not
the model axis. Pre-register that read.

---

## 4. Verify-before-citing table

| Item | Status | URL fetched |
|---|---|---|
| MACHIAVELLI, Pan et al., ICML 2023, arXiv:2304.03279 | VERIFIED | https://arxiv.org/abs/2304.03279 |
| Agentic Misalignment blog, Anthropic, Jun 20 2025 | VERIFIED | https://www.anthropic.com/research/agentic-misalignment |
| Agentic Misalignment paper, Lynch et al., arXiv:2510.05179 | VERIFIED | https://arxiv.org/abs/2510.05179 |
| Towards Understanding Sycophancy, Sharma et al., arXiv:2310.13548 | VERIFIED (ICLR 2024 venue UNVERIFIED on page) | https://arxiv.org/abs/2310.13548 |
| FlipFlop, Laban et al., arXiv:2311.08596 | VERIFIED | https://arxiv.org/abs/2311.08596 |
| TRUTH DECAY, Liu et al., arXiv:2503.11656 | VERIFIED | https://arxiv.org/abs/2503.11656 |
| Challenging the Evaluator, Kim & Khashabi, EMNLP 2025 Findings, arXiv:2509.16533 | VERIFIED | https://arxiv.org/abs/2509.16533 |
| SPINE, Tang et al., arXiv:2609.09090 | VERIFIED | https://arxiv.org/abs/2609.09090 |
| Beyond Sycophancy, Wang & Koch, arXiv:2607.21558 | VERIFIED | https://arxiv.org/abs/2607.21558 |
| Persona Vectors, Chen et al., arXiv:2507.21509 | VERIFIED | https://arxiv.org/abs/2507.21509 |
| Knowing But Not Doing, Huang et al., arXiv:2601.07972 | VERIFIED | https://arxiv.org/abs/2601.07972 ; https://arxiv.org/html/2601.07972 |
| Mind the Value-Action Gap, Shen et al., EMNLP 2025, arXiv:2501.15463 | VERIFIED | https://arxiv.org/abs/2501.15463 ; https://aclanthology.org/2025.emnlp-main.154/ |
| Pseudo-Deliberation, Rakshit et al., arXiv:2605.09893 | VERIFIED (venue: arXiv only) | https://arxiv.org/abs/2605.09893 ; https://arxiv.org/html/2605.09893 |
| Alignment Revisited, Gu et al., arXiv:2506.00751 | VERIFIED | https://arxiv.org/abs/2506.00751 |
| Judgment-Consequence Gap, Hosseini et al., AIES 2026, arXiv:2608.05583 | VERIFIED | https://arxiv.org/abs/2608.05583 |
| When Ethics and Payoffs Diverge, Backmann et al., arXiv:2505.19212 | VERIFIED | https://arxiv.org/abs/2505.19212 |
| Superficial Beliefs, Freedman & Toni, arXiv:2606.11016 | VERIFIED | https://arxiv.org/abs/2606.11016 |
| Model-Adaptive Tool Necessity, Cheng et al., arXiv:2605.14038 | VERIFIED | https://arxiv.org/abs/2605.14038 |
| Interpretability without actionability, Basu et al., arXiv:2603.18353 | VERIFIED | https://arxiv.org/abs/2603.18353 |
| Faithfulness Gap, Wang, arXiv:2606.00476 | VERIFIED | https://arxiv.org/abs/2606.00476 |
| LLMs are Greedy Agents, Schmied et al., arXiv:2504.16078 | VERIFIED | https://arxiv.org/abs/2504.16078 |
| STOCKTAKE, Deb & Krishnan, arXiv:2607.13618 | VERIFIED | https://arxiv.org/abs/2607.13618 |
| Moral Competence Before Moral Content, Libert et al., arXiv:2609.05036 | VERIFIED | https://arxiv.org/abs/2609.05036 |
| Moral Reasoning Trajectories, Huang, Kwak, An, arXiv:2603.16017 | VERIFIED | https://arxiv.org/abs/2603.16017 |
| Model Editing as a Double-Edged Sword, Huang et al., AAAI 2026, arXiv:2506.20606 | VERIFIED | https://arxiv.org/abs/2506.20606 |
| Moral Alignment for LLM Agents (ICLR 2025, arXiv:2410.01639) | UNVERIFIED (search hit only, not fetched) | none |
| SycEval, SYCON Bench, MedPRESS (search hits) | UNVERIFIED (not fetched) | none |

Not fetched and therefore not to be cited from this pass: anything not in the table.
