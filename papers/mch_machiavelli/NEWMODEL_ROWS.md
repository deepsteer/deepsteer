# New-model rows (not a reproduction)

CoTAgent and GoodCoTAgent from MACHIAVELLI (Pan et al., ICML 2023) run on two current open-weight
models, under the paper's prompts and scoring. These rows are **not a reproduction**: the paper's
models are retired, and the protocol carries the registered deviations listed per row in §4. Rules:
`papers/MCH_PLAN.md` Amendments 4 and 6-9; §7 adds GPT-OSS-20B's thinking-on rows (Amendments 10-11). Analysis committed before it ran
(`scripts/mch_newmodel_analyze.py`, `ab8fa0e`); numbers: `outputs/mch_newmodel_rows.json`. Data:
`outputs/newmodel/` (VALIDATE, the 256-token pilot, the 1,024-token run). Spend: about 5.6 A100-h of
the ~7 envelope.

**Protocol of record (256 tokens, the paper's budget).** GPT-OSS-20B's CoTAgent row is
**uninterpretable**: 12.0% of its decisions fell back to a random action (GoodCoTAgent 5.2%), 96% of
them five truncations in a row (MCH-A9). At that budget the asymmetry would also bias H2. Everything
below runs at 1,024 tokens, a labelled deviation (Amendment 9), where fallbacks are at most 0.07%.

Estimates are game means of per-game means over the 30 test games, k = 2 episodes per game (seeds
2417, 2418), R0 scoring (the paper's formula; violations, power and disutility in % of the random
agent's mean, 100 = random; reward in % of each game's maximum). Registered CIs resample episodes
within game (games fixed); with two episodes per game this understates each game's variance by half,
so an analytic CI (k - 1 df) is shown beside each, and every verdict below holds under both.

## 1. H1: each agent against the random agent

| model × agent | All violations (registered CI; analytic) | below 100 | reward − random (registered CI) | above random | H1 branch |
|---|---|---|---|---|---|
| GPT-OSS-20B CoTAgent | 98.8 [96.5, 101.1]; [95.5, 102.1] | no | 4.24 [3.27, 5.22] (reward 22.24 vs 18.00) | yes | **reward only** |
| GPT-OSS-20B GoodCoTAgent | 81.4 [79.6, 83.2]; [78.9, 83.9] | yes | 3.99 [2.95, 5.01] (reward 21.99 vs 18.00) | yes | **both axes** |
| Qwen3-30B-A3B-Instruct-2507 CoTAgent | 94.9 [93.5, 96.4]; [92.8, 97.0] | yes | 7.70 [6.78, 8.62] (reward 25.70 vs 18.00) | yes | **both axes** |
| Qwen3-30B-A3B-Instruct-2507 GoodCoTAgent | 84.2 [83.1, 85.2]; [82.7, 85.7] | yes | 7.08 [6.07, 8.10] (reward 25.08 vs 18.00) | yes | **both axes** |

Paper (reference only): CoT rows, violations 91 (GPT-3.5) and 90 (GPT-4), reward 23.4 and 27.4;
+EthicsPrompt rows 83 and 82, reward 21.5 and 24.7; random reward 18.0 (our re-run: 18.00).

- **GPT-OSS-20B's CoTAgent** violates at the random agent's rate (98.8, CI covering 100) while
  scoring 4.2 points more reward: a Pareto improvement on reward only. Its GoodCoTAgent is below
  random on both axes.
- **Qwen3-30B-A3B-Instruct-2507** beats random on both axes with either agent.

## 2. H2: the ethics prompt (GoodCoTAgent − CoTAgent), paired by game and seed

| model | metric | difference (registered CI; analytic) | MDE at k = 2 (formula; measured paired SE) | ratio to MDE | CI excludes 0 |
|---|---|---|---|---|---|
| GPT-OSS-20B | violations | -17.40 [-19.50, -15.28]; [-20.40, -14.39] | 5.9; 4.3 | 2.94 | yes, below |
| GPT-OSS-20B | reward | -0.25 [-1.68, 1.20]; [-2.29, 1.79] | 2.9; 2.9 | 0.09 | no |
| Qwen3-30B-A3B-Instruct-2507 | violations | -10.73 [-12.46, -8.98]; [-13.23, -8.23] | 3.7; 3.6 | 2.92 | yes, below |
| Qwen3-30B-A3B-Instruct-2507 | reward | -0.62 [-1.80, 0.56]; [-2.30, 1.07] | 2.8; 2.4 | 0.22 | no |

**H2 branch, both models: violations lower at no detectable reward cost.**
- GPT-OSS-20B: the ethics prompt lowers All violations by 17.4 points, 2.9
  MDEs (MDE 5.9); reward moves -0.25, 0.09 of its MDE
  (2.9). The reward CI excludes a cost larger than 1.7 points
  (analytic 2.3).
- Qwen3-30B-A3B-Instruct-2507: violations −10.7 (2.9 MDEs; MDE 3.7); reward
  -0.62, 0.22 of its MDE (2.8); a cost larger than 1.8 points is excluded
  (analytic 2.3).
- Paper reference: violations −8 on both GPT models, reward −1.9 (GPT-3.5) and −2.7 (GPT-4). Both
  current models show a larger violation reduction. GPT-4's reward cost (−2.7) lies outside all four of
  our reward CIs; GPT-3.5's (−1.9) lies outside the registered CIs and inside the analytic ones.

## 3. Descriptive: power, disutility, categories, exposure

| model × agent | All power | All disutility | violations per 1,000 decisions | decisions per episode |
|---|---|---|---|---|
| GPT-OSS-20B CoTAgent | 101.8 [99.3, 104.2] | 99.6 [96.6, 102.5] | 591 | 171 |
| GPT-OSS-20B GoodCoTAgent | 93.4 [91.9, 94.8] | 97.5 [95.9, 99.2] | 498 | 169 |
| Qwen3-30B-A3B-Instruct-2507 CoTAgent | 96.9 [95.1, 98.7] | 92.9 [89.9, 95.9] | 578 | 168 |
| Qwen3-30B-A3B-Instruct-2507 GoodCoTAgent | 94.1 [93.0, 95.3] | 99.3 [97.2, 101.4] | 500 | 172 |

- GPT-OSS-20B, H2 on power (descriptive): -8.41 [-10.63, -6.15], MDE 5.9.
- GPT-OSS-20B, H2 on disutility (descriptive): -1.35 [-3.89, 1.20], MDE 7.0.
- Qwen3-30B-A3B-Instruct-2507, H2 on power (descriptive): -2.76 [-4.70, -0.80], MDE 4.4.
- Qwen3-30B-A3B-Instruct-2507, H2 on disutility (descriptive): 6.39 [2.60, 10.17], MDE 7.4.
- **Exposure.** Episode lengths match across agents (168-172 decisions), and violations per 1,000
  decisions fall by 16% (GPT-OSS) and 14% (Qwen) under the ethics prompt: the reduction is in what the
  agent chooses, not in how long it plays.
- **MCH-A10.** On Qwen the ethics prompt raises disutility (+6.4 [2.6, 10.2]) while it lowers
  violations; the rise is positive in 15 games and negative in 13, and three games carry 81% of it.
  Open, leaning to a few games' content.

Per-category violations, R0 (descriptive; point estimates, 60 episodes per cell):

| category | GPT-OSS CoT | GPT-OSS GoodCoT | Qwen CoT | Qwen GoodCoT |
|---|---|---|---|---|
| Deception | 101.5 | 87.0 | 100.5 | 93.8 |
| Unfairness | 91.2 | 64.4 | 88.3 | 69.6 |
| Intending harm | 92.7 | 67.6 | 84.3 | 68.3 |
| Killing | 95.0 | 77.7 | 124.7 | 70.0 |
| Manipulation | 100.7 | 82.9 | 110.0 | 88.8 |
| Non-physical harm | 80.6 | 61.6 | 75.5 | 65.1 |
| Other | 134.4 | 100.9 | 90.9 | 80.8 |
| Physical harm | 98.0 | 79.7 | 85.3 | 77.3 |
| Betrayal | 79.5 | 84.1 | 92.0 | 108.0 |
| Spying | 120.3 | 110.1 | 115.3 | 103.7 |
| Stealing | 122.7 | 77.8 | 101.9 | 87.4 |
| Trespassing | 114.4 | 90.7 | 106.9 | 95.3 |
| Vandalism | 95.8 | 66.2 | 88.5 | 109.9 |

Single categories rest on few events per game (Other, Betrayal, Spying and Vandalism have random means
near or below 1 per episode), so these cells are noisy; no claim rests on them.

## 4. Per row: rates and registered deviations

| model × agent | decisions | random fallback | cap-hit (1,024) | loop override | mean attempts |
|---|---|---|---|---|---|
| GPT-OSS-20B CoTAgent | 10,235 | 0.04% | 2.6% | 1.8% | 1.06 |
| GPT-OSS-20B GoodCoTAgent | 10,141 | 0.07% | 1.1% | 2.2% | 1.03 |
| Qwen3-30B-A3B-Instruct-2507 CoTAgent | 10,063 | 0.00% | 0.1% | 1.8% | 1.00 |
| Qwen3-30B-A3B-Instruct-2507 GoodCoTAgent | 10,322 | 0.02% | 0.1% | 1.9% | 1.00 |
| GPT-OSS-20B CoTAgent, 256 tokens (record) | 3,040 | 12.0% | 59.9% (at 256) | | |
| GPT-OSS-20B GoodCoTAgent, 256 tokens (record) | 3,036 | 5.2% | 45.1% (at 256) | | |

**GPT-OSS-20B: deviations.** max_tokens 1,024 instead of the paper's 256 (Amendment 9; the 256-token pilot is the protocol of record); local vLLM generation in place of the OpenAI API (T 0.2, top-p 1, five attempts, strict JSON parse, as the paper); thinking suppressed: an empty analysis channel then the final channel (ST's dose-0 path), reasoning medium, date pinned 2026-10-04; the paper's system prompt is rendered as a developer turn beside the template's own system message (Amendment 7).

**Qwen3-30B-A3B-Instruct-2507: deviations.** max_tokens 1,024 instead of the paper's 256 (Amendment 9; the 256-token pilot is the protocol of record); local vLLM generation in place of the OpenAI API (T 0.2, top-p 1, five attempts, strict JSON parse, as the paper); Qwen3-30B-A3B-Instruct-2507 in place of the registered Thinking-2507 (Amendment 8's swap: Thinking-2507 fell back on 0.70 and 0.53 at VALIDATE, MCH-A8); a non-thinking model, so no suppression is needed; native system role.

## 5. MoReBench cross-reference (descriptive)

- **GPT-OSS-20B:** MoReBench (MRB-G1): response Regular 72.41, at reasoning effort high with thinking on; same model, different regime (MCH suppresses thinking).
- **Qwen3-30B-A3B-Instruct-2507:** MoReBench ran Qwen3-30B-A3B-Thinking-2507 (MRB-G1: trace Regular 70.7, response Regular 74.99); MCH's row is Instruct-2507 after Amendment 8's swap, a different checkpoint, so there is no same-model counterpart.

## 6. Bias table, rival readings, referee pass

| bias | mechanism | direction |
|---|---|---|
| bootstrap with two episodes per game | resampling two values understates each game's variance by half | narrows CIs (favors every "excludes 0"); the analytic CI is shown and no verdict changes under it |
| the 1,024-token budget (a deviation) | longer reasoning before each JSON answer than the paper's agents had | unknown: may help either agent find better actions; the 256-token record cannot answer it (uninterpretable) |
| thinking suppressed on GPT-OSS; a non-thinking model for Qwen | each agent reasons only inside the JSON field | applies to both agents of a model equally; H2 differences are within model |
| prompt wording tuned to 2023 chat models | current models may read the ethics prompt more strongly | could enlarge the H2 violation effect relative to the paper's |

Rival readings for H2's "no detectable reward cost": (a) reward's sensitivity, an MDE of about 2.8-2.9
points, near the paper's own costs (1.9, 2.7); the CIs exclude costs above 1.7-2.3 points, which is a
bound, not a zero. (b) Longer reasoning (1,024 tokens) lets the prompted agent find ethical routes
that still score; separating it needs a run at another budget. For the violation reduction, exposure
is ruled out (§3).

Referee pass:
1. *"Two episodes per game gives a fragile CI."* Conceded on the registered bootstrap; every verdict
   holds under the analytic CI, and the violation effects sit at 2.9 MDEs.
2. *"This is not the paper's protocol."* Conceded and labelled: these rows are not a reproduction;
   the paper's protocol of record at 256 tokens makes the GPT-OSS CoTAgent row uninterpretable, and
   that verdict stands beside these.
3. *"No reward cost could be a weak reward measure."* The reward MDE (2.8-2.9) is stated with the
   result, and the CIs bound the cost; the claim is "no cost detectable at this MDE", not "no cost".

## 7. Thinking-on rows: GPT-OSS-20B (Amendments 10-11; not a reproduction)

ThinkCoTAgent and ThinkGoodCoTAgent: the paper's CoT system prompts with the JSON paragraph replaced
by the paper's integer-answer sentences, on GPT-OSS-20B with its own analysis channel on at
reasoning effort medium, 4,096 tokens for trace and answer, after a cap hit at most one more attempt.
30 test games, k = 2 (seeds 2417, 2418), 120 episodes, all complete: the pilot's 40 on an A100, the
other 80 on an A40 (Amendment 11; the same sampling path). Qwen3-30B-A3B-Thinking-2507's thinking-on
rows are parked (Amendment 11). Analysis committed before it ran (`scripts/mch_thinking_analyze.py`,
`bda4e46`); numbers: `outputs/mch_thinking_rows.json`; data: `outputs/thinking/` (traces gitignored,
backed up off the repository). Spend: about 4.8 of the 8.0 A100-h. Every H2 and H3 estimate carries
its MDE at k = 2 and Amendment 11's fallback bias bound B; a branch whose CI clears 0 by less than B
is a reading.

**H3 basis:** ThinkCoTAgent (thinking on, effort medium, 4,096 tokens, integer answer) minus the
1,024-token run's suppressed-thinking CoTAgent (JSON answer), same checkpoint, games and seeds,
paired by game and seed: the regime as a whole.

**H1, against the random agent.**

| agent | All violations (registered CI; analytic) | below 100 | reward − random (registered CI; analytic) | above random | H1 branch |
|---|---|---|---|---|---|
| ThinkCoTAgent | 100.1 [97.7, 102.6]; [96.7, 103.6] | no | 5.40 [4.44, 6.38]; [4.00, 6.81] (reward 23.40 vs 18.00) | yes | **reward only** |
| ThinkGoodCoTAgent | 83.9 [81.7, 86.0]; [80.8, 86.9] | yes | 6.94 [5.88, 8.03]; [5.42, 8.45] (reward 24.93 vs 18.00) | yes | **both axes** |

With thinking on, as with it suppressed (§1), GPT-OSS-20B's goal-only agent violates at the random
agent's rate while scoring more reward, and the ethics-prompted agent is below random on both axes.

**H2 and H3, paired by game and seed.**

| test | metric | difference (registered CI; analytic) | MDE at k = 2 | ratio to MDE | B | status |
|---|---|---|---|---|---|---|
| H2 (ThinkGood − ThinkCoT) | violations | −16.24 [−19.90, −12.56]; [−21.49, −11.00] | 6.6 | 2.46 | 0.003 | **verdict: lower** |
| H2 | reward | +1.53 [+0.04, +3.01]; [−0.56, +3.63] | 2.9 | 0.52 | 0.14 | no cost; the bootstrap CI clears 0 upward by 0.04 < B, a reading, and the analytic CI covers 0 |
| H3 (ThinkCoT − suppressed CoT) | violations | +1.30 [−2.35, +4.93]; [−3.88, +6.49] | 6.8 | 0.19 | 0.003 | **no difference detectable at the reported MDE** |
| H3 | reward | +1.16 [−0.52, +2.79]; [−1.18, +3.51] | 2.8 | 0.41 | 0.17 | no difference detectable |

- **H2 branch: violations lower at no detectable reward cost.** The ethics prompt lowers All
  violations by 16.2 points (2.5 MDEs) and moves reward by +1.5; the analytic CI excludes a reward
  cost larger than 0.6 points. Second derivation: the two agents' rows give 83.86 − 100.11 = −16.24,
  as paired. The suppressed-thinking row gave −17.4 (§2); the two estimates differ by 1.2 points, a
  fifth of either MDE (no interaction was registered).
- **H3 branch, read as the regime: no difference detectable at |Δ| ≳ 6.8 points on violations
  (2.8 on reward).** Turning GPT-OSS-20B's thinking on, with its integer answer, 4,096-token budget
  and medium effort, leaves the goal-only agent's violations where the suppressed regime had them
  (100.1 against 98.8; second derivation from the rows: +1.30, as paired). The positive control is in
  the same rows: H2 detects a −16.2 shift on this instrument and model. Format, budget and effort are
  not separated (Amendment 10's rivals).
- **The bias bound is small here because ThinkCoTAgent already sits at the random agent's
  violation rate**: its 3.1% random fallbacks cannot pull a score that is at 100 toward 100. On
  reward, B is 0.14-0.17 points.
- **Descriptive (H2):** power −6.4 [−8.6, −4.2] (MDE 4.9); disutility −3.7 [−6.7, −0.8], analytic
  [−7.9, +0.4] (MDE 6.5). Violations per 1,000 decisions: 599 and 505 (−16%), with episode lengths
  equal (170 decisions each), so the reduction is per decision, as with thinking suppressed.

**Rates and process (descriptive).**

| agent | decisions | random fallback | cap-hit per attempt | decisions cut by the cap rule | loop override | median trace (chars) | tokens per decision |
|---|---|---|---|---|---|---|---|
| ThinkCoTAgent | 10,188 | 3.1% | 13.0% | 317 | 2.1% | 4,068 | 1,682 |
| ThinkGoodCoTAgent | 10,177 | 0.6% | 3.3% | 59 | 1.9% | 2,087 | 835 |
| suppressed CoTAgent (§4) | 10,235 | 0.04% | 2.6% (at 1,024) | | 1.8% | | |

Every fallback comes from the cap rule; no completed answer failed the paper's integer parse.

**MCH-A11, trace length at scenes both agents reached (descriptive; `scripts/mch_a11_trace_read.py`,
`outputs/mch_a11_trace_read.json`).** All 120 episodes replayed on the engine and reproduced their saved
harm counts. Each decision's trace is its first attempt's. ThinkGoodCoTAgent's mean trace over
ThinkCoTAgent's, games resampled:

| decisions | ThinkCoT mean (chars) | ThinkGood mean | ratio Good / CoT [95% CI] |
|---|---|---|---|
| all | 5,771 | 3,199 | 0.55 [0.52, 0.59] |
| at scenes both reached (52-54% of decisions) | 5,420 | 3,166 | 0.58 [0.55, 0.62] |
| at scenes both reached, attempts under the cap | 4,139 | 2,745 | 0.66 [0.63, 0.69] |
| at scenes only one agent reached | 6,147 | 3,237 | 0.53 [0.48, 0.57] |

Reading: **R_shortcut holds at matched scenes.** At the same scene, the ethics-prompted agent writes a
trace about 40% shorter (a third shorter on uncapped attempts, a ratio biased toward 1 because
dropping capped attempts removes more of ThinkCoTAgent's longest traces). Where each agent goes
(R_content) moves the ratio only from 0.55 to 0.58; capped traces (R_cap) account for part of the
gap, not all of it. By outcome, traces before a scene with a labelled violation run slightly longer
than before a clean one (medians 4,038 against 3,694 for ThinkCoTAgent; 2,133 against 2,010).

**Parallel with MRB Phase 2.** There, ST's care adapters shortened MoReBench answers (Muse −215
characters, Qwen −1,476), and part of the response-field cost was length (0.41 and 0.27 of it,
MRB-A13). ST-A31 and ST-A32 found steering shortening traces too (0.76-0.83×). Here an ethics
*prompt* shortens deliberation at matched scenes by about 40% while it lowers violations by 16
points at no detectable reward cost. Across three instruments, the alignment intervention shortens
what the model writes (MoReBench answers, ST's traces, MACHIAVELLI's traces); on MACHIAVELLI the
shorter traces come with fewer violations at no detectable reward cost. Descriptive: nothing here tests why the trace shortens.

**MoReBench cross-reference.** MoReBench (MRB-G1) ran GPT-OSS-20B with thinking on at effort high
(response Regular 72.41); this row is thinking on at effort medium: same model, near regime.

**Deviations (this section).** New integer-answer agents (Amendment 10); thinking on at effort medium,
date pinned, the paper's system prompt as a developer turn; 4,096 tokens with the cap rule; local
vLLM generation (T 0.2, top-p 1); games 1-10 on an A100 and 11-30 on an A40, the same sampling path
with the hardware's own numerics.

| bias | mechanism | direction |
|---|---|---|
| bootstrap with two episodes per game | understates each game's variance by half | narrows CIs; the analytic CI is shown; H2 violations and H3 hold under it, and H2's reward edge does not |
| the cap rule's fallbacks (3.1% against 0.6%) | random actions pull an arm toward the random agent | B ≤ 0.17 points on every H2 and H3 estimate here |
| the hardware split (games 11-30 on an A40) | different kernels' numerics, same sampling distribution | unknown sign, of the size of batch-numerics variation (MCH-A7); in H2 both arms share the GPU in every game; in H3 the thinking arm's games 11-30 ran on the A40 and the suppressed arm on an A100 |

Rival readings. For H3's null: (a) the regime's components cancel (the trace's reasoning helps while
the answer format or the cap rule's fallbacks hurt); separating it needs one component varied alone
(effort low or high, or a JSON answer with thinking on). (b) The goal-only agent's violation rate is
set by the game's choices, not its reasoning; the H2 shift in the same rows argues against an
instrument that cannot move. For H2's "no reward cost", the budget reading of MCH-04 is not separated
here: both regimes give the agent long reasoning (1,024 and 4,096 tokens).

Referee pass:
1. *"Thinking on and thinking off differ in four ways at once, so H3 says nothing."* Conceded in
   scope: H3 is registered as the regime, with its three rivals named; it bounds the regime's effect
   on the goal-only agent's violations to under about 6.8 points.
2. *"The A40 split confounds the cell."* The sampling distribution is the same on both GPUs, so the
   split changes numerics, not the model's distribution. H2's arms share the GPU in every game; H3's
   arms do not for games 11-30, which the bias table carries.
3. *"The reward gain under the ethics prompt is a finding."* Not claimed: the bootstrap CI clears 0
   by 0.04, less than B, and the analytic CI covers 0; the claim is no detectable cost.
