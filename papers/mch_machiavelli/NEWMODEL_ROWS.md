# New-model rows (not a reproduction)

CoTAgent and GoodCoTAgent from MACHIAVELLI (Pan et al., ICML 2023) run on two current open-weight
models, under the paper's prompts and scoring. These rows are **not a reproduction**: the paper's
models are retired, and the protocol carries the registered deviations listed per row in §4. Rules:
`papers/MCH_PLAN.md` Amendments 4 and 6-9. Analysis committed before it ran
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

