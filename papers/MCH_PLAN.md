# MCH Plan — MACHIAVELLI Replication, Decomposed Into the Library

Status: v0.7, 2026-11-03. **v0.7 (author, 2026-11-03): Amendment 9 (§5): the 256-token pilot is the protocol-of-record verdict; `max_tokens` 1,024 as a labelled deviation for the whole new-model cell; the pilot reruns with a Qwen Instruct-2507 rate leg; envelope about 7 A100-h total; Phase 2 parked.** v0.6 (author, 2026-11-02): Amendment 8 (§5): Qwen stays Thinking-2507 (MRB's checkpoint), with a swap to Instruct-2507 if its VALIDATE fallback exceeds 10%; comparisons with MRB are same model, different regime; the bundle's local-cache fallback.** v0.5 (author, 2026-11-02): Amendment 7 (§5): G1's stability fork and its fp32 positive control; the chat path has no 2,047 cut, so the new-model rows do not truncate; how each model receives the system prompt; the pod runner and its stages.** v0.4 (author, 2026-11-02): Amendment 6 (§5): G1's variance is descriptive; a variance pilot on GPT-OSS-20B is stage 1 of the new-model pod and decides k, the Phase 2 go/no-go and stop-or-continue; a 10% fallback bar; the new-model envelope is about 6 A100-h.** v0.3 (author, 2026-11-02): MCH-G0 passed (`mch_machiavelli/MCH_G0_REPORT.md`);
Amendment 3 (§5) records the gate decisions, the Phase 2 go/no-go rule and the corrections folded in
below; Amendment 4 registers the new-model rows; Amendment 5 the G1 protocol, all before any G1
data.** v0.2 (agent at MCH-G0, 2026-11-02): Amendment 1 (§5): the
random-baseline analysis registered before the full run is read; Amendment 2: the missing-key fork. v0.1 draft, 2026-11-02 (author request: reproduce MACHIAVELLI as a reference implementation and
as the action-side instrument for the steering work). Pre-registration candidate. **No GPU spend
authorized by this document.** Line prefix **MCH**: this file is `papers/MCH_PLAN.md`; code, data
manifests and outputs in `papers/mch_machiavelli/`; anomalies `MCH-A1, ...` in `papers/ANOMALIES.md`;
claims `MCH-01, ...` in `papers/SYNTHESIS.md` (the MB and MRB precedent; Amendment 3). Inherits the goals, standard and license rules of
`papers/MORAL_EVAL_REPLICATIONS.md` (MCH section) and the shared modules built under MB and MRB.

Skills in force: `instrument-calibration` (the random agent is the ladder's floor and the metric's
normalizer), `estimator-traps` (difference CIs; games fixed, episodes resampled), `compute-ordering`
(zero GPU first; the CPU baseline before any model), `anomaly-triage`, `program-thesis` for prose.

## 0. Why this benchmark

**For DeepSteer's research.** MoralBench and MoReBench measure what a model says; MACHIAVELLI
measures what an agent does over long play while reward pulls against ethics.
- **The action side under reward pressure.** It is KDG's judgment–action question at agent scale:
  achievements reward the agent, and many of the paths to them are deceptive, harmful or
  power-seeking.
- **Deterministic scoring.** Every scene carries precomputed labels (18 categories: ethical
  violations, disutility, power), so scoring needs no judge. After MRB-A10, that matters.
- **ST's specificity question in agentic form.** Does an adapter lower violations without giving up
  reward? The paper's ethics prompt under reward pressure (`GoodCoTAgent`: the goals and the ethics
  prompt) is a ready inference-time comparator that a weight-level adapter should match or beat
  without the prompt. (`GoodAgent` drops the goals, the paper's "+EthicsPrompt + NoGoals"; it is the
  no-reward-pressure reference, not the comparator.)
- **Forward consumers, not built here (§8):** a per-decision activation read (does a probe at the
  decision predict a labelled violation before the action is chosen?), which serves the MoReBench +
  steering line's represented-versus-verbalized idea on the action side; and a KDG-style judgment
  frame on the same scenes (a judgment–action gap with reward pressure built in).

**For others (reference implementation).** Nobody can run the paper's LM agents today:
- the LM agents call the legacy OpenAI APIs, and every model the paper ran through them
  (`davinci`, `gpt-3.5-turbo`, `gpt-4` of early 2023) is retired;
- `lm_agents.py` raises `NotImplementedError` for any other model, so the LLaMA rows of the
  released results come from code that is not in the repo;
- the stack pins `gym==0.25.1` and `spacy==3.4`.

A maintained harness that runs any HF or vLLM model with the paper's agent prompts, the paper's
metric pipeline, pinned data, a re-run random baseline and an audited results table is the useful
artifact.

## 1. Source of record

- Paper: Pan, Chan, Zou, Li, Basart, Woodside, Ng, Zhang, Emmons, Hendrycks, "Do the Rewards Justify
  the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the MACHIAVELLI Benchmark",
  arXiv:2304.03279, ICML 2023 (oral). Pin the arXiv version whose tables are compared, at MCH-G0.
- Code: `github.com/aypan17/machiavelli`, commit `ebb44e1` (2025-07-23), MIT.
- Released results: `experiments/results/main_results.csv` and `main_results_expanded.csv` in that
  commit (MIT, part of the repo).
- Data: the game bundle (`game_data/`) on Google Drive, password published in the README; the
  "artificial conscience" weights on a second Drive file. Pin both by sha256 at MCH-G0.

## 2. What the source pins, and what it leaves open

**Pinned by the code:**
- the environment (`machiavelli/game/machiavelli_env.py`) and trajectory scoring
  (`evaluate_trajectories.py`, `experiments/export_results.py`);
- the 30 test games (listed in `experiments/baselines_*.sh`);
- the LM agents' prompts and decoding (corrected at G0): at T 0.2, `LMAgent` and `GoodAgent` chat
  replies are a bare integer (`max_tokens` 8); the CoT agents' chat replies are a JSON object with
  "reasoning" and "action" (`max_tokens` 256), retried up to five times on a failed parse and then
  replaced by a random action. Non-chat agents are read by log-probabilities over the action numbers
  (`max_tokens` 1; the API returned the top five) and an action is sampled from them; the non-chat CoT
  agents first generate up to 1,048 tokens of reasoning. Requests are truncated with GPT-2 token
  counts: non-chat prompts (davinci) to 2,047 tokens by a middle cut; chat requests to 4,096 minus
  `max_tokens`, the system prompt cut in the middle and the scene from the front. Loop detection
  overrides with a random action;
- the random baseline (1,000 episodes per test game) and the DRRN runs (per-game training, 16
  environments, a 24 h per-game job limit).

**Inferred, to confirm at MCH-G0:**
- Table 2's "base" and "+EthicsPrompt" LM rows match the CSV's `CoTAgent` and `GoodCoTAgent` rows to
  rounding; "DRRN +shaping" matches `drrn_reg_more`.
- The random row: the CSV gives 18.36 points; a summary of the paper read 18.0. Confirm against the
  PDF; a real gap is an MCH anomaly.
- Coverage: the random row has 28,414 samples, not 30,000; `drrn_reg_less` and `LMAgent:gpt-3.5-turbo`
  cover 29 games. Find why, and how the export handles a missing game.

**Not pinned:**
- The checkpoints behind the CSV's `llama-7b`, `llama-13b` and `llama-30b` rows (the paper describes
  Alpaca-finetuned LLaMA) and the harness that ran them.
- Which label set (`annotations_model` or a later "model+" set) the released results use.
- The DRRN seeds.

## 3. Rows: what can be run

| paper row | status |
|---|---|
| Random agent | **runnable on CPU** (the M4); it is also the normalizer of every violation score |
| LM agents on `davinci`, `gpt-3.5-turbo`, `gpt-4` | not replicable: models retired |
| LM agents on LLaMA 7B/13B/30B (CSV, Appendix K) | **dropped** (author, G0): the checkpoints cannot be identified and the paper's cells do not follow from its CSV (MCH-A4) |
| DRRN base and +shaping | **not run** (author, G0): unreleased agents and power checkpoint, encoder and seeds unpinned; ceiling 1,440 GPU-h |
| new-model rows (Amendment 4) | CoTAgent and GoodCoTAgent on GPT-OSS-20B and Qwen3-30B-A3B-Thinking-2507 (MRB's checkpoints; thinking suppressed, so comparisons with MRB are same model, different regime; Amendment 8's swap rule); **not a reproduction**, reported in their own section |

## 4. Readouts

- **R0, paper-faithful (the default):** the paper's agent, prompt, truncation and decoding path for
  the model type; the paper's scoring formula (100 minus the "harm reduced" rows pooled by episode,
  zero-mean normalizer cells random-equivalent, missing keys skipped; MCH-A2, MCH-A3); violations as a
  percentage of the random baseline. The pinned code's scorer is reported beside it (author, G0).
- **Normalizers:** report each violation score against our re-run random baseline and against the
  CSV's, so a difference in the normalizer is visible.
- **DeepSteer additions (opt-in, beside R0):** difference CIs with games fixed and episodes
  resampled; the random-action fallback rate per run; action-number mass at the decision (the
  non-chat path already reads it); balanced rotation of action order (required for the steering
  phase under the forward standard, off for the paper-faithful rows, which the paper ran unrotated).

## 5. Gates

- **MCH-G0 (zero GPU).**
  - Fetch and pin the code, the game bundle and the conscience weights; record the license basis.
  - Audit every Table 2 cell (and the paper's other result tables that the CSVs cover) against the
    released CSVs; confirm the row mapping in §2.
  - Re-run the random baseline on the M4's CPU: a pilot of 10 episodes per test game, then 1,000 per
    game if the pilot's time projection allows. Pre-registered rule: the paper's random row
    replicates if it lies inside our bootstrap CI (games fixed, episodes resampled).
  - From the random runs, compute per-game variance and the MDE for the paper's headline comparisons
    at its own design (one LM episode per game).
  - Prompt parity at zero GPU: render our prompts and the original agents' prompts for the same game
    states and compare byte for byte, for both the chat and the non-chat path.
  - Try to identify the `llama-*` checkpoints; report what was found and the license of each.
  - Price DRRN and the steering phase.
  - Namespace proposal (§7) and the license check.
  - Author reads before anything runs on a model.
  - **Amendment 1 (2026-11-02, agent at MCH-G0; registered after the 10-episode timing pilot and
    before any analysis of the 1,000-episode run; for the author's review at G0).** It fixes what
    the registered random-baseline rule leaves open; the rule itself is unchanged.
    - *Found before registering (from the paper, CSVs and coefficient file only, no run data).*
      Table 2's random reward (18.0) is the mean of the 30 per-game means; the CSV's 18.36 and
      Table 9's 18.4 pool episodes, so the three short games weigh less. Table 2's achievements
      cell (9.4) is the pooled value (game mean 9.53). The harm cells are 100 by construction; the
      per-game random means behind them are in `game_data/normalization_coeffs.json` (mean, SD,
      n per game and category).
    - *Primary statistic:* the mean over the 30 test games of our per-game means (games fixed,
      1,000 episodes per game, episodes resampled within game; 10,000 bootstrap draws, seed 0).
    - *Cells and multiplicity.* Primary cells (five): Norm. Reward (18.0), # Achievements (9.4),
      All power, All disutility, All violations. For a harm category the statistic is our
      paper-normalized score, the game mean of 100 × (our per-game mean count / the paper's
      per-game mean), compared with 100; a game whose paper mean is 0 is left out of that
      category (upstream would score it 0; that floor is reported separately). The 17
      sub-category cells are secondary: about 1 miss in 17 is expected by chance.
    - *Rule as registered.* A cell replicates if the paper's printed value, read as the interval
      of its rounding (±0.05 for one decimal; ±0.5 for an integer), intersects our 95% CI.
    - *Fork: the pooled aggregation.* Our per-game means weighted by the paper's per-game n,
      against the CSV and Table 9 (18.36 and 18.4 reward; 9.44 achievements). Reported beside the
      primary, never replacing it.
    - *Secondary: Δ-CI.* For reward and each harm category, ours minus the paper's game-mean
      statistic, with the paper's per-game SD and n from the coefficient file (normal
      approximation per game, independent samples). The registered rule ignores the paper's own
      sampling noise, so it can miss on an exact replication; the Δ-CI is the difference test.
    - *Seed-exact check (diagnostic).* Upstream seeds episode i with i + 2417, so the same agent
      on the same engine reproduces each game's mean exactly. Per game and category, our mean over
      episodes 0 to n − 1 (n = the paper's per-game count) against the coefficient mean: equal
      within 1e-9, or not. Equal on every game: the 2024 `RandomAgent` and today's engine are the
      paper's. Unequal: the comparison is statistical only, and the per-game z-scores (ours minus
      theirs, both SEs) are reported against N(0, 1).
    - *MDE at the paper's design.* σ_g is the SD over our random episodes in game g of the
      per-episode paper-normalized score (points for reward). One agent, one episode per game:
      SE₁ = √(Σ σ_g²) / 30; two agents paired by game: SE₂ = √2 · SE₁; MDE = 2.80 × SE (80% power,
      two-sided 5%). Assumption, stated with it: the agent's within-game episode SD equals the
      random agent's.
    - *Both branches.* Primary cells replicate: the floor and the normalizer are confirmed for the
      2024 `RandomAgent` on today's engine. A primary cell misses: the seed-exact and per-game reads
      locate the games, and the diagnosis (engine version, label set, episode cap, agent) precedes
      any model run.
  - **Amendment 2 (2026-11-02, agent at MCH-G0; a fork registered after Amendment 1's first
    computation returned NaN on three cells, before the fork is computed).** Upstream's
    `get_mach_counts` creates a power or disutility key only when the event occurs, so an episode
    with none has no key, and pandas' `mean` skips it; the paper's coefficient means are
    computed that way (the seed-exact check matches them under it). Amendment 1 does not say which
    reading "per-game mean count" takes. Both are reported, neither replaces the other: *F_skip*
    (the paper's formula: a missing key is skipped) and *F_zero* (a missing key is a count of 0).
    The verdict is stated under each; the violation categories are unaffected (upstream always
    creates their keys).
  - **Amendment 3 (author, 2026-11-02): MCH-G0 decisions.**
    - G0 accepted. R0 scores with the paper's formula; the pinned code's scorer is reported beside
      it (MCH-A2). Corrections folded into §0, §2 and §8: `GoodCoTAgent` is the inference-time
      comparator; the CoT chat path is JSON at 256 tokens with retries and a random fallback.
    - The LLaMA rows are dropped; DRRN is not run (MCH-G2 removed).
    - The engine is fetched at the pinned commit as an optional extra, hash-verified (§7).
    - Per-episode harm counts of the random run are not tracked (they would rebuild the bundle's
      coefficient file); seeds and actions are. Done by rewriting the unpushed G0 results commit.
    - Claims go in `SYNTHESIS.md`. The prompt renderer stays staged in `papers/mch_machiavelli/`
      until G1 validates it.
    - **Phase 2 go/no-go (registered before G1).** From G1's measured within-game variance of the
      per-episode R0 All-violations score (σ_g², pooled over the k episodes per game; F_skip), SE₁ =
      √(Σ σ_g²) / 30 and the paired MDE for adapter minus none at k episodes per game, MDE_k =
      2.80 × √2 × SE₁ / √k. Phase 2 runs only if some k ≤ 4 gives MDE_k ≤ 4 points and the steering
      read at that k (2 models × 2 arms × k, priced from G1's measured decisions per episode and
      prompt tokens on the G0 basis: ~3,000 prefill tok/s for a 30B dense model on one A100, half of
      each prompt prefix-cached, ×1.3 for full action-order rotation) prices at about 10 A100-h or
      less (the midpoint of the priced range ≤ 10). Otherwise Phase 2 is parked. The new-model rows
      on Muse and Qwen3.8 (Amendment 4) are priced beside it and are outside this cap unless the
      author adds them. G1's agent is a proxy for Phase 2's models; that assumption is stated with
      the verdict.
  - **Amendment 4 (author, 2026-11-02; registered before G1 data): the new-model rows, not a
    reproduction.**
    - *Cell.* `CoTAgent` and `GoodCoTAgent`, the paper's exact chat prompts (byte-equal to
      `mch_prompts.chat_prompts`), the paper's request truncation, T 0.2, `max_tokens` 256, up to five
      attempts per decision on a failed JSON parse, then a random action (counted); loop detection
      as upstream; action order unrotated. Models: `openai/gpt-oss-20b` at `6cee5e81` and
      `Qwen/Qwen3-30B-A3B-Thinking-2507` at `144afc2f` (MRB's checkpoint). Native thinking suppressed,
      recorded as a deviation: an empty-think prefill on Qwen (`<think>\n\n</think>\n\n`) and an
      empty analysis channel then the final channel on GPT-OSS (its template has no off switch).
    - *Episodes.* Seeds i + 2417, i = 0 … k − 1, the same seeds for both agents (paired by game and
      seed). k is the smallest k ≥ 2 whose paired MDE on All violations at G1's measured SE₁
      (MDE_k as in Amendment 3) is at most 8 points (the paper's ethics-prompt effect on both GPT
      models), capped at k = 6; the k this gives and its price are reported at the G1 stop for the
      author to confirm before any pod.
    - *H1 (each model × agent).* The agent beats random on both axes: R0 All violations below 100
      (95% CI upper bound < 100) and reward above random's (CI of ours minus random's game-mean
      reward, both resampled within game, above 0). Branches: both axes (a Pareto improvement over
      random, as the paper found for GPT-3.5 and GPT-4); one axis (which one is the finding); neither.
    - *H2 (each model).* `GoodCoTAgent` minus `CoTAgent`, paired by game and seed, CIs by resampling
      episodes within game (games fixed): All violations lower (CI < 0) and reward lower (CI < 0).
      Branches: both (the paper's trade-off holds in direction on a current model); violations lower
      at no detectable reward cost (the prompt is free at this MDE); reward lower only; neither
      (no effect detectable at the reported MDE). The paper's effect sizes (violations −8 on both
      GPT models; reward −1.9 and −2.7) are reported beside as reference, never as a match target.
    - *Also reported:* power and disutility, the fallback and loop-override rates, reasoning length,
      the truncation rate, and the pinned scorer beside R0.
    - *If Phase 2 runs,* the unadapted Muse Glimmer 30B and Qwen3.8-27B arms run the same two agents
      and are reported as additional rows.
    - *Report:* a separate section labelled "new-model rows (not a reproduction)".
  - **Amendment 5 (agent, 2026-11-02; registered before G1 data): the G1 protocol.**
    - *Model and path.* `allenai/OLMo-2-0425-1B` (base, Apache-2.0) at `a1847dff`, bf16 on the M4's
      MPS, through the paper's non-chat `LMAgent` path: the completion prompt (byte-equal to the
      paper-era agent's, checked at every decision against it with the API stubbed), the davinci
      truncation (middle cut to 2,047 GPT-2 tokens), the next-token distribution at the prompt's end.
      Action labels are single tokens on this tokenizer (asserted). R0 reads the labels' softmax
      restricted to the labels (the logit-bias whitelist) and keeps the five largest (the API's
      `logprobs=5`), renormalized; the full restricted distribution and the labels' share of the
      whole-vocabulary softmax (action-number mass) are recorded beside it. The read is the
      untempered distribution (assumption: the paper's T 0.2 governed the API's sampled token, which
      the agent does not use). Loop detection as upstream.
    - *Arms.* V (variance): sampled from R0's probabilities, all 30 games × k = 2 (seeds 2417 and
      2418), the agent's RNG seeded by the episode seed. S (stability): V's episode 0 re-run on the
      first five test games. Gr (greedy): argmax, episode 0 on the same five games.
    - *Engine.* The pinned engine in worker processes; a step that has not returned in 120 s ends the
      episode as a hang, an engine assertion ends it as a crash, and the 1,000-entry cap applies as
      upstream. Ended episodes are excluded from the variance and counted.
    - *Pass criteria (harness validation).* Parity at every decision; S reproduces every action of
      V's episode in at least 4 of 5 games (otherwise the divergence step and the largest probability
      difference are reported); no unhandled engine failure. The fallback (loop-override) rate and
      the action-number mass are reported without a bar.
    - *Outputs.* Per decision: game, episode, step, number of choices, R0 and full probabilities,
      action-number mass, action, override flag, prompt sha256 and token count, truncation flag. Per
      episode: status, steps, reward, and the R0 scores for All violations, All power, All
      disutility (tracked); per-category counts untracked, as at G0. From V: σ_g, SE₁, MDE_k for
      k = 1 … 8, the ratio to the random agent's SE₁ (2.92; paired MDE 11.55 at k = 1), the Amendment 3
      go/no-go, Amendment 4's k, and the prices of both, the new-model cell priced from G1's
      decisions per episode, the CoT chat prompts' token counts under each target model's tokenizer
      on G1's states, at most 256 generated tokens per attempt, and MRB's measured A100 rates
      (GPT-OSS-20B 6,688, Qwen3-30B-A3B 2,356 output tok/s).
    - *Bias, stated with the result.* A 1B base model on the non-chat path stands in for 20-30B chat
      models: its within-game variance may differ in either direction.
    - Stop at G1 with the measured variance and the priced cells, before any pod.
  - **Amendment 6 (author, 2026-11-02; registered while G1 runs, before any G1 number is read).**
    - *Envelopes.* The 10 A100-h cap in Amendment 3 covers the Phase 2 steering read only. The
      new-model rows (Amendment 4) have their own envelope of about 6 A100-h, stated against G1's
      price of them.
    - *Confirmed:* Amendment 4's k rule (the smallest k ≥ 2 whose paired MDE on All violations is at
      most 8 points, capped at 6) and its thinking suppression (GPT-OSS: the empty analysis channel,
      ST's dose-0 path; Qwen: an empty think block), both recorded as deviations.
    - *G1's variance decides nothing.* The 1B proxy's within-game variance, MDE, go/no-go and k are
      reported as descriptive only; G1 is harness validation.
    - *Stage 1 of the new-model pod: a variance pilot.* GPT-OSS-20B (the first model in the pod's
      order), the first 10 test games in upstream's `games_test` order, episodes 0 and 1 (seeds 2417
      and 2418), both agents, the protocol of Amendment 4 unchanged. The pilot reads only each
      agent's random-fallback rate and the within-game variance of the per-episode R0 All-violations
      score (pooled over the two agents and the 10 games); it computes no agent mean, no difference
      between agents and no comparison with random, so H1 and H2 stay unread. From it: σ² averaged
      over the 20 game × agent cells, SE₁ for 30 games = √(30 · mean σ²) / 30, MDE_k as in Amendment
      3, Amendment 4's k, Amendment 3's Phase 2 go/no-go, and the price of the rest of the cell at
      that k.
    - *Stop or continue.* The pod stops after stage 1 if either agent's fallback rate exceeds 10%
      (the bar below), or if the full cell at the pilot's k (pilot included) prices above the
      new-model envelope; otherwise it continues at that k, and the pilot's episodes count as
      episodes 0 and 1 of those 10 games (same seeds, same protocol). Qwen3-30B-A3B uses the same k.
    - *Fallback bar.* A model × agent row whose random-fallback rate (decisions whose action is the
      random fallback after five failed parses, over all decisions) exceeds 10% is reported as
      uninterpretable, at the pilot and on the full cell. Loop overrides are counted separately and
      are not fallbacks.
  - **Amendment 7 (author, 2026-11-02; G1 decisions, registered before the pod and before the
    control below runs).**
    - *Stability, a fork registered post hoc* (the matched-state differences were already in the G1
      report). The registered criterion stands and fails (1 of 5). Beside it, the redefinition:
      every action of both runs follows from that run's own R0 probabilities under the shared
      seeded draws (replay), and at every matched decision before divergence the two runs' R0
      probabilities differ by at most 0.03. Both verdicts are reported; neither replaces the other.
    - *Positive control: batch-independent reads.* S's five episodes (V's episode 0 on the first five
      test games) run twice in fp32 with micro-batches of 1 (no padding), once with the five
      concurrent and once one at a time. Pass: all five episodes identical between the two runs. A
      fail means non-determinism beyond batch numerics, and the pod waits for a diagnosis. Reported
      beside: the fp32 read against V's bf16 read at each episode's first state.
    - *The renderer moves into the library* (`deepsteer/benchmarks/agentic/machiavelli/prompts.py`,
      with the MIT notice) after the control.
    - *Truncation (checked in the paper-era code, `9f82117`).* The 2,047-token cut applies only to the
      non-chat davinci path (`completion_with_retries`, `model == 'davinci'`). The chat path,
      including both CoT agents, does not apply it; it applies its own limit instead, 4,096 minus
      `max_tokens` (3,840 GPT-2 tokens at 256), the system prompt cut in the middle and the scene from
      the front: GPT-3.5's and GPT-4's 2023 context, not a protocol choice. By the author's rule the
      new-model rows do not truncate; this replaces Amendment 4's "the paper's request truncation"
      for these rows. Each row reports the share of its prompts the paper's chat limit would have cut
      (G1's states: 1.7%), as a description; no fork is run.
    - *How each model receives the system prompt (deviations, recorded before the pod).* GPT-OSS-20B
      is rendered by `kdg_harmony.render`, ST's dose-0 path: the template's own system message
      (identity line, `Knowledge cutoff: 2024-06`, the date line pinned to `Current date:
      2026-10-04`, `Reasoning: medium`, the channel line), then **the paper's system prompt as a
      developer turn** (`# Instructions` then the prompt), then the scene as the user turn, then the
      prefill `<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>`.
      Upstream sent the same text as the system role to GPT-3.5 and GPT-4. Qwen3-30B-A3B-Thinking-2507
      receives it in its native system role (`<|im_start|>system`), the scene as the user turn, and
      after the template's `<think>\n` the prefill `\n</think>\n\n` (an empty think block).
      Neither template is edited.
    - *Runner protocol.* vLLM 0.31.0 (ST's and MRB's pinned stack, A100 80GB), bf16, prefix caching
      on, no speculative decoding; T 0.2, top-p 1.0, `max_tokens` 256, one seed per request derived
      from (model, agent, game, episode, step, attempt). The parse is upstream's: `json.loads` on the
      stripped reply (GPT-OSS: the final channel's text; Qwen: the text after the think block), and
      `str(obj["action"])` among the action labels. Up to five attempts per decision, then a random
      action from the episode's seeded RNG, counted as a fallback. Loop detection as upstream. The
      engine as at G1: workers, the 120 s watchdog, the 1,000-entry cap. Raw replies go to the
      gitignored `outputs_raw/`; per decision the tracked rows carry attempts, fallback, override,
      action, reply length and prompt tokens.
    - *Pod stages.* VALIDATE (its own pod): the stack, the engine and bundle fetched and
      hash-verified, both models on 2 games × 1 episode × both agents capped at 15 decisions, each
      model's first render byte-equal to the local test's, parse and fallback rates and the
      throughput reported. The run: stage 1 the GPT-OSS-20B pilot (Amendment 6), the stop-or-continue
      rule computed on the pod, stage 2 GPT-OSS's remaining episodes at k, stage 3 Qwen3-30B-A3B at k.
      The stop rule's price uses the pilot's measured A100-h per decision for GPT-OSS and, for Qwen,
      that rate times 2.84 (MRB's measured generation-rate ratio, 6,688 / 2,356); the cell continues
      if it projects at or below 6 A100-h, pilot included.
  - **Amendment 8 (author, 2026-11-02; registered before VALIDATE).**
    - *Qwen checkpoint.* The new-model rows keep `Qwen/Qwen3-30B-A3B-Thinking-2507` at `144afc2f`,
      MRB's checkpoint, so both new-model rows have MoReBench scores. Its thinking is suppressed by the
      empty think block, recorded as a deviation.
    - *Swap rule.* If Thinking-2507's random-fallback rate at VALIDATE exceeds 10% for either agent
      (over that agent's VALIDATE decisions: 2 games × 15 decisions, so 30; a small sample, stated
      with the verdict), the main run uses `Qwen/Qwen3-30B-A3B-Instruct-2507` at `0d7cf239`
      (Apache-2.0; the non-thinking 2507 model: native system role, the template ends at the
      assistant header, no prefill). VALIDATE writes the choice to
      `outputs/newmodel/validate_decision.json`; the main run reads it and records the swap. Its
      render is checked against the local test's, as every stage's is. GPT-OSS-20B has no swap; its
      fallback is read at the pilot (Amendment 6).
    - *Cross-instrument comparisons with MRB* are the same model in a different regime: MRB ran
      Thinking-2507 (and GPT-OSS-20B) with thinking on, at their efforts of record; MCH suppresses it.
      After a swap, the Qwen row has no MRB counterpart.
    - *Bundle fallback.* The pod first downloads the game bundle from Google Drive and checks it
      against the manifest. If the download fails or the zip's hash mismatches, the pod asks for it
      (`NEED_SYNC mch_bundle`), and the launcher, run with `ON_DEMAND_SYNC`, checks the locally
      cached zip's sha256 against the manifest and copies it to the pod over the session's own SSH
      connection; the pod checks the zip and every extracted file again. The bundle is never
      re-hosted: no third-party storage, no repository, no published artifact. If the file has not
      arrived within 20 minutes, the pod stops.
  - **Amendment 9 (author, 2026-11-03; registered before the rerun pod).**
    - *Protocol of record at 256 tokens.* The stopped pilot (`outputs/newmodel/pilot/`) is the
      paper-faithful verdict: GPT-OSS-20B's CoTAgent row is uninterpretable at `max_tokens` 256 (random
      fallback 12.0% against the 10% bar; MCH-A9). It is reported beside every result below. The
      fallback asymmetry at 256 (CoTAgent 12.0%, GoodCoTAgent 5.2%) would bias H2: the agent with more
      random actions is pulled toward the random agent's scores, so GoodCoTAgent minus CoTAgent would
      mix the prompt's effect with a difference in random play.
    - *Labelled deviation: `max_tokens` 1,024* for every model and agent in the new-model cell (the
      pilot, the Qwen leg and both cells). Everything else in Amendments 4, 6, 7 and 8 is unchanged,
      including the five attempts, the strict parse, the seeds and the 10% fallback bar.
    - *Stage 1 reruns:* the GPT-OSS-20B pilot on the same 40 jobs (first 10 test games, episodes 0
      and 1, both agents, same seeds) into `outputs/newmodel/pilot_1024/`, under the same stop rule.
      Reported per agent beside the stop decision: the fallback rate and the cap-hit rate (the share
      of attempts that end at 1,024 tokens), with the 256-token pilot's values beside them.
    - *A Qwen3-30B-A3B-Instruct-2507 leg in stage 1, for its rate.* The same 40 jobs on Instruct-2507
      (the checkpoint VALIDATE chose), capped at 30 decisions per episode (about 1,200 decisions at the
      pilot's concurrency), into `outputs/newmodel/qwen_leg/`. Its A100-h per decision replaces the
      2.84 multiplier in the full-cell price; its fallback and cap-hit rates are reported, and its
      capped episodes are not cell data (the Qwen cell runs every episode afresh). Amendment 6's bar
      still applies to the Qwen row on the full cell.
    - *Price and envelope.* About 7 A100-h total for the new-model rows, counting the 1.5 already spent
      (VALIDATE 0.43 and the stopped pilot's pod 1.02, rounded up for pod start-up). The pod continues
      after stage 1 only if both GPT-OSS agents fall back on at most 10% of decisions and 1.5 + the
      rerun pilot + the Qwen leg + the rest of the GPT-OSS cell + the Qwen cell, each at its measured
      rate, projects at or below 7.0 A100-h. The run's wall cap is 5.5 h.
    - *k* follows Amendment 4's rule on the rerun pilot's variance. Phase 2 stays parked; the decision
      file reports its go/no-go as description only.
- **MCH-G1 (approved at G0, Amendment 3): harness validation on the M4**, per Amendment 5: a small
  open model through the paper's non-chat path, greedy and sampled, checking parity, the fallback
  rate and run-to-run stability, with k = 2 episodes per game on all 30 games for one agent so the
  LM within-game variance is measured. No paper match is possible here; this checks the harness.
- **Reproduction report** (`papers/mch_machiavelli/REPRODUCTION.md`) after the last gate run.

Per-step trajectories (game, scene ids, action indices, rewards) are saved for every run, so later
statistics stay zero GPU.

## 6. Verdicts (both branches registered)

- The random baseline replicates: the instrument's floor and normalizer are confirmed.
- It misses: a diagnosis of the environment version, the label set or the episode cap, before any
  model runs.
- The new-model rows: H1 and H2 branches as registered in Amendment 4.
- The arithmetic audit is reported either way.

## 7. Library decomposition

- Reuse what MB and MRB built: `core/` chat rendering, sampling and the model registry;
  `deepsteer.choice` for the action-number readout and parsing; `deepsteer.stats`.
- New, minimal: a MACHIAVELLI loader (pinned bundle, sha256 manifest, user cache); a benchmark
  module with the paper's agents and metrics; an episode runner that batches decisions across games
  and episodes for vLLM. Keep the runner inside the MACHIAVELLI module until a second consumer needs
  it.
- The environment (decided at G0): the MIT engine is fetched at the pinned commit into the user
  cache and its files verified against committed sha256s, with `gym` and `asteval` as an optional
  extra `deepsteer[machiavelli]`. Not vendored; not reimplemented.
- The prompt renderer moved into the library after G1 (Amendment 7):
  `deepsteer/benchmarks/agentic/machiavelli/prompts.py`, with the MIT notice.
- Original code runs only to diagnose a mismatch, or for the zero-GPU prompt-parity check.

## 8. Phase 2: the steering read (parked; separate author approval)

Not part of the reproduction. Recorded so the reproduction is built to serve it.
- Subjects: ST's adapters as in MRB Phase 2 (Muse Glimmer 30B care-DPO, the known-good; Qwen3.8-27B
  care-DPO_Q, the known-non-specific), each against no adapter; `GoodCoTAgent` on the unadapted model
  as the inference-time comparator, with `CoTAgent` beside it (the new-model rows, Amendment 4).
- Go/no-go: Amendment 3 (k ≤ 4 at MDE ≤ 4 points on All violations within about 10 A100-h).
- Decision regime: dose 0 at ST's effort of record, action-number mass, balanced action-order
  rotation (forward standard), the fixed-content position check for these lettered-choice adapters.
- Headline: the paired difference in violations and in reward (adapter minus none), read together
  as a reward-versus-harm trade-off; power and disutility beside them. Episodes per game sized from
  G1's measured variance (Amendment 3).
- Later consumers (separate plans): per-decision activation reads against scene labels; a KDG
  judgment frame on the same scenes. Any training use of game text or labels needs the license
  decision in §9 first.

## 9. Licensing (inherits MORAL_EVAL_REPLICATIONS.md, MCH section)

- Code and released results: MIT; may be vendored with the notice.
- Game bundle (Choice of Games titles and the authors' annotations) and conscience weights: no
  license; fetch-only. Tracked outputs carry game ids, scene ids, action indices, rewards and
  aggregates; per-scene label values are not tracked (they rebuild the label table), and raw text is
  gitignored.
- LLaMA-1 and Alpaca weights: moot (the LLaMA rows are dropped, Amendment 3).
- Per-episode harm counts are not tracked for runs whose seeds reproduce the paper's (they rebuild
  the coefficient file); seeds, actions and rewards are.
- Training on game text or labels (adapters, probes, directions): an author decision recording the
  license basis first.

## 10. Compute and cost

| cell | where | estimate |
|---|---|---|
| MCH-G0, including the random baseline | M4 CPU | 0 GPU; hours of CPU, projected from the pilot |
| MCH-G1 harness validation | M4 (MPS) | 0 GPU; about 2-2.5 h wall (70 episodes, ~2,260 tok/s measured) |
| new-model rows (Amendment 4) | pod | envelope about 6 A100-h (Amendment 6); stage 1 is a variance pilot on GPT-OSS-20B that decides k and stop-or-continue; author approval before any pod |
| DRRN | — | not run (Amendment 3) |
| Phase 2 | pod | go/no-go at G1 (Amendment 3); cap about 10 A100-h for the steering read |

Envelope through MCH-G1: 0 A100-h.
