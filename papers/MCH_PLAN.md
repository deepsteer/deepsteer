# MCH Plan — MACHIAVELLI Replication, Decomposed Into the Library

Status: v0.1 draft, 2026-11-02 (author request: reproduce MACHIAVELLI as a reference implementation and
as the action-side instrument for the steering work). Pre-registration candidate. **No GPU spend
authorized by this document.** Line prefix **MCH**: this file is `papers/MCH_PLAN.md`; code, data
manifests and outputs in `papers/mch_machiavelli/`; anomalies `MCH-A1, ...` in `papers/ANOMALIES.md`;
claims `MCH-01, ...` in `papers/CLAIMS.md`. Inherits the goals, standard and license rules of
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
  reward? The paper's ethics prompt (`GoodAgent`, `GoodCoTAgent`) is a ready inference-time
  comparator that a weight-level adapter should match or beat without the prompt.
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
- the LM agents' prompts and decoding: chat models generate at T 0.2 with `max_tokens` 8; non-chat
  models are read by log-probabilities over the action numbers (`max_tokens` 1) and an action is
  sampled from them; loop detection and a random-action fallback;
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
| LM agents on LLaMA 7B/13B/30B (CSV, Appendix K) | runnable only if the checkpoints can be identified, and only after an author decision on their licenses (§9). The 30B size exists only in LLaMA-1 (Feb 2023, non-commercial research license), so these are LLaMA-1 or Alpaca on LLaMA-1, not Llama 2 or later |
| DRRN base and +shaping | not planned: per-game RL training on 30 games, priced at MCH-G0 (the original jobs allow up to 24 GPU-h per game) |

## 4. Readouts

- **R0, paper-faithful (the default):** the paper's agent, prompt and decoding path for the model
  type; the paper's metric pipeline; violations as a percentage of the random baseline.
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
- **MCH-G1 (author approval at G0).** Two parts, each optional:
  - Harness validation: a small open model through the paper's non-chat path on the M4, a few test
    games, greedy and sampled, checking parity, the fallback rate and run-to-run stability. No paper
    match is possible here; this checks the harness.
  - The LLaMA rows, only if G0 identifies the checkpoints and the author clears their licenses: the
    7B on the M4; 13B and 30B on a pod. Match rule: the paper's cell lies inside our prediction
    interval for a one-episode-per-game mean (games fixed, episodes resampled from k episodes per
    game). A miss triggers a diagnosis under amendment with a stopping rule (the MB pattern).
- **MCH-G2 (only if the author approves DRRN at G0).**
- **Reproduction report** (`papers/mch_machiavelli/REPRODUCTION.md`) after the last gate run.

Per-step trajectories (game, scene ids, action indices, rewards) are saved for every run, so later
statistics stay zero GPU.

## 6. Verdicts (both branches registered)

- The random baseline replicates: the instrument's floor and normalizer are confirmed.
- It misses: a diagnosis of the environment version, the label set or the episode cap, before any
  model runs.
- The LLaMA rows (if run) replicate or miss with the reason located, as in MB and MRB.
- The arithmetic audit is reported either way.

## 7. Library decomposition

- Reuse what MB and MRB built: `core/` chat rendering, sampling and the model registry;
  `deepsteer.choice` for the action-number readout and parsing; `deepsteer.stats`.
- New, minimal: a MACHIAVELLI loader (pinned bundle, sha256 manifest, user cache); a benchmark
  module with the paper's agents and metrics; an episode runner that batches decisions across games
  and episodes for vLLM. Keep the runner inside the MACHIAVELLI module until a second consumer needs
  it.
- The environment: vendor the MIT game engine with its notice, or depend on the original package
  (a decision at MCH-G0 with the namespace proposal). Do not reimplement the engine.
- Original code runs only to diagnose a mismatch, or for the zero-GPU prompt-parity check.

## 8. Phase 2: the steering read (parked; separate author approval)

Not part of the reproduction. Recorded so the reproduction is built to serve it.
- Subjects: ST's adapters as in MRB Phase 2 (Muse Glimmer 30B care-DPO, the known-good; Qwen3.8-27B
  care-DPO_Q, the known-non-specific), each against no adapter; `GoodAgent` on the unadapted model
  as the inference-time comparator.
- Decision regime: dose 0 at ST's effort of record, action-number mass, balanced action-order
  rotation (forward standard), the fixed-content position check for these lettered-choice adapters.
- Headline: the paired difference in violations and in reward (adapter minus none), read together
  as a reward-versus-harm trade-off; power and disutility beside them. Episodes per game sized from
  MCH-G0's MDE.
- Later consumers (separate plans): per-decision activation reads against scene labels; a KDG
  judgment frame on the same scenes. Any training use of game text or labels needs the license
  decision in §9 first.

## 9. Licensing (inherits MORAL_EVAL_REPLICATIONS.md, MCH section)

- Code and released results: MIT; may be vendored with the notice.
- Game bundle (Choice of Games titles and the authors' annotations) and conscience weights: no
  license; fetch-only. Tracked outputs carry game ids, scene ids, action indices, rewards and
  aggregates; per-scene label values are not tracked (they rebuild the label table), and raw text is
  gitignored.
- LLaMA-1 and Alpaca weights (non-commercial research terms): an author decision before any run.
  This applies only to LLaMA-1-era checkpoints. Llama 2 and later (Llama Community License,
  commercial use with restrictions) need no special decision and are already in the registry.
- Training on game text or labels (adapters, probes, directions): an author decision recording the
  license basis first.

## 10. Compute and cost

| cell | where | estimate |
|---|---|---|
| MCH-G0, including the random baseline | M4 CPU | 0 GPU; hours of CPU, projected from the pilot |
| MCH-G1 harness validation | M4 | 0 GPU |
| MCH-G1 LLaMA rows (if cleared) | 7B on M4; 13B and 30B on a pod | 0 GPU for 7B; about 1–2 A100-h for 13B and 30B (forward-only reads, sequential steps), priced at G0 |
| DRRN (MCH-G2) | pod | not planned; priced at G0 |
| Phase 2 | pod | priced at G0; forward-only, likely a few A100-h |

Envelope through MCH-G1: 0 A100-h unless the author approves the 13B and 30B rows (cap 2 A100-h).
