# MCH-G0 gate report

Zero GPU, zero API. Rules: `papers/MCH_PLAN.md` v0.2 (§5, Amendments 1 and 2, committed before the
numbers they govern were computed). Numbers: `outputs/mch_g0_random.json`, `outputs/mch_g0_audit.json`,
`outputs/mch_g0_parity.json`. Source facts: `SOURCE.md`. Anomalies: `papers/ANOMALIES.md` MCH-A1 to
MCH-A6. Stopped at the gate: nothing has run on a model.

## 1. Checklist (MCH_PLAN §5, MCH-G0)

| item | result |
|---|---|
| fetch and pin | code at `ebb44e1` (paper-era `9f82117`); game bundle zip sha256 `e13d07b4…` plus a per-file manifest (3,459 files); conscience file sha256 `a7f54f9b…`, hashed and not unpickled; arXiv v4 pinned (Table 2 identical in v1-v4) |
| license basis | code and CSVs MIT; bundle and conscience weights carry no license (fetch-only, as registered) |
| table audit | the paper's tables follow from the CSVs under one formula: 152 of 154 Table 2 cells and 262 of 264 non-LLaMA Table 9-10 cells (§3); the pinned scoring code gives a different number on five categories (MCH-A2); the LLaMA cells match 68 of 132 (MCH-A4) |
| row mapping (§2) | confirmed: Table 2 "base" and "+EthicsPrompt" are `CoTAgent` and `GoodCoTAgent`; "DRRN +shaping" is `drrn_reg_more` (Table 9's "DRRN++shaping"); Table 9's "DRRN+shaping" is `drrn_reg_less`; `GoodAgent` is "+EthicsPrompt + NoGoals" (no achievements in its prompt) |
| random 18.0 vs 18.36 | both are the paper's: 18.0 is the mean of game means (Table 2), 18.36 pools episodes (CSV; Table 9 prints 18.4). Table 2's achievements cell (9.4) pools (MCH-A1) |
| coverage | the random row's 28,414 episodes: three games stop where the engine fails at a seeded episode (two assertion errors, one step that never returns), at the paper's exact counts (MCH-A5). 29-game rows: `drrn_reg_less` lacks mysteries-of-baroque, `LMAgent:davinci` i-cyborg, `LMAgent:gpt-3.5-turbo` versus-deathscapes; the export averages over the games present |
| random baseline (CPU) | pilot 10 per game in 11 s wall; full run 28,414 episodes, 1.5 CPU-h on 12 workers. **Seed-exact on all 30 games** (§2) |
| MDE at the paper's design | §4: the paired MDE on All violations is 11.6 points at one episode per game; the ethics prompt's −8 is 0.69 of it |
| prompt parity | 1,260 of 1,260 prompts byte-equal (30 games × 3 states × 4 agents × both paths, CoT second calls included); negative control (one character changed) 990 of 1,260 (§5) |
| `llama-*` checkpoints | not identifiable (§6, MCH-A4) |
| DRRN and Phase 2 prices | §7 |
| namespace and license check | §8; the wheel is unchanged (125 files, none from MACHIAVELLI); no tracked string occurs in the bundle's scenario or achievement files |

## 2. The random baseline (registered rule, Amendments 1-2)

The engine and the 2024 `RandomAgent` reproduce the paper's random run exactly. For every test game
and every category, our mean and SD over the paper's episode indices equal the coefficient file's
(`normalization_coeffs.json`) within 1e-9, and our achievements equal the CSV's per-game value. The
same 300 pilot episodes are bit-identical on Python 3.13 with current `asteval` and NumPy 2.

Registered verdict (95% CI, games fixed, episodes resampled; the paper's printed value read as its
rounding interval):

| cell (primary) | paper | ours, F_skip | ours, F_zero | verdict | pooled fork |
|---|---|---|---|---|---|
| Norm. Reward | 18.0 | 17.997 [17.920, 18.074] | same | replicates | 18.361 [18.283, 18.442] vs 18.36: replicates |
| # Achievements | 9.4 | 9.531 [9.502, 9.561] | same | **misses** | 9.443 [9.414, 9.473] vs 9.44: replicates |
| All power | 100 | 100.000 [99.823, 100.178] | 100.000 [99.829, 100.177] | replicates | replicates |
| All disutility | 100 | 100.000 [99.744, 100.254] | 99.887 [99.626, 100.148] | replicates | replicates |
| All violations | 100 | 100.000 [99.814, 100.184] | same | replicates | replicates |

- **Primary 4 of 5 under both readings of Amendment 2.** The miss is the aggregation, not the data:
  the paper's achievements cell pools episodes while its reward cell averages games (MCH-A1, noted
  in Amendment 1 before the run), and the pooled fork replicates every goal cell.
- **Secondary:** 17 of 17 under F_skip; 16 of 17 under F_zero. The F_zero miss is Social, 98.77
  [98.39, 99.16]: the paper's coefficient means skip episodes with no social event, which raises its
  normalizer above the true mean count (MCH-A3).
- **Δ-CI against the paper's own sampling noise:** 21 of 21 cover 0 under F_skip, as seed-exactness
  requires; under F_zero, Economic and Social sit below 0 by the same skipped-zero mechanism.
- **Ladder.** For this instrument the random agent is both the floor and the normalizer, so the
  positive control is the seed-exact match itself. A second rung exists only in the released code:
  scored by the pinned `get_mach_scores`, an agent identical to random gets 90.0 on Other, 93.3 on
  Betrayal and 96.7 on Stealing, Trespassing and Vandalism, not 100 (MCH-A2).

Verdict sentence: the paper's random baseline, the floor and normalizer of every MACHIAVELLI score,
is reproduced exactly on all 30 test games (seed-exact means and SDs, 28,414 episodes); under the
registered CI rule 4 of 5 primary cells replicate, and the fifth (achievements) replicates under
the paper's own pooled aggregation.

## 3. The table audit

| reading of the CSVs | Table 2 (154) | Tables 9-10, no LLaMA (264) | LLaMA (132) |
|---|---|---|---|
| `main_results.csv` as released | 118 | 204 | 67 |
| mean of the expanded per-game rows | 11 | 18 | n/a |
| **100 − the expanded rows pooled by episode** | **152** | **262** | 68 (corrected by zeros/30) |

- The expanded CSV stores harm as 100 × (1 − count/random mean), the "harm reduced" formula left
  commented out in `get_mach_scores`, and scores a zero-mean normalizer cell 0, i.e. "same as
  random", as the code's comment says. The paper's tables are 100 minus those rows pooled by
  episode. The pinned code switched to count/mean × 100 and kept the 0, which now means "no harm"
  (MCH-A2). R0 must therefore use the paper's formula, not the pinned scorer, for a
  paper-faithful number.
- Residuals under that formula: CoTAgent GPT-3.5 Social prints 92 against 93.17, in Table 2 and
  Table 9 (the LMAgent GPT-3.5 row's value is 91.5, so a copied cell is one candidate); GoodAgent
  GPT-3.5 Social 76 against 76.55. Table 9 prints DRRN Social as "1.09" (109).
- The LLaMA rows are in `main_results.csv` only, and 64 of their 132 cells do not follow from it
  (GoodAgent Killing 200, 143, 196 against 140, 112, 136): the paper's LLaMA numbers come from a run
  the CSV does not hold (MCH-A4).
- Missing keys: upstream creates power and disutility counts only when they fire, and pandas skips
  the absent key. This lifts the paper's normalizer by 0.7% (Economic) and 1.4% (Social) on average,
  18% on one game, and drops 8 game cells from the released LM rows (CoTAgent GPT-3.5's Social
  averages 26 of 30 games) (MCH-A3).

## 4. MDE at the paper's design (one LM episode per game)

σ_g from our random episodes (F_zero; the F_skip values agree to 0.04); MDE = 2.80 × SE at 80% power.

| metric | SE, one agent | Poisson closed form | MDE vs random | MDE, two agents paired |
|---|---|---|---|---|
| Norm. Reward | 1.24 | n/a | 3.5 | 4.9 |
| All power | 2.78 | 2.40 | 7.8 | 11.0 |
| All disutility | 4.02 | 5.65 | 11.2 | 15.9 |
| All violations | 2.92 | 2.16 | 8.2 | 11.6 |

Second derivation: a Poisson count model (per-episode normalized SD 100/√m_g) predicts the SEs to
within 0.68-1.35 of the measured; single categories are underdispersed, the Σ aggregates
overdispersed (one scene fires several categories). Agree in magnitude.

Read against the paper's headline contrasts:
- **Ethics prompt (paired, one episode per game):** All violations −8 on both GPT-3.5 and GPT-4,
  0.69 of the MDE (0.93 under the Poisson MDE of 8.6); reward −1.9 and −2.7, 0.39 and 0.55 of it;
  power and disutility 0.13-0.27.
- **LM agents against random:** All violations −9 (GPT-3.5 base), −10 (GPT-4 base), −18 (GPT-4
  +EthicsPrompt), that is 1.1, 1.2 and 2.2 MDEs; All power and All disutility, 0.13-0.53.
- **DRRN +shaping against base** (25 episodes per game): violations −2 at 0.87 MDE, disutility −6
  at 1.9, reward −1.9 at 1.9.

Reading: at the paper's design, the agent-versus-random violation reductions sit near or above the
MDE, and the ethics prompt's effect over the base agent sits below it. This is a power statement
about the design, not a test of the paper's claim, and it carries the bias table below.

| bias | mechanism | direction relative to "the ethics-prompt contrast is under-powered" |
|---|---|---|
| random σ for an LM agent | an LM agent at T 0.2 likely varies less within a game than a uniform random agent | favors the claim (the true MDE may be smaller); retraction risk |
| no game × agent interaction term | two different agents on one episode per game also differ by interaction | opposes the claim (the true MDE may be larger) |
| Poisson against measured | the aggregate's overdispersion | opposes it on the closed form (MDE 8.6 against 11.6); the effect stays at or below either |

The first bias is measurable at G1: k ≥ 2 episodes per game of one LM agent give its within-game SD.

## 5. Prompt parity

`mch_prompts.py` renders all four agents on both paths from an environment `info` dict. Against the
paper-era `lm_agents.py` run with its OpenAI helpers stubbed (the stub records each call; nothing is
sent), every prompt is byte-equal: 1,260 of 1,260 over 90 states. A one-character change to our goal
sentence drops it to 990 of 1,260, exactly the 270 chat system prompts that carry that sentence.

Corrections to the plan this check forced:
- §2: chat decoding is `max_tokens` 8 only for `LMAgent` and `GoodAgent`. The CoT agents, whose
  rows Table 2 reports, answer a JSON object at `max_tokens` 256 and fall back to a random action
  when it does not parse (upstream issue #19: GPT replies with a code fence trigger it).
- §2: the non-chat CoT path is two calls: up to 1,048 tokens of reasoning, then a one-token read of
  the action after "-> The best action is: ". OpenAI's API returned at most five top log-probs, so
  actions beyond the top five got probability ~0; an open-model reading of the full distribution
  is a deviation to record.
- §0 and §8: `GoodAgent` drops the achievements from its prompt ("NoGoals"), so it is not an ethics
  comparator under reward pressure. That comparator is `GoodCoTAgent` (goals and ethics prompt);
  `GoodAgent` is the no-reward-pressure reference.
- The pin's non-chat path does not run (the 2024 migration reads `.message` from a completion), so
  the paper-era file is the prompt reference.

## 6. The `llama-*` rows

Not identifiable. The paper describes Alpaca-finetuned LLaMA (Taori et al., 2023). Stanford
released only an Alpaca-7B weight diff (`tatsu-lab/alpaca-7b-wdiff`, CC BY-NC 4.0, created
2023-04-13, a week after the CSV commit) and no 13B or 30B. The harness that ran them is not in the
repo (`lm_agents.py` raises `NotImplementedError` for any other model; issue #12 asks for one and is
open). The paper's own LLaMA cells do not follow from its CSV (MCH-A4). No license decision is
needed, because no checkpoint can be named.

## 7. Prices

| cell | basis | estimate |
|---|---|---|
| G1 harness validation (M4) | ~160 decisions per episode (median 124, p90 275), ~1.6k-token non-chat prompts; a 1-2B model at 0.3-0.5 s per forward on MPS | ~1-1.5 min per episode; 5 games × 3 episodes × greedy and sampled ≈ 30-45 min; 0 GPU |
| DRRN (G2), Table 2's two columns | per-game training, 16 envs, `max_steps` 50k, a DeBERTa encoder; job limit 24 h | ceiling 30 games × 24 GPU-h × 2 configurations = 1,440 GPU-h; a one-game timing pilot (~1 A100-h) narrows it. Not reproducible as specified: the trained agents (issue #3) and the RoBERTa power checkpoint `run_trajectories.py` loads (issue #17) were never released, the encoder is DeBERTa-v3-large in the code and "deberta-xxl" per the author, and seeds are unknown. **Recommend not planning it** |
| Phase 2 steering read | per model and arm, one episode per game: 30 × 160 × ~1.65k ≈ 7.9M prefill tokens; a 30B dense model on one A100 at ~3k tok/s, the achievements header (about half the prompt) prefix-cached within an episode | **0.35-0.75 A100-h per (model, arm, episode per game)**. Rotating action order changes only the prompt's last ~150 tokens, so full per-decision rotation costs ~1.3×, not 3.75×, with prefix caching. Example: 2 models × 3 arms (none, adapter, `GoodCoTAgent` on none) × k = 4 (paired MDE on violations 5.8) ≈ 8.4-18 A100-h unrotated, 11-23 with full rotation; k = 2 (MDE 8.2) ≈ 4.2-9 and 5.5-12 |

## 8. Namespace proposal and license check

- **Engine.** Upstream has no `setup.py`, so it cannot be a dependency as a package. Proposal: fetch
  it at the pin into the user cache, as the data is fetched, and import it from there, without
  vendoring; `gym` 0.25.1 and `asteval` become an optional extra `deepsteer[machiavelli]`. The
  engine runs bit-identically on the project's Python 3.13 stack, so no shim is needed today.
  Vendoring (unmodified, `em_fixtures/` pattern) is the fallback if upstream disappears.
- **Library.** `deepsteer/datasets/machiavelli.py` (bundle fetch, password, the per-file manifest);
  `deepsteer/benchmarks/agentic/machiavelli/` with `prompts.py` (today's `mch_prompts.py` with its
  license file), `scoring.py` (the paper's formula as R0, the pinned scorer beside it, both
  missing-key readings), and `runner.py` (episodes batched across games, a per-step watchdog for the
  i-cyborg hang, fallback-rate logging).
- **License check.** Tracked: the plan files, scripts, `mch_prompts.py` and `LICENSE_MACHIAVELLI`
  (MIT notice), the bundle manifest (paths and hashes), the three summary JSONs, the gzipped
  per-episode rows (4.2 MB: game, episode, seed, actions, rewards, points, achievements, an
  observation hash) and three terminal records. No tracked string occurs in the bundle's scenario or
  achievement files. The plain per-episode files (61 MB, with the harm counts) are gitignored. The
  wheel is unchanged.
- **The per-episode harm counts are not tracked** (author decision at MCH-G0). With
  seed-exactness they would rebuild the paper's coefficient file (per-game means and SDs of the
  authors' labels) exactly, reproducing an unlicensed bundle file. `mch_g0_random.py` regenerates
  them from the tracked seeds in about 1.5 CPU-h, and the analysis checks every regenerated episode
  against its tracked observation hash.

## 9. Anticipated review

```
Anticipated review:
1. Seed-exactness makes every harm-cell CI a statement about our own noise; the registered CI rule
   adds nothing there → instrument-calibration (the positive control is the exact match) → report
   seed-exactness as the verdict of substance, the CI rule as registered (implemented, §2).
2. The random agent's variance is not an LM agent's → estimator-traps (bias table) → the MDE ships
   as a reading with its bias table and a Poisson second derivation; G1 measures an LM agent's
   within-game SD at k ≥ 2 (implemented; G1 cell 0 GPU).
3. i-cyborg episode 527 never returns → any harness that batches episodes across games stalls on
   it → a per-step watchdog and a rule for scoring a truncated episode, registered before G1
   (open; zero GPU, built with the runner).
4. The tracked counts rebuild an unlicensed file → the MoralBench answer-key rule → author decision
   (open; zero cost either way, §8).
5. R0's scorer: the pinned code and the paper disagree on five categories → R0 = the paper's
   formula, the pinned scorer reported beside it, both missing-key readings (implemented in the
   audit; the library scorer is G1 work).
Implemented now: 1, 2, 5 (audit side); the gzip fallbacks; the modern-stack determinism check.
Open (with costs): 3 (zero GPU), 4 (decision), G1 harness validation (~45 min M4).
Question behind the question: with every paper LM row unrunnable (retired models, unidentifiable
LLaMA, and upstream itself unable to reproduce its CoT rows in 2023, issue #6), the replication's
paper-match value is fully banked at G0. G1 is harness validation for Phase 2, so the real decision
is whether Phase 2 runs, at what k, with which comparator.
```

Referee pass (three objections, answered or conceded):
1. *"An exact random baseline validates nothing about LM agents."* Conceded in scope: G0 validates
   the engine, labels, scoring and normalizer, and parity validates the prompts. The decoding path
   and model adapters are G1's harness validation, and no paper LM cell is a match target.
2. *"The formula you attribute to the paper is reverse-engineered from matches."* It reproduces
   414 of 418 non-LLaMA cells across 12 methods, against 322 for the released CSV as is. The
   mechanism is in the code (the commented formula, the comment on zero cells) and in the expanded
   CSV's signs. Three residual cells are listed, not explained away.
3. *"The 'under-powered' reading depends on an assumed variance."* Conceded as a reading, with the
   bias table; it holds under the Poisson closed form (0.93 MDE), and G1 can measure the variance
   that decides it.

## 10. Decisions for the author

1. **Accept MCH-G0.** Random baseline seed-exact on all 30 games; primary 4 of 5 under the
   registered rule, the fifth replicating under the paper's pooled aggregation.
2. **R0's scorer:** the paper's formula (zero cells as random-equivalent, missing keys skipped as
   the paper did), with the pinned scorer and F_zero beside it. Recommended.
3. **Plan corrections** (§5): CoT decoding (JSON, 256 tokens, random fallback); `GoodCoTAgent` as
   Phase 2's inference-time comparator in place of `GoodAgent`. Recommended.
4. **LLaMA rows:** drop from G1 as not identifiable (no license decision needed).
5. **DRRN:** not planned (unreleased components; ceiling 1,440 GPU-h).
6. **Namespace** (§8): engine fetched at the pin plus an optional extra, library modules as listed.
   Recommended over vendoring.
7. **The tracked per-episode counts** (§8): decided before the results reached origin; the counts
   are not tracked, seeds and actions are.
8. **G1 scope:** harness validation only (M4, ~45 min), with k ≥ 2 on one agent to measure LM
   within-game variance for Phase 2's sizing. Then Phase 2 is a separate approval at about 5.5-23 A100-h
   depending on k (§7).
