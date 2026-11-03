# MACHIAVELLI reproduction report (MCH, 2026-11-03)

Paper: Pan, Chan, Zou, Li, Basart, Woodside, Ng, Zhang, Emmons, Hendrycks, "Do the Rewards Justify
the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the MACHIAVELLI Benchmark",
arXiv:2304.03279v4 (ICML 2023): Table 2 and Tables 9-10. Rules of record: `papers/MCH_PLAN.md`
(Amendments 1-9). Gate reports: `MCH_G0_REPORT.md`, `MCH_G1_REPORT.md`; new-model reports:
`NEWMODEL_VALIDATE.md`, `NEWMODEL_PILOT.md`, `NEWMODEL_ROWS.md`. Source facts: `SOURCE.md`. No game
text is in this report; the game bundle is fetched, never committed.

## 1. Verdict

**The instrument is reproduced exactly; no LM row of the paper can be reproduced.**

| part of the paper | verdict | evidence |
|---|---|---|
| the random baseline (the floor and normalizer of every score) | **reproduced exactly** | the pinned engine and the 2024 `RandomAgent` regenerate every per-game mean and SD of the paper's coefficient file on all 30 test games within 1e-9 (28,414 episodes), on Python 3.11 and on 3.13 with NumPy 2; under the registered CI rule 4 of 5 primary cells replicate, and the fifth (achievements) replicates under the paper's own pooled aggregation (MCH-A1) |
| where the paper's random run stopped | **reproduced** | three games end at seeded engine failures at the paper's exact counts: two assertions (hollywood-visionary 584, vampire parliament-of-knives 303) and one step that never returns (i-cyborg 527) (MCH-A5) |
| the scoring arithmetic of Tables 2, 9, 10 | **reproduced, with the formula located** | 100 minus the per-game "harm reduced" rows pooled by episode, zero-mean normalizer cells read as random-equivalent: 152 of 154 Table 2 cells and 262 of 264 non-LLaMA Table 9-10 cells; three residual Social cells listed in the G0 report |
| the LM agents' prompts | **reproduced byte for byte** | 1,260 of 1,260 prompts at G0 (90 states, four agents, both paths) and 11,669 of 11,669 at G1 against the paper-era `lm_agents.py` (`9f82117`) with its API stubbed; library test on every agent |
| the paper's LM rows (GPT-3.5, GPT-4, davinci, LLaMA) | **not reproducible** | §2 |
| DRRN rows | **not run** | §2 |

## 2. Rows that cannot be reproduced, and why

- **GPT-3.5-turbo, GPT-4 and davinci (Table 2's LM columns, Tables 9-10).** The 2023 models are
  retired from the API, and the pinned code's non-chat path no longer runs (a 2024 migration reads a
  chat field from completions).
- **Upstream could not reproduce its own CoT rows (MCH-A6).** In August 2023 an upstream author
  confirmed that re-running `CoTAgent:gpt-3.5-turbo` (and the pinned `-0301`) gave weaker results
  than the released CSV (issue #6, open); issue #19 adds a mechanism, replies in a JSON code fence
  failing the parse and becoming random actions. The paper's CoT rows were therefore not reproducible
  even before the models retired.
- **LLaMA 7B/13B/30B (Appendix K).** The checkpoints cannot be named: the paper describes
  Alpaca-finetuned LLaMA, Stanford released only an Alpaca-7B weight diff (after the CSV commit) and
  no 13B or 30B, and the open-model harness is not in the repo. 64 of the 132 LLaMA cells also do not
  follow from the released CSV (MCH-A4). Dropped (Amendment 3).
- **DRRN base and +shaping.** The trained agents and the RoBERTa power checkpoint that the evaluation
  path loads were never released, the encoder differs between paper, code and author, and the seeds
  are unknown; ceiling 1,440 GPU-h. Not run (Amendment 3).

## 3. Scoring: the released code and the paper disagree (MCH-A2)

The released `get_mach_scores` scores a test-game category whose random mean is 0 as 0 for every
agent, although its own comment says such cells are "same as random". Eight test-game cells are
affected, so an agent identical to random scores 90.0 on Other, 93.3 on Betrayal and 96.7 on
Stealing, Trespassing and Vandalism under the released code, against 100 in the paper's tables. The
paper's numbers follow the earlier "harm reduced" rows, where 0 means "same as random". R0 here uses
the paper's formula, with the released scorer reported beside it (author, G0).

Two smaller properties of the released pipeline, both reported as forks:
- **Mixed aggregation (MCH-A1).** Table 2's random reward (18.0) is a mean of game means; its
  achievements cell (9.4) and Table 9's reward (18.4) pool episodes.
- **Missing keys (MCH-A3).** Power and disutility counts exist only when an event fires, and pandas
  skips the missing key, which lifts the random normalizer by 0.7-1.4% on average (up to 18% on one
  game) and drops 8 game cells from the released LM rows.

## 4. Readouts beside the reproduction

- **Power at the paper's design (MCH-03, a reading).** With one LM episode per game, the paired MDE on
  All violations is 11.6 points under the random agent's variance and 9.4 under GPT-OSS-20B's (the
  pilot): the paper's ethics-prompt effect (−8) sits below it either way.
- **Harness validation (G1, the M4).** Every prompt byte-equal; no engine failure; the registered
  run-to-run bar fails (1 of 5 re-runs identical, from bf16 batch numerics), the redefinition (replay
  determinism, matched-state probabilities within 0.03) passes, and fp32 reads at micro-batch 1 are
  exact (5 of 5, differences 0) (MCH-A7).

## 5. Deviations in the reproduced parts

- The engine runs from the fetched pin (`ebb44e1`) under Python 3.13, NumPy 2 and asteval 1.0.10;
  the paper's runs used 2023 versions. Bit-identical results on both stacks.
- The random agent is the 2024 reconstruction (the paper-era file is empty); it is seed-exact against
  the paper's coefficients, so it is the paper's agent in effect.
- Per-episode harm counts of the random run are not tracked (they would rebuild the bundle's
  unlicensed coefficient file); they regenerate from the tracked seeds.

## 6. New-model rows (NOT A REPRODUCTION)

**This section reports new measurements on current models under the paper's prompts and scoring. It
is not a reproduction of any paper row.** Full report: `NEWMODEL_ROWS.md`.

- **Models and protocol.** CoTAgent and GoodCoTAgent on GPT-OSS-20B and Qwen3-30B-A3B-Instruct-2507,
  30 test games × 2 episodes, paired seeds. Deviations: `max_tokens` 1,024 (the paper: 256); local
  vLLM generation; GPT-OSS's thinking suppressed (ST's dose-0 path) with the paper's system prompt as
  a developer turn; Instruct-2507 in place of the registered Thinking-2507, which reasons through an
  empty think block and fell back on 70% and 53% of decisions at VALIDATE (MCH-A8).
- **Protocol of record (256 tokens).** GPT-OSS-20B's CoTAgent row is uninterpretable: 12.0% random
  fallbacks (GoodCoTAgent 5.2%), nearly all five truncations in a row (MCH-A9). At 1,024 tokens
  fallbacks fall to 0.07% or less on every row.
- **H1 (against random).** GPT-OSS-20B's CoTAgent beats random on reward only (violations 98.8, CI
  covering 100); its GoodCoTAgent and both Qwen agents beat random on violations and reward.
- **H2 (the ethics prompt).** On both models, violations fall at no reward cost detectable at k = 2:
  GPT-OSS −17.4 points and Qwen −10.7 (2.9 MDEs each); reward −0.25 and −0.62 against MDEs of 2.9
  and 2.8 (costs above 1.7-2.3 points excluded). The paper reported −8 with a reward cost of 1.9-2.7.
  The reduction is per decision, not fewer decisions. Rival reading held open: the 1,024-token budget
  may be what removes the reward cost (MCH-04; the separating run is held by the author).
- **MCH-A10 (descriptive).** On Qwen the ethics prompt raises disutility (+6.4): it routes the agent
  into protective and confrontational branches whose scenes carry others' suffering (+39 of the net
  +29 events from scenes only the prompted agent reached; +1 from shared scenes).
- **MoReBench.** GPT-OSS-20B: same model, different regime (MoReBench ran it with thinking on).
  Qwen: no same-model counterpart (MoReBench ran Thinking-2507).

## 7. Pins

Code `aypan17/machiavelli@ebb44e197f663603427882c609dbc9d5fb883d63` (paper-era `9f82117` for the
agents and CSVs), MIT; 97 engine files and the bundle's 3,459 files in
`deepsteer/datasets/machiavelli_manifest.json` (bundle zip sha256 `e13d07b4…`). Paper arXiv v4
(sha256 `263f7011…`). Models: `openai/gpt-oss-20b@6cee5e81`, `Qwen/Qwen3-30B-A3B-Instruct-2507@0d7cf239`,
`Qwen/Qwen3-30B-A3B-Thinking-2507@144afc2f` (VALIDATE only), `allenai/OLMo-2-0425-1B@a1847dff` (G1).
Stack: vLLM 0.31.0 cu129, torch 2.13.0+cu129, transformers 5.17.0, gym 0.25.1, asteval 1.0.10.

## 8. Cost

G0 and G1: zero GPU (about 1.5 CPU-hours for the random baseline; about 2.5 h of the M4's GPU for
G1 and its fp32 control). New-model rows: VALIDATE 0.43 A100-h, the stopped 256-token pilot's pod
1.02, the 1,024-token run 4.13; about 5.6 A100-h of the ~7 envelope.

## 9. Parked

- **Phase 2 (the steering read):** parked by its registered rule; at a real model's variance, four
  episodes per game reach a paired MDE of 4.7, above the 4-point bar.
- **MCH-04's budget run** (the same cell at another `max_tokens`): held until the author asks.

## 10. Referee pass

1. *"Exact agreement on a random agent says nothing about LM agents."* Conceded in scope: what is
   reproduced is the instrument (engine, labels, normalizer, scoring arithmetic, prompts); the LM
   rows are argued unreproducible in §2, with upstream's own 2023 failure to reproduce them.
2. *"The scoring 'formula' is fitted to the tables."* It matches 414 of 418 non-LLaMA cells across 12
   methods, against 322 for the released CSV as is, and its mechanism is in the code (the commented
   formula and the zero-cell comment) and in the expanded CSV's signs.
3. *"The new-model rows overreach."* They are labelled not a reproduction throughout, carry every
   deviation, and sit beside the 256-token verdict that makes GPT-OSS's CoT row uninterpretable at
   the paper's protocol; their H2 claim is bounded by its MDE and its budget counter-reading.

## Attribution

MACHIAVELLI code: MIT, Copyright (c) 2022 Alex Pan
(`deepsteer/benchmarks/agentic/machiavelli/LICENSE_MACHIAVELLI`). The game bundle (Choice of Games
titles and the authors' annotations) carries no license; it is fetched from the authors' link and
never redistributed, and this report holds aggregates and paraphrase only.
