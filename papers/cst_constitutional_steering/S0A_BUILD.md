# CST S0a build (2026-11-07; Amendments 7-11 applied to `S0_HARNESS.md`)

S0a gates G1: A0 (`none`) and A0s (`C_anth` in the system message) on Qwen3.8-27B, read with the
instruments every later arm is read with. This file is the build of record; `S0_HARNESS.md` is the
draft it replaces for S0a.

## 1. Settings of record

| setting | value | source |
|---|---|---|
| subject | `Qwen/Qwen3.8-27B@1d4bf0f2`, thinking on, effort medium | §3.1, D5 |
| engine | vLLM 0.31.0, MTP off, `max_num_seqs` 128 (A0s: 32), prefix caching on, `max_model_len` 44,032; chunks of 1,024 (A0s: 256) | Amendment 10 item 4; Amendment 12 addendum (CST-A13) |
| cap | 4,096 for every arm and condition; a truncated rollout is "no conclusion", parse status reported | Amendment 8 item 3 |
| sampler | KDG v2 (T 0.7, top-p 1, top-k 0), seeded per rollout | §8 |
| bulk scorer | `gpt-oss-120b@b5c939de`, blind (final answer never shown), temperature 1, effort low, letter-mapped; S_concl's reader | Amendments 9, 10, 11 |
| S_tenet reader | Claude through the account login, blind, one rollout per call, up to three attempts, unscored reported | Amendment 11 and addendum |
| 10% reader | `Olmo-3.1-32B-Instruct@ac0587e4`, blind, temperature 1; disagreement reported, never voted | Amendment 9 item 1 |

## 2. Components (S0a: 1-8 and 12)

| # | component | runs where | rollouts or calls per arm |
|---|---|---|---|
| 1 | Generation: panel 586 × (plain 4, user 2, operator 4, operator_pr 6); H1_held 132 × (plain 8, operator 8); cell A′ 102 × 8 | pod | 12,304 |
| 2 | Bulk scoring of every panel and H1 rollout by gpt-oss (S_concl, the o3 share, the no-conclusion rate) | pod | 11,488 |
| 3 | Retest: 100 seeded A0 operator rollouts re-scored twice by gpt-oss, same order and permuted | pod | 200 calls |
| 4 | The relabel (718 items, v4 prompt): a consistency read by gpt-oss | pod | 72 calls |
| 5 | Third-family votes on `T_core`'s 207 splits and `T_conf`'s 13 disputed pairs | **open (§4)** | 22 calls |
| 6 | Recall read on A0: teacher-forced log-probs of the true continuation and its two paraphrases (240 windows) | pod | 720 sequences |
| 7 | f: the applied-tenet fraction, from Claude's S_tenet read | local | — |
| 8 | ST's outcomes: ΔP on H1's 82 clear items (splits beside); the forced-choice read on cell A′; E at dose 0 on the panel | pod (rollouts and letter reads), local (analysis) | dose-0 letter reads |
| 12 | Capability: 800 items (400 GSM8K, 400 MMLU, test splits), seeded, disjoint from ST's gate 200 and selection 100 | pod | 800 |
| — | Layer A: gpt-oss on the 51 Scruples overlap items | pod | about 6 calls |
| — | Olmo's 10%: a seeded 10% of each arm's scored rollouts | pod | about 1,150 |
| — | S_tenet: Claude on the seeded 200 (`data/stenet_subset_v1.json`) × k 4, operator cell | local, after the pod | 800 rollouts, about 952 calls |
| — | Frontier second opinion: Claude and Codex on 200 seeded rollouts, both tasks; the validity bar of record | local, after the pod (Codex after its quota returns) | 200 per family |

## 3. Sessions

S0a prices at 22.6 A100-h on the engine of record (`CST_REFORECAST.md`), of which A0s generation is
about 13 h. Every generation chunk is saved when it finishes and skipped on resume, so a session cut
short continues in the next. Proposed split (8-12 h chunks):

- **S0a-1 (about 7 h):** A0's generation, dose-0 reads and recall; capability on A0; gpt-oss bulk
  scoring of A0, the retest, the relabel, Layer A; Olmo's 10% of A0.
- **S0a-2 (about 10 h):** A0s on the panel (9,376 rollouts).
- **S0a-3 (about 6 h):** A0s on H1_held and cell A′, its dose-0 reads and capability; gpt-oss and
  Olmo on A0s.

## 4. Decided (Amendment 12, 2026-11-07)

1. Component 5: Olmo-3.1-32B-Instruct votes twice on the 220 pairs; self-agreement reported; its
   vote decides only if it agrees with itself, else the splits stay disputed and the disputed
   `T_conf` pairs stay excluded from training.
2. E at dose 0 on the panel is included (component 8).
3. Capability at 4,096 + 256 (ST read 1,536 + 256): the absolute number is not comparable to ST's;
   the bar is paired within CST.
4. Three sessions as in §3, each resuming from saved chunks.

## 5. Saves (per unit)

Every rollout: arm, set, condition, scenario id, rollout index, letter order, seed, prompt token
count, generated ids, text, finish reason, parse status, trace length, letter, option. Every judge
call: rollout id, pass, temperature, seed, effort, prompt token count, raw reply, out tokens, cap
hit, parsed option and tenets. Every letter read: the letter log-probs. Recall: per window the
summed log-prob of the continuation and of each paraphrase. Capability: as ST's `score_cap`.
