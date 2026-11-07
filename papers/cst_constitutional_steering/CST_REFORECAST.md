# CST line re-forecast at measured rates (Amendment 8 item 5; revised for Amendment 9 and the probe pod, 2026-11-07)

For the author before S0a launches. Source: `scripts/t09_reforecast.py` writes
`outputs/t09_reforecast.json`; the second VALIDATE's records are in `outputs/validate/`, the probe pod's
in `outputs/probe/` (engine rates at cap 4,096). Dollars at
$1.6 to $2.0 per A100-80GB hour; envelope $500 to $1,000; envelope of record 150 A100-h; the dollar
cap governs. Spent so far: the two VALIDATE pods and the probe, 4.1 pod-hours (about $7). Scoring follows
Amendment 9: gpt-oss-120b alone, Olmo-3.1-32B-Instruct on a 10% sample, the DPO pair filter on the
frontier tier (account logins, no GPU; its Codex calls are counted in §4).

## 1. Unit prices

| unit | value | basis |
|---|---|---|
| A0 generation, thinking on, cap 4,096 | ST's engine 781 tokens/s; tuned 847 (6,722 rollouts/h at 448 tokens) | measured, probe, 128 rows |
| Generation with `C_anth` in context (A0s), cap 4,096 | ST's engine 178 tokens/s; tuned 322 (987 rollouts/h at 1,237 tokens) | measured, probe, 64 rows; CST-A11 |
| SDF / SFT / DPO training throughput | 411 tokens/s; 10M tokens in 6.8 h | measured (V5) |
| Scoring one arm (12,304 rollouts) | gpt-oss-120b 0.9 h; Olmo-3.1-32B-Instruct's 10% 0.2 h | measured |
| Full read of one trained arm | 5.9 h (generation 2.5 at 600 tokens, scoring 1.1, capability 1.2, N 1.0, dose-0 0.1) | generation and scoring measured; capability carried; N estimated |
| Finalist bars | 3.6 h (MRB 500 k 2 1.5, MACHIAVELLI k 2 1.75, refusal 0.3) | carried from MRB and MCH |
| DPO, one draw of 3,000 pairs, 3 epochs | 16.1 h (1 epoch: 5.4) | 411 tokens/s, 1,100-token pairs, reference forward at 0.2 |

The plan's §8 unit prices were 3.5 to 4 h per adapter read and about 2 h per DPO draw. The
measured read is 5.9 h, of which capability and N are carried or estimated. A DPO draw at §4.3's
2,000 to 4,000 pairs and the 3-epoch cap costs 8 times ST's 40-pair draw.

## 2. Forecast by session (A100-h; cap 4,096)

| session | ST's engine | tuned engine (MTP off, 128 sequences) |
|---|---:|---:|
| Probe pod (run) | 1.5 | 1.5 |
| S0a (gpt-oss-120b, Olmo's 10%, cap 4,096, 10% margin) | 34.4 | 22.6 |
| S0b (A0s riders, N on A0 and A0s, A0s's length-matched fork) | 10.5 | 8.4 |
| S1 (§4.1 with the document in context and its self-critique pass, §4.2, A1, A1c, recovery) | 49.2 | 35.0 |
| S2 (A1 and A1c reads, A1-without-recovery format read, the winner's bars) | 16.3 | 15.9 |
| S3 (A2 two draws, A2 without the rewrite, A3; reads and bars) | 104.8 | 97.5 |
| **Partner recipe, S0a-S3** | **216.6** | **181.0** |
| S4 (B1 on the sibling; estimate) | 29.0 | 29.0 |
| S5 (A4, A4f; gated by R3; estimate) | 49.1 | 48.7 |
| **Whole line** | **294.8** | **258.7** |

The probe measured the engine case (CST-A11): MTP off at 128 sequences raises A0s from 178 to 322
tokens/s and A0 from 781 to 847. A0s still runs at 38% of A0's per-token rate, so the earlier
"engine fixed" column (A0s at A0's rate, at cap 1,536) was optimistic by about 25 A100-h on the recipe.

## 3. Ranked cut list (applied cumulatively to the partner-recipe path)

| rank | cut | saves (tuned / ST's) | recipe after (tuned / ST's) | loses | status |
|---|---|---:|---:|---|---|
| 1 | DPO at 1 epoch, not 3 | 32.1 / 32.1 | 148.8 / 184.5 | dose; ST-A38 moved protection with pair count, not epochs | taken (Amendment 9 item 5) |
| 2 | Defer A2 without the rewrite | 12.7 / 13.0 | 136.2 / 171.5 | the lab's rewrite-ablation replication | taken (Amendment 9 item 5) |
| 3 | A0s on the operator condition only | 11.3 / 19.7 | 124.9 / 151.8 | A0s's plain, user and operator_pr cells | not taken |
| 4 | One A2 draw, not two | 16.3 / 19.6 | 108.5 / 132.2 | the draw-to-draw read ST-A38 asks for | not taken |
| 5 | SDF corpus 5M tokens, not 10M | 17.0 / 24.1 | 91.5 / 108.1 | dose | not taken |
| — | S5 (already gated by R3) | 48.7 | — | GRPO | decided at G3 |
| — | S4, B1 on the sibling | 29.0 | — | the stage question on the sibling | decided at G3 |

**The path of record (Amendment 9 item 5):** cuts 1 and 2; envelope 150 A100-h for S0a-S3.

| | recipe of record, S0a-S3 | against 150 A100-h |
|---|---:|---|
| tuned engine | 136.2 ($218-272) | inside, 13.8 to spare |
| ST's engine, with A0s's user cell dropped | 168.9 ($270-338) | over by 18.9 |

The tuned engine (one line of engine configuration) brings the path of record inside its envelope
with no further cut; the margin is thinner than the earlier estimate.

## 4. The frontier tier's Codex calls (Amendment 9 items 2-3)

| task | Codex calls (10 items per call) |
|---|---:|
| DPO pair filter, per draw (3,000 pairs at a 0.7 pass rate) | 429 |
| The 200-pair pilot | 20 |
| Frontier second opinion, about 200 S0a rollouts | 20 |

The full 718-item labeling was 72 Codex calls and took several 5-hour windows. Two draws of the
pair filter are 858 calls, about twelve times that, before the weekly limit is counted. The weekly
quota, not the GPU, bounds item 3 at the registered pair counts; cut 4 (one draw) halves it, and
the 200-pair pilot measures the calls each window holds.

## 5. Found while pricing

- §5 says the A2-without-rewrite ablation rides along "at zero extra training". It needs its own DPO
  run (chosen = a second sample), so it is priced here as one draw at the same dose (cut 2).
- §4.1's corpus and its self-critique pass are generated with `C_anth` in context, so their price
  moves with CST-A11; the 1M-token pilot in S1 measures it at the engine of record.

## 6. Bias-direction table for the forecast

| known bias | mechanism | direction |
|---|---|---|
| A0s's rate applied to every condition | measured on the operator condition; plain and user prompts are shorter | overprices S0a |
| Long-prompt calls priced at the probe's seconds per call | 8 calls ran concurrently; 72 calls batch better | overprices (small) |
| Trained arms' generation at 600 tokens per rollout | the toy SDF adapter's median; a stronger dose may lengthen traces toward A0s's 1,197 | either way; at 1,197 tokens each arm read costs 2.5 h more |
| DPO throughput from packed 2,048-token SDF sequences | 1,100-token pairs pad and pack less well | underprices DPO |
| Capability, MRB, MACHIAVELLI carried from other lines | other models, other engines | unknown |
| S4 and S5 | estimates, not built from measured units | unknown; S5 is gated |

## 7. Scoring under Amendment 11 (the scorer mini-pod, `outputs/scorer_high/`)

| item | effort low | effort high |
|---|---:|---:|
| gpt-oss bulk scoring, one arm read (12,304 rollouts) | 0.9 A100-h (14,062 calls/h) | 14.4 A100-h (853 calls/h) |
| S_concl retest kappa (single, same-order / permuted) | 1.00 / 0.98 | 0.98 / 0.94 |
| path of record, S0a-S3, tuned engine | **136.2** | **231.1** (+94.9 over seven reads) |

Effort high adds nothing to S_concl and its tenet naming is less stable, so the 94.9 A100-h buys no
measurement the path of record uses; at high the path of record exceeds its 150 A100-h envelope by
81.

**S_tenet's judge of record (Claude, account login, no GPU).** Per arm read: 800 rollouts (200
items × k 4, operator cell), about 952 calls with retries (the retest's 1.19 calls per rollout), 93%
scored, about 1.6 h of calls one at a time at 6 s each. Path of record (seven reads): about 6,660
calls; with S4 and S5 (four more reads), about 10,470. **Codex's 10% sample:** 80 rollouts per read,
560 calls on the path of record, after the weekly quota returns.

## 8. Recommendation

1. Every subject read and every generation with the document in context runs on the engine of
   record (Amendment 10 item 4).
2. The bulk scorer runs at effort low: S_concl's reads are the same at both efforts, and effort
   high costs 94.9 A100-h on the path of record (an amendment to Amendment 10 item 1).
3. Claude's calls are the S_tenet budget to watch: the account's limits, not the GPU, set how fast
   a read's 952 calls run; the 7% classifier loss is stochastic per call (no rollout lost in both
   retest passes), so a third attempt on a stopped rollout is the cheap recovery.
