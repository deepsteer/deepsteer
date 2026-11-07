# CST line re-forecast at measured rates (Amendment 8 item 5; revised for Amendment 9, 2026-11-07)

For the author before S0a launches. Source: `scripts/t09_reforecast.py` writes
`outputs/t09_reforecast.json`; the second VALIDATE's records are in `outputs/validate/`. Dollars at
$1.6 to $2.0 per A100-80GB hour; envelope $500 to $1,000; envelope of record 150 A100-h; the dollar
cap governs. Spent so far: the two VALIDATE pods, 2.8 pod-hours (about $5). Scoring follows
Amendment 9: gpt-oss-120b alone, Olmo-3.1-32B-Instruct on a 10% sample, the DPO pair filter on the
frontier tier (account logins, no GPU; its Codex calls are counted in §4).

## 1. Unit prices

| unit | value | basis |
|---|---|---|
| A0 generation, thinking on | 819 tokens/s; 6,888 rollouts/h at 428 tokens | measured, 256 rows |
| Generation with `C_anth` in context (A0s) | 150 tokens/s; 452 rollouts/h at 1,197 tokens | measured, 128 rows; CST-A11 |
| SDF / SFT / DPO training throughput | 411 tokens/s; 10M tokens in 6.8 h | measured (V5) |
| Scoring one arm (12,304 rollouts) | gpt-oss-120b 0.9 h; Olmo-3.1-32B-Instruct's 10% 0.2 h | measured |
| Full read of one trained arm | 5.9 h (generation 2.5 at 600 tokens, scoring 1.1, capability 1.2, N 1.0, dose-0 0.1) | generation and scoring measured; capability carried; N estimated |
| Finalist bars | 3.6 h (MRB 500 k 2 1.5, MACHIAVELLI k 2 1.75, refusal 0.3) | carried from MRB and MCH |
| DPO, one draw of 3,000 pairs, 3 epochs | 16.1 h (1 epoch: 5.4) | 411 tokens/s, 1,100-token pairs, reference forward at 0.2 |

The plan's §8 unit prices were 3.5 to 4 h per adapter read and about 2 h per DPO draw. The
measured read is 5.9 h, of which capability and N are carried or estimated. A DPO draw at §4.3's
2,000 to 4,000 pairs and the 3-epoch cap costs 8 times ST's 40-pair draw.

## 2. Forecast by session (A100-h)

| session | current engine | engine fixed (CST-A11's R_engine) |
|---|---:|---:|
| Probe pod: engine tuning, the letter-mapped scorer on gpt-oss and Olmo | 1.5 | 1.5 |
| S0a (measured; gpt-oss-120b, Olmo's 10%, cap 4,096, 10% margin) | 38.0 | 13.6 |
| S0b (A0s riders, N on A0 and A0s, A0s's length-matched fork) | 11.1 | 6.9 |
| S1 (§4.1 with the document in context and its self-critique pass, §4.2, A1, A1c, recovery) | 54.9 | 24.7 |
| S2 (A1 and A1c reads, A1-without-recovery format read, the winner's bars) | 16.0 | 16.0 |
| S3 (A2 two draws, A2 without the rewrite, A3; reads and bars) | 106.5 | 93.6 |
| **Partner recipe, S0a-S3** | **228.1 ($365-456)** | **156.3 ($250-313)** |
| S4 (B1 on the sibling; estimate) | 29.0 | 29.0 |
| S5 (A4, A4f; gated by R3; estimate) | 48.9 | 48.9 |
| **Whole line** | **306.0 ($490-612)** | **234.2 ($375-468)** |

The engine case is still the largest single uncertainty, about 72 A100-h across S0a, S0b, S1 and
S3: §4.1's documents and §4.3's rewrites are generated with the document in context, the path that
runs at 150 tokens/s now. The probe that decides it costs about 0.3 A100-h.

## 3. Ranked cut list (applied cumulatively to the partner-recipe path)

| rank | cut | saves (fixed / current) | recipe after (fixed / current) | loses | amendment? |
|---|---|---:|---:|---|---|
| 1 | DPO at 1 epoch, not 3 | 32.1 / 32.1 | 124.2 / 195.9 | dose; ST-A38 moved protection with pair count, not epochs | no (inside "≤ 3") |
| 2 | Defer A2 without the rewrite | 12.8 / 12.8 | 111.4 / 183.2 | the lab's rewrite-ablation replication | yes |
| 3 | A0s on the operator condition only (panel k 4, H1 operator k 8) | 4.8 / 22.5 | 106.6 / 160.6 | A0s's plain, user and operator_pr cells; A0s is a comparator and its band is provisional until S0b | yes |
| 4 | One A2 draw, not two | 14.2 / 20.7 | 92.3 / 139.9 | the draw-to-draw read ST-A38 asks for | yes |
| 5 | SDF corpus 5M tokens, not 10M | 11.9 / 26.9 | 80.5 / 113.0 | dose (the 1M-token pilot prices it anyway) | yes |
| — | S5 (already gated by R3) | 48.9 | — | GRPO | registered gate |
| — | S4, B1 on the sibling | 29.0 | — | the stage question on the sibling | yes (author) |

With the engine fixed, cut 1 alone brings the partner recipe to 124 A100-h ($199-248), inside the
envelope of record. With the engine as it is, cuts 1-4 reach 140.

**The path of record (Amendment 9 item 5):** cuts 1 and 2; cuts 3-5 not taken; without the engine
fix, A0s drops its user cell only; the recipe's envelope is 150 A100-h.

| | recipe of record, S0a-S3 | against 150 A100-h |
|---|---:|---|
| engine fixed | 111.4 ($178-223) | inside, 38.6 to spare |
| engine as run | 180.2 ($288-360) | **over by 30.2** |

So the probe pod decides more than a column: if the engine cannot be fixed, the path of record
exceeds its envelope, and a further cut (cut 4, one A2 draw, 20.7; or cut 5, 26.9) is the
author's at that point.

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

## 7. Recommendation

1. The probe pod first (1.5 A100-h, about $3): three engine configurations on A0 and A0s (prefix
   caching off for A0, MTP off, `max_num_seqs`), and the letter-mapped scorer on gpt-oss-120b and
   Olmo-3.1-32B-Instruct. Its result picks the column of §2.
2. Cut 1 (DPO at 1 epoch) needs no amendment and saves the most; cut 2 needs one sentence. The
   Codex count in §4 is the other budget the cut list answers to.
