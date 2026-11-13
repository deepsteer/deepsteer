# MACHIAVELLI throughput on one A100: design note

2026-11-13. Zero GPU. **Ruled (Amendment 45):** the concurrency cap with the longest game first for
every future MACHIAVELLI read (38 for one candidate and the base, 52 for two), a scheduling change
with an in-sample forecast and a standing price exception at the capped forecast; priority without a
cap, the MTP tail swap and a second GPU not built; the short-game screen not built (its reason of
record below). Source: the finalist MACHIAVELLI pod's run
(config 2 and the base, 30 games, k 2, 120 episodes, Qwen3.8-27B with LoRA on vLLM 0.31.0, the
4,096-token budget; `outputs/finalist_c2_rule_right_mach`), modeled by
`scripts/mach_throughput_model.py` from the runner's 60-second progress lines and vLLM's
10-second engine stats. The run was still finishing its longest game when this was written; that
game is taken at 590 decisions.

## What the run measured

- **The engine is KV-bound, not batch-bound.** During the bulk of the read the KV cache sat at
  about 99%, about 60-70 requests ran, and up to 65 waited. Generation throughput plateaus at about
  500 tokens/s from about 30-40 running sequences (prompt processing about 1,000 tokens/s).
- **Decisions per hour against active episodes, D(A):** 373 at 4, 1,169 at 15, 1,717 at 27, 1,941
  at 34, 2,074 at 43, **2,145 at 53 (the peak)**, about 1,900-2,050 from 64 to 120. Past about 45
  active episodes, adding episodes adds waiting, not throughput: each episode's rate falls from
  46 per hour at 40 active to 16 at 120.
- **The tail.** The longest game (luminous-underground, about 590 decisions) runs at 16-19
  decisions per hour while all 120 episodes are in flight. Its four episodes are still running when
  everything else has finished, at about 93 per hour each on a nearly idle GPU (about 25 tokens/s
  per sequence, 7% of the KV cache).
- **Model check (in-sample).** Fed this curve and the run's own decision counts, the model's
  active-episode count is within about 4 of the observed count at hours 6-10, and the model
  finishes the read at 13.3 h, past the pod's 12.75 h deadline, as the run is doing.

## Options

Read = the MACHIAVELLI stage; pod = read + 0.56 h (setup, gate tests, smoke and parity to the
read's start, 0.41 h measured; provisioning and download, 0.15 h estimated). Throughput floor at
the curve's peak: 9.63 h for one candidate and the base (20,653 decisions), 14.47 h for two
candidates and the base (31,032).

| option | 1 candidate + base: read, A100-h | 2 candidates + base: read, A100-h |
|---|---|---|
| as run: all episodes at once, first come first served | 13.3 h, 13.9 | 18.4 h, 19.0 |
| **concurrency cap, longest game first (best cap)** | **10.8 h, 11.3 (cap 38)** | **14.6 h, 15.2 (cap 52)** |
| cap 30 / 40 / 50 / 60 | 12.0 / 10.9 / 11.4 / 12.1 h | 17.8 / 15.5 / 14.7 / 15.7 h |
| priority for the longest game(s), no cap | 12.2 h, 12.8 | 16.0-16.1 h, 16.6-16.7 |
| priority plus the best cap | same as the cap alone | same as the cap alone |
| tail swap to MTP at 8 or fewer active (speedup assumed 1.5x / 2.0x, 0.2 h restart) | 12.5 / 12.0 h, 13.0 / 12.5 | 17.5 / 16.9 h, 18.1 / 17.5 |
| longest game on its own GPU, best cap on the main GPU | 8.7 h wall, 16.1 | 13.1 h wall, 21.4 |

- **The cap is the lever.** Capping at about 40 episodes keeps the engine at its peak throughput
  and runs the longest game at about 46 decisions per hour from the start instead of 16, so the
  read finishes about 1 h above the throughput floor instead of about 3.7 h. Fewer than about 30
  starves the engine (cap 20: 15.3 h).
- **Priority adds nothing once the read is capped**: with the longest game first in the queue and
  a cap above its episode count, its requests never wait. Without a cap, priority recovers about
  1.1 h of the 3.7.
- **Several adapters per engine amortize the tail only with a cap.** All at once, the second
  candidate slows every episode (18.4 h). Capped at 52, the second candidate costs about 3.9
  A100-h (15.2 against 11.3), about 7.6 A100-h per candidate.
- **The tail swap to MTP** depends on a speedup this run did not measure, and is a protocol change
  (Amendment 43 rules it out for this pod).
- **A dedicated GPU for the longest game** buys wall time at about 1.4 times the A100-h.

## A short-game screen tier

The shortest games, both arms at k 2, at the best cap for each subset. Resolution: Amendment 3's
paired MDE at k 2 from this run's within-game variance (both arms, the 29 games with both episodes
finished), the full read's MDE being 6.49 on All violations and 3.48 on reward. **No candidate-base
difference was read for this note** (the registered read is not complete).

| games | read | pod | decisions covered | MDE violations | MDE reward |
|---|---|---|---|---|---|
| 13 shortest (to 137 decisions) | 2.8 h | 3.4 h | 25% | 7.88 (1.21x) | 4.76 (1.37x) |
| 15 shortest (to 139) | 3.3 h | 3.9 h | 31% | 9.38 (1.45x) | 4.38 (1.26x) |
| **17 shortest (to 147)** | **3.8 h** | 4.4 h | 36% | **8.35 (1.29x)** | **4.07 (1.17x)** |
| 18 shortest (to 151) | 4.0 h | 4.6 h | 39% | 8.28 (1.28x) | 3.87 (1.11x) |

Second derivation: with equal within-game variance the MDE grows as the square root of 30 over the
games read; for 17 games against the 29 read here that is 1.31, against the measured 1.29 (agree).
The 17-game screen's games: avatar-of-the-wolf, pendragon-rising, ratings-war, t-rex-time-machine,
death-collector, versus-the-lost-ones, hero-project-redemption-season, undercover-agent,
hero-of-kendrickstone, kung-fu, midsummer-nights-choice, tower-behind-the-moon, battlemage,
versus-deathscapes, mysteries-of-baroque, trials-of-the-thief-taker, seventh-sea-a-pirates-pact.

What the screen costs beyond resolution, measured where the answer is known: it reads a different quantity (the mean over the short
games, 36% of the decisions), so it cannot stand in for §6.4's 30-game read; it could gate one
(run the full read when the screen shows a move, or for every finalist). Whether short games
predict long ones is testable on this run once its read is complete: the screen's estimate against
the full read's, both arms, zero GPU.

### The screen against a known effect (Amendment 45 item 3; the reason of record for not building it)

MCH's ethics-prompt cells, the ethics-prompt agent minus the plain agent, paired by game and seed;
**MCH's estimator of record**: the mean over games of per-game means, the CI resampling episodes
within each game (games fixed, 10,000 draws, seed 0); a bootstrap over games beside it (games
treated as a sample). The screen is the 17 games above (12 of them in each 20-game GPT-OSS cell).

| cell | games | All violations: full | screen | long games | screen minus long (games fixed) | by game |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 (MCH's new-model row, not the CST subject), `cell_qwen3_30b_instruct` | 30 | −10.7 [−12.5, −9.0] | **−6.2 [−8.9, −3.5]** (by game [−17.1, +5.4]) | −16.6 [−18.6, −14.7] | **+10.4 [+7.0, +13.7]** | [−3.3, +25.0] |
| GPT-OSS-20B thinking, `thinking/cell_gpt_oss_20b` | 20 | −15.3 [−20.3, −10.3] | −15.4 [−22.6, −8.0] | −15.2 [−20.8, −9.8] | −0.2 [−9.4, +9.2] | [−14.2, +14.1] |
| the same, pilot and cell joined | 30 | −16.2 [−20.0, −12.6] | −15.4 [−21.2, −9.5] | −17.4 [−21.3, −13.5] | +2.1 [−5.0, +9.1] | [−8.4, +12.4] |
| GPT-OSS-20B at 1,024, `newmodel/cell_gpt_oss_20b` | 20 | −19.1 [−22.0, −16.3] | −20.7 [−23.6, −17.7] | −16.8 [−22.0, −11.6] | −3.9 [−10.1, +2.3] | [−15.6, +8.2] |
| the same, pilot and cell joined | 30 | −17.4 [−19.5, −15.3] | −17.4 [−19.5, −15.3] | −17.4 [−21.2, −13.6] | +0.0 [−4.5, +4.4] | [−9.6, +9.5] |

The ethics prompt has not been read on the CST subject (Qwen3.8-27B); every row here is another
model's. Reward, screen minus long (games fixed; by game beside): Qwen3-30B-A3B-Instruct-2507 +6.1 [+3.8, +8.4] ([−0.9, +13.7]);
GPT-OSS thinking cell +5.4 [+1.9, +8.9] ([+0.1, +11.1]), joined +3.9 [+1.1, +6.6]; GPT-OSS at 1,024
cell +2.0 [−1.3, +5.2], joined +1.8 [−0.9, +4.4]. The sign is the same in every cell (two of the
three models are GPT-OSS-20B). The joined GPT-OSS-1,024 row's three equal points are a near-tie, not
an error: −17.39, −17.41 and −17.40 at four places.

**Reading.** A quiet short-game screen does not clear a candidate. On the Qwen3-30B-A3B-Instruct-2507
cell the screen reads
58% of the full effect (−6.2 against −10.7): detected with games fixed, not detected by game. **The
short-long gap on violations is established on the Qwen3-30B-A3B-Instruct-2507 cell with games
fixed (+10.4 [+7.0,
+13.7]) and not by game, and on no GPT-OSS cell under either.** Which inference applies depends on
the question: for a screen that always uses these 17 games, the games-fixed CI says these games
under-read that model's effect; whether short games under-read effects in general is a question over
games, which the by-game CI answers and does not establish.

**Against the author's scratch numbers** (a plain per-game mean): the points agree on every row the
scratch gave (Qwen3-30B-A3B-Instruct-2507 full −10.7, screen −6.2, long −16.6, gap +10.4; GPT-OSS thinking −15.3 and −15.4;
GPT-OSS at 1,024 −19.1 and −20.7). Two differ, and these are of record: the thinking cell's reward
gap is +5.4 [+0.1, +11.1] by game (the scratch: +5.8 [+0.3, +11.8]); the 1,024 cell's reward gap is
+2.0 on the 20-game cell (the scratch's +1.8 matches the 30-game join). The scratch's CIs match the
by-game bootstrap, not MCH's estimator of record, so the scratch's "known effect read as not
detected" holds by game and not with games fixed.

### An interim look inside the full read (Amendment 45 item 5; model only)

The capped read ordered the longest game first, then the 17 shortest, then the rest: at cap 38 the
17 finish at **4.8 h** and the read at **11.05 h** (10.78 h longest first, +0.3 h); at cap 40, 4.9 h
and 10.84 h (10.89 h longest first). Ordered longest first throughout, the 17 do not finish until
about 10.3-10.6 h. The interim look costs about 0.0-0.3 h and gives the screen's games at about
4.8 h without a separate pod.

## Bias and limits

| assumption | effect on the numbers | direction |
|---|---|---|
| the curve is fitted and checked on the same run (in-sample) | the model's fit to its own run is optimistic | favors every option equally |
| D(A) depends only on the number of active episodes, not on which games or scenes | games with longer prompts would cut throughput when grouped | unknown; the cap sweep's minimum is flat between about 36 and 44 |
| the second candidate's decisions and lengths equal the first's | a candidate with longer traces costs more | favors the two-candidate rows |
| the longest game at 590 decisions (its episodes still running) | about 10 decisions either way moves the tail by minutes | neutral |
| MTP's speedup assumed (1.5x, 2.0x) | not measured on this benchmark | could favor the tail swap |
| the screen's MDE from 29 games' variance at k 2 | the 30th game (the longest) is missing from the variance | neutral |

## The in-context constitution on MACHIAVELLI (Amendment 45 item 6; held; forecast only)

Measured on the CST engine (single-turn rollouts, prefix caching on): A0s at 32 sequences, the
largest batch its ~35k-token prefix allows without overcommitting the KV cache (CST-A13), generates
**286-335 tokens/s** (the loop pod, S0a, the DPO pod); A0 at 128 sequences about 1,050 tokens/s.
On the loop pod's content cell A0s's rollouts ran 1,107 tokens against A0's 655-675 (about 1.67x).

**Assumption about prefix caching.** The forecast assumes every MACHIAVELLI request puts the agent's
system prompt and the constitution first and identical, so vLLM caches that prefix once and each
decision prefills only its scene (about 1,850 tokens, as now; this run's prefix hit rate was 52%).
Without that sharing each decision prefills about 36k tokens (about 390M prompt tokens in all) and
the read does not fit any pod. Qwen3.8 is a hybrid model whose recurrent layers cache at 800-token
block boundaries ("align" mode), so a hit needs the shared prefix to cover whole blocks; the A0s
rates above were measured under the same engine and cache mode.

**Forecast** (the in-context arm, 60 episodes, the base's decisions per game, about 1.06 attempts per
decision as GPT-OSS's thinking run, about 1,290 tokens per attempt at A0s's 1.67x):
- **Throughput:** about 14M generated tokens at 286-335 tokens/s is about 12-14 h of generation,
  plus about 20M scene tokens of prefill.
- **The serial chain:** the longest game's episode is about 590 x 1.06 x 1,290 = about 0.8M tokens in
  one sequence; at A0s's per-sequence rate in a batch of 32 (about 10 tokens/s) about 21 h, and at a
  small batch (an assumed 15-20 tokens/s over a 36k context) about 11-15 h.
- **As its own pod:** about 15-22 h for the read (16-23 A100-h with the pod's 0.56 h), the serial
  chain the binding term.
- **As a third arm on a capped read** (with a candidate and the base): the candidate and the base
  finish near their capped 10.8 h if the long-context sequences do not slow the shared step much
  (they do slow it: each step attends over their 36k contexts); the in-context arm's serial chain
  sets the read at about 15-22 h, so the third arm adds about 4-11 h to the pod (about 16-23 A100-h
  in all), less than its own pod plus the base's read.
- **Second derivation** (from the base's curve): scaling D(A) by A0s's decode rate at an equal
  batch (about 0.5) and by the token ratio (0.6) gives about 580 decisions/h at 32 active, about
  18 h for the arm's 10,300 decisions, inside the range.

These rates come from single-turn rollouts, not from MACHIAVELLI's episodes, so the forecast is
out of sample: an extrapolation the note's own curve never covered. An earlier estimate in this
session (about 3-7 A100-h, before the measurements above) is withdrawn: it assumed about 650 tokens
per decision and no serial chain. A pilot on the pod's gate at load would measure the arm's rate
before the read commits.

## What a build would need (not started)

The cap and the longest-first order are a parameter and a sort in the finalist pod's episode loop
(`concurrency`, `mach_jobs`); several adapters per engine need one LoRA slot per adapter
(`max_loras`) and per-arm request routing, which the loop already does for two arms. The cap is
ruled (Amendment 45 item 1); building it changes the pod's scheduler only, with its forecast printed
at the gate.
