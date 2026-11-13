# MACHIAVELLI throughput on one A100: design note

2026-11-13. Zero GPU, nothing built or registered. Source: the finalist MACHIAVELLI pod's run
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

What the screen costs beyond resolution: it reads a different quantity (the mean over the short
games, 36% of the decisions), so it cannot stand in for §6.4's 30-game read; it could gate one
(run the full read when the screen shows a move, or for every finalist). Whether short games
predict long ones is testable on this run once its read is complete: the screen's estimate against
the full read's, both arms, zero GPU.

## Bias and limits

| assumption | effect on the numbers | direction |
|---|---|---|
| the curve is fitted and checked on the same run (in-sample) | the model's fit to its own run is optimistic | favors every option equally |
| D(A) depends only on the number of active episodes, not on which games or scenes | games with longer prompts would cut throughput when grouped | unknown; the cap sweep's minimum is flat between about 36 and 44 |
| the second candidate's decisions and lengths equal the first's | a candidate with longer traces costs more | favors the two-candidate rows |
| the longest game at 590 decisions (its episodes still running) | about 10 decisions either way moves the tail by minutes | neutral |
| MTP's speedup assumed (1.5x, 2.0x) | not measured on this benchmark | could favor the tail swap |
| the screen's MDE from 29 games' variance at k 2 | the 30th game (the longest) is missing from the variance | neutral |

## What a build would need (not started)

The cap and the longest-first order are a parameter and a sort in the finalist pod's episode loop
(`concurrency`, `mach_jobs`); several adapters per engine need one LoRA slot per adapter
(`max_loras`) and per-arm request routing, which the loop already does for two arms. Each is a
protocol change to a registered read and would be registered before it is built.
