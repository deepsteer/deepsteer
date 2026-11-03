# MCH-G1 gate report: harness validation on the M4

Zero GPU. Rules: `papers/MCH_PLAN.md` v0.4, Amendments 3-6 (pushed before any G1 number was read).
Numbers: `outputs/mch_g1.json` (registered analysis), `outputs/mch_g1_stability.json` (post-hoc
diagnosis, MCH-A7), per decision `outputs/g1/decisions.jsonl.gz`, per episode `outputs/g1/episodes.jsonl`.
Stopped at the gate: no pod has run.

## 1. Verdict

The harness reproduces the paper-era `LMAgent` prompt at every one of 11,669 decisions, runs all 70
episodes to completion with no engine failure, and its agent logic is deterministic under replay.
It **fails the registered run-to-run criterion**: 1 of 5 re-runs reproduce their reference episode
against a bar of 4. The cause is located (MCH-A7): bf16 reads on MPS shift the label probabilities
by up to 0.027 between batch compositions, which flips a sampled action about once every 40-70
decisions; the shared seeded draws and each run's own probabilities reproduce every action of both
runs.

| criterion (Amendment 5) | result | bar | verdict |
|---|---|---|---|
| prompt parity at every decision | 11,669 of 11,669 byte-equal | all | pass |
| S re-runs identical to V | 1 of 5 (divergence at steps 24, 14, 9, 2; one identical over 95) | ≥ 4 of 5 | **fail** (MCH-A7: numerics, not logic) |
| unhandled engine failure | 0 of 70 episodes (no crash, no hang) | 0 | pass |
| fallback (loop override) rate | V 2.8%, S 0.6%, Gr 0.6% | reported | — |
| action-number mass (labels' share of the next-token distribution) | median 0.166 (p10 0.096, p90 0.268) | reported | — |
| davinci truncation (2,047 GPT-2 tokens, middle cut) | 24% of V's decisions | reported | — |

Run: `allenai/OLMo-2-0425-1B` at `a1847dff`, bf16, MPS, torch 2.14.0; 16 episodes at a time,
micro-batches of 8; 8,125 s wall; engine fetched and hash-verified by
`deepsteer.datasets.machiavelli`.

## 2. The stability failure (MCH-A7)

- **Replay.** With each episode's `RandomState(seed)`, every action of V and of S follows from that
  run's own R0 probabilities, loop overrides included. At each divergence the uniform draw sat
  0.0008-0.021 from a choice boundary while the two CDFs differed by 0.015-0.025.
- **Second derivation.** The matched-state CDF differences predict a flip probability of 0.0141 per
  decision; observed 4 divergences in 148 compared decisions, 0.027 (exact Poisson 95% CI [0.0074,
  0.069]). Agree.
- **Reading.** R_numerics over R_logic: a leak or sampler bug would show large probability gaps or
  replay failures, and neither occurs. A batch-invariant re-run (fp32, micro-batch 1) is the open
  positive control (~15 min on the M4; it needs an amendment first).

## 3. Descriptive reads (Amendment 6: the 1B proxy decides nothing)

LM within-game variance from V (k = 2 on all 30 games, all 60 episodes complete), R0 scoring:

| metric | SE₁, LM proxy [95% CI over games] | SE₁, random (G0) | ratio | paired MDE, k = 1 / 2 / 4 |
|---|---|---|---|---|
| All violations | 1.89 [1.47, 2.27] | 2.92 | 0.65 | 7.5 / 5.3 / 3.7 |
| All power | 1.87 [1.30, 2.46] | 2.78 | 0.67 | 7.4 / 5.2 / 3.7 |
| All disutility | 3.87 [2.58, 5.07] | 3.98 | 0.97 | 15.3 / 10.8 / 7.7 |
| Norm. reward | 1.17 [0.83, 1.48] | 1.24 | 0.94 | 4.6 / 3.3 / 2.3 |

- **MCH-03 (the ethics-prompt contrast at the paper's design), proxy datum.** At this variance the
  paired MDE at one episode per game is 7.5 points, so the paper's −8 sits at 1.07 MDE, not below
  it as under the random agent's variance (0.69). The reading weakens toward "at the MDE"; a 20-30B
  model's variance, measured at the pilot, decides it.
- **Phase 2 go/no-go, at the proxy (descriptive).** k = 4 reaches MDE 3.74 ≤ 4 but prices at
  7.1-14.3 A100-h (midpoint 10.7 > 10); k = 3 gives MDE 4.32 > 4. No k passes: no-go at the proxy,
  narrowly and by price. The pilot applies the rule at a real model's variance.
- **Amendment 4's k, at the proxy (descriptive):** k = 2 (paired MDE 5.3 ≤ 8).

## 4. Prices

Measured at G1: 171 decisions per LM episode; non-chat prompts 1,445 tokens (OLMo); the CoT chat
prompts on G1's own replayed states (10,138 prompts, both agents) 1,949 tokens under GPT-OSS's
template with the empty-analysis prefill and 1,958 under Qwen's with the empty think block (p90 2,864
and 2,902); upstream's 3,840-token chat cut would apply to 1.7% of them.

**New-model rows, envelope about 6 A100-h (Amendment 6), at k = 2, generation at the 256-token cap:**

| model | registered basis (prefill 3,000 tok/s, a 30B dense model) | reading: prefill scaled by MRB's measured MoE/dense generation ratio |
|---|---|---|
| GPT-OSS-20B | 3.92 A100-h (prefill 3.71, generation 0.22) | ~1.0 (×4.7) |
| Qwen3-30B-A3B | 4.34 (3.72, 0.62) | ~2.9 (×1.7) |
| total | **8.3, above the envelope** | **~3.9, inside it** |
| stage-1 pilot (GPT-OSS, 10 games × 2 episodes × 2 agents, 6,846 decisions) | 1.3 | ~0.35 |

The registered basis overstates both models, which activate ~3-4B parameters per token; the pilot
measures the real rates, and Amendment 6's stop rule compares the full cell at the pilot's k against
the envelope. At the scaled basis k = 3 costs ~5.8 A100-h (at the envelope) and k = 4 ~7.7 (over).

**Phase 2 (descriptive, for the pilot's go/no-go):** the steering read prices 1.8-3.6 A100-h per
episode per game (2 models × 2 arms); its additional rows (CoTAgent and GoodCoTAgent on unadapted
Muse and Qwen3.8) at k = 2 price 4.8 and 5.3 A100-h upper, outside the 10 A100-h cap.

## 5. New-model rows (not a reproduction): the cell as it will run

No data yet. Registered in Amendments 4 and 6:
- `CoTAgent` and `GoodCoTAgent`, the paper's exact chat prompts and request truncation, T 0.2, 256
  tokens, up to five parse attempts then a random action, unrotated; GPT-OSS-20B (`6cee5e81`) and
  Qwen3-30B-A3B-Thinking-2507 (`144afc2f`), thinking suppressed (deviations).
- **Deviation to add before the pod:** GPT-OSS's template renders the paper's system prompt as a
  developer turn beside the template's own system message (`# Instructions`); upstream sent it as
  the system role to GPT-3.5 and GPT-4.
- Stage 1: the variance pilot on GPT-OSS-20B (10 games × 2 episodes × both agents), reading only
  fallback rates and within-game variance; k, the Phase 2 go/no-go and stop-or-continue are applied
  there. The fallback bar (> 10% uninterpretable) is checked there and on the full cell.
- H1 (each agent beats random on violations and reward) and H2 (the ethics prompt lowers both,
  paired by game and seed), both branches registered; the paper's effects beside as reference.

Not yet built: the vLLM pod runner for the chat CoT path (the G1 harness reads the non-chat path).
It needs a local test and a VALIDATE pod before stage 1, under the pod-boundary gates.

## 6. Anticipated review

```
Anticipated review:
1. S failed its bar → intervention on the instrument's validity → replay + second derivation locate
   it in bf16 numerics, not logic (implemented, MCH-A7); the batch-invariant positive control is
   open (~15 min M4, amendment first).
2. Numeric re-randomization could inflate the measured within-game variance → it is a fresh draw
   of the same sampler, not added noise on the outcome; the variance is descriptive anyway
   (Amendment 6) and the pilot runs on vLLM, which is also batch-nondeterministic.
3. 24% of non-chat decisions lose the middle of their prompt to the davinci cut (often achievements)
   → a property of the paper's davinci path, reported; Phase 2's read on long-context models should
   register whether it keeps the cut (open; decision at Phase 2 registration).
4. The new-model price straddles the envelope by basis (8.3 vs ~3.9) → the pilot measures the
   rates and the stop rule applies; the pilot itself costs 0.35-1.3 A100-h plus VALIDATE.
5. GPT-OSS sees the system prompt as a developer turn → a deviation to register before the pod
   (open; zero cost).
Implemented now: 1 (diagnosis), the descriptive reads, the prices on both bases.
Open (with costs): the batch-invariant S re-run (~15 min M4); the vLLM CoT runner and its local
test (zero GPU); VALIDATE + stage 1 (~0.6-1.6 A100-h).
Question behind the question: the run-to-run criterion asked for trajectory identity, which no
batched bf16 harness has; what G1 has to establish is that equal inputs give equal decisions up to
numeric noise, which the replay shows. The choice is whether to re-register the criterion as that,
or to require batch-invariant reads.
```

Referee pass:
1. *"A harness that cannot reproduce its own episodes is not validated."* Conceded as registered: the
   criterion failed. Answered on substance: replay reproduces every action from each run's own
   probabilities; the divergence is a ~0.02 probability shift at a draw near a boundary, at the rate
   the measured shifts predict.
2. *"A 1B base model's variance says nothing about 20-30B chat models."* Conceded (Amendment 6);
   it is reported as a proxy datum and decides nothing.
3. *"The prices are assumptions."* Conceded in part: decisions per episode and prompt tokens are
   measured; throughput is MRB's measured generation rate and an assumed prefill rate, bracketed by
   two bases; the pilot replaces both with measured rates before the stop rule.

## 7. Decisions for the author

1. **The stability criterion.** (a) Accept G1 with MCH-A7's reading and re-register run-to-run
   stability as replay determinism plus matched-state probabilities within 0.03, or (b) require
   batch-invariant reads and re-run S (~15 min M4) first. Recommend (a), with (b) run as the positive
   control, since the pod's vLLM batching cannot meet trajectory identity either.
2. **The renderer** passed parity at 11,669 decisions: move `mch_prompts.py` into the library
   (`deepsteer/benchmarks/agentic/machiavelli/prompts.py`) now, or after (1). Recommend after (1).
3. **The GPT-OSS developer-turn deviation:** register it before the pod (recommended), or render the
   paper's system prompt into the template's system message instead (a template edit).
4. **Next build (zero GPU):** the vLLM runner for the CoT chat path with its local test, then a
   VALIDATE pod and stage 1 under Amendment 6's stop rule. Pod launches stay yours.
