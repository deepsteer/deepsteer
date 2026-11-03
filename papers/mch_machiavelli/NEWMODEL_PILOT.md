# New-model rows: stage-1 pilot report (stopped)

Pod of 2026-11-03 (one A100-SXM4-80GB, 05:29-06:31 UTC, about 1.0 A100-h), `remote_mch_newmodel.sh`
at `6ac5b1f`, VALIDATE=0, Qwen checkpoint Instruct-2507 by `validate_decision.json`. Rules:
`papers/MCH_PLAN.md` Amendments 6-8. Outputs: `outputs/newmodel/pilot/` (`decision.json`),
`session_run.log`. The pilot reads only fallback rates and within-game variance: no agent mean, no
difference between agents and no comparison with random has been computed (H1 and H2 unread).

## 1. The stage-1 decision (computed on the pod): STOP

| quantity | value | rule | |
|---|---|---|---|
| CoTAgent random-fallback rate | **0.120** (365 of 3,040) | ≤ 0.10 | **fails** |
| GoodCoTAgent random-fallback rate | 0.052 (159 of 3,036) | ≤ 0.10 | passes |
| projected full cell, pilot included | 8.45 A100-h | ≤ 6 | fails (not the reason of record) |
| k (Amendment 4's rule) | 2 | | |
| Phase 2 go/no-go (Amendment 3) | **no-go** | some k ≤ 4 with MDE ≤ 4 within ~10 A100-h | k = 4 gives MDE 4.72 |

Reason of record: fallback above the bar. By Amendment 6, GPT-OSS-20B's CoTAgent row is
uninterpretable at the paper's protocol, and the pod stopped before stages 2 and 3.

## 2. Why CoTAgent falls back (MCH-A9)

Every reply opens with JSON; none carries a code fence or opens with prose. 60% of CoTAgent attempts
(45% of GoodCoTAgent's) end at the paper's 256-token cap with the "reasoning" field still open, and
352 of the 365 CoTAgent fallbacks are decisions whose five attempts all hit the cap (157 of 159 for
GoodCoTAgent). The model follows the format; the paper's answer budget is shorter than its reasoning.

## 3. Variance and prices from the pilot

- **Within-game variance (a real model, 20 game × agent cells):** SE₁ for 30 games 2.38, 0.82 of the
  random agent's (2.92) and above G1's 1B proxy (1.89). Paired MDE on All violations: 9.4 at k = 1,
  6.7 at k = 2, 4.7 at k = 4. **MCH-03** (the paper's ethics-prompt contrast at one episode per game):
  −8 sits at 0.85 of this MDE, below it, as under the random agent's variance.
- **Rates:** 6,076 decisions in 2,641 s, 1.21e-4 A100-h per decision with 40 episodes at once and
  fewer in the tail; 433 generated tokens per decision (retries included); 152 decisions per episode.
- **Price of the full cell at k = 2:** GPT-OSS 2.2 A100-h. The stop rule priced Qwen at 2.84 times
  GPT-OSS's rate (MRB's Thinking-2507 generation ratio), 6.25, for 8.45 in total. Instruct-2507 does not
  think; at GPT-OSS's own rate the cell would cost about 4.4.
- **Spend so far:** VALIDATE 0.43 A100-h, this pod about 1.0; about 1.45 against the ~6 envelope.

## 4. Anticipated review

```
Anticipated review:
1. The stop is a length budget, not the model's format → MCH-A9 discriminator on saved replies
   (implemented: 96% of fallbacks are five cap hits; zero fence or prose openings).
2. The price projection used a thinking model's rate for a non-thinking one → the 8.45 overstates
   Qwen; the fallback bar stopped the pod first, so no verdict rests on it (implemented: both bases
   reported).
3. A rerun at a larger budget is a deviation from the paper → only as a registered fork beside the
   paper-faithful verdict, never replacing it (open; an amendment before any pod).
4. The Phase 2 no-go uses GPT-OSS's CoT variance for a different decision regime (action-number mass
   on other models) → the rule as registered says park; whether to re-measure Phase 2's own variance
   is the author's call (open; zero GPU to decide).
Question behind the question: do the new-model rows exist at the paper's protocol, or only under a
budget fork? At 256 tokens one of four model × agent rows on GPT-OSS is uninterpretable; the paired
H2 needs both agents on the same model.
```

## 5. Decisions for the author

1. **The GPT-OSS rows.** (a) Accept the stop: GPT-OSS-20B CoTAgent is uninterpretable at the paper's
   protocol (12.0%), and the new-model cell ends here with MCH-A9 as its finding. (b) Register a
   `max_tokens` fork (for example 1,024) as a deviation, beside the paper-faithful verdict, and rerun the
   pilot under the same stop rule (about 0.5-0.8 A100-h with setup); under R_budget the fallback falls
   well below 10%, and generated tokens per decision may fall too (today's 433 include up to five capped
   attempts). Recommend (b), with the 256-token verdict reported as the protocol-of-record result.
2. **The stop rule's Qwen multiplier.** If the cell reruns, replace the 2.84 Thinking-2507 ratio with a
   rate measured on Instruct-2507 (a short Qwen pilot slice in stage 1), registered before the pod.
3. **Phase 2:** parked by the registered rule (k = 4 gives MDE 4.72 > 4). Re-measuring variance under
   Phase 2's own regime is possible but not registered.
