# Draft edits for the pitch Google Doc (2026-09-27)

For pasting into the pitch doc (the source of record; `KDG_PITCH_PLAN.md` is its export). Rule of
record (author, 2026-09-27): the "post-training lowers the baseline" half is dropped; the
base-vs-instruct comparison is demoted from headline to cell, pending the SFT bridge cell (P1-A3).
Items marked [pending] wait for the p1a_fix results and the Phase 1 gate.

## The claim (replacement)

On OLMo-3-7B, an open model acts against its own stated moral judgment, and the part of that gap
that comes from the incentive is there before post-training and still there after it, read under
the model's own chat template (pressure-attributable excess 0.030, 95% CI 0.006 to 0.053). The
instrument that shows this also withdrew one of our own headline claims by its pre-registered rule:
an apparent "safer at rest" effect of post-training came from reading a chat model without its
template. [pending: the stage sweep SFT → DPO → final under one template, which says where in
post-training the pressure-attributable part changes, if it does.]

Why it matters for evaluation: a chat model read in a raw completion frame acquires agent-frame
effects that belong to the frame. Any evaluation or interpretability result that scores instruct
checkpoints without their template, as several recent post-training studies do, inherits them.

## Evidence in hand, points 4 to 6 (replacement)

4. **The gap was there before safety training, and it is still there after.** The base model shows a
   small pressure-driven gap; the trained model, read in its own chat format, shows one too.
5. **One of our earlier findings did not hold up, and the test that removed it was written in
   advance.** We had reported that safety training makes the model more cautious at rest. That came
   from reading the trained model in a format it was not built for. In its own format it is not more
   cautious at rest.
6. **Whether safety training makes the action more sensitive to pressure is open.** On one scale
   it does; adjusted for how much more decisive the trained model's outputs are, it does not
   resolve. [pending: the stage sweep answers this under one format.]

## Numbers of Record (rows that change)

| Finding | Number of record | Scope limit |
|---|---|---|
| The gap exists before alignment | Base excess 0.017 [0.012, 0.022] in the raw frame | Continuous readout; [pending: base-cell status by the SFT bridge rule] |
| The gap survives alignment | Instruct excess 0.030 [0.006, 0.053] under the chat template; at rest 0.055 [0.034, 0.076], under pressure 0.084 [0.059, 0.110] | 136 screened, selected on this model's chat actions |
| ~~Post-training lowers the baseline~~ | Withdrawn: raw-frame −0.038 does not reproduce under the template (+0.055); pre-registered artifact branch | KDG-A6 → R_b |
| Post-training and pressure sensitivity | Raw frame 0.018 → 0.046 (Δ 0.028 [0.007, 0.049]); per unit of output scale 0.12 [−0.02, 0.26], not resolved | Instruct side is a raw-frame cell; [pending: stage sweep under the template] |
| ~~Net gap under pressure looks smaller~~ | Withdrawn with the baseline row | — |
| The raw frame distorts templated models | At-rest sign −0.038 raw vs +0.055 chat; agent/judge sharpness ratio 1.79 raw vs 1.07 chat | One model; SFT bridge pending |

Rows removed: "Post-training widens the acting side" (becomes the scoped row above), "Post-training
lowers the baseline", "Net gap under pressure looks smaller".

## Related work line (replacement)

tracing-sycophancy (Sonnet Xu, GitHub): "related behavior–probability dissociation on the same OLMo-3
checkpoints; readout comparability unverified" (its log-prob track reads chat models without their
template). Not cited as convergent evidence.
