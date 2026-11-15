# Constitution distillation against operator pressure: a one-page recipe

Draft for the author's review, 2026-11-15 (CST Amendment 53 item 5). Every number comes from one
model (Qwen3.8-27B, thinking on) and one public document (the constitution pinned as `C_anth`).
Intervals are 95% bootstraps, paired by scenario (by game for MACHIAVELLI).

## The recipe

The model answers operator-pressure scenarios with the document in its system prompt. A LoRA
adapter (rank 16, alpha 32, learning rate 1e-4, 200 steps over 797 of those answers, about 1.9M
tokens) is trained on them with the document removed. At inference the adapter runs with no
document in context. We call it P-ctx.

## What the adapter does

On the operator cell (458 scenarios in which an operator's order conflicts with the document), the
base model follows the order in 0.34 of its answers. With the document in context it follows in
0.03, and with the adapter alone in 0.07. The adapter carries 0.86 to 0.90 of the in-context gain
(0.86 [0.80, 0.91] on its first read, 0.88 in a second session, 0.90 on a second GPU provider).
A second training draw gives the same drop (−0.286 against −0.275), and the second provider
reproduces it against its own base (−0.283 [−0.318, −0.248]). The same recipe with a neutral text
of the same length as the teacher raises order-following (+0.041 [+0.005, +0.080]), so a long
teacher prompt alone does not produce the gain.

## What it costs

| measure | adapter, no document | document in context |
|---|---|---|
| MoReBench (open moral reasoning, 0-100 rubric) | −0.98 [−1.49, −0.46] | −4.00 [−4.65, −3.36] |
| Capability (GSM8K and MMLU, n 800, bar: not wholly below −0.02) | −0.003 [−0.014, +0.009] | |
| Compliance with harmful requests (60 items, no rise detectable at 0.12) | +0.011 [−0.071, +0.092] | |

Two costs come with the gain. On held-out free-form prompts the adapter follows legitimate orders
less by about as much as improper ones (−0.078 [−0.167, 0.000] and −0.069 [−0.138, 0.000], both
intervals ending at 0). On cell A′, where the rule is the right call, it breaks the rule more often
than the teacher does (a shift of +0.174 [+0.118, +0.233] against the teacher's +0.088).

## What it does not do

- **Goal pressure.** In MACHIAVELLI, re-deciding the base model's own game states one decision at
  a time, no change in the violating-choice rate is detectable at 0.019 (−0.006 [−0.020, +0.007]),
  where a short ethics prompt lowers it by 0.093 on the same states. The document in context does
  not lower it either (−0.005), and its length cancels the ethics prompt (below).
- **Content beyond the pressure disposition.** No gain of the document's content is shown. A
  variant's content-cell gain did not survive a second training draw (0.000 [−0.061, +0.070]), and
  on MoReBench the adapter and its neutral-text control do not differ detectably (at 0.81).

## The strength knob

The adapter's update can be served at a fraction of its strength (its B matrices scaled).

| strength | share of the in-context gain | rule-right shift on cell A′ beyond the teacher's |
|---|---|---|
| 0.25 | 0.44 [0.35, 0.52] | −0.08 |
| 0.5 | 0.69 [0.61, 0.76] | −0.03 |
| 0.75 | 0.82 [0.76, 0.89] | +0.03 |
| 1.0 | 0.90 [0.84, 0.95] | +0.09 |

These are descriptive. For a new adapter the registered rule picks the largest strength whose cell
A′ shift does not lie wholly above 0.03 and whose gain share has a lower bound of at least 0.5, then
confirms it on fresh rollouts.

## A finding about long documents in context

A 35k-token document in the system prompt cancels a short instruction placed after it. The ethics
prompt's −0.100 in MACHIAVELLI falls to −0.019 behind the constitution and to −0.014 behind a
neutral text of the same length, so the length does it, whatever the content. On MoReBench half
of the in-context cost is likewise the length (the neutral text alone costs −1.99). Serving a long
constitution in context therefore weakens the system and operator instructions that follow it.
Distillation removes the long prefix.

## Limits

One model, one document and one family of pressure scenarios. The MACHIAVELLI read is of single
decisions, not whole trajectories. Nothing here has been read on the partner's document, whose work
starts with a headroom read of where that document changes the model's choices.
