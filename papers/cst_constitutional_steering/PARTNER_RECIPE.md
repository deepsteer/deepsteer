# Constitution Distillation: First Results

Draft for the author's review, 2026-11-15.

## The goal and where this fits

The aim is a model that holds the values a constitution sets out across the situations it meets in
deployment: what businesses instruct it to do, what users ask in chat, and the cases where the two
pull apart. This page reports the first step toward that, and the method it rests on.

The obvious way to give a model a constitution is to put the document in its system prompt. That
works to a degree, but the document we tested is about 35,000 tokens long, so every request carries
it, and (as the last section shows) a prompt that long weakens the instructions placed after it.
We tested whether the model can learn to behave as if the document were present, with the
document removed.

Everything here comes from one open model (Qwen3.8-27B, with its reasoning turned on) and one
public constitution. Numbers in brackets are 95% intervals.

## The method

We give the model the constitution in its system prompt and collect its answers to a set of
scenarios. Then we train a small add-on to the model's weights (a LoRA adapter: rank 16, alpha 32,
learning rate 1e-4, 200 steps over 797 of those answers, about 1.9M tokens) to give those answers
without the document. At use time the adapter is switched on and the document is left out. This is
known as context distillation.

## The first test: business instructions that conflict with the constitution

We started with a direct test of a constitution: an operator (the business deploying
the model, writing its system prompt) orders something the constitution says not to do. We built
458 such scenarios. Each offers a few choices, one of which is to follow the order.

The base model follows the conflicting order in 0.34 of its answers. With the constitution in its
prompt it follows in 0.03. With the adapter and no constitution it follows in 0.07, so the adapter
keeps 0.86 to 0.90 of the improvement the document gives (0.86 [0.80, 0.91] on its first read, 0.88
in a second session, 0.90 on a second GPU provider). Training a second copy with a different seed
gives the same drop (−0.286 against −0.275), and the second provider reproduces it against its own baseline
(−0.283 [−0.318, −0.248]).

To check that the gain comes from the constitution, we ran the same recipe with a neutral text of
the same length in the prompt (lectures on the chemistry of a candle). That adapter follows the
orders slightly more than the base model does (+0.041 [+0.005, +0.080]), so a long prompt alone
does not produce the gain.

## What it costs

| measure | adapter, no document | constitution in the prompt |
|---|---|---|
| Moral reasoning on open dilemmas (MoReBench, 500 dilemmas, 0-100 rubric) | −0.98 [−1.49, −0.46] | −4.00 [−4.65, −3.36] |
| General capability (math and knowledge questions, 800 items) | −0.003 [−0.014, +0.009] | |
| Compliance with harmful requests (60 items, a rise of 0.12 would have shown) | +0.011 [−0.071, +0.092] | |

The adapter costs about one point of moral-reasoning score against four for the document in the
prompt. No capability loss and no rise in harmful compliance is detectable.

It also over-corrects in two ways. On new free-form prompts it follows legitimate operator orders
less, by about as much as improper ones (−0.078 [−0.167, 0.000] and −0.069 [−0.138, 0.000]). Both
intervals just reach zero, so neither drop is settled. And in scenarios where following a rule is
the right call, it breaks the rule more often than the model with the document in its prompt does
(a shift of +0.174 [+0.118, +0.233] against +0.088), so about half of its shift goes beyond what
the document in the prompt does.

## What it does not do yet

The adapter does not yet hold values under pressure from a goal. We used MACHIAVELLI, a set of
text-adventure games where the model pursues a goal and can take unethical shortcuts, and re-asked
the model at decision points from its own play. With the adapter, no change in how often it picks
the unethical choice is detectable: the effect is −0.006 [−0.020, +0.007], and we could have
detected one as small as 0.019. A short ethics instruction in the prompt lowers it by 0.093 at the
same points. The constitution in the prompt does not lower it either (−0.005), and the last section
shows that a prompt this long cancels even the short ethics instruction.

Nor does it yet carry the document's content beyond resisting conflicting orders. In our tests
the adapter did not detectably change the model's choices on everyday questions the constitution
speaks to. A variant's gain on such questions did not survive a second training run (0.000
[−0.061, +0.070]), and on moral reasoning no difference between the adapter and its neutral-text
control is detectable at 0.81 points.

## A strength dial

The adapter can be applied at part strength. Weaker settings keep less of the improvement and
over-correct less.

| strength | share of the improvement kept | extra rule-breaking where the rule is right, beyond the document's own |
|---|---|---|
| 0.25 | 0.44 [0.35, 0.52] | −0.08 |
| 0.5 | 0.69 [0.61, 0.76] | −0.03 |
| 0.75 | 0.82 [0.76, 0.89] | +0.03 |
| 1.0 | 0.90 [0.84, 0.95] | +0.09 |

These numbers describe this one adapter. For each new adapter we will pick a strength by a fixed
rule (keep at least half the improvement, with little extra rule-breaking) and confirm the choice
on a fresh run.

## What we learned about long prompts

A 35,000-token document in the system prompt cancels a short instruction placed after it. In the
games, the ethics instruction's effect of −0.100 fell to −0.019 with the constitution in front of
it, and to −0.014 with the candle lectures in front of it. Two very different documents of the
same length had the same effect, so the length does it. On moral reasoning, half of the constitution's in-prompt cost is also the length
(the candle lectures alone cost −1.99 points). A long constitution in the prompt weakens the
operator and system instructions that follow it, and training the behavior into the model avoids
that.

## Next steps with your constitution

Operator pressure was the first, narrow test. The goal is a model that holds your constitution
across many operator and chat situations, and three planned steps lead there.

First, a headroom read on your document: for each group of its principles, we write ordinary
scenarios with no pressure and measure whether putting the document in the prompt changes the
model's choices. Only the groups where it does are worth training, since training can only move
behavior the document actually changes.

Next, training on those groups' scenarios with the same recipe. Each result has to hold on two
independent training runs before it counts.

Last, applying the strength dial and checks from this page (moral reasoning, capability, safety,
over-correction) to every new adapter.

Goal pressure needs a different approach. A long document in the prompt did not carry it in our
tests, so it needs a short conduct text or a different teacher, and that is open work.

## Limits

One model, one constitution and one family of conflicting-order scenarios so far. The game results
re-ask single decisions and do not replay whole games. Nothing has yet been run on your document.
