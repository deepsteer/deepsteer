# Constitution Distillation: First Results

Draft for the author's review, 2026-11-15.

## The goal and where this fits

The aim is a model that holds the values a constitution sets out across the situations it meets in deployment: what businesses instruct it to do, what users ask in chat, and the cases where the two pull apart. This page reports the first step toward that, and the method it rests on.

The obvious way to give a model a constitution is to put the document in its system prompt. That works to a degree, but the document we tested is about 35,000 tokens long, so every request carries it, and (as a later section shows) a prompt that long weakens the instructions placed after it. We tested whether the model can learn to behave as if the document were present, with the document removed.

Everything here comes from one open model (Qwen3.8-27B, with its reasoning turned on) and one public constitution. Rates are percentages, and changes in a rate are in percentage points. Numbers in brackets are 95% intervals.

## The method

We collect the model's answers to a set of scenarios with the constitution in its system prompt, then train a small add-on to its weights (a LoRA adapter, rank 16, 200 steps over 797 of those answers) to give the same answers without the document. At use time the adapter is switched on and the document is left out, a technique known as context distillation.

## The first test: operator pressure

We started with a direct test of a constitution: an operator (the business deploying the model, writing its system prompt) orders something the constitution says not to do. We built 458 such scenarios. Each offers a few choices, one of which is to follow the order.

The base model follows the conflicting order in 34% of its answers. With the constitution in its prompt it follows in 3%. With the adapter and no constitution it follows in 7%, so the adapter keeps 86% to 90% of the improvement the document gives (86% [80%, 91%] on its first read, 88% in a second session, 90% on a second GPU provider). Training a second copy with a different seed gives the same drop (28.6 points against 27.5), and the second provider reproduces it against its own baseline (a drop of 28.3 points [24.8, 31.8]).

To check that the gain comes from the constitution, we ran the same recipe with a neutral text of the same length in the prompt (lectures on the chemistry of a candle). That adapter follows the orders slightly more than the base model does (+4.1 points [+0.5, +8.0]), so a long prompt alone does not produce the gain.

## What it costs

| Measure | Adapter, no document | Constitution in the prompt |
|---|---|---|
| Moral reasoning on open dilemmas (MoReBench, 500 dilemmas, score out of 100) | −0.98 [−1.49, −0.46] | −4.00 [−4.65, −3.36] |
| General capability (accuracy on 800 math and knowledge questions) | −0.3 points [−1.4, +0.9] | |
| Compliance with harmful requests (60 items, a rise of 12 points would have shown) | +1.1 points [−7.1, +9.2] | |

The adapter costs about one point of moral-reasoning score against four for the document in the prompt. No capability loss and no rise in harmful compliance is detectable.

It also over-corrects in two ways. On new free-form prompts it follows legitimate operator orders less, by about as much as improper ones (−7.8 points [−16.7, 0.0] and −6.9 points [−13.8, 0.0]). Both intervals just reach zero, so neither drop is settled. And in scenarios where following a rule is the right call, it breaks the rule more often than the model with the document in its prompt does (+17.4 points [+11.8, +23.3] against +8.8), so about half of its shift goes beyond what the document in the prompt does.

## What it does not do yet

The adapter does not yet hold values under pressure from a goal. We used MACHIAVELLI, a set of text-adventure games where the model pursues a goal and can take unethical shortcuts, and re-asked the model at decision points from its own play. With the adapter, no change in how often it picks the unethical choice is detectable: the effect is −0.6 points [−2.0, +0.7], and we could have detected one as small as 1.9 points. A short ethics instruction in the prompt lowers it by 9.3 points at the same decisions. The constitution in the prompt does not lower it either (−0.5 points), and a later section shows that a prompt this long cancels even the short ethics instruction.

Nor does it yet carry the document's content beyond resisting conflicting orders. In our tests the adapter did not detectably change the model's choices on everyday questions the constitution speaks to. A variant's gain on such questions did not survive a second training run (0.0 points [−6.1, +7.0]), and on moral reasoning no difference between the adapter and its neutral-text control is detectable at 0.81 points.

## A strength dial

The adapter can be applied at part strength, and weaker settings keep less of the improvement and over-correct less. At 25%, 50%, 75% and 100% strength it keeps 44% [35%, 52%], 69% [61%, 76%], 82% [76%, 89%] and 90% [84%, 95%] of the improvement, with −8, −3, +2 and +9 points of extra rule-breaking where the rule is right, beyond the document's own. These numbers describe this one adapter. For each new adapter we will pick a strength by a fixed rule (keep at least half the improvement, with little extra rule-breaking) and confirm the choice on a fresh run.

## What we learned about long prompts

A 35,000-token document in the system prompt cancels a short instruction placed after it. In the games, the ethics instruction lowered unethical choices by 10.0 points on its own, by 1.9 points with the constitution in front of it, and by 1.4 points with the candle lectures in front of it. Two very different documents of the same length had the same effect, so the length does it. On moral reasoning, half of the constitution's in-prompt cost is also the length (the candle lectures alone cost 1.99 points). A long constitution in the prompt weakens the operator and system instructions that follow it, and training the behavior into the model avoids that.

## Next steps with your constitution

Operator pressure was the first, narrow test. The goal is a model that holds your constitution across many operator and chat situations, and three planned steps lead there.

First, a headroom read on your document: for each group of its principles, we write ordinary scenarios with no pressure and measure whether putting the document in the prompt changes the model's choices. Only the groups where it does are worth training, since training can only move behavior the document actually changes.

Next, training on those groups' scenarios with the same recipe. Each result has to hold on two independent training runs before it counts.

Last, applying the strength dial and the checks from this page (moral reasoning, capability, safety, over-correction) to every new adapter.

Goal pressure needs a different approach. A long document in the prompt did not carry it in our tests, so it needs a short conduct text or a different teacher, and that is open work.

## Limits

One model, one constitution and one family of conflicting-order scenarios so far. The game results re-ask single decisions and do not replay whole games. Nothing has yet been run on your document.
