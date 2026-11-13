# A headroom read for any constitution: design and price

2026-11-13. Zero GPU, nothing built or registered (CST Amendment 47 item 7). For the partner's
document; not to be run on `C_anth` (its in-context content gain, +0.096 on a selected subset, leaves
too little room for a two-draw test).

## What it is for

Context distillation moves what the document does in context into the weights, so it can only move
a gain the document has in context. The headroom read finds, for a new document, the tenet groups
where the document in context changes the subject's plain choices against the base, and only those
groups' scenarios become the kind that distillation trains on. It is the line gate of Amendment 47
item 5's standing rule (a kind enters a training mix only after a gate shows the teacher differs
from the base on it), built before any scenario is written for training.

## Design

1. **The document's tenets.** The document is decomposed into tenets with verbatim quotes and a
   kind (priority, directive, prohibition), with a second model family's review, as `C_anth` was at
   CST-G0 (`data/tenets_C_anth_v2.json`). Tenets are grouped before any data, by kind and by topic
   (a group holds about 20 tenets), and the grouping is registered with the read.
2. **Plain scenarios, written by the subject.** For each tenet, m scenarios that the subject writes
   from a brief: an everyday situation the tenet bears on, with no operator order and no pressure,
   and two or three lettered options of which one is the option the tenet directs. The checks are
   the ones CST's writers already use: no lettered option names the tenet or the document, lengths
   in range, no 8-gram shared with the evaluation index (`outputs/_sync/eval_8gram_hashes_v1.npy`), no
   5-gram Jaccard above 0.35 with any panel, cell or held-out set or with an earlier scenario. **Every
   written scenario is read**, so no scenario is selected on what the base chose (the content cell's
   selection on A0's first rollouts needed a later-rollout reference to undo the regression; this
   read needs none).
3. **The read.** The document in context (A0s's render: the header and the document first in the
   system message) against the base, k 4 each, the share of rollouts taking the directed option,
   paired by scenario.
4. **The line gate per tenet group:** the group's document-minus-base directed share, CI by
   bootstrapping tenets within the group. **A group passes when its lower bound exceeds 0 and its
   point is at least twice the two-draw test's MDE** (the room the later distillation test needs),
   and the base's share in the group is below 0.85 (a ceiling leaves no room to train toward).
5. **Confirmation before training.** The passing groups get a fresh set of scenarios from the same
   writer and checks; the gate is re-read on it, and a group trains only if it passes again (two
   draws, as Amendment 47 item 5 asks of recipe arms). The confirmation set is never trained on; it
   stays as that kind's evaluation cell.
6. **Output.** The passing groups' tenets and the brief, from which the training kind is written
   (fresh scenarios again, disjoint from both read sets), and the per-group teacher gains that the
   distilled arm is later read against.

## Power

From the content cell's confirmation (A0s against A0's later rollouts, 57 scenarios in 36 tenets
at k 4: +0.096 [+0.024, +0.171], an SE of about 0.037), one scenario's document-minus-base share has
an SD of about 0.28. A group's MDE at k 4 is about 2.80 x 0.28 / sqrt(n scenarios):

| scenarios in the group | MDE | gain needed to pass (twice the MDE) |
|---|---|---|
| 40 (20 tenets x m 2) | 0.124 | about 0.25 |
| 80 (20 tenets x m 4) | 0.088 | about 0.18 |
| 160 (40 tenets x m 4) | 0.062 | about 0.12 |

The SD comes from a selected subset of one document; a pilot of two groups gives the document's own
SD before the full read is priced.

## Price

Rates measured on the CST engine: the document in context at 286-335 tokens/s (32 sequences, a
~35k-token prefix; a shorter document runs faster) with about 1,100 tokens per rollout on plain
lettered scenarios (the content cell), the base at about 1,050 tokens/s with about 660 tokens per
rollout. Per tenet at m 4 and k 4 (16 rollouts per arm): about 0.016 h for the document arm and
0.003 h for the base, plus writing (about 0.001 h per scenario). Plus 0.5 h of pod overhead:

| tenets | the gate read | with the confirmation draw (its own pod) |
|---|---|---|
| 100 | about 2.8 A100-h | about 4.0 A100-h (a third of the groups passing) |
| 200 | about 5.0 A100-h | about 7.0 |
| 400 (`C_anth`'s size) | about 9.5 A100-h | about 13.0 |

The partner's document's size sets which row applies; a document much shorter than `C_anth` cuts
the document arm's cost roughly in proportion to its prefix.

## Anticipated review

1. **A subject writes scenarios it finds easy.** Its own scenarios may sit where the base already
   agrees with the document, which understates headroom. → A second writer (another model family)
   on two groups as a sensitivity check, about 0.3 A100-h.
2. **Many groups, one gate each.** A group passing by chance would train on noise. → The
   confirmation draw (item 5) is the guard; the expected false passes at a lower bound above 0 are
   about 2.5% of groups per draw, about 0.06% across both.
3. **The directed option is the tenet's, as the writer read it.** A writer that misreads a tenet
   produces a directed option the document does not direct. → The two-family labeling CST already
   uses (`label_v4` under two groupings) on each group's scenarios before the read, zero GPU.
4. **Plain scenarios are not operator pressure.** The gate finds a content kind, not a pressure
   kind; P-ctx's gain came from pressure scenarios. → Report both: the read is for the content the
   document adds, and the pressure line's existing cells stay the pressure kind's gate.
