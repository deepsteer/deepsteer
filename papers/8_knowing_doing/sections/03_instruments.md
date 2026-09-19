# 3. Instruments: harness, screen, ladder, and two readouts {#instruments}

## 3.1 Harness and calibration

One parser and one classifier (`kdg_harness` 1.0.0) serve every cell, so no cell can drift onto
a private classifier. The parser resolves a reply to an option by a fixed ladder of rules
(answer line, bare letter, tool name, leading letter, unique mention, contrastive commitment)
and leaves refusals, hedges, and unmarked two-option mentions unparsed rather than guessed. It
was calibrated in two stages: on 200 synthetic reply formats built on the panel's own scenarios,
where it agreed with the construction labels on all 200 and with an independent judge on 197; and,
before any verdict, on 200 real replies from each pod, oversampling every reply the parser had
not resolved by a clean rule, against two independent judges (one Claude, one GPT). Agreement
was 0.99 and 1.00 with the judges and 0.99 between them on the full panel, with the harness
the conservative side of every disagreement (\Cref{app:calibration}). On the pods of record the
parse rate was 0.9997 on 10,240 action rollouts and 1.000 on 2,880 judgment replies.

## 3.2 Screen and gates

The panel is useful only where pressure engages. A scenario passes the **screen** if its
judgment is stable and its violating fraction across the 32 action rollouts lies in
$[0.15, 0.85]$ (mixed outcomes), or exceeds 0.85 with a stable norm-consistent judgment (a clean
gap); scenarios the model never violates (no pressure) or never judges stably (no reference)
are dropped and counted. The pre-registered **pilot gate** required at least 14 of 48 gate-family
primaries to pass with at least two families contributing; the **full gate** requires at least 60
screened primaries across at least three gate families, a family whose gap CI excludes zero,
harness agreement of at least 0.95, and no family whose result reverses across generator. The
screen's binomial properties at 32 rollouts (a true 0.10 scenario lands in the mixed band one
time in five) are why the panel's unit of inference is the scenario-level bootstrap, not the
screen label.

## 3.3 The calibration ladder

No gap or absence of a gap is stated bare. Every headline sits inside four rungs measured on the
same instrument and model. The **floor** is re-elicitation noise on the reference: the greedy
judgment against the greedy judgment on a paraphrased frame. The **matched null** is the gap
statistic on the pressure-removed twins: what frame change alone, with no incentive, produces.
The **measurement** is the statistic on the screened panel. The **positive band** is the
statistic under a known-gap control in which the system prompt orders the violating action:
what the instrument returns when a gap is forced. Verdict sentences carry their rung and their
detection bar; comparisons are paired difference CIs on the same scenarios, never CI overlap.

## 3.4 Reference strictness

A self-referenced gap is only as good as its reference. The judgment reference is therefore
elicited on four third-person frames (the original and three paraphrases) and the gap is read
at three strictness levels: **L0**, the original frame with sampled stability (the
pre-registered primary rule); **L1**, plus agreement of the four frames by majority on the
violating/non-violating binary; **L2**, plus all four frames naming the same option. The verdict
sentence is stated at the strictest level with at least 40 paired scenarios, a bar chosen from
the pilot's bootstrap width. Both branches were written before data: a gap that survives the
strictest level, or one that decays toward the null as the reference tightens, which would mean
the pilot's gap was reference noise.

## 3.5 Two readouts

The **binary readout** is the majority-vote gap of \Cref{panel}. Because every rollout
saved the full next-token distribution at its decision position, a **continuous readout** was
pre-registered as a secondary instrument (amendment A15): $p_D(s)$ is the violating option's
normalized probability mass at the action position, averaged over the 32 order-randomized
rollouts; $p_J(s)$ the same at the judgment position over the four frames; the continuous gap
is $g(s)=p_D-p_J$. It was admitted only after a coherence check it could have failed:
thresholding $p_D$ at 0.5 reproduces the majority action on 95.5% of scenarios (bar 95%), and
its mean on the screened panel (0.436) matches the rollout-level violating fraction (0.444)
within 0.01. The binary readout keeps primacy in every table; the continuous one is reported
beside it, instrument named, and no sentence uses "established" without saying which instrument.
