# 3. Instruments: harness, screen, ladder, and two readouts {#instruments}

## 3.1 Harness and calibration

One parser and one classifier (`kdg_harness` 1.0.0) serve every cell, so no cell can drift onto
a private classifier. The parser resolves a reply to an option by a fixed ladder of rules (answer
line, bare letter, tool name, leading letter, unique mention, contrastive commitment) and leaves
refusals, hedges, and unmarked two-option mentions unparsed rather than guessed. It was calibrated
in two stages: on 200 synthetic reply formats built on the panel's own scenarios, where it agreed
with the construction labels on all 200 and with an independent judge on 197; and, before any
verdict, on 200 real replies from each pod, oversampling every reply the parser had not resolved
by a clean rule, against two judges from different providers (one Claude, one GPT). Agreement
was 0.99 and 1.00 with the judges and 0.99 between them on the full panel, with the harness the
conservative side of every disagreement (\Cref{app:calibration}). The judges are language
models; the only human labels in the panel are the author's blind read of one family
(\Cref{structure}). On the pods of record the parse rate was 0.9997 on 10,240 action rollouts
and 1.000 on 2,880 judgment replies.

**Harness corrections.** Two defects in the panel harness were found after the first version was
posted and corrected; neither changes a verdict of record. First, the batched forward pass that reads the
letter distribution did not pass position ids, so a left-padded prompt was read at positions shifted by
its pad count. Rotary embeddings cancel the shift in exact arithmetic but not at 16-bit precision. With
positions derived from the attention mask (readout version 2, which matches generation exactly on all four
panel models), the original readout's per-row error stayed within the 0.05-nat validation bar on OLMo-3,
Llama-3.1 and Tulu 3 (0.016 to 0.031) and reached 0.094 on Qwen2.5, whose re-read left the excess unchanged
(paired difference +0.0002, $-0.0001$ to +0.0004); no padding contrast aligned with the twin design on any
model. Second, the cells that sample (the reasoning traces of the dose arms and free replies) were
generated one scenario at a time with a single seed, so rollout $k$ of every scenario drew from the same
random stream, and sampling applied each model's default nucleus on top of the stated temperature. Under a
two-way bootstrap over scenarios and rollout indices, which treats those shared streams as a second source
of variance, the intervals of the sampled cells widen by 10 to 48%, the point estimates are unchanged, and
three verdicts move: OLMo-3 against its own truncated filler goes from branch (a) to branch (b)
(\Cref{deliberation}), and the binary paired excess over the twins at L0 and at L1 now has its lower bound
at zero (\Cref{gap}, \Cref{reference}); the continuous readout, read before any token is sampled, is
unaffected. Two-way intervals are the intervals of record for sampled cells in v2; the scenario-only
intervals they replace are listed in \Cref{app:calibration}. The harness now seeds each scenario separately and samples at pure
temperature.

## 3.2 Screen and gates

The panel is useful only where pressure engages. A scenario passes the **screen** if its
judgment is stable and its violating fraction across the 32 action rollouts lies in
$[0.15, 0.85]$ (mixed outcomes), or exceeds 0.85 with a stable norm-consistent judgment (a clean
gap); scenarios the model never violates (no pressure) or never judges stably (no reference)
are dropped and counted. On the panel of record 74 of 208 gate-family primaries pass; 38 show no
pressure, 22 are violated and judged violating (no contradiction), and 74 have an unstable
judgment (\Cref{app:pods}). The pre-registered **pilot gate** required at least 14 of 48
gate-family primaries to pass with at least two families contributing; the **full gate** requires
at least 60 screened primaries across at least three gate families, a family whose gap CI
excludes zero, harness agreement of at least 0.95, and no family whose result reverses across
generator. The screen's binomial properties at 32 rollouts (a true 0.10 scenario lands in the
mixed band one time in five) are why the panel's unit of inference is the scenario-level
bootstrap, not the screen label. The screen selects on the model's own actions, so every rate on the
screened set is a rate on an outcome-selected subset, not a prevalence on the panel; the claims that do not
depend on that selection are the paired excess over the same scenarios' twins and the whole-panel readouts
of \Cref{recipe}. On OLMo-3 the screened set is 136 scenarios on this sampled-rollout
screen, 110 on the letter-only screen of \Cref{recipe}, and 130 in the dose arm (the 136 less the six
multi-turn F2 scenarios, which have no dose arm); each table names which.

## 3.3 The calibration ladder

No gap or absence of a gap is stated bare. Every headline sits inside four rungs measured on the
same instrument and model. The **floor** is re-elicitation noise on the reference: the greedy
judgment against the greedy judgment on a paraphrased frame. The **matched null** is the gap
statistic on the pressure-removed twins: what frame change alone, with no incentive, produces.
On the chat cells that frame change includes a deliberation asymmetry, because the judgment is
read after the model's own reasoning and the action is read immediately, so the null rung is not
small and is not expected to be; the number the paper uses is the paired excess of the
measurement over it. The **measurement** is the statistic on the screened panel. The
**positive band** is the statistic under a known-gap control in which the system prompt orders
the violating action: what the instrument returns when a gap is forced. On OLMo-3 the chat
template injects a default system prompt when none is given, and the known-gap control replaces
it, so the band differs from the measurement in the presence of an operator prompt as well as in
its content; the band bounds the instrument's range, it is not a matched contrast. Verdict
sentences carry their rung and their detection bar; comparisons are paired difference CIs on the
same scenarios, never CI overlap.

## 3.4 Reference strictness

A self-referenced gap is only as good as its reference. The judgment reference is therefore
elicited on four third-person frames (the original and three paraphrases) and the gap is read at
three strictness levels: **L0**, the original frame with sampled stability (the pre-registered
primary rule); **L1**, plus agreement of the four frames by majority on the violating versus
non-violating binary; **L2**, plus all four frames naming the same option. The verdict sentence
is stated at the strictest level with at least 40 paired scenarios, a bar chosen from the
pilot's bootstrap width. Both branches were written before data: a gap that survives the
strictest level, or one that decays toward the null as the reference tightens, which would mean
the pilot's gap was reference noise.

## 3.5 Two readouts

The **binary readout** is the majority-vote gap of \Cref{panel}. Because every rollout saved the
full next-token distribution at its decision position, a **continuous readout** was
pre-registered as a secondary instrument: $p_D(s)$ is the violating option's normalized
probability mass at the action position, averaged over the 32 order-randomized rollouts;
$p_J(s)$ the same at the judgment position over the four frames; the continuous gap is
$g(s)=p_D-p_J$. The amendment was committed after the majority readout's strictest-level result
was known to be unresolved and before any saved vector was read, with a coherence check that
could have voided the instrument: thresholding $p_D$ at 0.5 reproduces the majority action on
95.5% of scenarios (bar 95%), and its mean on the screened panel (0.436) matches the
rollout-level violating fraction (0.444) within 0.01. The binary readout keeps primacy in every
table; the continuous one is reported beside it, instrument named, and no sentence uses
"established" without saying which instrument. In the raw frame, where both models are read by
log-probability, the continuous statistic is the natural one and the argmax majority is its
binary shadow.

## 3.6 Estimands

\Cref{tab:estimands} states each quantity the paper reports, per scenario, with how ties, neutral options and
unstable judgments are handled.

| estimand | per scenario | ties, neutral options, unstable judgments | reported as |
|-------|------------|------------|----------|
| binary directional gap | 1 if the stable judgment names a non-violating option and most of the 32 action rollouts take the violating one; 0 if the action majority matches the judgment | a neutral option counts as non-violating on both sides; undefined if the judgment is unstable (the greedy option does not hold a majority of the 8 sampled judgments), the action is unparsed, the violating fraction is exactly 0.5, or the action is non-violating against a violating judgment | the mean over defined scenarios (a rate on the screened, outcome-selected subset), and its paired excess over the same scenarios' twins |
| continuous gap $g$ | $p_D - p_J$: the violating option's renormalized letter mass at the action position (32 order permutations) minus the same at the judgment position (four frames) | read before any token is sampled, so no rollout noise enters; neutral options carry their own mass and are not violating | the mean over scenarios, and its paired excess over the twins |
| pressure-attributable excess $E$ | $(p_D - p_J) - (p_{D,\mathrm{twin}} - p_{J,\mathrm{twin}})$, letter-only readout under each model's template, eight option orders | scenarios kept when the option mass is at least 0.5 in all four cells | the mean over the 586 scenarios screened by no model, or over a named screened set |

Table: The three estimands. Action and judgment are always closed-menu letter choices, posed in separate
calls. {#tab:estimands}
