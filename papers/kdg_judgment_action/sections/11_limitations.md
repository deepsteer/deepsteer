# 11. Limitations {#limitations}

**Four models, one panel.** The panel was written and screened on OLMo-3, and each model finds its own
pressuring scenarios (\Cref{recipe}). All four are 7--8B models and scale is not varied; nothing here
says what holds at larger sizes. Whether the pressures in this panel are the ones that matter for the
recipes that show no gap is a construct question the positive control does not answer. The positive
control shows each readout moves when the action changes; the detection bar for a small gap comes from
each null's own interval (about 0.01 on Tulu 3, 0.02 on Qwen2.5). On its own screened scenarios Tulu 3 shows an excess no larger
than the same screen produces on its pressure-removed twins; Qwen2.5's is unresolved against that null
(0.083, $-0.028$ to 0.195; $n = 47$, bar about 0.16). Resolving it needs more scenarios that engage Qwen2.5: at its measured screen rate (0.08), about
1,700 new scenarios would bring its screened set to about 185, where an excess of 0.083 would clear its
bar, plus about one A100-hour of letter-only forward passes (KDG-A19). The base cells are raw-frame readings,
descriptive by the bridge rule, so the same-base contrast says that the recipe decides whether the aligned
model carries the gap, not what either recipe did to the base. A recipe is a bundle of data, method and
template; which part decides is an ablation on one base, not a question this panel can answer. Huang et
al. [-@huang2026knowing] report near-perfect cross-model agreement on enacted choices; on this panel the
models disagree about whether the gap exists at all, which is a difference in construct (a self-referenced
gap under pressure, not a value profile) worth stating rather than resolving here.

**The dose arm is two results, not one.** OLMo-3's arm reads truncated reasoning (the model finishes
within the budget on 8% of rollouts) and Llama-3.1's reads largely completed reasoning (90%), on different
scenario sets (each model's own screen). Their sizes are therefore not compared. The norm-salience share is
measured on OLMo-3 only.

**The reference is the model's own judgment.** What is measured is the model contradicting its
own judgment, not wrongness by an external standard. External labels are recorded as a covariate
and agree with the construction on 262 of 288 primaries, with every disagreement but three a
preference for the neutral option; the paper makes no claim about which option is right. The
judgment is also a deliberated readout while the action is immediate; the twin absorbs that
asymmetry in every paired number, but the absolute rate and the absolute mass gap include it.

**The binary readout is under-powered where the effects are small.** At the strictest reference
and throughout the raw frame the majority readout does not resolve what the continuous readout
resolves. The continuous readout is a secondary instrument by pre-registration, admitted after a
coherence check and after the strictest-level binary result was known; a reader who weights only
the primary instrument should read the strictest-level chat result as unresolved at 43 pairs and
the raw-frame comparison as a point-estimate reading.

**Family power.** With 18 to 41 screened scenarios per family the binary contrast MDE is about
0.40; structure smaller than that is not excluded, and the continuous-instrument contrasts are
exploratory. The gate the panel cleared is a screening gate, not a structure gate.

**The raw frame is format-invalid for the instruct model on 155 of 397 scenario-frames.** The
missing mass is on the chat end-of-turn token, not on refusal or hedge tokens, so the shared
subset is the set of scenarios the instruct model engages without its template. The selection
check bounds the consequence for base's number; it cannot say what the instruct model would do
on the scenarios it declined to engage.

**F4 and the amendment count.** The one generator-dependent family was resolved as a difference
of degree by three zero-GPU legs and an amended reversal clause; the swap cell that would have
separated construction from register cleanly was too small, and the amendment is a post-hoc
change with both verdicts recorded. Of the seventeen panel amendments, eleven precede any model data,
four were committed before the computation they license, and two are post-hoc forks. Of the fourteen
Phase 1 amendments, six precede the data they govern, four precede the computation they license, three
are post-hoc forks (P1-A1, P1-A5, P1-A14), and one is a post-hoc addition (P1-A13, written after P1-A11's result and
pushed before its own computation). A reader may prefer the non-F4 panel, which meets the original gate.

**Labels and readers.** The scenarios were written by two large models under one prompt, so the
pressures they contain are the pressures those models think of; the harness judges are language
models from two providers, not humans; and the blind read of F4 is a single reader who knew the
hypothesis even with labels and origins hidden.

**No causal cell.** The panel measures a behavioral gap; it does not intervene on
representations. The mechanistic cells the pre-registration schedules (the rank of the moral read
at the action position; persona steering as a lever) are what this gap exists to score, and they
are not in this paper.

**What would change these results, and what it costs.** Each open reading has a priced discriminator.
The pressure-removed twins under deliberation (P1-A12, run for this version) leave one question open: a
pressure-specific effect of reasoning smaller than the ratio bars (about a fifth on OLMo-3, a quarter on
Llama-3.1). Narrowing it needs more scenarios per model (the bars shrink as $1/\sqrt{n}$, and OLMo-3's
screen holds 136) or a reasoning arm that names the incentive itself.
A like-for-like dose run, both carrying models on one scenario set with a budget long enough for both to
finish (about three GPU-hours), says whether completed reasoning does more than truncated reasoning and
licenses any side-by-side statement of the two dose effects. Extending the salience and truncation controls
to the full union (about two GPU-hours) narrows the salience share. The recipe question, which part of a
recipe decides whether the gap survives, needs a same-base ablation (one base, recipes that differ in one
component at a time) and is the subject of separate work. None of these needs a new instrument.
