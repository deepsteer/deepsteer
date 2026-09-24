# 9. Limitations {#limitations}

**One model, one lineage.** Every number is on OLMo-3-7B-Instruct and its base checkpoint. The
pre-registration names a second tier (Qwen2.5-7B, Llama-3.1-8B) and a deliberation-dose arm;
neither was run, for budget, and nothing here speaks to whether the gap generalizes across
families or moves under deliberation. Huang et al. [@huang2026knowing] report near-perfect
cross-model agreement on enacted choices; if that holds, a cross-model table of this gap would
show little variance along the model axis, and the family structure, which is flat here, would be
the only live quantity. The base-versus-instruct result is likewise one lineage's post-training
recipe; a second base-and-instruct pair is the cheapest test of whether "lowers the baseline,
raises the pressure sensitivity" is a property of alignment or of OLMo-3's.

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
change with both verdicts recorded. Of the seventeen amendments, eleven precede any model data,
four were committed before the computation they license, and two are post-hoc forks; a reader may
prefer the non-F4 panel, which meets the original gate.

**Labels and readers.** The scenarios were written by two large models under one prompt, so the
pressures they contain are the pressures those models think of; the harness judges are language
models from two providers, not humans; and the blind read of F4 is a single reader who knew the
hypothesis even with labels and origins hidden.

**No causal cell.** The panel measures a behavioral gap; it does not intervene on
representations. The mechanistic cells the pre-registration schedules (the rank of the moral read
at the action position; persona steering as a lever) are what this gap exists to score, and they
are not in this paper.

**What would change these results, and what it costs.** Each open reading above has a priced
discriminator. A letter-only judgment readout under the chat template on the pressure-removed
twins (one cell, about five minutes on a loaded OLMo-3) separates an installed agent-frame
caution from a raw-frame artifact. The deliberation-dose arm with its filler control (three arms
on the 136 screened scenarios, about ninety minutes on one A100) separates goal-following from a
moral read. Sixteen more F3 and F5 scenarios per generator (one generation batch and about an
hour of GPU) bring the exploratory family excesses to a family MDE near 0.10 on the continuous
instrument. The second tier (two models, about nine A100-hours for the full ladder) tests the
cross-model rival, and a second base-and-instruct pair (about two hours) tests whether the
post-training signature is OLMo-3's. None of these needs a new instrument.
