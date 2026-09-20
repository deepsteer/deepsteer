# 8. Limitations {#limitations}

**One model.** Every number is on OLMo-3-7B-Instruct and its base checkpoint. The
pre-registration names a second tier (Qwen2.5-7B, Llama-3.1-8B) and a deliberation-dose arm
(answer immediately, think briefly, reason carefully, with a length-matched filler control);
neither was run, for budget, and nothing here speaks to whether the gap generalizes across
families or moves under deliberation. Huang et al. [@huang2026knowing] report near-perfect
cross-model agreement on enacted choices; if that holds, a cross-model table of this gap would
show little variance along the model axis, and the family structure, which is flat here, would
be the only live quantity.

**The reference is the model's own judgment.** What is measured is the model contradicting its own judgment, not
wrongness by an external standard. External labels are recorded as a covariate and agree with
the construction on 262 of 288 primaries, with every disagreement but three a preference for the
neutral option; the paper makes no claim about which option is right.

**The binary readout is under-powered at the strictest reference**, and the continuous readout
that resolves it is a secondary instrument by pre-registration. The two agree everywhere the
first has power, and the coherence check passed, but a reader who weights only the primary
instrument should read the strictest-level result as unresolved at 43 pairs.

**Family power.** With 17 to 25 screened primaries per family the contrast MDE is about 0.40;
structure smaller than that is not excluded. The gate the panel cleared is a screening gate,
not a structure gate.

**F4.** The one generator-dependent family was resolved as a difference of degree by three
zero-GPU legs and an amended reversal clause; the swap cell that would have separated
construction from register cleanly was too small, and the amendment is a post-hoc change with
both verdicts recorded. A reader may prefer the non-F4 panel, which meets the original gate.

**Scenario provenance.** All scenarios were written by two large models under one prompt; the
pressures they contain are the pressures those models think of. The harm-matched twins and the
cross-labeling bound some construction biases, not all.

**No causal cell.** The panel measures a behavioral gap; it does not intervene on
representations. The mechanistic cells the pre-registration schedules (the rank of the moral
read at the action position; persona steering as a lever) are what this gap exists to score,
and they are not in this paper.
