# 6. Conclusion

We tested the Wang et al. (2025) toxic-persona mechanism — observed
at the 32B scale with SAE-decomposed feature sets — at the OLMo-2 1B
scale with a linear probe, a paired secure-code control, and a
matched training-time intervention primitive. The mechanism, the
measurement, and the intervention all engage as designed; the
behavioral phenomenon Betley et al. (2025)'s recipe is supposed to
induce does not. The persona-probe direction and persona-voice
behavior decouple at 1B both at training time (§4.1) and under
training-time direction-suppression (§4.3). The recipe is not
mechanistically inert (§4.4 fragility-locus signature) — it just
operates on representational structure that does not project onto
the linear direction the probe extracts. We name this a **compound
scaling boundary** because the failure is in the coupling between
otherwise-functional components, not in any one of them.

This paper closes on two falsifiable Phase E predictions; both
follow directly from the §5 reading and both are within the
open-checkpoint compute envelope of a single workstation plus
access to a public SAE.

**Phase E prediction 1: Coupling returns at 7B.** Repeating the §3.2
controlled insecure-code LoRA replication on OLMo-3 7B (or another
open-checkpoint 7B base with a documented training trajectory)
should close the persona-probe / behavioral-EM gap if Wang et al.
(2025)'s mechanism engagement scales with model size. The specific
prediction is that the C10 v2 paired delta in probe activation will
exceed Cohen's *d* = 1.0 (the PROBE PASS threshold from this paper),
and that the behavioral coherent-misalignment rate will exceed 5 %
with non-overlapping Wilson 95 % CIs vs. secure. Either outcome at
7B refines our 1B null — a mechanism *engagement* at 7B confirms the
scaling boundary; another null at 7B with engagement only at 32B
shifts the boundary location.

**Phase E prediction 2: Suppression captures behavior at 7B with
SAE-decomposed features.** Replacing the linear
`PersonaFeatureProbe` with an SAE-feature-target version of the
§3.4 `gradient_penalty` primitive — penalizing the activation of
the SAE features Wang et al. identify rather than a single linear
direction — should make the §4.3 dissociation close at 7B. Specific
prediction: the per-condition behavioral judge score will move in
the predicted direction (suppression decreases persona-voice rates
by ≥ 1 SD, amplification increases them) when the suppression
target is the SAE feature set rather than the linear analog.

Both predictions are concrete enough to falsify at 7B without
re-deriving the methodology, just scaling it. A Phase E researcher
inheriting this paper's instrument stack would: (a) train the
linear probe on the 7B base checkpoint using the same 240-pair
dataset (§3.1); (b) run the same controlled C10 v2 protocol (§3.2);
(c) extend `gradient_penalty` to operate on a public SAE feature
set; (d) re-run §4.2-§4.3 with the SAE-feature target. We have
released the four scripts that bootstrap this pipeline alongside
the present paper (§D.5).
