# 8. Anticipated objections {#objections}

Three objections a methods reviewer raises on first read, answered or conceded.

1. *"Your anomalies come from one program on the field-default OLMo / Qwen / Llama / GPT-OSS
   panel. How do you know they generalize?"* Each anomaly is stated with its architectural
   trigger, not asserted as universal: A3 fires on reordered-norm models (detected via
   `post_feedforward_layernorm`), A1 on massive-activation families (the outlier dim's variance
   share is the diagnostic), A2 on control-token positions (the participation ratio is measured,
   not assumed). The claims are cautions keyed to a detectable structure, which is how a reader
   checks whether their model is in scope. Conceded: the panel is the field default, and the note
   demonstrates the failure modes rather than surveying their prevalence.

2. *"'The bottleneck finding and the instrument failure are the same fact' is rhetoric. Isn't the
   low PR just your `V_moral` being mis-constructed?"* The participation ratio is
   outcome-independent: a property of the activations at the position, not of any direction of
   interest, and it is low at the decision token while full-rank at content positions on the same
   model. The band-below-null is a positive-control property (held-one-out moral directions
   project below the null), and the decision-direction cosine (R3), immune to the projection null,
   reads as active separation. A mis-constructed subspace would not give a clean positive-control
   band at content positions and a below-null one only at the decision token. Conceded: the reframe
   is load-bearing, so it ships with its certifying cell (R3), not asserted.

3. *"You reclassified your own headlines after seeing data (n=11 → n=23, +0.82 → −0.28). Isn't that
   post-hoc?"* Each reclassification is a dated, committed, pre-registered amendment with both
   result branches written and publishable before the recompute, and the superseded number is
   retained in the VOID register so it cannot silently re-enter prose. The discipline is the
   mechanism that catches post-hoc drift; §6.1 shows it catching an error in the program's own
   published paper. Conceded: the program is the case study, so the note cannot claim independent
   replication of its own discipline; that is what the companion skills and external review are for.
