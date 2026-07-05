# 6. Case study: one program's amendments as caught failures {#case-study}

The program asked two questions of the refusal decision: *what* moral content it
reads, and *how* it commits. Its pre-registration trail is a sequence of caught
failures, each a dated amendment. Read in order, it is the methods note in miniature. The public amendment trail is a
credibility asset; it is cited from the flagship, not hidden.

- **Null degeneracy.** The instruct-model covariance null saturated on
  Qwen/Llama. Fix: standardized recompute; OLMo unchanged raw→standardized certified it.
- **Position gate.** The decision site is a PR-14.7 bottleneck with the
  positive control below the null. Fix: PR<30 position-invalid flag; V_moral re-typed as
  format-robust (invalid-position artifact at `final_pre_assistant`, band matches at the valid
  `mean_content` position), and the content-projection numbers re-typed as non-verdict.
- **Referee-pass hardening.** Before any asset was built, a referee pass
  re-typed the twin stimulus (request-twins carrying the judgment outcome, Δrefusal expected-flat),
  added a transport positive control to the decisive cell, made the head-score null channel-matched
  (mean/resample ablation, not zeroing), and added a behavioral-discrimination pilot screen.
- **OV overshoot.** Per-head attribution reconstructed at 3.05 on OLMo-3's
  reordered norm. Fix: two-sided gate + exact RMSNorm fold, reconstruction 3.05 → 0.9999.
- **MDE-crossing headline.** The `reads_non_vmoral_features` verdict (n=11)
  rested on an absolute transport comparison. Fix: within-outcome ratio at n=23 → the honest
  `under_transfer`.
- **Under-transfer superseded.** A rank sweep replaced the point comparison:
  as k ∈ {1, 3, 8, 16}, R_judgment climbs 0.05 → 0.46 → 0.59 → 0.66 while R_refusal saturates
  0.01 → 0.31 → 0.26 → 0.27 at the harm-rank-1 level (harm_rank1_R 0.31), random-null ~0 at
  every rank. The one-knob model `R_refusal(k) ≈ min(harm_ceiling, R_judgment(k))` fits the
  plateau (k≥3) at RMSE 0.036, and PC1 (highest variance, purity 0.974, most harm-aligned at
  cos 0.35) is causally inert (rank-1 moves neither readout, 0.01 / 0.05), the lesson that
  variance is not causal relevance. Both the one-knob RMSE 0.036 and the PC1-inert reading are
  illustrative point estimates, reported without CIs. Verdict: `harm_saturating`.
- **GPT-OSS commit axis.** The first session banked the position gate (PR 12.8),
  consequential engage deliberation (benign→refuse 7/7), and the first-run disengage 0/7 that
  looked irreversible.
- **Power table.** The saved-array power computation ruled the Llama same-design
  re-run futile before the compute run (ratio-of-ratios CI [−2.3, 4.9] on a latched denominator).
- **One-root diagnosis.** The Llama chaos was diagnosed by a single root split,
  judgment-delta coherence, not a grab-bag of probes: the orthogonal judgment cell is coherent,
  so the refusal chaos is saturation.
- **Denominator-latched voids.** The Llama "reads beyond harm" hint (R_refusal
  0.44 vs harm-rank-1 0.14 at rank 16) was voided, its denominator saturated; the three branches
  re-entered unweighted and were resolved by the depth-matched `broad_moral` read.
- **Nomenclature + early-commitment.** Fixed engage = harm-add /
  disengage = harm-remove; defined the asymmetry statistic A; the patch-layer sweep gave the
  EARLY-COMMITMENT verdict (Llama disengage coherent at 8/12/14, incoherent at 16) and the
  read-layer cross-model asymmetry A_Llama − A_OLMo = 1.03 (later depth-re-attributed, §5).
- **Depth-indexed verdict.** The +0.82 read-layer asymmetry collapsed to −0.28
  at matched layer 12 (§5). This amendment started this note.
- **Harm-coextensive hardening.** The reads-broad verdict survived the rank-1
  harm-coextensive alternative: a single harm cue spans only 3.6% of the engage-driving moral
  basis (the rank-2/4 severity-ladder version is a stated extraction rider on unsaved contrasts).
- **Graded disengage.** The step-gate saturation trap was de-confounded: GPT-OSS
  is a reversible reader, violating→comply 6/10 with monotone projection in all 10 items (§4.1).
- **Confound-named hypothesis.** The n=3 categorical co-occurrence
  ("harm-readers reversible, broad-reader early-commits") was replaced by a falsifiable
  dimensionality→reversibility hypothesis with an explicit architecture confound: the read↔commit
  pairing is confounded by lineage/scale/tokenizer/reasoning-vs-instruct at three points, and is
  deconfounded only by varying one axis at a time. The measured two-axis table stands; its
  interpretation is a follow-on hypothesis, not an n=3 claim.

## 6.1 Reflexive discipline: the program audits its own published paper {#reflexive-discipline}

A cold-boot re-read turned the same scrutiny on the program's own published work. Paper 1 (Reblitz-Richardson, 2026, arXiv:2606.11375v1, 9 Jun 2026) stated a raw layer-depth fragility gradient as its abstract-level Finding 2: late layers
were reported as far more fragile than early ones, with a raw late/early σ* ratio up to ~14.7×
(the claim ledger records the range as 7–15×; Table 2 late 10.0 / early 1.8), plus a raw post-saturation
σ* decline from 18.3 to 4.7. A post-submission control (§4.4, RMS normalization) shows the
gradient is largely an activation-scale artifact: under RMS normalization the ratio collapses to
~1.8–2× (the residual ~2× is not claimed as a genuine gradient, since RMS controls scale not
covariance shape), the cross-checkpoint ordering fails at 8/37 checkpoints, and the post-saturation
decline is withdrawn (flat, ~13.8 → 15.0). The lesson is exact: raw σ* is valid within-layer (same activation
scale) but activation-scale-confounded cross-layer; RMS-normalize for any cross-layer claim.

Two things about *how* it was caught belong in this note, and they are not the same thing.
First, what caught the error was a cold-boot (fresh-context) ledger re-read after publication,
not the calibration protocol firing at authoring time: the confound surfaced when a
fresh-context audit re-read a result the warm working sessions had produced and re-read many
times without flagging. The fresh-context reviewer's advantage is real, and mechanically
recreating it caught an abstract-level error. That the written calibration protocol would have
flagged this prospectively is a separate and weaker claim the note does not establish; the
honest record is that a published number stood until a fresh re-read caught it. Second, it
triggers a v2 erratum on a published paper. A program that runs an instrument-calibration discipline on other people's panels has to
run it on itself; the same scale confound named in the covariance null (magnitude is not
the signal) is the one that inflated Finding 2. This is the reflexive instance, and it is the
reason the note leads with "instruments before verdicts" rather than presenting the direction
results as settled.

## 6.2 Claim hygiene: a traceable claim ledger {#claim-hygiene}

Every number in the program traces to an anchored-sentence row in a claim ledger: if a draft states
a scalar that ledger does not carry, the draft is wrong until a row is added. The ledger also
carries a register of superseded claims retained *with their
replacements*, so they cannot re-enter prose as findings. The three the reader of a naive draft
would most likely resurrect are all there: the un-folded 3.05 head anatomy (replaced by the
folded 0.9999), the n=11 `reads_non_vmoral_features` headline (replaced by `under_transfer`
at n=23), and the `A = +0.82` read-layer asymmetry (replaced by the depth-matched −0.28).
A separate set of number-integrity flags blocks specific *scalars*
whose value is still in dispute across documents, without blocking the verdicts (the shapes and
signs are robust; only a printed number waits on the flag). Voided results may be discussed as
methods lessons in this note; they are never findings in the flagship.

# Limitations {#limitations}

The discipline in this note is drawn from one research program, and its scope should be read
accordingly.

**Per-mode model coverage.** The summary "six failures across four architectures" is honest
only about the panel as a whole; each individual mode is established on one or two models, not
on all four. The band-below-null position-invalid instrument (§2.1) is shown on
OLMo-3-Instruct at its decision token, with the PR < 30 gate applied on OLMo, Qwen, and Llama.
The massive-activation outlier's position-dependence (§2.2) is a Llama-3.1 finding cross-checked
against OLMo. The covariance-matched null degeneration (§2.3) is a Qwen-and-Llama result with
OLMo as the clean control. The reordered-norm OV overshoot (§2.4) is OLMo-only, since pre-norm
Llama and Qwen reconstruct near 1.0 natively. The deliberation/prefill asymmetry (§4.1) is
GPT-OSS-only. The read-layer depth artifact (§5) is Llama-versus-OLMo. So the note is six
protocols, each demonstrated on one or two members of a four-model panel, not six effects each
seen on four models.

**The position-validity gate can flag a real direction.** The PR < 30 gate declares a position
invalid for content projection-fraction tests, but its false-invalid rate is unquantified. A
genuine content direction present at a narrow position would be flagged the same way as a
weak-instrument artifact. The gate is calibrated to catch band-below-null cases; it is not
calibrated against a bank of known-present directions at narrow positions, so it can suppress a
real read.

**Standardization can destroy anisotropic signal.** Per-dimension standardization rescues the
covariance-matched null in massive-activation families (§2.3), but z-scoring flattens genuine
anisotropy along with the artifactual kind. The note's own boundary case, where standardization
and top-k projection-out disagree on the same cell (§2.3), is the symptom: when a subspace is
genuinely low-rank, standardization can report structure that projection-out does not, with no
guarantee the standardized space preserves a real anisotropic direction.

**Survivorship bias.** Only caught failures appear here. The note has no estimate of how many
measurement failures the discipline missed, and a protocol that reports only its catches cannot
state its own false-negative rate. The six are the ones a fresh-context re-read or a positive
control happened to surface; failures that produced a plausible number and were never
re-examined would not be in this note by construction.

**Single program, single panel.** All of the evidence is from one causal interpretability
program on one model panel (OLMo-3, Qwen2.5, Llama-3.1, GPT-OSS). The protocols are offered as
portable within that program and worth testing elsewhere, not as externally validated across
programs, tasks, or architectures beyond this panel.
