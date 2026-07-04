# arXiv v2 comment (draft — for the submission "Comments" field)

**Draft for the arXiv v2 submission "Comments" field.** Paste (a trimmed form of) this when
submitting v2. Held for Orion's final sign-off before upload. Template A approved
(2026-07-04): Finding 2 is the confound-isolation; neither sub-claim is retained as a
positive robustness finding.

## Full form

> v2: Adds Section 4.4, an activation-scale (RMS-normalization) control for the
> critical-noise (fragility) metric. Two v1 sub-claims are corrected. (i) The
> post-saturation decline of critical noise (raw σ* 18.3→4.7) is **withdrawn**: it is flat
> under RMS normalization (13.8→15.0), a scale artifact, not continued representational
> change. (ii) The layer-depth fragility gradient (raw late/early 7–15×) is **largely
> activation scale**: under RMS normalization it attenuates to ~2× and the late≥mid≥early
> ordering fails at 8 of 37 checkpoints; the residual ~2× is **not claimed** as a genuine
> gradient (RMS controls per-layer scale, not covariance shape; a whitened or
> participation-ratio-matched control would be the test). Finding 2 and the abstract are
> scoped accordingly, and we recommend RMS-normalized critical noise for cross-layer claims
> and raw critical noise for within-layer comparisons. Findings 1 (staged emergence) and 3
> (data curation, a within-layer comparison at matched layers) are unaffected. Minor:
> citation formatting fix.

## Short form (if arXiv comment length is a concern)

> v2: Adds an activation-scale (RMS-normalization) control (Section 4.4). The v1
> post-saturation critical-noise decline is **withdrawn** as a scale artifact (flat under RMS,
> 13.8→15.0); the layer-depth gradient (raw 7–15×) is **largely activation scale** (attenuates
> to ~2× under RMS, ordering fails 8/37), with the residual **not claimed** as a genuine
> gradient. Finding 2 and the abstract are scoped; we recommend RMS-normalized critical noise
> cross-layer and raw within-layer. Findings 1 and 3 unaffected. Minor citation fix.

## Notes (surfaced by the W0 cold-boot audit)

- The two sub-verdicts differ and are stated separately per rider: the **decline vanishes**
  (withdrawn), the **gradient attenuates** (largely scale, residual not claimed).
- v1 already carried a one-sentence hedge (Section 5.2: "a low critical noise can also
  reflect activation-scale changes"), so the erratum is a quantified confirmation of a
  stated caveat, not a reversal.
- Residual-gradient follow-up (whitened / PR-matched control) is logged as an optional
  priced thread (OPEN_THREADS OT-9); it is not required for the v2 claims.
- Cross-paper dependency: Paper 3 (moral_geometry) already cites this paper's §4.4, which
  exists only in v2 — so v2 is the version Paper 3's citation resolves against.
