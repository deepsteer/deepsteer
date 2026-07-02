---
name: compute-ordering
description: >-
  Sequence experiments for maximum information per GPU-hour under a fixed compute envelope.
  Use this skill when planning any experiment sequence, phase plan, paper plan, or pod
  session; when a publication or framing decision is pending; and when writing any plan that
  mixes analysis with model runs. Fires on: "plan", "phase", "session", "RunPod", "A100",
  "GPU budget", "what should we run next", "sprint". Enforces zero-GPU-first ordering,
  ranking experiments by which decisions their results change, batching by loaded model,
  pilot gates before scaling generation, explicit cross-session data dependencies, and
  per-pair/per-rollout artifact saving so future statistics stay zero-GPU.
---

# Compute Ordering

**Core principle: rank experiments by the decisions their results change, then by GPU cost —
and never let a write-up decision wait on a computation that is free.**

## The ordering algorithm

1. **Zero-GPU on existing artifacts, first — always.** Especially anything that gates a
   publication, framing, or verdict decision. Before any pod is provisioned, enumerate what
   is computable from committed .npz/JSON right now. (This program's Phase A — positive
   bands, mutual projections, variance percentiles, continuity cosines, Δ-bootstraps — was
   an entire calibration layer with zero GPU spend, and it changed how the flagship number
   gets written up.)
2. **Rank the GPU items by ΔDecision/GPU-hour.** For each candidate, answer: *which pending
   decision does each branch of this result change?* An experiment decisive in **both**
   branches (e.g. "sensitivity-confirmed validates seven papers / category-mismatch replaces
   the instrument") outranks an experiment that only matters if it comes out one way. If
   neither branch changes any decision, it doesn't get scheduled.
3. **Batch by loaded model.** Group everything that touches the same weights into one
   session (extractions, controls, sweeps ride along with the keystone measurement). Session
   1 = loaded-model batch; session 2 = generation-heavy work.
4. **Pilot-gate anything generation-heavy.** Before scaling rollouts, run the smallest
   version that can kill the design (8 prompts × 8 rollouts) with an explicit proceed
   criterion ("≥ 6 prompts show mixed outcomes, else switch model and record the invariance
   as a datapoint"). Never scale into a design that a 30-minute pilot could have falsified.
5. **Encode dependencies in the plan text.** If session 1 needs data specced in a session-2
   task (this program: the XSTest subset specced in B2, needed by B5's over-refusal arm),
   the plan says so explicitly — agents execute the file, not your intentions.
6. **Novelty claims are gated on a zero-GPU lit pass.** Before building anything framed as
   "first to X," a verification pass against primary sources belongs in the zero-GPU layer:
   finding your Stage-1 method in prior art *before* the pod (as happened with the C1
   head-attribution stage) re-centers the claim while re-centering is still free.
7. **End sessions at clean checkpoints.** Don't start a build blind in a session's last
   stretch; a committed, rsync'd checkpoint plus a scoped next step beats a half-built
   harness every time.

## Artifact hygiene (what keeps future work zero-GPU)

- **Save per-unit arrays by default** (per-pair diffs, per-prompt activations, per-rollout
  traces), not just the mean direction. Every bootstrap, Δ-CI, and re-analysis then costs
  nothing. This program had to queue re-extractions solely because early runs saved only
  the means.
- **MISSING_ARTIFACTS.md ledger:** when an analysis needs an artifact that doesn't exist,
  log it and queue regeneration into the next batched session — never silently regenerate
  inline (conventions drift, and the pod bill grows).
- Pin model revisions, dataset commits, and extraction SHAs in every artifact's metadata so
  a re-run is a decision, not an archaeology project.

## Session plan template

```
SESSION n (est. X h, model group: …)
  keystone:      the ΔDecision-ranked item this session exists for
  riders:        everything else needing these weights (controls, sweeps, re-extractions)
  pilot gates:   criterion → proceed/switch
  depends on:    data/artifacts that must exist before pod start (owner: which phase)
  saves:         per-unit arrays list (enforced)
  gate after:    which human decision this session's results feed
```

## Ship-blockers (for any plan document)

- [ ] Zero-GPU items enumerated and scheduled before any GPU item
- [ ] Every GPU item has the "which decision does each branch change?" line
- [ ] Sessions grouped by loaded model; riders attached to keystones
- [ ] Generation-heavy items have pilot gates with explicit criteria
- [ ] Cross-session dependencies stated in the plan text
- [ ] Per-unit artifact saving specified per task
- [ ] "First to X" framings preceded by a committed lit-verification pass

Pairs with: `anomaly-triage` (priced discriminators enter the ranking),
`instrument-calibration` (calibration is the canonical zero-GPU-first layer).

*Changelog — v2 (2026-07-02): added novelty-gating (case: the pre-build Sahara catch) and
clean-checkpoint session discipline.*
