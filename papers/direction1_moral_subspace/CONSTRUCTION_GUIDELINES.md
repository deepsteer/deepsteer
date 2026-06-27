# Direction 1 — Pair-construction guidelines (Phase 1, step 5)

**Date:** 2026-06-26 · **Frozen before generation.** Per-source rules for building the
contrastive pairs that feed `V_moral`. Base rules are `deepsteer/datasets/DATASET_GUIDELINES.md`
(§1.1 relational structure, §1.2 structural parallelism, §1.5 accidentally-moral neutral,
§4 pair-level, §3 registers) — all apply here. This doc adds the Direction-1 specifics and
one fable-only control.

Companion: `PREREGISTRATION.md` (§1 salience framing, §3A GATE G-AXIS), `audit_runner.py`
(the gates that enforce these), `partition_manifest.json` (the train/eval pools).

## Core: salience contrast throughout

Every pair, every source, is a **moral-salience** contrast: a moral-valence-**present**
side vs a valence-**stripped** (neutral) side, holding scenario, participants, and
**discourse type** constant. The pair must differ in moral valence and nothing else. This
is what makes `V_moral` a moral-salience subspace and keeps the 0.1044 Paper 5 result its
salience baseline (`PREREGISTRATION.md` §1).

The contrast is moral-vs-neutral (valence present vs stripped), so **§1.5 applies in its v2
accidentally-moral form**: the valence-stripped side must not smuggle moral weight back in.

## Per-source construction

### MORABLES — fable-internal derivation (NOT novel neutral construction)

Derive **both** sides from the fable's own narrative material:

- **Moral side:** a concrete retelling of the fable's central event with its moral valence
  present.
- **Neutral side:** a concrete retelling of the **same** event with the valence stripped —
  **same actors, same scene, same domain**, only the moral charge removed.

**Hold discourse type constant: both sides are concrete event-retellings.**

**Do NOT** contrast the fable's abstract **stated moral** (the aphorism) against its
narrative. That confounds moral valence with discourse type (aphorism vs narrative) and
hands the probe a genre/abstraction shortcut. The earlier `binary` config (correct-vs-
opposite stated moral) is **not** used for direction extraction for this reason.

#### Fable abstraction/genre-match control (new, fable-only)

The moral and neutral sides must **match on concreteness/abstraction level** — both
concrete event-retellings, neither an abstract aphorism. This legislates specifically
against the fable abstraction shortcut. Enforced by audit gate `g_abstraction_match`
(`audit_runner.py`, pair-type `fable_salience`).

> MORABLES's inclusion in `V_moral` is **not** assumed. After extraction it must clear
> **GATE G-AXIS** (cosine ≥ 0.67 with the Moral-Stories axis) to pool; otherwise it reverts
> to construct-anchor evaluation only. See `PREREGISTRATION.md` §3A.

### Moral Stories — situation-held, valence-stripped action

Hold `situation` (+ `intention`) constant; contrast the `moral_action` (valence present)
against a **valence-stripped** version of the action in the same situation. The reference
axis for G-AXIS. (Not moral-action-vs-immoral-action — the contrast is salience, not
polarity.)

### ETHICS commonsense — minimal edit that FULLY strips valence

Derive the neutral as **the smallest edit that removes every trace of moral charge** —
valence-removal first, minimality second. This is the faithful reading of "derive
minimally": the goal was always low-risk clean neutrals, and on ETHICS's register
minimality and valence-stripping point opposite ways (`20000→2000 calories` is low-edit
but high-risk — the exact §1.5 defect). ETHICS has §1.2 headroom to spend on §1.5.

**First-batch finding (register limitation; pre-registered fallback triggered).** After
the valence-first prompt, ETHICS §1.2 stayed perfect (0.00 — bigger edits cost no
parallelism) but **§1.5 did not clear** (≈0.40 fail, n=40 and n=10 agree). Mechanism: ETHICS
commonsense scenarios are *moral-judgment-native* — the scenario frame itself (feeding
children, tending a person on the ground, a tired sibling, a dog's needs) carries residual
care valence that survives any same-frame edit. The prompt removed the egregious cases
(no score-1 remain) but cannot strip the frame without abandoning the same-setting constraint.

**Decision: ETHICS contributes ZERO to the training direction** (2026-06-26). Its
register limitation (above) plus the clean-subset selection bias make it unsuitable as a
bulk direction source; rather than force a biased contribution, it is removed from training
entirely. Consequences:

- **ETHICS eval role upgrades from in-distribution check to generalization probe.** With
  zero ETHICS in training, its bias-flagged eval pairs (the non-care-frame slice that
  clears the gates) now test whether a **Moral-Stories + MORABLES** `V_moral` generalizes
  to ETHICS's **abstract-judgment register — a register absent from training.** This sits
  alongside Social Chemistry 101's deferred OOD role (below).
- **G-AXIS is a two-source check** (Moral Stories reference + MORABLES); no ETHICS
  direction is extracted or pooled. See `PREREGISTRATION.md` §3A and the zero-ETHICS amendment.

### Social Chemistry 101 — NOT constructed here

Deferred OOD generalization probe (Track 4). No directions extracted from it. Together with
the ETHICS eval set, it forms the held-out generalization battery for a
Moral-Stories + MORABLES `V_moral`.

## Cross-source rules

- **Training is two-source by construction: Moral Stories (workhorse bulk) + MORABLES
  (full clean yield, pooled iff it clears G-AXIS).** ETHICS is zero-in-training; it and
  Social Chemistry 101 are eval-side generalization probes.
- **Relational structure (§1.1) and parallelism (§1.2)** apply to every pair, both sides.
- **Three registers (§3)** declarative / narrative / dialogue, per-source and per-register
  balance tracked in each split independently. MORABLES is register-narrow (event-retellings
  are narrative; declarative possible; **no native dialogue**) — dialogue coverage comes
  from Moral Stories + constructed registers.
- **Eval-set register composition is reported** (per source × register), so a paraphrase gap
  cannot be a register-shift confound.
- Paraphrases (eval-source only) follow `PARAPHRASE_PROTOCOL.md`.

## First-batch calibration (required before scaling)

The anti-triviality gates (§1.1/§1.2 via the LLM score) use an **uncalibrated** fail
threshold (score ≤ 3). On the **first generated batch**, run the audit and confirm the
threshold actually separates trivially-separable pairs from genuine ones; adjust via
`audit_runner.py --fail-at` if the score distribution shows it mis-discriminates. Do not
scale to full generation until this threshold pass is done.

## Exit criterion (step 5)

Generated pairs (both splits, per source) clear the audit at v2 thresholds; fable pairs
additionally clear `g_abstraction_match`; eval paraphrases clear `PARAPHRASE_PROTOCOL.md`
C1+C2; register composition reported. Then Phase 2: per-source extraction → GATE G-AXIS →
finalize `V_moral`.
