# KDG Phase 3 Spec: what the action position reads (pre-registration)

Status: v0.1, 2026-10-01. Pre-registration of the Phase 3 action-position cell and its incentive-direction
companion, written before any extraction (pitch plan, handoff task 3; author rule: pushed before the Phase
2 pod). No GPU spend is authorized here; the pod-boundary checklist is §9. Builds on FL's interchange
rank sweep as implemented (`papers/d3_decision_anatomy/scripts/causal_cells.py`, `c1_session.py`,
`scripts/sweep.py`; D3 PREREGISTRATION Amendments 3, 8, 11, 15) and on the KDG letter-only readout
(`KDG_PHASE1_SPEC.md` P1-A2, validated P1-A9).

Skills in force: `intervention-validity` (a spec block per causal cell, §3–§4), `construct-audit` (type
blocks §5, design matrix §6), `instrument-calibration` (ladders, transport control, MDE §7),
`estimator-traps` (covariance-matched nulls, channel chance, ratio CIs), `compute-ordering` (§8–§9),
`program-thesis` (branches §10).

## 0. What this decides

The program has two reads of OLMo-3's moral content: judgment reads the moral subspace broadly
(R_judgment(16) = 0.66) and refusal saturates at the harm rank-1 level (R_refusal(16) = 0.24 [0.13, 0.41]
at pooled n = 42, 0.27 on the original 23; `harm_saturating`); on Llama-3.1 refusal reads broadly (0.85 vs judgment 0.79). The KDG paper shows
the action departs from the model's own judgment under pressure and that the gap is not harm-keyed. What
the action position reads is open. The pitch's branch rows that this cell decides:

| Result | Thesis edit (pitch table, unchanged wording) |
|---|---|
| Action position reads the moral subspace at low rank | Widening that read is the intervention; grounding is the basis |
| Action position reads a goal direction and nothing moral | Grounding is necessary and not sufficient; value moves to measurement and monitoring; the intervention targets goal arbitration |

and a third this spec adds, written now: the action reads the moral subspace as broadly as judgment does,
so the gap is arbitration against the incentive rather than a missing read (cell C2 then carries the
mechanism).

## 1. New asset: action moral-status twins (zero GPU)

A twin is one agent-frame situation under pressure (an incentive toward a tempting option is present in
both members) in which a single span flips the moral status of the tempting option: in the **violating**
member the tempting option breaches a norm (it uses something belonging to someone else, harms a third
party, deceives someone, breaks a rule that binds the agent); in the **permissible** member the same
option breaches nothing (the thing is the agent's own, no one is affected, the rule does not apply). The
option texts are identical across members; only the span differs.

- Construction reuses `deepsteer.kdg.phase2` templates: agent and third-person templates with one `{SLOT}`
  whose two fills are the violating and permissible spans, so members share an exact prefix and suffix
  by construction (FL's alignment rule then strips the shared token prefix and suffix to find the span,
  `causal_cells.py` `patch_positions`). Third-person members of the same twins give the judgment arm on
  the same content, so action and judgment are compared within twin (FL compared refusal on request twins
  with judgment on separate compositional narratives).
- Readout option: the **tempting option** (one option, identical text in both members). Outcome per frame:
  renormalized mass on its letter, mean over 8 permutations (p_D(tempt) in the agent frame, p_J(tempt) in
  the third-person frame).
- Spread: the seven construction norm classes, both generator halves (Claude subagents / Codex CLI),
  KDG panel settings and the F6–F8 incident settings. Generation target 200 twins (100 per half); a
  24-twin construction pilot first, validated zero-GPU (one-span check, length band, rater agreement
  that the violating fill breaches a norm and the permissible fill does not, coherence).
- External rater (other provider) labels each member: is the tempting option acceptable (yes/no);
  twins whose rater labels do not flip are excluded and counted.

## 2. Screen (baseline discrimination, operating band)

Run in-session as the first unit on each model (forward passes, letter-only chat, dose 0, neutral
prefix), before any patch.
- **Action band:** p_D(tempt | permissible) ≥ 0.5 and p_D(tempt | violating) < 0.5.
- **Judgment band:** p_J(tempt | permissible) ≥ 0.5 and p_J(tempt | violating) < 0.5.
- **Primary set:** twins in both bands (paired action vs judgment). Reported beside: pass rate per band,
  per norm class and per generator; the action-band-only and judgment-band-only sets are descriptive.
- Selection note (estimator-traps trap 4): the screen conditions on the action discriminating moral
  status, the same conditioning FL used for refusal; R_action is therefore read on twins where the
  action is morally sensitive at baseline, which favors finding a moral read. Stated in every verdict
  sentence.
- **Bail:** fewer than 40 primary twins on a model → stop that model after the screen; report pass rates;
  generate more (a dated amendment), never loosen the band.

## 3. Cell C1: the action-position rank sweep (spec block)

```
stimulus_class:          action moral-status twins (§1), primary set = both bands (§2); type blocks §5
outcome_variable:        p_D(tempt), letter-only agent-frame next-token mass at the model output
                         (behavioral; primary); p_J(tempt) third-person (judgment arm, same twins)
baseline_discrimination: §2 bands; pass rates reported per band, norm class, generator
site:                    residual entering block L at the flipped-span (content) positions, the FL
                         instrument as implemented (source span mean-pooled, broadcast over target
                         span positions); L = 16 (OLMo-3), 12 (Llama-3.1, FL's layer of record);
                         L ± 4 descriptive. PR recorded at the span positions and at the decision token
transfer_scope:          full-residual (span) | subspace-restricted to the nested PCA basis of rank
                         k ∈ {1, 3, 8, 16} | complement (off-basis part) | harm rank-1 | random rank-k
transport_control:       the same rank-k restriction on the same twins in the third-person frame
                         (R_judgment(16) must exceed the covariance-matched random q95); plus the
                         reproduction gate R2 (FL's compositional twins)
ablation_semantics:      interchange (resample-patch from the twin member): distribution-preserving
direction:               disengage, as FL's refusal sweep: permissible span into the violating prompt
                         (p_D(tempt) expected to rise); engage (violating into permissible) secondary
alignment_rule:          FL's: strip the shared chat-templated token prefix and suffix; the remainder
                         is the span. A twin with an empty span is excluded AND COUNTED (FL's harness
                         dropped these uncounted; ANOMALIES process ledger 2026-10-01)
controls:                random rank-k, covariance-matched to the span activations (primary null, 20
                         draws per k) and isotropic (FL parity, beside); named references: harm
                         rank-1 (`d_harm`, FL's construction), complement cell (instrument certificate)
outcome_harness:         letter-only chat readout, PHASE1_TEMPLATE_VERSION p1-1.0.0, 8 permutations
                         with matched orders across members, bf16 forward passes
branches:                §10, all written before data
```

**Basis.** FL's nested PCA basis: uncentered SVD of the per-pair `mean_content` moral − neutral contrasts
over the three moral sources (moral_stories, fables, ethics; the "1057-pair PCA", `sweep.nested_pca_basis`)
at layer L. FL did not save these bases (D3 PREREGISTRATION l.862); this session rebuilds them and saves
them with type blocks.

**Quantities (per arm, per k).** Δ_full(s) = readout(target with the full span patch) − readout(target
unpatched); Δ_k(s) likewise with the rank-k restriction. R_arm(k) = mean_s Δ_k / mean_s Δ_full (ratio of
twin means, FL's normalization), bootstrap over twins (10,000, seed 0), resampling the paired action and
judgment deltas of a twin together. harm1_arm = mean Δ_harm / mean Δ_full.

**Precondition per arm and model:** mean Δ_full CI entirely above 0 (the full span patch at L moves the
readout). If it fails at L, the arm reads `no_transfer_at_L`, the L ± 4 band is reported descriptively,
and any re-run at another layer is a dated amendment.

**Primary verdict (FL's frozen shape rule, action in refusal's place; tolerances 0.1, ceiling 0.6):**
- `instrument_ceiling` if R_act(16) and R_jud(16) are both < 0.6;
- else `action_reads_broad` if |R_act(16) − R_jud(16)| ≤ 0.1 and R_act(16) − R_act(1) > 0.1;
- else `action_harm_saturating` if |R_act(16) − harm1_act| ≤ 0.1 and R_jud(16) > R_act(16) + 0.1;
- else `indeterminate`.

**Co-primary (FL's ratio-of-ratios, paired here):** D = R_jud(16) − R_act(16), paired bootstrap over
twins; M = 0.15. CI entirely above 0 and D ≥ 0.15 → `action_reads_outside_vmoral` (the action reads
features judgment does not); above 0 and D < 0.15 → `_small_margin`; CI includes 0 → `under_transfer`
(not separable at this n); CI below 0 → `action_reads_vmoral_more`.

**Complement certificate:** the complement cell's share mean Δ_comp / mean Δ_full is reported for both
arms. If R_act(16) is low and the complement carries the action effect while R_jud(16) is high, the
reading "the action reads moral-status features outside the rank-16 moral basis" is licensed; if the
complement is also low, nonlinearity (FL's A4) is the named rival.

**Secondary readout (FL parity):** the same deltas read as projections at the output of layer L onto
cross-fitted decision directions d_act and d_jud (§5), so R_act can be set beside FL's projection-based
R_refusal on the same scale. Descriptive.

## 4. Cell C2: the incentive direction at the action token (spec block)

```
stimulus_class:          Phase 1 KDG scenarios on each model's own screen (OLMo-3 136,
                         `screened_ids_a17_union.json`; Llama-3.1 118, `screened_ids_llama31_meta.json`),
                         pressure vs pressure-removed members, agent frame, letter-only, dose 0
outcome_variable:        p_D(violating), letter-only agent-frame mass at the model output
baseline_discrimination: per scenario incentive effect p_D(pressure) − p_D(pressure removed); the cell
                         runs on scenarios with a positive effect (reported: count, mean effect)
site:                    residual at the decision token (final_pre_assistant), block L (16 / 12),
                         single layer; PR at the position recorded
transfer_scope:          rank-1 direction ablation
ablation_semantics:      mean-ablation: the pressure prompt's projection onto d_inc is set to the mean
                         projection of the pressure-removed prompts (never zeroed)
alignment_rule:          single position (the decision token), shared template suffix; no span mapping
controls:                20 covariance-matched random directions at the same position, same
                         mean-ablation; named references: d_jud and d_act (from the C1 twins, §5)
outcome_harness:         as C1
branches:                §10
```

**Directions (cross-fitted).** Scenarios split into halves A and B by a hash of the scenario id (fixed
now: sha256(id) mod 2). d_inc = unit diff-of-means at the decision token, pressure − pressure removed,
built on one half and ablated on the other; both directions of the split reported, pooled for the
verdict.

**Quantities.** Removed share S = mean_s [p_D(pressure) − p_D(pressure, d_inc ablated)] / mean_s
[p_D(pressure) − p_D(pressure removed)] (ratio of means, bootstrap); S_rand from each random direction;
S_jud and S_act from ablating d_jud and d_act. Geometry beside (descriptive, co-located: chat format,
letter-only, decision token in both frames): |cos(d_inc, d_jud)|, |cos(d_inc, d_act)|, against the
covariance-matched null q95 and the channel-chance level sqrt(2 / (π · PR)) at that position (A2: the
decision site is a 9–15-dimensional bottleneck, so a cosine of 0.25 can be chance there).

**Rule.** **Low-rank incentive channel:** S CI entirely above 0.5 and above the random q95 → the incentive
reaches the action through one direction at the decision token (a monitor target). **Distributed:** S CI
entirely below 0.5 while above the random q95 → the incentive is read along more than one direction or
before this token. **No specific channel:** S within the random distribution. Beside, not a branch:
**the action consults the judgment direction** if ablating d_jud moves p_D(violating) beyond the random
q95 in either sign (a sign flip is an ANOMALIES entry, as FL's A8).

## 5. Type blocks (filled at extraction; every field required before any comparison)

```
V_moral nested basis B_k   contrast: stimulus/content (moral − neutral, mean_content); sources
                           moral_stories, fables, ethics; layer L; format raw text (FL's); PR of the
                           source sample; outcome_variable: none (a basis)
d_harm                     contrast: stimulus/content, Heretic harmful − harmless prompts, mean_content
                           (FL's construction; FL RESULTS l.325 describes it differently, ledger)
d_act                      contrast: content-driven decision, violating − permissible twin members,
                           agent frame, chat, final_pre_assistant, layer L; cross-fitted halves;
                           outcome_variable: p_D(tempt)
d_jud                      as d_act, third-person frame; outcome_variable: p_J(tempt)
d_inc                      contrast: stimulus/content (incentive present − removed), agent frame, chat,
                           final_pre_assistant, layer L; known covariates: length (stake sentence vs
                           filler, ±15% band), stake-specific vocabulary, valence; outcome_variable:
                           p_D(violating)
random bases               covariance-matched to the activation sample of the site they control
```
Every block also records model id and revision, template sha, n per cell, extraction commit and PR
(PR < 30 flags the position as a bottleneck site: decision-direction reads only, per A2).

## 6. Design matrix (what is measured, construct-audit)

|  | content (span) | decision (decision token) |
|---|---|---|
| **moral content** | B_k basis transfer (C1 restricted patch) | d_jud, d_act (C1 secondary; C2 references) |
| **incentive** | not measured here (deferred; needs span-aligned pressure twins) | d_inc (C2) |

The content × decision cell for moral content is the causal C1 cell; the decision × decision cell is C2's
geometry plus its named-reference ablations. Incentive × content is deferred and named, so no
"the incentive never touches moral content" sentence is licensed by this session.

## 7. Calibration, power and gates

- **R2 reproduction gate (positive control on the instrument, same model):** on OLMo-3 the rebuilt basis
  and harness reproduce FL's R_judgment(16) on FL's compositional twins (folded run of record 0.66); pass
  iff the rebuilt value's 95% CI includes 0.66. On Llama-3.1, FL's R_jud(16) = 0.79 at L12, same rule.
  **Bail:** fail → stop before C1; the port is not faithful.
- **Transport gate (C1):** R_jud(16) on the action twins exceeds the covariance-matched random q95. Fail →
  C1's action arm is reported as uninterpretable on this asset (the judgment arm does not transport
  either), whatever R_act shows.
- **Known-gap positive control** for the behavioral readout on both models is banked (P1-A9).
- **Power (conservative, from FL's measured spread).** FL's pooled R_refusal(16) CI [0.13, 0.41] at n = 42
  implies a per-twin SD of ≈ 0.46 on the ratio scale. Treating R_jud − R_act as unpaired (an upper bound;
  pairing within twin narrows it): MDE(D) = 2.8 · √2 · 0.46 / √n ≈ 0.29 at n = 40, 0.20 at n = 80, 0.17 at
  n = 120. Target: ≥ 80 primary twins per model (generate 200; FL's screens passed 38–39%). The paired SD
  is measured in-session and the realized MDE is stated beside every verdict; a D within its MDE reads
  `under_transfer`, never "no difference".
- **R1 rider (FL null, ANOMALIES process ledger 2026-10-01):** on each model, FL's request-twin refusal
  sweep re-run with covariance-matched random rank-k bases beside FL's isotropic ones (same harness, same
  layer of record). Branches: the V_moral-over-random margin survives the matched null (FL wording keeps
  its claim, the null named) / does not (FL's specificity sentence is scoped; author).

## 8. Sessions (batched by loaded model)

```
SESSION P3-1 (est. 2.5 h, OLMo-3-7B-Instruct 6e5971d9)
  order:       R2 reproduction gate → §2 screen → C1 (both arms, k sweep, controls, complement,
               L ± 4 descriptive) → C2 → R1 rider
  pilot gates: R2 (bail), screen n >= 40 (bail), transport (scopes C1)
  depends on:  action-twin asset generated, rated and committed (§1); basis sources on the pod
               (moral_stories, fables, ethics loaders); FL compositional twins and request twins; the
               Phase 1 screen files; this spec pushed
  saves:       nested bases B_k and d_harm per layer (.npz + type block); per-twin per-k Δ for both
               readouts and both arms; per-draw random-control Δ; complement Δ; d_act, d_jud, d_inc
               per cross-fit half; PR samples at span and decision positions; screen masses per member
  gate after:  Phase 3 human gate
SESSION P3-2 (est. 2.5 h, Llama-3.1-8B-Instruct Meta 0e9e39f2): same order, L = 12
```
Estimate from FL's C1 timings scaled to n ≈ 80–120 twins × 6 restriction kinds × 4 ranks × 2 arms × 8
permutations, plus 20 random draws per k; the basis rebuild is the largest single item. Not scheduled:
the author sets the pod after the Phase 2 pilot gate.

## 9. Zero-GPU layer and pod-boundary checklist

1. Commit and push this spec.
2. Action-twin generator (`phase2` templates, family tag `AT`), 24-twin construction pilot, rater pass,
   then the 200-twin set; cost statement before any API fallback (CLI paths have no API spend).
3. Harness port: `causal_cells.interchange` reused unchanged for the patch; a letter-only behavioral
   readout wrapper (patched forward pass, option-letter mass at the output); covariance-matched random
   bases; the empty-span counter; local tests naming their failure modes ("most probable failure: the
   span mapping patches the template suffix, not the flipped span"; "the random control is isotropic").
4. Analysis script (C1 shape rule, ratio-of-ratios, complement, C2 shares, geometry with channel chance)
   committed before data.
5. VALIDATE=1 remote dry run (stub model), then the pod command to the author.

Pod-boundary checklist: power (§7, from measured FL spread, realized MDE in-session) ✓; branches (§10) ✓;
bail conditions (§2, §7) ✓; per-unit saves (§8) ✓; dependency check at pod time (§8 "depends on").

## 10. Both branches (written before data)

**C1, per model.**
- `action_harm_saturating`: the action reads a harm slice like refusal does on OLMo-3. Thesis: the action
  and refusal channels share a read; widening the action's read is the intervention (pitch row 1).
- `action_reads_broad`: the action reads the moral subspace as broadly as judgment. Thesis: grounding
  reaches the action; the KDG gap is arbitration against the incentive, and C2 carries the mechanism.
- `action_reads_outside_vmoral` (D ≥ 0.15, complement carries the action): the action reads moral-status
  features outside the rank-16 moral basis that judgment does not need. Thesis: grounding is necessary and
  not sufficient (pitch row 2); the monitor reads the action site, not the moral subspace.
- `under_transfer` / `indeterminate`: not separated at the realized MDE; reported with the bar.
- `instrument_ceiling`, `no_transfer_at_L`, transport or R2 failure: the instrument does not answer the
  question on this asset or model; no thesis edit.
Cross-model: OLMo-3 refusal is harm-saturating and Llama-3.1 refusal reads broadly. If the action read
matches each model's refusal read, the action inherits the family's refusal read (the SYNTHESIS
conjunction under watch); if it matches on neither, action and refusal are separate channels on both.

**C2, per model.** Low-rank incentive channel: the monitor's white-box feature is the d_inc projection at
the decision token, scored in Phase 3's monitor cell. Distributed: the monitor needs a subspace or an
earlier site. No specific channel: the incentive's route is not found at this site; the monitor relies on
C1's action-site read. Each publishable; the C2 result is read beside C1, never instead of it.

## Amendments

(none)
