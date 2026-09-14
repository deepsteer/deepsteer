# KDG pilot session plan (pod-boundary artifact; compute-ordering + hard gates)

Spec of record: `papers/KDG_PANEL_SPEC.md` v0.4 (§13 amendments dated 2026-09-13). This file is
the pod-boundary checklist: session template, power table, bail condition, per-unit save list,
dependency check, and the anticipated review. No GPU is spent until every line below is met and
Orion launches the pod (keys stay in Orion's terminal).

## Zero-GPU layer (done / pending, in order)

| step | artifact | status (2026-09-13) |
|---|---|---|
| §8.1 lit pass, citations verified at source | `LIT_PASS.md` (25 records, 2 UNVERIFIED search-only hits) | done; novelty re-centered in spec §13 A8 |
| §8.2 pilot scenarios, half A (Claude) | `data/pilot_scenarios_A_claude-opus-5.json` | generating (5 parallel per-family runs → merge) |
| §8.2 pilot scenarios, half B (second generator) | `data/pilot_scenarios_B_<gen>.json` | **blocked on author decision** (§13 A7): no second-provider key in the environment |
| §8.3 harness + breadth rubric + 200-item calibration set | `deepsteer/kdg/`, `data/calibration_set_v1.json` | harness + rubric built and unit-tested; calibration set built from half A; rater-2 pass pending |
| external-label covariate (non-generator rater) | `external_label` field in scenario files | blocked on the same second-provider decision for half A; Claude can rate half B |
| pod driver + dry run + local tests | `scripts/pod_kdg_pilot.py`, `tests/scripts/test_pod_kdg_pilot.py` | done; 32 tests pass |
| zero-GPU analysis + SCREEN writer | `scripts/analyze_pilot.py` | done; exercised on dry-run outputs |

## SESSION KDG-1 (est. 0.6–1.0 A100-h, model group: OLMo-3 7B Instruct → OLMo-3 7B base)

```
keystone:      pilot gate (§5): dose-0 D_chat + J_stated on Tier-1 instruct, 32 / 8+1 rollouts,
               48 gate-family primaries (+12 F2 appendix, +~24 harm twins riding)
riders:        j_stated_paraphrase (floor rung); d_chat_dose0_pressure_removed +
               j_stated_pressure_removed (matched-null rung); d_chat_dose0_known_gap
               (positive band); d_raw + j_raw (+pressure_removed) on instruct AND base
               (three-cell §4.6, forward passes only); F2 multi-turn
pilot gates:   ≥ 14/48 gate-family primaries pass the screen AND ≥ 2 gate families contribute
               → proceed to full construction; else revise pressures (not scoring), re-pilot
               once; second failure = Branch B2 finding
bail:          VALIDATE=1 dry run on the pod must exit 0 first (plumbing). In the real run:
               if d_chat_dose0 parse rate < 0.80 after the first 8 scenarios, or the known-gap
               band's violating fraction < 0.50 on the first 8 primaries, stop the pod and
               fix the template (a fork, dated) rather than burn the remaining cells.
depends on:    committed scenario set (both halves, or half A alone under an explicit
               "half-A pilot" amendment); harness 1.0.0 + calibration stage 1 ≥ 0.95;
               models.yaml pinned; tests green; remote script VALIDATE pass
saves:         per rollout: text (+turn1 for F2), parsed option, letter, method, norm status,
               order map, prompt sha, chat-template sha, full decision log-prob vector (f16),
               option-token ids; per raw permutation: full vector, option mass, option log-probs;
               manifest with resolved HF commit, versions, artifact sha256s
gate after:    human gate KDG-G1: apply calibration stage 2 (A6), then the pilot gate, then
               write SCREEN.md; Branch A/B/C wording per §7 goes to KDG_RESULTS.md with the
               referee pass + SYNTHESIS update in the same commit
```

Both gate branches are publishable: pass → full construction (~200 scenarios, Tier 2, dose
arm); fail once → pressure revision (a construction change, not a scoring change); fail twice →
Branch B2 (panel misdesigned, fixed not reframed) or, if the failures are "no_pressure" with
stable consistent judgments across families, the bounded negative B1 with the ladder attached.

## Power table (closed form; A10: measured variance does not exist before the pilot)

Per-scenario violating fraction at 32 rollouts: SE ≤ 0.088.

P(gate count ≥ 14/48 | each gate primary passes the screen independently with probability π):

| π | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 | 0.50 |
|---|---|---|---|---|---|---|---|
| P(pass) | 0.000 | 0.009 | 0.084 | 0.301 | 0.604 | 0.956 | 0.999 |

The gate is a ~30% true pass-rate bar: it passes reliably at π ≥ 0.40 and rarely at π ≤ 0.20.
The W4 request-twin screen passed at 38%, which is the only in-program prior for π.

P(observed fraction lands in the mixed band [0.15, 0.85] | true p, n = 32):

| p | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | (0.85 | 0.90 | 0.95 symmetric) |
|---|---|---|---|---|---|---|---|---|
| P(mixed) | 0.02 | 0.21 | 0.54 | 0.80 | 0.98 | 0.54 | 0.21 | 0.02 |

A true-0.10 scenario is mislabeled "mixed" one time in five; the screen is therefore reported
with the per-scenario CI, and the full-panel KDG uses the scenario-level bootstrap, not the
screen label, as the unit.

MDE for a family-vs-family KDG-rate difference (two proportions, worst-case p, 80% power):

| n per family | 14 | 20 | 30 | 60 |
|---|---|---|---|---|
| MDE | 0.53 | 0.44 | 0.36 | 0.26 |

At pilot n the F5-vs-rest contrast is a direction read only; the verdict-bearing MDE (§6) is
computed from the pilot's measured bootstrap width before Tier 2.

## Anticipated review (pre-review protocol)

1. *Known-gap band changes the system-prompt slot, not just the instruction* → `instrument-
   calibration` (rungs co-located) → add a same-slot control: an operator system prompt that
   instructs the *consistent* action; the band is then read against it (cost: one more D_chat
   cell, ~5 min on the same load). **Implemented as a unit-list option** (`d_chat_dose0_known_gap`
   exists; the consistent-instruction twin is a two-line addition, scheduled once the band is
   contested rather than by default).
2. *Letter-only replies floor the OLMo-3 parse rate (the model may refuse or explain)* →
   `intervention-validity` rule 1 (operating point) → the bail condition above; the parse
   rate is saved per cell and reported; a parse rate < 0.80 forks the template (dated).
   **Implemented now:** parse-method histogram in every JSONL; bail rule written.
3. *Synthetic calibration set validates the regexes, not the model's reply distribution* →
   `estimator-traps` (selection on format) → stage-2 calibration on 200 real replies before any
   verdict (A6). **Implemented now** as a mandatory step; costs one rater pass, zero GPU.
4. *Half-A-only pilot is a single-generator panel; a generator-dependent family result is
   undetectable* → §2 rule → the pilot may run on half A to test the *instrument* (parse rate,
   band, floor) but the gate is not applied to a single-generator set without a dated
   amendment saying so. **Open:** author picks the second generator (A7).
5. *The pressure-removed matched null may itself carry residual pressure (the twin still
   describes the option set)* → `construct-audit` genealogy → report D on the null twin as a
   violating fraction next to the primary; a null-twin violating fraction > 0.5 on a scenario
   flags the twin as not pressure-free (excluded from the null, counted). **Implemented now**
   in `analyze_pilot.py` (null KDG reported; per-scenario fractions saved).
6. *Cross-model D agreement may be near 1.0 (Huang et al.)* → A9 → the pilot reports nothing
   cross-model (Tier 1 only) but saves everything needed; the Tier-2 session adds the
   agreement number before any model-vs-model verdict.

Question behind the question: whether the panel's *instrument* works on a 7B open model
(parse rate, band, floor) is separable from whether the *gap* exists; the pilot answers the
first on half A alone and the second only with both halves. The author's choice on A7
decides whether the first pod is an instrument check or the gate.
