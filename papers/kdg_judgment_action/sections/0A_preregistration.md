# Appendix A. Pre-registration record {#app:prereg}

The panel specification (`papers/KDG_PANEL_SPEC.md`) was committed as the pre-registration of
record before any scenario was generated. Every construction or analysis decision taken after
that is a dated amendment in the same file; a change made after data were seen is a fork and
carries verdicts under both choices. \Cref{tab:amendments} lists the seventeen amendments in one
line each, split by whether any model data existed when they were committed. Verdict rules and branch
names are quoted as registered ("installed", "widened", "inherited, not installed"); where the body
describes a model on its own terms it says the model "carries" the gap, which claims presence and not
the step that produced it.

| id | date (2026) | committed | content |
|---|---|---|---|
| A1 | 09-13 | before any model data | decision-position conventions: letter-only reply, first generated token; `Answer:` anchor for judgment frames; raw-frame prefix; single-token letters verified |
| A2 | 09-13 | before any model data | F3 tool menu presented as lettered tools; tool-name parse accepted |
| A3 | 09-13 | before any model data | twins are derived variants, not counted in the primaries; harm twins ride every cell, never the gate |
| A4 | 09-13 | before any model data | F2 implemented as a two-turn exchange with fixed pushback; final letter is the action |
| A5 | 09-13 | before any model data | OLMo-3 template injects a default system prompt; the known-gap band replaces it (recorded) |
| A6 | 09-13 | before any model data | two-stage harness calibration; stage 2 on real replies before any verdict |
| A7 | 09-13 | before any model data | generator split resolved: GPT-5.5 for half B; cross-provider external labels |
| A8 | 09-13 | before any model data | novelty re-centered after the verified literature pass |
| A9 | 09-13 | before any model data | rival reading from Huang et al.: the model axis may collapse |
| A10 | 09-13 | before any model data | pilot power from closed form; measured MDE after the pilot |
| A11 | 09-13 (ratified 09-14) | before any model data | neutral options count as non-violating for the reference; harm level stratified within generator |
| A12 | 09-14 | after the pilot (fork, both verdicts) | judgment-stability rule: option-level stays primary; binary rule reported as a fork in the pilot only |
| A13 | 09-14 | before the full-panel pod | reference-strictness ladder L0/L1/L2 with four judgment frames; verdict at the strictest level with 40 paired scenarios |
| A14 | 09-15 | before generation | construction round 2 (F1/F3/F5) and the F4 paraphrase-swap cell with both branches |
| A15 | 09-15 | before computation | continuous log-prob readout as a secondary instrument with a coherence gate and three branches |
| A16 | 09-19 | after the data (fork, both verdicts) | reversal clause evaluated on both instruments |
| A17 | 09-19 | before computation | three-cell weights contrast with paired CIs on both readouts, raw-frame matched null, sub-branch verdict rules; exploratory items labelled |

Table: Amendments to the pre-registration. "Before computation" means the amendment was
committed before the analysis it licenses was run on data already on disk. {#tab:amendments}

The multi-model work of \Cref{base}, \Cref{recipe} and \Cref{deliberation} ran under a second specification
(`papers/KDG_PHASE1_SPEC.md`), pre-registered and pushed before its first computation, with seventeen dated
amendments of its own (\Cref{tab:phase1}). Four of them are forks made after data were seen (P1-A1, P1-A5,
P1-A14, P1-A15), each reported beside the choice it forks; P1-A13 was added after P1-A11's result and is labelled post-hoc
where it is used; the others were committed before the data they govern existed or
before those data were read.

| id | date (2026) | committed | content |
|---|---|---|---|
| P1-A1 | 09-26 | after the scale-check arrays were seen (fork) | frame-specific output-scale normalization beside the averaged-scale verdict of record |
| P1-A2 | 09-26 | during the first pod, before its data were read | verdict sets, mass floors and anchored-rollout rule for the chat twin cell, the stage sweep and the dose arm |
| P1-A3 | 09-27 | before the pod | bridge cell at the first templated checkpoint (raw vs chat), with the rule for the base cell's status |
| P1-A4 | 09-27 | before the pod | every instruct model in the second session also read under its template; no raw-only instruct finding |
| P1-A5 | 09-27 | after the dose probe failed its bail (fork) | forced-answer readout at the pre-registered budgets; "truncated reasoning" label |
| P1-A6 | 09-27 | before the rider's data were read | 2,048-token rider at 8 rollouts per scenario, per-scenario agreement as its readout |
| P1-A7 | 09-28 | before computation | at-rest lean on a model-free set, and per unit of output scale |
| P1-A8 | 09-28 | before the pod | truncated-filler control and norm-salience arm, with branch rules |
| P1-A9 | 09-28 | before computation | known-gap positive control per model with its validation rule; per-model screens; dose arm on a second recipe |
| P1-A10 | 10-01 | before computation (04d604f) | stage steps of the excess per unit of output scale on the model-free set (RL-step sharpening rival) |
| P1-A11 | 10-05 | before computation (c263eb2); post-review addition | each instruct model's excess on its own letter-only screen, with its bar |
| P1-A12 | 10-05 | before the pod (c263eb2; execution note 4d6b020); post-review addition | dose arm on the pressure-removed twins: reasoning's change in the pressure-attributable part, with branch rules |
| P1-A13 | 10-05 | after P1-A11's result, before computation (1d2616d); post-hoc, post-review addition | selection-matched null for the own-screen excess: the same screen on the twins, contrast reversed |
| P1-A14 | 10-05 | after P1-A12's result, before its interval (3500a2e); post-hoc fork, post-review addition | ratio reading of the deliberation twins (proportional-reduction rival), reported beside the registered probability-scale verdict |
| P1-A15 | 10-08 | after the shared sampling stream was found, before any re-analysis; post-hoc fork, both reported | two-way bootstrap over scenarios and rollout indices for the sampled dose arms |
| P1-A16 | 10-08 | before the recomputations it schedules | the two-way bootstrap is the estimator of record for sampled-cell claims; §9's truncated-filler verdict corrected in v2 |
| P1-A17 | 10-15 | after an external review of v2, before any recompute it lists | exact bounds for the stage steps; the paired Meta-vs-Tulu 3 excess; two-way intervals primary, with the audit bounded as in \Cref{app:calibration}; clustered overlap sensitivity; three-option movement and GPT-OSS-20B transition accounting on the rule-harms set; the \Cref{app:three-cell} crossing audit; sample flow and per-rollout export |

Table: Amendments to the Phase 1 specification. {#tab:phase1}

GPT-OSS-20B (\Cref{recipe}, \Cref{rule-harm}) ran under a third specification (`papers/KDG_GPTOSS_SPEC.md`,
v0.1 approved 2026-10-04 and pushed before any cell ran), with 29 dated amendments (\Cref{tab:gptoss}). The
paper uses two of its results: the dose-0 row of \Cref{tab:recipe} and the rule-harms set of
\Cref{rule-harm}. Three amendments were written after a verdict they bear on (G-A10, G-A12, G-A13).

| id | date (2026) | committed | content |
|-----|-----|----------|------------------------|
| G-A1 to G-A5 | 10-07 | before any build output or cell | C0 tie rule; C0 printed with its Wilson interval; decision-token PR references; bail conditions, timing gate, save list; local-test assertions |
| G-A6 to G-A9 | 10-07 | before the data they govern | operational details; C0 near-miss discriminator; rerun branches after a failed forward-vs-generate gate; batch-invariance arm |
| G-A10 | 10-07 | after the C0 verdict, before any diagnostic number (post-hoc) | C0 failure diagnosis tree; the registered verdict stays of record |
| G-A11 | 10-07 | before any C0-dm code or pod | dose-matched C0 (C0-dm) and a wording rule |
| G-A12 | 10-08 | after C0-dm's verdict, before the prediction (post-hoc fork) | sampling-noise prediction for C0-dm, both verdicts reported |
| G-A13 | 10-08 | after both C0 verdicts, before any ceiling was computed (post-hoc calibration correction) | C0 clears for the dose-0 readout through token identity and G-A12; model-relative bar $0.9\,\kappa^*$, prospective, with past C0s keeping their 0.80 verdicts |
| G-A14 | 10-08 | before computing | $\kappa^*$ under each model's effective sampler |
| G-A15 to G-A20 | 10-08 | before any dose-stated code or pod | dose-stated readout of GPT-OSS's reasoning mode, its second primary, the main stage after the pilot, re-derivation after a crashed pod, batch-1 forward readout |
| G-A21 to G-A23 | 10-09 to 10-10 | before the labels or counts they govern | letter balance and risk-reversed cell; split risk constructs and actor-cost cell; actor-cost set of record and judgment readout at the default reasoning level |
| G-A24 | 10-10 | before the continuation pod and before the set existed | the rule-harms (compliance-harm) set: generation, two-rater keep rule, pilot gate |
| G-A25 | 10-10 | before the pilot | an instrument gate from an author sample; compliance-harm split of the 586; pilot guardrails and the author's spot check for the rule-harms set |
| G-A26 | 10-11 | before the pod | the rule-harms cell on GPT-OSS-20B, its branches and scope |
| G-A27, G-A28 | 10-12 | before any tag, split or pod | recoverability tag; deferral split both ways; judgment readout on the crossings; dose curve; carrier pricing |
| G-A29 | 10-12 | before the pod | the rule-harms set under OLMo-3's and Llama-3.1's forced dose arms, paper-matched design |

Table: Amendments to the GPT-OSS-20B specification, grouped where consecutive amendments share a purpose
and a commitment status. {#tab:gptoss}

**The C0 bar, both outcomes.** C0, the construct check of GPT-OSS's dose-0 readout, was registered with a
fixed bar of 0.80. Both registered checks failed it: C0 0.688 (Wilson 0.566 to 0.788) and the dose-matched
C0-dm 0.766 (0.649 to 0.853, a near-miss). After those verdicts, G-A12's fork found the C0-dm miss to be what
a valid readout under temperature-0.7 sampling produces, and G-A13 replaced the fixed bar with
$0.9\,\kappa^*$, where $\kappa^*$ is the agreement an exact readout would reach under the check's own
sampling (\Cref{app:calibration}). Under that bar, which is a post-hoc calibration correction, GPT-OSS
passes (0.668). Both outcomes stand: fail at the registered 0.80, pass at the model-relative bar; the dose-0
row is cited with both.

Every result document carries a referee pass (three damaging objections, answered or conceded)
and updates the program's synthesis file in the same commit; anomalies live in a ledger with a
priced discriminator each. Six anomalies were opened by this panel: reference instability
(resolved on the continuous instrument), the instruct model's raw-frame mass floor (resolved by
calibration: end-of-turn mass), the pilot's family structure (resolved as noise), the F4 generator
dependence (resolved as a graded register difference), the exploratory family structure on the
continuous instrument (open, priced), and the instruct model's negative no-pressure frame gap in
the raw frame (open, priced).
