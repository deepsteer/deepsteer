# CST-G0 gate report

Zero GPU, zero API billing (model calls ran on account logins: Claude Code subagents, `claude -p`,
`codex exec`). Rules: `papers/CST_PLAN.md` v0.1, §8 Tier 0. Stopped at the gate: no pod, no
generation, no training. Sources: `SOURCES.md` (primary-source verification), `T04_POWER.md`,
`T06_T08_SPECS.md`, `pins.yaml`, `outputs/t0*.json`. Anomalies: `papers/ANOMALIES.md` CST-A1 to
CST-A4. **Decisions: `CST_PLAN.md` Amendment 1 (author, 2026-11-04).**

## 1. Checklist (CST_PLAN §8, Tier 0)

| item | result | section |
|---|---|---|
| T0.1 tenets, `T_core`, document `T_conf` | v2 sets: `C_anth` 401, `C_spec` 430 tenets, every quote verified against the pinned body; v1 (260, 269) reviewed by a second family and revised flag by flag; v2 re-reviewed (inaccurate 0.10 and 0.07, from 0.44 and 0.33); `T_core` 116 pairs at two of two, 207 splits for the third judge; document `T_conf` 52 and 55 tenets; matching reliability below the 0.8 bar (CST-A3); author sheet of 15 | §2 |
| T0.2 judge panel spec, Layer A licenses | `deepsteer.judges.panel` (retest, cross-family κ, pooled label, the `T_conf` disagreement rule; a float bug that admitted every 2-1 split fixed and tested); MoralExceptQA CC BY-NC-SA 4.0, Scruples data unlicensed (code Apache-2.0): both evaluation-only by id, recorded in `DATASET_LICENSES.md`; panel families are the author's choice (§5 D4) | §5 |
| T0.3 pressure triple | `scripts/cst_pressure.py`: the operator cell is the frontier exhibit's `dl_chat_known_gap` byte for byte (tested); renders for Qwen3.8-27B's template (sha `c3cf9e34…`, matches the registry); two construct findings (§3) | §3 |
| T0.4 power table | `T04_POWER.md`: panel k = 4 (MDE on ΔS 0.041 at f 0.5), H1_held k = 8 (0.059), no S on the 17 engaged items; MoReBench 200 cannot resolve the −0.87 bar (MDE 2.05); second derivation agrees within 3% | §4 |
| T0.5 corpus spec, prompts, coverage, overlap | `deepsteer.corpus` (13 registered templates with shas, coverage tracker that records `T_conf` leaks, 8-gram index); the constitution body shares 0 of its 8-grams with 5.4M evaluation 8-grams across seven sets | §6 |
| T0.6 reward-pair classes, N spec block | `T06_T08_SPECS.md`: A4's rubric as drafted reads the trace (a CoT term); registered fix. Spec block for the span-removal check written | §7 |
| T0.7 lit verification | `SOURCES.md`: all eight author lists match; seven summaries corrected; "first" is not safe for the format contrast or for saying-versus-doing; it survives, scoped, for the necessity read | §8 |
| T0.8 VALIDATE plan | `T06_T08_SPECS.md` V1-V8, about 2 A100-h; local tests for the Tier 0 library (16 tests) | §7 |
| T0.9 pins | `pins.yaml`: sibling base `1b75feb7`, Thinking-2507 `144afc2f`, gpt-oss-120b `b5c939de`, judge candidates, `faithful-cot` `f9e1f295` (no license: re-implement) | §5 |
| T0.10 A0s rides S0 | noted; its render needs the context and channel decisions in §3 | §3 |
| document re-fetch (§2.1) | `C_spec` identical; `C_anth` differs in the site footer only (CST-A1); bodies cut by the page's own rule ("the preface and the acknowledgements are not part of the official constitution") | §2 |

## 2. Tenets, `T_core`, `T_conf` (T0.1)

**Documents.** Both pins verify. The bodies are cut by the page's own rule (`C_anth`: from
"## Overview" to "## Acknowledgements", 39 sections, body sha `c2a56f02…`; `C_spec`: from
"## Overview" to the end, 81 sections, `6bae7045…`). The live `C_anth` page differs from the pin
in one footer sentence (CST-A1).

**Procedure.** Jakkli, Rajamanoharan and Nanda describe section-by-section extraction with Claude
Opus 4.5 and a prompt asking for "verifiable behavioral claims with surrounding context
preserved", then manual review, merging and priority assignment; the prompt is unreleased and
their tenet files carry no license, so CST follows the description and copies nothing. Each tenet
carries a verbatim quote (checked mechanically against the pinned body: every one passes), a
one-sentence claim, its conditions, a kind and the document's own priority level.

| stage | C_anth | C_spec |
|---|---|---|
| v1 extraction (Claude Opus 5.5 subagents, 2 per document) | 260 tenets, 6 of 39 sections without one | 269, 9 of 81 |
| second-family review of v1 (codex, GPT family) | 10 not supported by their quote; 114 inaccurate; 58 missed behaviors; 70 merge/split notes | 2; 90; 100; 100 |
| v2 revision (every flag answered, accepted or rejected with a reason) | **401**: 109 kept, 84 revised, 156 split parts, 52 added, 3 dropped, 8 missed items rejected | **430**: 116, 71, 184, 58 added, 1 merge, 27 rejected |
| second-family re-review of v2's 606 revised, split or added tenets (2026-11-04, after G0) | 292 re-reviewed: 2 unsupported; 28 inaccurate (0.10, from 0.44); 109 further missed behaviors; 35 merge/split | 314: 1; 22 (0.07, from 0.33); 138; 49 |

- **What the review caught.** Dropped qualifiers ("by default", scope), hedges hardened into rules
  ("might" and "should probably" written as unconditional), examples generalized into blanket
  rules, and quotes carrying only part of the claim. Over-hardened tenets would inflate S_tenet
  (departures from rules the document does not state) and would train the SDF corpus on them, so v1
  is superseded. v2 is the candidate set of record once its re-review lands.
- **Deference rider.** The revisers accepted every accuracy flag (some in part) while rejecting
  most merge proposals and some missed items. The re-review of v2 flags 0.10 and 0.07 of the
  changed tenets as inaccurate (from 0.44 and 0.33), so the revisions answer the reviewer's own
  reading. The reviewer keeps finding further behaviors (109 and 138 missed), so the decomposition
  has no natural floor at this granularity (D16).
- **Granularity.** v2 is about twice the Nanda group's counts (205 and 197, on a December 2025
  Spec revision), because splitting kept each hedge and condition in its own tenet. Split parts keep
  their v1 root (`A012`, `A012b`), so a coarse layer (about 310 and 330 roots) exists without
  another pass. The plan's ~200 was a target, not a bar; the granularity enters cost at S1 (k
  documents per tenet) and at every judge prompt that lists the tenets (D16).
- **Author calibration (2026-11-05; `outputs/t01_author_marks.json`).** `T01_AUTHOR_CHECK.md`: 15
  v2 tenets (10 `C_anth`, 5 `C_spec`), seeded and stratified by kind, one question each, a split
  tenet judged with its sibling set. The author marks 13 of 15 items faithful (0.87; 24 of 28
  tenets counting sibling sets, 0.86). The two misses are both `C_anth`: a claim that drops the
  quote's severity and reversibility terms, and a condition that should read "when Claude reasons
  before responding".

**`T_core`.** Candidate pairs are TF-IDF top-3 in both directions united with free matching by
each family against the full other list. The lexical filter alone was not enough (CST-A3): it held
4 of 13 matches in a 25-tenet recall check (0.31). On v2: 2,601 candidate pairs (1,885 lexical,
Claude's 269 fresh free matches, 850 free matches carried from v1 where both ids survive). With
the tightened matching prompt (Amendment 1, D17) Claude votes "same behavior, same conditions" on
323 pairs; codex votes on all 323 and on a seeded 15% of Claude's no votes (342 of 2,278): it
agrees on 116 of Claude's yes votes (0.36) and says yes on 7 of the sampled no votes (0.02,
projecting about 47 of the 2,278). **`T_core` at two of two: 116 pairs (77 `C_anth` tenets, 79
`C_spec`).** 207 split pairs and the projected codex-yes, Claude-no pairs wait for the third
family (D4, GLM-4.7-Flash at S0).
The matching judgment's reliability is below the plan's 0.8 panel bar: Claude retest κ 0.78
(agreement 0.89, 120 balanced pairs), cross-family κ 0.69 on the v1 lexical set, and Claude
voted yes on 0.73 of the pairs it had itself proposed (CST-A3). No claim rests on `T_core`'s size.

**Document-internal `T_conf`.** The rule is narrowed to explicit, specific gating (§9 rider 6).
Claude's pass on v2: `C_anth` 13 competing pairs and 31 explicit conditioning links, 52 of 401
tenets; `C_spec` 19 and 26, 55 of 430. Codex's check of each competing pair is pending (quota; rerun scheduled). Panel-
detected conflicts join at S0 (D7).

## 3. The pressure triple (T0.3)

Implemented on `kdg_pod_lib.letter_chat_messages`, so every cell is the KDG/ST cell it names:

| set | condition | messages |
|---|---|---|
| panel | plain | agent frame, pressure-removed twin (`dl_chat_neutral_pressure_removed`) |
| panel | user | agent frame, primary: the incentive in the user turn (`dl_chat_neutral`) |
| panel | operator | the frontier exhibit's cell exactly: the known-gap operator system prompt over the primary (`dl_chat_known_gap`) |
| panel | operator_pr | the same operator prompt over the pressure-removed twin (the fourth cell of incentive × order) |
| H1 | plain | agent frame, primary |
| H1 | operator | the known-gap sentence frame naming the **rule-following** option, over the primary |

Findings that change the plan's text:
1. **The operator condition is an order, not the incentive moved to another principal.** The
   frontier cell names the violating option and orders it; the KDG incentive is woven into the
   scenario narrative (the twins differ in two or three clauses, not a separable sentence), so "the
   same incentive as an operator instruction" cannot be rendered mechanically. On Qwen at dose 0
   the order moves the action 0.655 and the user-turn incentive 0.031 (CST-A2). Under `C_anth`, an
   operator's instruction is sometimes what the applicable tenet directs, so S(operator) can fall
   by deference without any gap between trace and action (CST-A2's R_authority).
2. **On H1 the frontier prompt would order protection.** H1's `violating` option is the protective
   rule-break and its twin removes the harm, not an incentive. The H1 operator cell orders the
   rule-following option instead (tested), and H1 has no user cell.
3. **Effort and channel.** At effort `medium` (ST's effort of record on Qwen) the template adds no
   system block; at `xhigh` (the shipped default the plan names) it injects an effort system block
   and the operator prompt is appended inside it. Under A0s the constitution would also sit in the
   system channel, i.e. in the operator's voice, merged with the operator's order.
4. **Context.** `C_anth`'s page text is 36,313 tokens on the subject's tokenizer; ST's vLLM config
   caps context at 8,192. A0s and corpus generation need `max_model_len` ≥ 44k (VALIDATE V1-V2:
   64 KiB of KV per token on the 16 full-attention layers, 2.4 GB for the cached prefix).
5. The known-gap text carries a doubled period (`…form.".`), inherited from the KDG renderer and
   kept for byte parity with the frontier rows.

## 4. Power (T0.4; `T04_POWER.md`)

- **Panel, k = 4:** MDE on ΔS 0.041 at an applied-tenet fraction f of 0.5 (0.066 on worst-case
  variance). **H1_held, k = 8:** 0.059. **Cell A′ forced, k = 8:** 0.069. **H1 engaged (17):** no S;
  ΔP's null MDE there is 0.142, and re-running `none` on the selected items moves P by −0.088.
- **Second derivation:** the scenario bootstrap of ST's thinking-on `none` twin on H1_held agrees
  with the closed form within 3% at k = 2, 4, 8 (k 8: 0.037 against 0.038); the twin shows no
  run-to-run excess (τ² = 0).
- **f is the largest unknown.** Below f 0.3 the panel MDE at k 4 passes 0.06. Measure f on A0's
  first 20 panel scenarios in S0 and re-run the table before any arm trains.
- **MoReBench:** at 200 scenarios the matched-length MDE is 2.05 (Qwen basis), so a 200-subset
  read cannot place an arm on either side of −0.87; the full 500 gives 1.37. "Beats −0.87" at 500
  needs a true cost above about +0.5.
- **Capability:** the bar is a point rule (Δ ≥ −0.02) at paired SE 0.021: an arm with no true loss
  fails it 17% of the time.
- **MACHIAVELLI k = 2:** MDE 3.7 on violations, 2.8 on reward, measured on Qwen3-30B-A3B-
  Instruct-2507 (another model; bias table in `T04_POWER.md` §8).
- Ungrounded until S0: thinking-on behavior on the panel's pressure items and under the operator
  cell; f; the panel's κ; A0s's effect size.

## 5. Judge panel and pins (T0.2, T0.9)

- `deepsteer.judges.panel` takes labels as data: retest κ (bar 0.8, S0's gate), pairwise
  cross-family κ, pooled majority label (ties give no label), Layer A accuracy, and the `T_conf`
  rule. With three families disagreement takes only the values 0, 1/3 and 2/3, so the
  calibration-quantile threshold reduces to one of two rules: any dissent, or a three-way split.
- **Premise balance.** Each candidate judge family was post-trained toward one document or none:
  gpt-oss-120b toward the Model Spec (ST Amendment 47's premise check), Claude toward `C_anth`, the
  OLMo-3.1 Think line toward neither. A panel of all three balances the two documents' home
  fields; a panel without Claude has no judge trained on the document of record.
- Pins in `pins.yaml`, all read from the Hub API on 2026-11-04; the subject and the MRB/ST pins are
  unchanged.
- `faithful-cot` has no license at its root (Apache-2.0 covers only its vendored VERL fork), so
  vendoring is not licensed; re-implementing `grad_cot_necessity` (torch and transformers only) is
  the open route.

## 6. Corpus tooling (T0.5)

- 13 templates across the four layers, each with an id and sha (`deepsteer.corpus.manifest()`);
  the rewrite template forbids lettered options (MRB-A16, ST-A39); no template names an
  evaluation set (tested).
- Overlap: the CST rule (drop on any shared word 8-gram) is stricter than MRB's registered rule
  (drop at a 10% share), which the plan cites as its source. Indexed: every KDG scenario file,
  H1, cell A′ and its twins, refusal retention, MoReBench's 500, MACHIAVELLI's 30 test games (5.4M
  8-grams). The `C_anth` body shares none of its 8-grams with them; `C_spec` shares three, all
  stock phrases from MACHIAVELLI ("is there anything else i can help you"). Not indexed yet: the
  capability items (ids only on disk) and Layer A.
- Coverage: the tracker counts per tenet and kind and records any `T_conf` tenet that seeds a
  training item instead of counting it.

## 7. Objectives, the N block and VALIDATE (T0.6, T0.8; `T06_T08_SPECS.md`)

- **A4's rubric reads the trace as drafted.** "Rubric = applied tenets" scores the action against
  the tenet the trace invoked, so a rollout raises its reward by citing a permissive tenet: a CoT
  term, and the in-conflict class the plan forbids. Registered fix: the rubric is the panel's
  tenet reading of the scenario text, fixed before rollouts; the judge sees the action only.
- **Answer-only renders** for A1c, A2 and A3: the rewrite has no trace of its own, so DPO pairs are
  answer-only with the think block empty and masked; otherwise the chosen answer is trained beside
  a trace it did not come from.
- **B1 dependency gap:** "our post-training" from the 30B-A3B base to a thinking model needs a
  thinking-format SFT set the plan does not specify (nearest: Thinking-2507's own traces).
- **N:** the span-removal check's spec block is written (matched-length non-moral span from
  another rollout as the replacement; whole-trace removal as the upper rung; intact re-read as the
  floor; sentence-boundary alignment; branches). Grad-Nec is named "gradient necessity share",
  the authors' own description being a first-order linearization.
- **VALIDATE** (V1-V8, about 2 A100-h): context ≥ 44k with prefix caching; LoRA on the DeltaNet
  and attention projections; vLLM native LoRA serving against the merged path (MRB-A15: Qwen's
  merged checkpoint kept 0.60 of the attached adapter's letter shift); SDF throughput (second
  derivation: 4.1-7.1 h for 10M tokens at 20-35% utilization, consistent with the plan's ~8);
  format survival after recovery; Grad-Nec memory; render shas.
- The S0 harness is not built: its effort level, operator cells, k, judge families and context
  length are decided at this gate.

## 8. Literature and novelty (T0.7; `SOURCES.md`)

All eight author lists match their primary sources. Corrections to the plan's summaries:

| plan says | the source says |
|---|---|
| Teaching Claude Why: ablating the rewrite step raised misalignment about 2% → 19% | dropping the **response**-rewrite step gives 19% against about 1% for the full pipeline; dropping the prompt-rewrite step gives 2%, within error |
| documents beat chat-format SFT of the same content (with stories) | the chat comparator converted the constitution documents, with no stories; documents won on recall and hallucination evals, with "less of a delineation" on open-ended questions; the advice sets had extended thinking off |
| gains persisted through RL (the 3M-token set) | persistence through RL is reported for constitutional SDF |
| Jakkli et al.: failures where the document gives competing directives (authority conflicts, credential-gated safety, think-then-ignore) | five categories: authority conflicts, credential-gated safety, form-over-substance, think-then-ignore, unilateral agentic action; think-then-ignore is a GPT-family failure on the Model Spec |
| Kaufmann et al.: length penalties and preference rewards already produce obfuscation | in toy environments, simplified length-penalty and preference CoT rewards can be in conflict; the paper also reports counter-evidence |
| OpenAI: Scruples sycophancy is the least monitorable domain | "very low monitorability" on Scruples |
| Anwar et al.: on GPT-OSS-120B | not verifiable from the abstract (PDF unread) |
| Open Character Training: about 4,000 introspection transcripts (§4.4 scaling) | 12,000 (10,000 self-reflection, 2,000 self-interaction); the DPO stage also carries a KL term |

**Prior art the plan does not cite.**
- Li, Wichers, Price, Marks, Kutasov, "Model Spec Midtraining" (arXiv:2605.02087): spec-document
  midtraining on Qwen3-32B with reasoning, against a deliberative-alignment chat SFT baseline,
  with a spec based on Claude's constitution.
- "Teaching Qwen Why" (posted 2026-11-02): an open replication on Qwen 2.5 base models (story
  SDF, difficult-advice SFT; no constitution-document SDF, no documents-vs-chat test).
- Cho et al., Constitutional Midtraining (arXiv:2607.26654); Khursheed, Sosis, Roger
  (arXiv:2606.04413), SDF on Qwen3 with reasoning on.
- For saying versus doing under a system-prompt mandate: "The Profit Alignment Problem"
  (arXiv:2609.07731), Jakkli et al.'s think-then-ignore, and the program's own KDG paper
  (arXiv:2610.08670), which CST extends.

**Novelty scope after the pass.** "First" is not safe for the format contrast on an open thinking
model or for saying versus doing under operator pressure. A necessity read (gradient or
resampling) of the chain of thought on a constitution-trained model has no prior work in the
search (about eight searches plus direct reads, nothing after 2026-11-02 swept); "to our
knowledge the first", with that scope, is defensible there. What the line can still claim
without "first": the format contrast measured with instruments the prior work did not apply
(saying-to-doing under operator pressure, action under reward pressure, process at matched length,
necessity), on the full constitution, on a subject whose base is unreleased.

## 9. Anticipated review (pre-review protocol)

1. Prior art runs the format contrast on an open thinking model (Model Spec Midtraining on
   Qwen3-32B) → novelty gate → R1 is reworded as an extension measured with CST's instruments, and
   the paper leads with S under operator pressure and the necessity read (author, D1) (zero cost).
2. The operator cell orders the violating option, and `C_anth` sometimes sides with operators →
   construct audit → S(operator)'s headline is read on items where the document directs refusing
   the order, labeled by quote-and-classify (D2, zero GPU; CST-A2).
3. "S is arbiter-free" holds only for the trace's own conclusion; departure from a tenet's
   direction in a scenario needs the panel → split S into S_concl (action against the action the
   trace concludes it should take: arbiter-free) and S_tenet (against the applied tenet's
   direction: panel) (D3, a PRIMARY wording; zero cost).
4. A4's "applied tenets" rubric makes the reward read the chain of thought → reward-pair taxonomy
   → rubric fixed per scenario before rollouts (implemented in `T06_T08_SPECS.md`; zero cost).
5. The −0.87 process bar is not testable at 200 scenarios and the capability bar fails a
   no-loss arm 17% of the time → estimator traps → full 500 per finalist arm (+~1 A100-h per
   arm) and pass probabilities printed beside the capability bar (D9, D10).
6. (Not in the plan before this pass.) The conflict rule "conditioned on another tenet's priority
   ordering" would, read literally, admit most of `C_anth` (every helpfulness tenet yields to
   safety and ethics by the global ordering), and nothing in `T_conf` seeds training, so the SDF
   corpus would lose most of its coverage → `conditioned_on` only for explicit, specific gating
   (implemented in the extraction instructions and `deepsteer.constitutions.document_conflicts`).
7. (Not in the plan.) Answer-only renders for A1c, A2 and A3, or the chat and DPO arms train the
   subject to answer without reasoning or beside a trace it did not write (implemented as
   registered conditions in `T06_T08_SPECS.md`).

Implemented now: 4, 6, 7; the H1 operator inversion (§3); the float bug in the conflict rule;
license rows; pins; the novelty table. Open with costs: D1-D17 below.

**Question behind the question.** Is this line's first deliverable the partner recipe or the
paper? For the recipe, the prior art is good news: the published midtraining and advice-set
designs can be adopted rather than re-derived, and the format contrast A1c is optional (about 10
A100-h saved). For the paper, the instruments are the contribution and the arms are the means,
so S0 (the calibrated instrument on `none` and A0s) carries most of the value and could ship as a
measurement result before any adapter trains.

## 10. Decisions for the author

| # | decision | recommendation | cost |
|---|---|---|---|
| D1 | paper framing after the novelty pass (thesis-level) | R1 kept, worded as an extension of Model Spec Midtraining's comparison measured with CST's instruments; the paper leads with S under operator pressure and N | 0 |
| D2 | operator cells | frontier cell of record plus `operator_pr`; headline S(operator) on items where `C_anth` directs refusing the order (quote-and-classify labels, zero GPU) | +33% panel rollouts in S0 (about +2 A100-h) |
| D3 | S's definition | register S_concl (arbiter-free) as primary and S_tenet beside it | 0 |
| D4 | judge panel | gpt-oss-120b and Olmo-3.1-32B-Think self-hosted for bulk scoring; Claude on the account CLI for calibration, `T_core` and `T_conf` votes; an API cap only if the author wants Claude in bulk scoring | 0 self-hosted beyond pod time; API cap TBD |
| D5 | effort | `medium`, ST's effort of record on Qwen (no injected system block; ST rows comparable) | 0 |
| D6 | A0s render | the `C_anth` body (not the page) in the system channel under a fixed one-line header, the operator's order appended after it | context per V1 |
| D7 | `T_conf` panel rule | any dissent tags an item (conservative for training exclusion); the three-way-split rule reported beside | 0 |
| D8 | `faithful-cot` | re-implement Grad-Nec against `f9e1f29`, test against the reference on a tiny model | 0 |
| D9 | MoReBench | full 500 on every arm that passes the other bars; 200 only as a gross-failure screen | ~+1 A100-h per arm |
| D10 | capability bar | keep ST's point rule for comparability; print its pass probabilities beside it | 0 |
| D11 | 1M-token pilot go/no-go | proceed to 10M if the pilot projects ≤ 8 A100-h for 10M | — |
| D12 | k | panel 4, H1_held 8, cell A′ 8; re-run T0.4 once S0 measures f on 20 scenarios | — |
| D13 | B1 thinking-SFT data | specify before S4 (nearest source: Thinking-2507's own traces) | 0 now |
| D14 | overlap rule | any shared 8-gram, as the plan says; the pilot's drop rate reported; above 5% an amendment | 0 |
| D15 | `T_core` third family | the D4 panel's third member votes the split pairs (§2) | 0 |
| D16 | tenet granularity | v2 (401 / 430) as the scoring set, the v1-root layer (309 / 326) for corpus coverage targets, so k documents per tenet is counted per root | 0 |
| D17 | the matching discriminator (CST-A3) | run the tightened-prompt retest before the third family votes | 0 (account login) |

## 11. Referee pass

1. *"Your tenet decomposition is one model family's reading of the document; everything
   downstream inherits it."* Conceded as scope, and the reason for the second-family review (§2):
   the reviewer's disagreements are listed, not resolved by the extractor. Jakkli et al. call their
   own decomposition "opinionated"; ours follows their described procedure, whose prompt is
   unreleased, so the two sets are not interchangeable and no count of ours is compared with
   theirs as if they were.
2. *"Your operator condition measures obedience to a direct order, which the document you train
   on sometimes endorses."* Conceded; it is CST-A2. The registered response is D2 (the document's
   direction labeled per item, and the `operator_pr` cell); without D2 the headline S(operator) is
   not a saying-to-doing gap on the items where the document sides with the operator.
3. *"Nothing here is new: the lab and an open replication already compared documents with chat on
   Qwen."* Conceded for the comparison (§8). What prior work did not measure is the gap between
   the model's own trace and its action under an operator's order, the trace's necessity for the
   action, and the process and capability costs at matched length on the same rollouts; those are
   the paper's claims after D1.
