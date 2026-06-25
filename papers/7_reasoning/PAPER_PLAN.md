# Paper 7 — Plan: Functional vs Imitated Moral Coupling in Reasoning Refusal

> ## REVISION 2026-06-24 — Zhao et al. reopens the keystone (READ THIS FIRST)
> The causal keystone stalled: the refusal direction we could ablate (EOP / t_post-inst)
> is a WEAK causal handle — held-out validation showed no single direction cleanly ablates
> GPT-OSS refusal, so a moral null wasn't interpretable. **Zhao, Huang, Wu, Bau, Shi, "LLMs
> Encode Harmfulness and Refusal Separately" (arXiv:2507.11878; venue NeurIPS-2025 per the
> revision brief — VERIFY acceptance for the final cite; method verified vs the arXiv
> abstract + CHATS-lab repo 2026-06-24)** hands us the causal handle the keystone lacked.
>
> **Verified claims (primary-source checked):** (1) harmfulness encoded at **t_inst** (last
> instruction token), refusal at **t_post-inst** (last prompt token) — separate concepts;
> (2) **reply-inversion** — steering the t_inst harmfulness direction flips the model's
> harmful/not judgment (the strong causal yardstick the refusal direction lacks); (3)
> accept-harmful finetuning leaves the t_inst harmfulness representation ~intact while
> degrading refusal (= our distills: refusal degraded by R1-distillation, harm-comprehension
> survives — corroboration, not just a confound). Repo: github.com/CHATS-lab/
> LLMs_Encode_Harmfulness_Refusal_Separately (diff-of-means, `--use_inversion`).
>
> **NEW SPINE:** comprehension (harmfulness, t_inst) and refusal (t_post-inst) are encoded
> at different sites (Zhao); we EXTEND this into the reasoning regime — harm-comprehension is
> distributed across the CoT (peaks mid-trace, Phase 2a), the refusal decision is distributed
> and displaced from where comprehension peaks — and contrast deliberative (GPT-OSS) vs
> distilled (R1). Using Zhao's CAUSAL t_inst harmfulness direction (validated by reply-
> inversion), we test directly whether harm-comprehension is load-bearing for the reasoning
> refusal decision. Our novelty over Zhao: the reasoning-trace extension + the deliberative-
> vs-distilled axis + the load-bearing test + positioning MFT-moral vs Zhao-harmfulness.
>
> **NEW PHASES (build on the LOCAL working tree; extend extract_refusal.py / model_registry,
> don't fork; reuse random_ablation_control three-way design):**
> - **2c — position-parameterize extraction (prereq, cheap):** extract diff-of-means at BOTH
>   t_inst and t_post-inst (current code does only t_post-inst). Validity: reproduce Zhao
>   Fig 2 clustering flip (harmfulness clusters at t_inst, refusal at t_post-inst). HUMAN GATE.
> - **2d — extract + causally validate the t_inst harmfulness direction (reopened yardstick):**
>   reply-inversion (Zhao §3.5) is the NEW sensitivity yardstick replacing the one that
>   wouldn't fire. GATE: only proceed where reply-inversion fires cleanly (coherent, held-out;
>   sweep layers, Zhao found mid ~9-13). HUMAN GATE.
> - **2e — load-bearing test, done right (reopened keystone):** with the firing t_inst handle,
>   three-way ablate harmfulness vs random vs persona; measure refusal-outcome + harm-judgment
>   change; contrast t_inst (strong) vs t_post-inst (weak) grip. Coherence gate on. HUMAN GATE.
> - **2f — position MFT-moral vs Zhao-harmfulness (near-free, run w/ 2d/2e):**
>   cosine(MFT subspace, t_inst harmfulness dir) per model. HIGH→same object (convergence
>   w/ Zhao); LOW→moral-foundations distinct from harmfulness (sharp positioning). Either is
>   publishable; it's how we position vs Zhao.
> - **Recast distills (Zhao §5.2):** extract distill t_inst harmfulness dir, confirm
>   intact/causal (reply-inversion fires) DESPITE low refusal → dissociation under
>   distillation-degrades-refusal, parallel to Zhao Fig 7. Confound → corroboration.
>
> **Datasets:** verify real harmful/harmless provenance (our Arditi/Heretic set, or
> Advbench/JBB/Sorry-Bench/Alpaca/Xstest as Zhao uses) — not placeholders.
> **Prior phases (1, 2a, 2b) STAND** as decodability/geometry + distributed-refusal results;
> 2b's distributed finding is now externally grounded (Zhao App E.1 + Siu: t_post-inst is a
> weak handle). The OLD thesis below is superseded by the NEW SPINE above.

## One-paragraph thesis
Paper 5/6 showed refusal is ~99% residual / orthogonal to the moral subspace and forced
coupling fails — except Llama-3.1, where refusal is dose-dependently *behaviorally*
entangled with moral judgment (representation intact). Reasoning models refuse partly in
the chain-of-thought, where refusal is known to be multi-directional (Yamaguchi Fig 16).
**Paper 7 asks whether genuinely deliberative reasoning (learned via RL, as in GPT-OSS-20B's
deliberative alignment) routes refusal through moral cognition — and whether distilled
models that merely imitate reasoning traces reproduce the moral *form* without the
*function*.** Hoped-for result: in GPT-OSS-20B, moral-subspace activation concentrates at
the CoT sentences that determine refusal AND is causally load-bearing; in the distilled
models, moral activation is present but not load-bearing (imitated). That distinction —
**functional vs imitated moral coupling** — is the contribution, and it matters because
safety research leans heavily on open distilled models.

**Everything runs on `main`. No branch.** New work under `papers/7_reasoning/`.

## Scope note (light)
Paper 7 measures and explains *where* moral reasoning enters the refusal decision,
including characterizing the robustness that genuine moral coupling confers — finding
that moral-grounded refusal is harder to ablate is a result, not something to avoid.
One thing held for now: the Shairah extended-refusal construction (Phase 3a) is built
**as a measurement control only** — to separate moral routing from a structural
distribution effect — and is not packaged or released as a robustness technique. Whether
to later develop an un-removability *method* as a deliverable is an open decision to be
made deliberately, not by default; it is out of scope for this paper. Findings are
measured properties of released models; we make no claims about training recipes we don't
have.

---

## Model panel (LOCKED — primary + 2 contrasts)

| key | base | reasoning model | role / axis it isolates |
|-----|------|-----------------|--------------------------|
| gpt_oss_20b | (no separate base) | **GPT-OSS-20B** | **PRIMARY.** RL-trained, deliberative alignment, MoE, gradual-trace refusal. The only model where reasoning was learned via reward -> the only clean test of *functional* moral deliberation. |
| ds_r1_llama8b | Llama-3.1-8B (= Paper 6 anchor) | DeepSeek-R1-Distill-Llama-8B | CONTRAST: distilled (form-without-function candidate). Base-shared with Paper 6's moral-coupled Llama -> longitudinal "does distilled reasoning strengthen the Paper-6 coupling" probe. |
| ds_r1_qwen14b | Qwen2.5-14B (general) | DeepSeek-R1-Distill-Qwen-14B | CONTRAST: distilled, SAME R1 teacher as the Llama distill, general base -> shared-teacher transfer probe (clean: differs from the Llama distill only in base family, not specialization). |

**Axis logic.** GPT-OSS-20B = genuine deliberation. The two distills = imitated
deliberation (both distilled from R1). If moral coupling is *functional* it should be
load-bearing in GPT-OSS-20B; if the distills show moral *concentration* without
*load-bearing* causal effect, that is imitated-from-teacher moral-talk. The two distills
sharing R1 lets a transfer match corroborate "inherited, not independently arising."

**Acknowledged uncontrolled confound (state in paper, do not fake a control):**
GPT-OSS-20B is the only MoE. A GPT-OSS-vs-distill difference is confounded between
deliberative-alignment and MoE architecture; no non-distilled MoE reasoning model is
available to separate them. Address partially via Paper 2 MoE priors; otherwise
acknowledge as a limitation. Also note the 8B-vs-14B size gap between the two distills as
a minor transfer-probe caveat (much smaller than the math-base confound it replaces).

---

## Phase 0 — Extend registry + verify provenance + smoke (no big GPU)
- **0a Provenance verification (do not assume):**
  - Confirm ds_r1_qwen14b is on **general Qwen2.5-14B**, NOT a math variant.
  - Confirm GPT-OSS-20B and both distills' repos exist and load; HF_TOKEN forwarded for
    any gated weights. Confirm GPT-OSS-20B MoE expert structure is accessible for the
    per-expert orthogonalization Phase 3b needs.
- **0b Extend model_registry.py** (REUSE, extend — do not fork):
  - Add the three ModelSpecs; append to PANEL_ORDER.
  - Add a `think` input_format (chat template with reasoning enabled) alongside base-raw /
    instruct-chat.
  - Add TWO extraction sites per reasoning model: **end-of-prompt tokens** and **CoT
    tokens** (mirror Yamaguchi G.2). Keep the fractional-depth band/layer rule and
    assert_matches_model. Conventions identical across the panel (validity crux).
- **0c Smoke** on the lightest distill (one layer, tiny prompt set) to confirm activation
  hooks work on the <think>...</think> generation structure before any full pass.
- **Validity anchor:** run ds_r1_llama8b's END-OF-PROMPT decomposition; confirm it lands
  near Paper 6's Llama numbers (refusal ~99% residual at the prompt boundary). If not, a
  convention drifted — fix before proceeding.
- **0d GPT-OSS-20B precision gate (settle here, NOT as a Phase 3b surprise).** GPT-OSS-20B
  ships mxfp4-quantized experts. Quantization noise on the very weights an intervention
  edits would make a Phase 3b moral-component *null* uninterpretable ("not load-bearing"
  vs "quantization ate the edit"). Settle precision before any causal claim:
  1. **Fit check.** Confirm bf16-dequantized GPT-OSS-20B (~21B params -> ~42 GB bf16)
     fits ONE A100-80GB WITH activation hooks + the paired harmful/harmless forward
     passes Phase 1/3b need (weights + KV + hook buffers). Prefer running Phase 3b
     **dequantized** for a clean intervention; if it does not fit one A100, record the
     fallback (mxfp4 with documented caveat, or 2x GPU) before proceeding.
  2. **Mandatory positive control at the chosen precision.** Reproduce the known
     refusal-direction-ablation -> compliance effect on GPT-OSS-20B at that precision
     (ablate the refusal direction; confirm refusal drops / compliance rises, the
     Arditi/Heretic-universal effect). The moral-component ablation in Phase 3b is only
     interpretable if this refusal-direction positive control FIRES at the same precision.
     If it does not fire, a moral null is meaningless — fix precision first.
  - Pin the chosen precision in the runner so Phase 1/2/3 all use it identically.

**Human gate after Phase 0.**

## Phase 1 — Two-direction decomposition (does the single-direction gap open up?)
Reuse extract_refusal.py: subspace_projection_fraction (-> moral/persona/residual
fractions), linear_separability_gap (single-vs-full CV AUC — the gap that was 0.000 in
Paper 6), consolidation_at_layer, separation_margin. For each model, extract the refusal
direction at BOTH sites:
- **End-of-prompt direction** -> decomposition + gap. PREDICTION: ~Paper 6 (~99% residual,
  gap~0) — the reflexive boundary decision.
- **CoT direction** -> same. PREDICTION (interesting): single-vs-full gap OPENS UP
  (consistent with Yamaguchi Fig 16's low EOP<->CoT cosine 0.11-0.61), and may carry MORE
  moral projection than the EOP direction.
- Report **EOP<->CoT cosine** per model (reproduce/extend Yamaguchi Fig 16) and the
  **moral-projection asymmetry** = (CoT moral frac - EOP moral frac).

**CoT aggregation = TWO aggregations serving TWO distinct comparisons (user decision
2026-06-23 — do NOT collapse to "we tried both, they agreed").** The per-prompt CoT
pooling is computed two ways on purpose:
- **last-CoT-token** (token just before the boundary) -> the *matched-pooling* EOP<->CoT
  direction comparison. EOP is a last-input-token; matching the pooling (last token both
  sides) is what makes the EOP<->CoT cosine and the moral-projection asymmetry a fair
  like-for-like, not a pooling artifact.
- **mean-over-CoT** (mean across reasoning-trace tokens) -> trace-level moral *content* and
  the setup for the Phase 2 decision-sentence work (how moral activation distributes across
  the whole trace). The mean CARRIES A COHERENCE/LENGTH CONTROL (a long trace's mean can be
  length/coherence-confounded; exclude incoherent traces, report length stats, length-check
  the moral fraction) so trace-level moral content is not a length artifact.
- The **divergence between last-token and mean is itself a substantive finding** about how
  moral content is distributed through the trace (end-concentrated vs spread), reported
  PER MODEL. The interesting case is DISAGREEMENT, and per-model disagreement is part of the
  functional-vs-imitated story (e.g. genuine deliberation may spread moral content through
  the trace where imitation carries it only superficially). Frame the disagreement, never
  "both agreed, good".

**Pre-registered central hypothesis:** reflexive (EOP) refusal is non-moral (~Paper 6
residual); deliberative (CoT) refusal routes more through the moral subspace — and this
asymmetry is **larger in GPT-OSS-20B (genuine deliberation) than in the distills**.
Pre-register BOTH outcomes:
- Asymmetry present, strongest in GPT-OSS-20B -> deliberative reasoning re-couples morality.
- No asymmetry (CoT also ~99% residual) -> dissociation extends into the reasoning regime
  even where refusal is multi-directional. Clean negative, publishable.

**Human gate after Phase 1** (the asymmetry, and whether it is GPT-OSS-led, decides whether
Phase 2 trace-level compute is worth it).

## Phase 2 — Causal load-bearing of distributed trace morality (RELOCATED 2026-06-23)
**Target relocated after the Phase 1 result (user gate decision).** Phase 1 cleanly settled
the original Phase-2 question (decision-point concentration): moral asymmetry at the
last-token decision site is <= 0 in all three models, GPT-OSS most negative — so there is no
re-coupling to localize at the decision point. The live hypothesis moved to the
trace-distribution axis: GPT-OSS uniquely carries distributed moral content (CoT-mean, the
panel's highest) that is absent at the decision token. Phase 2's target is therefore the
CAUSAL question, brought forward: **is that distributed trace morality load-bearing
(functional) or decorative (imitated)?** — measured per model against random/persona
controls (the old 3b, now the core).

**PREREQUISITE 2a — trace-length disentanglement (MUST pass before the trace-distribution
axis carries any weight; user mandate).** GPT-OSS harmful traces are short/decisive
(median 54 tok, 64/64 close) vs the distills' 700+ token rambles, so the GPT-OSS-led
trace-distribution signal is confounded with length. Capture moral content vs trace
POSITION and LENGTH (position-binned + matched-prefix trace activations); test whether
GPT-OSS's distributed moral content survives at matched cognitive phase.

**The go/no-go is keyed to the FRACTIONAL-POSITION bins on CLOSED traces — NOT the
matched-length prefix (decision flaw caught at the Phase-1 gate review).** Why: matched-N
prefix (mean over first N tokens) is truncation-phase-confounded — at GPT-OSS's ~54-token
length the largest shared N is tiny, so it pits GPT-OSS's COMPLETE deliberation against the
distills' first ~32-64 tokens (their opening ramble, before they reach the harm
deliberation that may live mid-trace). Fractional bins (bin j = fraction j/K through THIS
trace's reasoning) compare mid-deliberation to mid-deliberation regardless of absolute
length — the comparison that actually controls for length. Restricted to CLOSED traces so
each binned trace is a complete start->decision deliberation (an unclosed trace's bins are a
truncated prefix). The decision reads `fractional_mean_moral` (equal weight per fractional
slice = the length-normalized analog of Phase 1's token-weighted CoT-mean); matched-length
is reported as a flagged-CONFOUNDED secondary diagnostic and CANNOT override (unit-tested:
on a constructed disagreement the decision follows fractional). Two guards added: precision
parity (all three profiled at the SAME bf16 — else a moral-fraction difference could be
precision not length) and closed-trace reporting (closed-fraction per model surfaced; if
the closed subset is thin/biased, flag and raise budget rather than trust it). Run WITHOUT
MAXTOK (full 768) so the fractional profile is the true full-trace shape, not a re-capped
one. If the fractional signal is a length artifact -> drop the axis, the Phase-1 negative
stands; if it survives -> proceed to the causal test (2b).

### Phase 2b — causal load-bearing of distributed trace morality (the test; user-specced)
Phase 2a PASSED (GPT-OSS carries more matched-phase trace moral content, 0.0255 vs
qwen 0.0183 / llama 0.0135; peaks frac-pos 0.25-0.35, fades to the decision). 2b asks
the causal question: is that distributed trace morality LOAD-BEARING (functional) or
DECORATIVE (imitated)? Intervene during generation — project out the moral subspace from
the residual stream across the reasoning trace (band layers, all generated positions) —
and measure whether the refusal OUTCOME changes, per model.

**Framing (user): Phase 1+2a dissociation is the ESTABLISHED SPINE; 2b adds a causal
layer. A 2b NULL reads as "the trace morality is decorative even in the deliberative
model" — a real, publishable finding — NOT a failure.** Functional-vs-imitated =
moral-ablation flips refusal MORE in GPT-OSS than in the distills.

**SCOPE NARROWED after the clean-refusal baseline (2026-06-24, user gate): GPT-OSS-only
causal test.** Baseline behavioral refusal: gpt_oss 1.0, llama-distill 0.17, qwen-distill
0.08 (NOT a budget/closure artifact — distills closed 22-23/24 at 1024 tok; they
acknowledge harm then reframe-to-defense, a soft mode the binary detector under-scores).
The causal ablation needs a clean strong refusal to ablate -> well-posed for GPT-OSS,
ill-posed for the distills (no clean binary refusal target). So the distill causal runs are
DROPPED.
- **Distill side = suggestive-but-CONFOUNDED behavioral contrast (do NOT over-claim).** The
  distills carry trace moral content (2a) and explicitly acknowledge harm in their reasoning
  yet do not robustly refuse. BUT the **R1-distillation-degrades-refusal alternative** is a
  live confound: a model that refuses almost nothing tells us nothing about whether its moral
  content WOULD be load-bearing if it did refuse. So "distill trace morality is decorative"
  CANNOT be claimed from the distills alone — frame as suggestive behavioral contrast, name
  the distillation-degradation alternative explicitly.
- **GPT-OSS causal test is the keystone, and a NULL is the more interesting result:** CoT
  moral reasoning present (2a) but decision-irrelevant even in a robustly-refusing
  deliberative model (which Phases 1+2a predict — GPT-OSS's decision representation is
  non-moral). Keep the three-way ablation (moral / random-floor / persona) + the refusal-
  ablation sensitivity yardstick so the null is interpretable: yardstick fires + moral does
  NOT beat the random floor/persona = "decorative even here", a real finding; yardstick
  doesn't fire = underpowered, fix the site, don't report a null.
- (Closure-robustness requirement is now moot for the dropped distill runs; it was
  satisfied diagnostically — the distills reach their decisions and still don't refuse.)
- **(2) Noise floor for a 2-4% component.** Three-way ablation: moral subspace (k dims) vs
  matched-dimension RANDOM subspaces (N draws -> a noise-floor distribution) vs persona.
  A load-bearing claim must beat BOTH control distributions (moral refusal-drop an outlier
  above the random floor AND above persona). The known-firing refusal-direction ablation
  (Phase 0d: drop +1.0) is the SENSITIVITY YARDSTICK — if even it doesn't fire at the
  chosen intervention site, the test is underpowered and a moral null is uninterpretable.
  Coherence filter applied; over-ablation incoherence reported, never a headline.

### (original Phase 2 plan, retained for the decision-sentence machinery)
Combine Yamaguchi's decision-sentence localization with our moral-subspace projection — no
prior work does this.
- **2a Locate decision sentences.** Adapt Yamaguchi 4.1 resampling: split CoT into
  sentences, condition on increasing-length prefixes, k=10 regenerations/prefix,
  StrongREJECT-score. Select high-variance prompts (their thresholds sigma[s_i]>=0.43, mean
  CoT-conditioned sigma<=0.03) where the CoT causally drives the outcome. Decision sentence
  = prefix position where outcome variance collapses.
- **2b Project at decision sentences.** Compute moral- and persona-subspace projection
  (subspace_projection_fraction) at the decision-sentence residual stream; compare to
  *non-decision* sentences in the same traces (within-trace control). HEADLINE QUESTION:
  does moral activation concentrate at refusal-determining sentences, and is the
  concentration strongest in GPT-OSS-20B?
- **2c Aggregation/stability (risk #1).** Decision-sentence location is per-prompt and
  stochastic. Aggregate across MANY prompts; report whether decision sentences are stable
  enough across rollouts for the signal to be meaningful; if unstable, aggregate over the
  distribution, not single traces. Bootstrap; report CIs (Yamaguchi used N=32 — small;
  treat n-limits like Paper 6's 24-prompt discipline).
- **2d GPT-OSS-20B coherence filter.** CoT intervention -> 40% incomplete/looping
  generations (Yamaguchi I.1). Apply a coherence filter; report dropped rollouts honestly;
  do not let looping/gibberish traces inflate or deflate the signal.

**Human gate after Phase 2.**

## Phase 3 — Functional vs imitated: the discriminators (3b MANDATORY)
- **3a Distribution confound (Shairah null).** Reproduce a Shairah-style extended-refusal
  model (justify-before-refuse SFT) on Llama-3.1-8B-Instruct (Paper 6 anchor base). Run the
  SAME decision-sentence moral-projection metric. DISCRIMINATING OUTCOME: reasoning model
  shows moral concentration AND extended-refusal model shows the same distribution/
  robustness WITHOUT moral concentration -> moral routing dissociated from mere
  distribution. (Comparator is measured-against only — built to compare, not released.)
- **3b Causal load-bearing — MANDATORY, run on ALL THREE models.** This is what
  distinguishes *functional* moral coupling from *imitated* moral-talk, and it is the whole
  reason the distilled-vs-RL panel exists — correlational concentration (2b) CANNOT tell
  them apart. **Precision for GPT-OSS-20B is settled in Phase 0d** (prefer bf16-dequant;
  the refusal-direction positive control must fire at the chosen precision or a moral null
  is uninterpretable). At decision sentences, ablate ONLY the moral-subspace component (project out
  the moral subspace at those positions) and measure whether the refusal outcome changes,
  vs the two Paper-6 controls: (i) matched-magnitude random direction, (ii) persona
  subspace. Reuse random_ablation_control.py design.
  - **The key result pattern:** moral-component ablation flips refusal MORE than
    random/persona in **GPT-OSS-20B** (functional, load-bearing) but NOT beyond controls in
    the **distills** (present-but-imitated) -> *genuine deliberative training routes refusal
    through moral cognition; distillation reproduces moral talk without the mechanism.*
    This is the sharp, publishable finding.
  - Other outcomes pre-registered: load-bearing everywhere (reasoning per se re-couples);
    nowhere (concentration is incidental in all). All honest.
- **3c Shared-teacher transfer corroboration (cheap).** Do the two R1 distills show
  *matching* moral-activation patterns at decision sentences? A match suggests
  inherited-from-teacher (imitated), corroborating 3b's distilled reading from a second
  angle. (Caveat the 8B-vs-14B size gap.)

**Human gate before Phase 3** (most compute; 3b is mandatory but only meaningful if Phase 2
shows concentration somewhere).

---

## Claim discipline (carry Paper 6's hard-won lessons)
- Anchor quantitative claims on COHERENT generation regimes; report degenerate
  over-ablation / high-incompletion regimes as caveated endpoints, NEVER headlines.
- Earn each word: "concentrated / dose-dependent / refusal-direction-specific / causally
  load-bearing (only if 3b passes)." Do NOT say "routes through" unless 3b establishes the
  causal direction for that model.
- **Functional vs imitated is a claim about GPT-OSS-20B vs the distills — frame it as
  measured properties of these released models, NOT a training-recipe claim.**
- Llama->DS-R1-Llama base-shared comparison: "does adding (distilled) reasoning to the
  Paper-6 moral-coupled base strengthen/preserve/wash out the coupling" — within-base
  observation, not a recipe claim.
- n-limits: decision-sentence subsets are small; bootstrap, report CIs, pre-register nulls.
- MoE confound on GPT-OSS-20B: acknowledged, partially addressed via Paper 2 priors, not
  faked.

## Novelty line
Non-reasoning work (incl. Paper 6) shows refusal is non-moral and dissociable; Yamaguchi
shows reasoning refusal is multi-directional and trace-carried (refusal-only); Shairah
shows distributed refusal resists ablation (distribution-only). Paper 7 tests whether
*genuinely deliberative* reasoning re-couples moral judgment into the refusal decision, and
separates that from *imitated* deliberation in distilled models — characterizing WHERE moral
reasoning enters refusal, and whether it is functional, the question all three prior works
leave open.

## Constraints
- Everything on `main`; new work under `papers/7_reasoning/`.
- Reuse Paper 6 tooling (model_registry.py, extract_refusal.py, random_ablation_control.py,
  run_phase1/2.py, ablation_strength_sweep.py); extend the registry, do not fork. Confirm
  real StrongREJECT / Arditi dataset provenance, not placeholders.
- Human gates after Phase 0, 1, 2, and before 3. Phase 3b is mandatory.
- Shairah comparator is measurement-only (built to compare, not released).

---

## Provenance ledger (Phase 0a — verified 2026-06-23 against HF primary sources)

| key | repo | model_type | n_layers | hidden | MoE | base (card-verified) |
|-----|------|-----------|----------|--------|-----|----------------------|
| gpt_oss_20b | `openai/gpt-oss-20b` | `gpt_oss` | 24 | 2880 | 32 experts / 4 active | none (RL-trained, no separate base) |
| ds_r1_llama8b | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `llama` (LlamaForCausalLM) | 32 | 4096 | dense | Llama-3.1-8B (= Paper 6 anchor; vocab 128256 / rope 5e5 match) |
| ds_r1_qwen14b | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | `qwen2` (Qwen2ForCausalLM) | 48 | 5120 | dense | Qwen2.5-14B GENERAL (card table; 7B distill is Math-7B, 14B is not) |

GPT-OSS MoE access: `model.layers.*.mlp.router` + per-expert weights present; experts ship
mxfp4-quantized (router/attn excluded) — Phase 3b per-expert orthogonalization must handle
dequant. Teacher for both distills: DeepSeek-R1 (shared) -> 3c shared-teacher transfer probe.

## Status
**PHASE 2b COMPLETE (2026-06-24) — experimental arc done; next is synthesis/write-up.**
Causal test reached a DEFINITIVE result via the held-out yardstick gate: **GPT-OSS refusal
is DISTRIBUTED across EOP+CoT — no single-direction bottleneck.** Held-out (harmful_eval,
disjoint+category-diverse, 24 baseline-refusable): eop clean-flip 0.042 (the in-sample 3/3
was overfit/prompt-homogeneity), cot_mean over-ablates (0.875 incoherent), cot_last dead.
The under-estimation alternative is ruled out. So the single-subspace load-bearing test
cannot cleanly run on GPT-OSS — but the distributed-ness is itself the answer: if refusal
is not bottlenecked through ANY low-dim direction (including the refusal direction itself),
a fortiori not through the low-dim moral subspace.

**Paper thesis (final, established by convergence):** Reasoning models morally deliberate
in their traces, but refusal is geometrically dissociated from and not bottlenecked by moral
cognition — the comprehension/compliance dissociation (Paper 5/6) extends into the reasoning
regime, in BOTH genuinely-deliberative (GPT-OSS) and distilled models. Evidence:
- **Phase 1:** EOP refusal ~99% residual (⊥ moral) in all 3; moral asymmetry (CoT-last −
  EOP) ≤ 0 in all 3, GPT-OSS most negative → no moral re-coupling at the decision.
- **Phase 2a:** GPT-OSS carries the most matched-phase trace moral content (0.025 vs distill
  0.013/0.018), peaking mid-trace, FADING to the decision (survived length disentanglement).
  So GPT-OSS morally deliberates, but that content is displaced from the decision point.
- **Phase 2b:** behaviorally GPT-OSS refuses 100% / distills 8-17% (soft-refuse-by-reframe;
  distill side suggestive-but-CONFOUNDED by R1-distillation-degrades-refusal). Causally,
  GPT-OSS refusal is distributed — not routed through any low-dim feature, moral included.
- **Functional-vs-imitated (reframed):** even genuine deliberative alignment (GPT-OSS) does
  NOT bottleneck refusal through moral cognition; distills reproduce the moral-talk form
  without robust refusal (confounded). The clean "functional vs imitated ablation asymmetry"
  the plan hoped for is not the result; the honest result is the extended dissociation +
  distributed refusal, which is publishable.

**Methodological note for write-up:** Phases 1/2/2a use the DECODABLE refusal direction
(correct for geometry/decodability). The Phase 2b causal probe showed decodable≠causal and
that GPT-OSS refusal resists single-direction ablation (held-out validated) — that is a
result, not a tooling failure. Over-ablation incoherence (cot_mean) reported, never a fire.

---

**PHASE 1 COMPLETE (2026-06-23) — all 3 models.**
Headline (matched-pooling, last-token): moral asymmetry (CoT-last − EOP) <= 0 in ALL three,
incl. GPT-OSS (−0.014 CI[−0.016,−0.010]). So the "genuine deliberation re-couples morality
at the decision" hypothesis is REFUTED — the comprehension/compliance dissociation EXTENDS
into the reasoning regime, genuine RL deliberation included. Clean pre-registered negative.
The LIVE thread is the trace-distribution axis: GPT-OSS uniquely carries distributed moral
content (CoT-mean mft ~0.027–0.038, highest of the panel) that is ABSENT at the decision
token (CoT-last ~0.005, lowest), with the largest last-vs-mean divergence (cos ~0.34 vs
distills 0.46–0.62) and trace-distribution gap (+0.032 vs +0.016/+0.017). I.e. GPT-OSS's
trace contains the most moral *talk* but it is not at the refusal-determining position —
the functional-vs-imitated question reframed for Phase 2. Confounds: the single-vs-full gap
opens at CoT in the DISTILLS (0.14/0.20) but not GPT-OSS (0.0) — but GPT-OSS harmful traces
are SHORT/decisive (median 54 tok, 64/64 close) vs distill 700+ token rambles, so the gap
difference partly tracks trace length, not deliberation quality (flag, don't headline).
Bootstrap CIs for cosines are attenuated by independent resampling (use point as central).

**PHASE 0 COMPLETE (2026-06-23).**
- 0a provenance: all 3 repos verified (ledger below).
- 0b registry + local gate: green; conventions locked (band-to-final-layer, subspace =
  reasoning-model raw, `think` format + EOP/CoT sites).
- 0c smoke: hooks fire at EOP + CoT on the reasoning generation structure.
- 0c anchor (ds_r1_llama8b EOP): refusal = **0.991 residual** (= Paper 6 Llama; no
  convention drift), single-vs-full gap **0.0** (consolidated single-direction refusal at
  the boundary — the Phase 1 EOP baseline the CoT site is compared against).
- 0d precision: GPT-OSS bf16-dequant **fits one A100** (45.9 GB); refusal-direction
  ablation positive control **FIRES** (refusal drop +1.0).
- **Caveat for Phase 3b:** all-layer directional ablation left 6/12 GPT-OSS rollouts
  incoherent (the anticipated 2d over-ablation effect). The positive control still fires on
  the coherent half, but it confirms all-layer projection is too blunt for clean causal
  claims — Phase 3b must ablate at DECISION SENTENCES (targeted), not all layers, and keep
  the coherence filter.

Tooling built (all under papers/7_reasoning/, Paper 6 untouched): `scripts/model_registry.py`
(reuses Paper 6 by path), `local_test.py`, `think_io.py`, `smoke_think_hooks.py`,
`gpt_oss_precision_gate.py`, `run_phase0.py`, `runpod/{run_session.sh,remote_phase0.sh}`.
Re-run any stage with `STAGE={smoke,anchor,precision} ./run_session.sh`.

### Phase 0 build/fix log (historical)

### Phase 0c smoke v1 result + fixes (2026-06-23)
First smoke (ds_r1_llama8b) CONFIRMED the real goal — hooks fire at END_OF_PROMPT and
CoT positions (finite, dim 4096). The structure check failed for three fixable reasons,
now fixed: (1) the R1 chat template appends `<｜Assistant｜><think>\n` to the PROMPT, so
the `<think>` opener is in the prompt and generation starts inside the trace (checked the
opener in the prompt, not the continuation); (2) 96 tokens never reached `</think>` (R1
reasoning is long) -> smoke budget raised to 512 + uses an easy factual prompt that closes
fast; (3) `decode(skip_special_tokens=False)` leaked `Ġ`/`Ċ` byte artifacts -> `<think>`/
`</think>` are ORDINARY text (verified from tokenizer_config), so decode with
`skip_special_tokens=True` gives clean text AND preserves the delimiters (harmony keeps
specials). Re-run the cheap smoke to confirm green before anchor/precision.

### Phase 0d precision: torch-stack fix (2026-06-23)
First precision run crashed at load: the base image ships **torch 2.4**, but its
`transformers` has the mxfp4 quantizer, whose `validate_environment` calls
`torch.accelerator.current_accelerator()` (added in torch 2.6) -> `AttributeError`.
Anchor/smoke were unaffected (the distills don't trigger the mxfp4 quantizer). Fixes:
(1) `remote_phase0.sh` upgrades the **torch trio** (torch 2.6.0 / torchvision 0.21.0 /
torchaudio 2.6.0, cu124) in place on the precision pod when `torch.accelerator` is missing
— the trio must move together: a torch-only bump leaves the image's torch-2.4 torchvision
with an unloadable `torchvision::nms`, which transformers imports via `image_utils` and
cascades into a `GptOssForCausalLM` import failure. Keeps the image's CUDA 12.4 /
host-driver requirement; lower risk than a cu12.8 image. (2) `WhiteBoxModel` gained an optional
`quantization_config` (default None, backward-compatible); (3) the gate passes
`Mxfp4Config(dequantize=True)` for a deterministic bf16 load. Fallback if torch 2.6 still
falls short: `IMAGE=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` (needs
a host driver >= 570). Re-run: `STAGE=precision ./run_session.sh`.

**FIT passed** (2nd precision run): GPT-OSS loaded bf16-dequant, 45.9 GB peak on an 80 GB
A100, experts realized in bf16. Positive control v1 was a parsing artifact
(`clean=1.0/ablated=0.0`): the harmony `analysis` opener is in the PROMPT and the model
ran past 256 tokens without reaching the `final` channel, so the splitter classified the
REASONING text (no opening refusal) as compliance. Fixes: classify the `final` channel
ONLY; "no final channel reached" -> **incomplete**, excluded from the rate (2d coherence
filter), not counted as compliance; budget raised to 768; verdict distinguishes
fired / didn't-fire / INCONCLUSIVE (too few coherent rollouts -> raise budget or soften
ablation, never fake a null); raw `rollout_head`/`final` samples dumped to the JSON for
inspection.

**Positive control v2 (the real result, masked by two more measurement bugs):** the
control DID fire — clean GPT-OSS refuses ("I'm sorry, but I can't help"), ablated engages
("step-by-step guide covering reconnaissance, scanning, exploitation"). Two bugs hid it:
(a) GPT-OSS emits curly apostrophes (U+2019), which greenblatt's straight-quote regex
misses -> refusals scored as compliance; (b) post-ablation compliance lives in the long
`analysis` channel and never reaches a short `final` answer, so requiring `final` excluded
the signal. Fix: a harmony-aware, apostrophe-normalized refusal detector over the WHOLE
rollout (greenblatt's single-response assumption doesn't fit analysis+final), reporting
**refusal drop** (clean - ablated); both error directions only shrink the drop, never fake
a pass. Validated on the real run's dumped rollouts: clean 1.00 -> ablated 0.00 = +1.00.
Also added: `hf_transfer` in the runner (resilient large-shard downloads after two CDN
stalls), budget trimmed to 512 (no need to reach `final`). Re-run
`STAGE=precision ./run_session.sh`; expect refusal_drop ~1.0 -> gate passes.

### Band convention (CONFIRMED by user 2026-06-23): band-to-final-layer
Qwen-14B (48L) band top pinned to L47 (its last layer), not the bare-rule L46. The
convention is "band runs to the FINAL layer inclusive" (the terminal residual state
feeding the unembedding, where late-layer refusal/moral structure consolidates), not
"apply round() literally." All Paper 6 models include their last layer (L31 of 32); the
bare rule breaks this ONLY at 48L (31/32*48==46.5 exactly -> round-half-to-even -> 46),
so pinning L47 is the MORE convention-faithful choice. Documented in model_registry.py.
