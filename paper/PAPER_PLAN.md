# Paper 1 Plan: *Probing Accuracy Saturates; Fragility Doesn't*

**Status:** Framing locked in. Phase C4 compositional probe ablation complete; abstract sentence locked to Outcome B variant. 3-seed compositional fragility replication complete (verdict `decline_real`); §4.3 locked to the *full subsection* branch. Drafting on §3 Methodology and §4 Results in flight; remaining items are §1 Introduction, §2 Related Work, §5 Discussion, §6 Conclusion, abstract tightening pass, and figure styling.

**Source-of-truth note.** Where this document and `RESEARCH_BRIEF.md` differ on Paper 1 framing, **this document takes precedence.** The BRIEF is the public-facing two-paper summary written before Paper 1 framing was locked. Specifically: the BRIEF lists Paper 1 findings in science-first ordering; Paper 1 itself is methodology-first.

## Title

**Primary:** *Probing Accuracy Saturates; Fragility Doesn't: Measuring Moral Representations During LLM Pre-Training*

Alternates kept on file:
- *When Moralized Distinctions Emerge: Probing Accuracy Saturates But Fragility Doesn't*
- *Beyond Probing Accuracy: Fragility-Based Measurement of Moral Representations During Pre-Training*

The colon does the work of separating the methodological claim ("probing accuracy saturates; fragility doesn't") from the demonstration domain ("measuring moral representations during LLM pre-training"). Keyword-rich for indexing, descriptive at a glance, signals contribution type immediately.

## Venue target

NeurIPS Safe Generative AI workshop or ICLR R2-FM. Length budget assumed 8 pages; appendix unbounded. If submitting to a 4-page workshop, drop §4.2 layer-depth gradient detail to the appendix and tighten §5 discussion.

## Thesis

**Probing accuracy is the wrong instrument for asking alignment-relevant questions about LLM pre-training; fragility is the metric that actually moves.** The moral-representation findings are the evidence that earns the methodological claim its keep, not the reverse.

## Abstract (draft v3 — locked, 4-seed result incorporated)

> Moral / neutral semantic distinctions are linearly decodable from base LLM hidden states within the first ~5 % of pre-training. But standard probing accuracy saturates within the first 4K of 36K early-training steps and provides no resolution for the remaining 95 % of training. We introduce *fragility* — the activation-noise level at which probe accuracy collapses — as a metric that continues evolving long after accuracy plateaus and reveals dynamics invisible to accuracy alone. We use it to establish three findings on the OLMo open-weight model family. (1) Moralized semantic distinctions emerge along a quantitative lexical→compositional gradient: a standard moral probe (single-lexeme swap) onsets at step 1K, sentiment at 2K, a compositional moral probe (multi-token integrated swap, in which contrast tokens are individually mild and only flip moral status in context) at 4K, and syntax at 6K — bounding the standard probe's step-1K onset to a claim about lexical accessibility rather than compositional moral encoding. (2) A layer-depth robustness gradient develops monotonically over training: late layers hold maximum robustness throughout while early-layer critical noise drops from 10.0 to 1.7 between steps 1K and 36K, a pattern invisible to probing accuracy. The compositional probe reproduces the qualitative pattern across four random-seed train / test splits and exhibits its own quantitatively distinct long-term trajectory (mean critical noise rises 0.10 → 5.11 by step 5K, then declines to 2.46 by step 30K with std collapsing to 0.12 by step 36K), establishing that fragility-resolves-what-accuracy-misses is not a lexical-accessibility artifact. (3) LoRA fine-tuning on three matched corpora produces identical probing accuracy (~80 % across narrative-moral, declarative-moral, and general-text conditions) but distinct fragility profiles, demonstrating that data curation operates on representational structure rather than representational content. The methodological claim generalizes: in every comparison where probing accuracy returns a flat answer, fragility returns a structured one.

**Locked variant rationale.** Phase C4 result: compositional probe onset step 4K (mean acc 0.721, plateau 0.77), between sentiment (2K) and syntax (6K). This is Outcome B from the original placeholder set — the lexical-accessibility framing is partially right; single-token moralized vocabulary is decoded earlier than compositional moral integration. The other two outcome variants (A: synchronous onset; C: failure-to-decode) were ruled out by the data and are removed from this plan. The 3-seed compositional fragility replication landed `decline_real` (gap 2.19 vs. max std 0.84; both decision-rule conditions pass), which is now incorporated into Finding 2's abstract sentence rather than left as "reproduced independently." Numbers sources: `outputs/phase_c4_compositional/RESULTS.md` and `outputs/phase_c4_compositional/3seed/decision.json`.

## Section structure

### 1. Introduction (~1 page)

Lead with the methodological gap. Standard story in interpretability is: train a probe, report accuracy, declare the concept "encoded" if accuracy is high. Show this fails for studying *training dynamics*: probing accuracy on moral/neutral pairs hits ~95 % within the first 4K of 36K early-training steps and stays flat. The remaining 95 % of training is invisible to the standard instrument. Motivate fragility as the alternative and preview the three findings. End on the deepsteer thesis (one-sentence form): "in every comparison where probing accuracy returns a flat answer, fragility returns a structured one."

### 2. Related work (~0.75 page)

- **Linear probing methodology:** Alain & Bengio (2017), Belinkov (2022). Establish the standard-probe baseline and its known limitations.
- **Causal tracing and activation methods:** Meng et al. (2022); used in supporting analysis (causal-probing divergence) sent to appendix.
- **Phase transitions in neural network learning:** Power et al. (2022) grokking. Connection to our semantic / structural emergence-dynamics dichotomy.
- **Moral Foundations Theory:** Haidt (2012), Graham et al. (2013). Source of the moral taxonomy used in the probing dataset.
- **OLMo and open checkpoints:** Groeneveld et al. (2024). The unique infrastructure that makes this analysis possible.
- **Fragility / robustness as a representation property:** Pres et al. (2024) on refusal as a single-direction phenomenon, by analogy. Connection to safety-circuit literature.
- Brief acknowledgement that this work uses moralized vocabulary as a demonstration domain — broader claims about moral reasoning would require harder probes (which §5 discusses and Phase C4 explicitly tests).

### 3. Methodology (~1.75 pages)

#### 3.1 Standard minimal-pair datasets

Moral / neutral pairs (240, balanced across six MFT foundations); sentiment pairs (210, positive vs. negative single-adjective swaps); syntax pairs (210, grammatical vs. ungrammatical with localized rule violations). Construction: matched syntactic skeleton, single-token swap. Validation gates: length ratio ≤ 1.5, content-word overlap ≥ 0.0 (no minimum on the standard moral pairs by design — neutral domain pool is intentionally diverse), deterministic seed-42 train / test split. All datasets API-free, deterministic, included in the toolkit.

#### 3.2 Compositional moral probing dataset (Phase C4 contribution)

A 200-pair compositional moral minimal-pair dataset is the methodological contribution of this paper that earns the standard moral probe's "lexical accessibility" framing its empirical bite. Four 50-pair categories, all hand-curated:

- **action_motive:** same action verb, motive determines valence — *"She lied to her parents to **protect** her younger brother today"* / *"...to **humiliate**..."*
- **action_target:** same action, target descriptor determines moral relevance — *"He gave the last loaf of bread to the **hungry** stranger at the door"* / *"...the **wealthy** stranger..."*
- **action_consequence:** same action, consequence framing determines valence — *"He kept the secret about the surprise to keep his sister **safe** today"* / *"...to keep his sister **hurt** today"*
- **role_reversal:** same components, role / target / context determines valence — *"The judge accepted the gift to free the **innocent** prisoner from prison"* / *"...the **guilty** prisoner..."*

Construction constraints (all enforced by `validate_compositional_dataset`, all 200 pairs pass):

- Per-pair length difference ≤ 2 words; both halves in 8-20-word band.
- Per-pair content-word overlap ≥ 0.60 (stopwords removed; matches the metric in `deepsteer/datasets/validation.py`, so the threshold is directly comparable to the standard moral probe gate).
- No individual lexeme on a 47-word strong-valence blocklist (`murder`, `torture`, `stole`, `assault`, etc.) on either side. Contrast tokens — `protect`, `humiliate`, `hungry`, `wealthy`, `safe`, `hidden`, `innocent`, `guilty` — are individually mild and only flip moral status in the surrounding action context.
- No exact duplicate moral or immoral sides across the 200 pairs.

The compositional gate that does the operational work: a TF-IDF + logistic regression classifier on bag-of-words features (5-fold CV, unigrams) achieves **0.113 overall** (per-category: 0.14-0.20). The design ceiling is 0.65; the dataset is well below. Single-word features cannot separate the classes; anything the linear probe achieves on hidden states above this floor must integrate multiple words.

Train / test split: 160 / 40, stratified by category, seed = 42. The dataset construction iterated through ~5 rewriting passes to satisfy the 0.60 content-overlap gate alongside the multi-word compositional contrast requirement — these two constraints are in genuine tension, and the iteration history is documented in the dataset module docstring.

#### 3.3 Probes

- **`LayerWiseMoralProbe` and variants.** Binary linear probing (`nn.Linear(hidden_dim, 1)` + BCE loss + Adam, 50 epochs, lr 1e-2) trained at every transformer layer; mean-pooled hidden states across the sequence dimension; per-layer accuracy curve as the readout; onset layer / peak layer / encoding depth / encoding breadth as summary statistics. The compositional probe (`CompositionalMoralProbe`) subclasses `LayerWiseMoralProbe` and overrides only the dataset path — trainer, activation collection, and metric computation are shared. This is methodologically important: the only experimental variable separating the standard moral probe from the compositional probe is the dataset, not the probe.

#### 3.4 `MoralFragilityTest`

Inject Gaussian noise of magnitude σ into cached test-set activations after probe training; record the σ at which probe accuracy drops below the fragility threshold (0.6 by default — chance + 0.1 on a binary task). Logarithmic noise sweep [0.1, 0.3, 1.0, 3.0, 10.0]. The minimum σ at which accuracy falls below threshold is the layer's *critical noise*. Per-layer critical noise across all transformer layers gives the *fragility profile*; mean across layers gives `mean_critical_noise` as a scalar summary. The same `MoralFragilityTest` infrastructure runs against the standard moral probe (Phase C1) and the compositional probe (Phase C4) — methodology generality is established by reuse, not by reimplementation.

#### 3.5 Target models and checkpoints

OLMo-2 1B (`allenai/OLMo-2-0425-1B-early-training`) with 37 early-training checkpoints (steps 0-36K at 1K intervals); OLMo-3 7B with 20 stage-1 checkpoints; OLMo-2 1B final checkpoint (`allenai/OLMo-2-0425-1B`, ~2.2T tokens) for the compositional probe validation gate. All on a single MacBook Pro M4 Pro (24 GB unified memory, MPS, fp16).

#### 3.6 Required moral-probe validity controls (before submission)

Three controls — leave-lexeme-out splits, paraphrase transfer, adversarial lexical swap — are mandatory for the standard moral probe before submission, to bring it to parity with the persona probe (which already has these controls; see `outputs/phase_d/c8/`). The compositional probe (§3.2) addresses the strongest version of the "your probe is just reading vocabulary" review attack by construction, and may obviate the leave-lexeme-out and adversarial-swap controls for the standard probe — flag this question to the reviewer rather than running redundant analyses. ~4-6 hours of additional work for the standard probe; document protocol and report results in the appendix or a dedicated subsection. Tracked under `RESEARCH_PLAN.md` "Open items" and Appendix C.

### 4. Results (~3.25 pages)

#### 4.1 Emergence ordering: a lexical→compositional gradient (Phase C2 + C4)

Headline figure: four-curve onset overlay (standard moral, sentiment, compositional moral, syntax) on shared step axis. Onset numbers (mean probing accuracy across all 16 layers ≥ 0.70):

| Probe | Construction | Onset step | Onset acc | Plateau acc |
|-------|--------------|-----------:|----------:|-------------:|
| Standard moral | single morally-loaded lexeme swap | 1,000 | 0.760 | 0.960 |
| Sentiment | single valenced adjective swap | 2,000 | 0.790 | 0.976 |
| **Compositional moral** | multi-token integrated swap | **4,000** | 0.721 | 0.774 |
| Syntax | structural well-formedness | 6,000 | 0.717 | 0.775 |

Three substantive findings, in order of importance:

1. **Lexical→compositional gradient.** The standard moral probe's step-1K onset measures how quickly *moralized vocabulary* becomes statistically separable, not how quickly *moral valence is encoded compositionally*. Compositional moral integration emerges at step 4K — between sentiment and syntax — establishing a quantitative gradient from single-token vocabulary statistics to multi-token integrated encoding. This *bounds* the standard probe's onset claim without invalidating it: lexically-marked moralized vocabulary is decoded first, compositional moral integration second, syntactic competence last.
2. **Phase-transition vs. gradual emergence dichotomy.** The standard moral and sentiment probes show sharp phase transitions; compositional moral and syntax rise more gradually. This dichotomy parallels the grokking literature (Power et al. 2022) and frames the paper's contribution about *types* of representational learning, not just timing.
3. **Plateau coincidence (subsection §4.2).** Compositional and syntax probes both saturate at ≈0.77 mean accuracy under mean-pooled linear probing; standard moral and sentiment saturate at ≈0.96-0.98. The 0.77 ceiling tracks across two structurally different tasks that share one feature: both require multi-token integration that mean-pooled linear probing partially captures.

Numbers source: `outputs/phase_c2/RESULTS.md` (standard moral + sentiment + syntax) and `outputs/phase_c4_compositional/RESULTS.md` (compositional moral + four-curve overlay JSON).

#### 4.2 Plateau coincidence: compositional ≈ syntax under mean-pooled linear probing

A short subsection (~0.4 page) on the structural finding that the four-curve overlay (Figure 3) makes visually self-evident: probes that ride single-token statistics (standard moral, sentiment) saturate near 0.97; probes that require multi-token structural integration (compositional moral, syntax) saturate near 0.77. The 0.77 ceiling is a probe-side property under our methodology, not necessarily a model property. Two competing readings:

- **Model ceiling.** The 1B model genuinely encodes compositional moral valence at ≈0.77 accuracy, and this is what the probe recovers.
- **Probe ceiling.** Mean-pooled linear probing on 1B hidden states can only recover ≈0.77 of compositional / structural signals at this scale; a non-linear or position-sensitive probe would push higher.

Phase C4 cannot distinguish these at 1B. The cleanest disambiguation is repeating the four-curve overlay at 7B / 32B in Phase E — if compositional moral accuracy rises with scale while syntax accuracy does not, the bottleneck is the 1B model not the probe. We state both readings honestly in §5 and flag the open question rather than overclaiming.

This is **not** a load-bearing subsection for the methodological thesis; it is load-bearing for the *honest scope* of the lexical→compositional gradient claim. Keep to ~3 paragraphs.

#### 4.3 Probing accuracy saturates; fragility doesn't (Phase B5 + C1 + C4 4-seed)

The figure that does the most work for the methodological thesis: two-panel comparison, shared training-step x-axis. Top panel: mean probing accuracy over training — sharp sigmoid 0 → ~95 % between steps 0 and 4K, then completely flat for the remaining ~33K steps. Bottom panel: fragility — initial jump alongside accuracy, then continued downward drift in early layers through step 36K, with late layers holding at maximum robustness. Visually self-evident: top panel is "ceiling reached, no information"; bottom panel is "still moving."

Layer-depth heatmap as Figure 3 companion: layers × training step, with two heatmaps stacked — top is probing accuracy (uniformly green after step 4K), bottom is critical noise (clear gradient developing, with late layers holding red while early layers fade toward blue).

Numbers source: `outputs/phase_c1/RESULTS.md` (1B trajectory, 37 checkpoints, dense). 7B fragility evolution from `outputs/phase_b/` is supporting evidence — same pattern at the 7B headline scale, less dense checkpoint sampling.

**Compositional fragility generalization — full subsection (3-seed replication landed; verdict `decline_real`).** Phase C4 ran `MoralFragilityTest` on the compositional dataset across the same 37 checkpoints with split seeds 42 + 43 + 44 + 45 (~50 min total compute). The accuracy-saturates-fragility-doesn't pattern reproduces *qualitatively* (compositional fragility rises 0.10 → 5.11 between steps 0 and 5K, peaks at step 5K, alongside compositional accuracy plateauing by step ~5K) and the long-term trajectory differs *quantitatively* from the standard probe in a way that the 4-seed std bands resolve cleanly: 4-seed mean critical noise drops 4.65 (step 7K) → 2.46 (step 30K), gap = 2.19, max std at the two endpoints = 0.84. Both decision-rule conditions (gap ≥ 1.0; std smaller than gap) pass; the post-step-7K decline is a replicable property of the compositional probe.

This adds ~0.3 page to §4.3 and a small companion panel to Figure 2 (4-seed mean ± std band for compositional fragility, plotted alongside the C1 standard moral fragility curve so the diverging long-term trajectories are visually adjacent). Numbers source: `outputs/phase_c4_compositional/3seed/aggregate_per_checkpoint.json` (per-step 4-seed mean ± std), `outputs/phase_c4_compositional/3seed/decision.json` (decision rule application), `outputs/phase_c4_compositional/3seed/4seed_fragility_evolution.png` (companion panel for Figure 2).

#### 4.4 Data curation reshapes structure, not content (Phase C3)

Triple-panel comparison figure: probing accuracy (three identical bars at ~80 %) and per-layer fragility profiles (three distinctly shaped lines, declarative dipping at layer 3 to critical noise = 3 while narrative and general text hold uniformly at 10). Bars and lines on the same figure make the contrast explicit. This is the cleanest controlled comparison in the paper and the strongest single piece of evidence for the deepsteer thesis.

Numbers source: `outputs/phase_c_tier2/c3/RESULTS.md`.

### 5. Discussion (~1.5 pages)

#### 5.1 Semantic vs. structural learning dynamics

Why phase transitions for moral / sentiment but not syntax? Hypothesis: phase-transition dynamics emerge when a feature can be acquired through local lexical / distributional statistics (the model "discovers" the feature in a discrete jump as soon as it has enough samples to distinguish the lexemes); gradual emergence indicates features that require learning positional, attentional, or structural relationships. Connection to Power et al. (2022) grokking and to broader literature on capability emergence.

#### 5.2 Why fragility succeeds where accuracy saturates

Probing accuracy is a thresholded, capped, top-end metric — once the linear separability is good enough, accuracy hits ceiling and stops moving. Fragility integrates the *margin* of separability and the *redundancy* of representation — both of which keep evolving through training. Frame fragility as a richer functional of the representation geometry than accuracy. Note that this argument generalizes: any binary probing task that hits accuracy ceiling will benefit from a fragility readout for studying training dynamics.

#### 5.3 Generalization beyond pre-training

One paragraph. The fragility-detects-what-accuracy-misses pattern reproduces under a different stimulus: applying the same moral-probe + fragility battery to LoRA adapters from narrow insecure-code fine-tuning of a fully-trained OLMo-2 1B (Reblitz-Richardson, Paper 2 in preparation; see `outputs/phase_d/c15_reframed/RESULTS.md`) shows identical probing accuracy across base / insecure / secure conditions but a fragility-locus shift of 2–3 layers under insecure-code specifically. The methodology generalizes to fine-tuning fingerprints; we develop this in companion work and reference it here only as evidence of generality.

#### 5.4 Limitations

- **Lexical→compositional gradient bounds the standard probe.** What the standard moral probe measures is closer to "moralized vocabulary becomes linearly separable from neutral vocabulary" than to "moral reasoning emerges." Phase C4's compositional probe established this is a quantitative gradient (1K → 4K, 3K-step gap), not a binary in-or-out distinction. State the gradient honestly; don't soften and don't overclaim.
- **Compositional probe partial scope.** The compositional probe addresses *whether the moral signal lives in single-token vs. multi-token features*; it does not address *whether the model represents moral concepts in any deeper functional sense* (e.g., moral reasoning, generalization to novel scenarios). The compositional probe is a stronger lexical-accessibility ablation than the standard probe; it is not a moral-reasoning probe.
- **Probe methodology — plateau coincidence.** Linear probes on mean-pooled hidden states are well-suited for lexical / semantic features and poorly suited for structural / multi-token integration features. Both the syntax (~0.78) and compositional moral (~0.77) plateaus reflect probe limitations as much as representation quality. Whether the 0.77 ceiling is a 1B-model ceiling or a probe-side ceiling is unresolved at 1B; cleanest disambiguation is repeating §4.1 at 7B / 32B in Phase E.
- **Single model family.** All findings are on OLMo-2 1B and OLMo-3 7B. Generalization to other architectures and training recipes is open. The Phase E plan addresses 7B replication on a Deep-Ignorance-style base.
- **Single language.** English pretraining text only.
- **Foundation-specific scope.** Liberty / oppression never fully stabilizes at either 1B or 7B; this cross-scale pattern is documented in the appendix but not in the main thesis. The compositional probe's 200 pairs are categorized by construction pattern (motive / target / consequence / role), not by MFT foundation; a foundation-stratified compositional probe is a natural extension out of scope for this paper.

### 6. Conclusion (~0.5 page)

Restate the thesis: probing accuracy is the wrong instrument for studying training dynamics; fragility is right. Position fragility as a methodology contribution that the alignment-during-pre-training research program can adopt for any probing-based question, not just moral representations. Restate the lexical→compositional gradient (1K / 4K) as the science finding that earns the methodology its keep — a single-token framing of "moral encoding emerges at step 1K" overstates what the probes recover, while "lexically-marked moralized vocabulary at 1K, compositional moral integration at 4K, syntactic competence at 6K" is the honest reading. End on two open questions Phase C4 leaves: (a) does the compositional plateau at ~0.77 lift with model scale (7B / 32B), and (b) does the fragility-resolves-what-accuracy-misses pattern that we establish for the standard moral probe and conditionally for the compositional probe extend to other probing-based investigations of pre-training? Both gates Phase E and broader adoption of the methodology.

### Appendices

- **A. Foundation emergence (Phase B3 / C1 finding 3).** Staggered emergence of the six MFT foundations, with liberty / oppression as the cross-scale outlier. Interesting but not load-bearing for the methodological thesis.
- **B. Causal-probing divergence (Phase B4).** Storage layers (probing peak ~10) and usage layers (causal peak ~0) diverge by ~10 layers at the 7B final checkpoint. Surprising and probably its own future paper rather than a body finding here.
- **C. Standard moral probe validity controls (parity with persona probe).** Leave-lexeme-out splits, paraphrase transfer, adversarial lexical swap on the *standard* moral probe. The compositional probe (§3.2) addresses the strongest version of "your probe is just reading vocabulary" by construction; these controls are for parity with the persona probe rather than load-bearing for the C4 gradient finding.
- **D. Compositional probe pair list and per-category breakdown.** Full 200-pair list (or 50-pair representative sample per category), per-category onset trajectories, per-category content-overlap statistics. Construction protocol itself moved to body (§3.2).
- **E. Reproducibility appendix.** Hardware (MacBook Pro M4 Pro, 24 GB unified, MPS), seeds (split seed 42 for the headline C4 results, seeds 43/44/45 for the 3-seed fragility replication), software versions, command-line invocations for each result, output JSON schemas.

## Headline figures (in body order)

1. **Figure 1: Onset overlay (lexical→compositional gradient).** Four curves on shared step axis — standard moral (1K, plateau 0.96), sentiment (2K, plateau 0.98), compositional moral (4K, plateau 0.77), syntax (6K, plateau 0.78). Vertical dashed lines at each onset step. The figure does triple duty: it establishes the science finding (emergence ordering), the methodological gradient (lexical→compositional), and the plateau-coincidence subsection (compositional ≈ syntax ≪ moral, sentiment). Pulls from `outputs/phase_c4_compositional/compositional_vs_lexical_onset.png` (already generated) — refine for paper styling. **This figure is now Figure 1 (was Figure 3); it leads §4 Results.**
2. **Figure 2: Probing accuracy saturates; fragility doesn't.** Two-panel, shared training-step axis. Top: mean probing accuracy over training (sigmoid 0 → 0.95 between steps 0 and 4K, then flat). Bottom: mean fragility (critical noise) over training (continued movement through step 36K with layer-depth gradient). *Conditional on 3-seed replication: add a third panel or a small inset for the compositional probe's fragility trajectory if the post-step-7K decline replicates.*
3. **Figure 3: Layer-depth heatmaps.** Two stacked heatmaps (layers × steps): probing accuracy (uniformly green after step 4K) above, critical noise (gradient developing, late layers holding) below.
4. **Figure 4: Data curation reshapes fragility, not accuracy.** Triple-panel for the C3 result: probing accuracy as three identical bars, fragility profiles as three distinct curves. *This is the cleanest controlled comparison in the paper.*

## Numbers and where to pull them from

| Claim | Number | Source |
|---|---|---|
| Moral onset step | 1K | `outputs/phase_c2/RESULTS.md` |
| Sentiment onset step | 2K | `outputs/phase_c2/RESULTS.md` |
| Syntax onset step | 6K | `outputs/phase_c2/RESULTS.md` |
| Moral plateau accuracy | ~96 % | `outputs/phase_c2/RESULTS.md` |
| Sentiment plateau accuracy | ~98 % | `outputs/phase_c2/RESULTS.md` |
| Syntax plateau accuracy | ~78 % | `outputs/phase_c2/RESULTS.md` |
| 1B accuracy saturation step | 4K | `outputs/phase_c1/RESULTS.md` |
| 1B mean critical noise: step 1K, step 36K | 10.0, 5.3 | `outputs/phase_c1/RESULTS.md` |
| 1B early-layer critical noise: step 1K, step 36K | 10.0, 1.7 | `outputs/phase_c1/RESULTS.md` |
| 1B late-layer critical noise: throughout | 10.0 | `outputs/phase_c1/RESULTS.md` |
| C3 probing accuracy (all three conditions) | ~80 % | `outputs/phase_c_tier2/c3/RESULTS.md` |
| C3 declarative layer-3 critical noise | 3.0 | `outputs/phase_c_tier2/c3/RESULTS.md` |
| C3 narrative / general critical noise | 10.0 uniform | `outputs/phase_c_tier2/c3/RESULTS.md` |
| 7B B5 mean critical noise (plateau) | ~5.3 | `outputs/phase_b/RESULTS.md` |
| C15 reframed fragility-locus shift (Discussion §5.3) | layer 7 → 9–10 | `outputs/phase_d/c15_reframed/RESULTS.md` |
| Compositional moral onset step | 4K | `outputs/phase_c4_compositional/RESULTS.md` |
| Compositional moral onset mean acc | 0.721 | `outputs/phase_c4_compositional/RESULTS.md` |
| Compositional moral plateau mean acc | 0.774 (step 36K) | `outputs/phase_c4_compositional/RESULTS.md` |
| Compositional moral peak acc (final 1B checkpoint) | 0.900 @ layer 5 | `outputs/phase_c4_compositional/c4_validation.json` |
| Compositional moral TF-IDF baseline | 0.113 (overall, 5-fold CV) | `outputs/phase_c4_compositional/RESULTS.md` |
| Compositional moral mean critical noise: step 5K, step 30K (seed 42 only) | 5.69, 2.26 | `outputs/phase_c4_compositional/compositional_per_checkpoint.json` |
| Compositional moral mean critical noise: step 5K (4-seed mean ± std) | 5.11 ± 0.95 (peak) | `outputs/phase_c4_compositional/3seed/aggregate_per_checkpoint.json` |
| Compositional moral mean critical noise: step 7K (4-seed mean ± std) | 4.65 ± 0.84 | `outputs/phase_c4_compositional/3seed/aggregate_per_checkpoint.json` |
| Compositional moral mean critical noise: step 30K (4-seed mean ± std) | 2.46 ± 0.28 | `outputs/phase_c4_compositional/3seed/aggregate_per_checkpoint.json` |
| Compositional moral mean critical noise: step 36K (4-seed mean ± std) | 2.49 ± 0.12 | `outputs/phase_c4_compositional/3seed/aggregate_per_checkpoint.json` |
| Compositional fragility 7K → 30K decline gap (4-seed) | 2.19 (max std 0.84; rule passes) | `outputs/phase_c4_compositional/3seed/decision.json` |
| Compositional moral validation gate (peak − baseline) | +78.7 pp (gate ≥ +10 pp) | `outputs/phase_c4_compositional/c4_validation.json` |

Use these exact values; don't paraphrase or round inconsistently.

## Framing decisions locked in

- **Methodology-led, not science-led.** The thesis is "probing accuracy is the wrong instrument; fragility is right." The moral-emergence science earns the methodology its keep.
- **Lexical→compositional gradient, not lexical-accessibility hedge.** Phase C4 turned the original "we frame this as lexical accessibility" hedge into a quantitative four-point gradient (1K / 2K / 4K / 6K). Every sentence that could be read as "moral reasoning emerges" gets rephrased as "moralized lexical distinctions become decodable at step 1K; compositional moral integration at step 4K." The compositional probe is now a real §3 subsection (§3.2) and a real §4 finding (§4.1, §4.2), not an appendix hedge.
- **Compositional probe construction protocol is a real §3.2 subsection.** Originally planned for Appendix D; promoted to body because the dataset-construction methodology (TF-IDF gate as the operational compositional check, 60 % content-overlap gate in genuine tension with multi-token contrast, 47-word strong-valence blocklist, ~5 iterative rewrite passes) is the contribution that earns the gradient finding. Appendix D becomes the full pair list.
- **C15 reframed gets one paragraph in §5.3 only.** No figure. Save for Paper 2.
- **Foundation emergence and causal-probing divergence go to appendix.** Not load-bearing for the methodological thesis.
- **Compositional probe partially obviates standard moral validity controls.** The original plan required leave-lexeme-out / paraphrase / adversarial swap controls for the standard moral probe before submission. The compositional probe (§3.2) addresses the strongest version of "your probe is just reading vocabulary" by construction. Recommended path: still run the three controls on the standard probe for parity with the persona probe, but they are no longer load-bearing — flag this question to the reviewer.
- **Honest about scope.** Single model family, single language, lexical→compositional gradient (not "moral reasoning emerges"), probe-methodology limitations for syntax *and* compositional moral (the plateau coincidence is a probe-side property until 7B replication). Stated in §5.4 without softening.

## Open items

- ~~**3-seed compositional fragility replication.**~~ **Done.** Verdict
  `decline_real`; §4.3 locks to the full-subsection branch. Numbers in
  `outputs/phase_c4_compositional/3seed/`.
- **Validity controls (deprioritized).** Leave-lexeme-out, paraphrase, adversarial swap controls for the standard moral probe — partially obviated by the compositional probe (§3.2). Still worth running for parity with the persona probe. ~4-6 hours; can run in parallel with prose drafting. Tracked under Appendix C and `RESEARCH_PLAN.md` Open items.
- **7B / 32B compositional probe replication (Phase E).** Gates the §4.2 plateau-coincidence interpretation (model ceiling vs. probe ceiling). Out of scope for Paper 1; flagged in §5.4 limitations and §6 conclusion.
- **Final figure styling.** Paper-ready figures (consistent color palette, font sizing, error bars on per-checkpoint values where applicable). The 4-seed band on the compositional fragility panel is now the standard; Figure 1 onset overlay can stay 1-seed for the standard moral / sentiment / syntax curves but should be marked clearly. One day of polish work after the prose draft is stable.
- **Submission target final selection.** NeurIPS Safe Generative AI workshop deadline / ICLR R2-FM deadline / arXiv preprint date — pick after the prose draft is complete.

## Cite list (anchor references for Claude Code)

Full citations in `REFERENCES.md`. Use these exact citations; don't fabricate or guess years.

- Alain & Bengio (2017) — probing methodology
- Belinkov (2022) — probing limitations survey
- Meng et al. (2022) — causal tracing (appendix only)
- Power et al. (2022) — grokking, phase transitions
- Haidt (2012); Graham et al. (2013) — Moral Foundations Theory
- Groeneveld et al. (2024) — OLMo
- Pres et al. (2024) — refusal direction (related-work fragility analogue)
- Reblitz-Richardson (Paper 2 in preparation) — C15 reframed reference in §5.3

## Drafting order recommendation for Claude Code

1. **§3 Methodology.** Concrete and constrained; least dependent on framing decisions. Builds momentum.
2. **§4 Results.** Pull numbers from the table above, write each subsection from the corresponding outputs/*/RESULTS.md file. §4.1 leaves a placeholder paragraph for the Phase C4 compositional probe result.
3. **§2 Related work.** Light pass; expand specific connections during §5 drafting.
4. **§5 Discussion.** §5.1 and §5.2 are interpretation of the §4 findings; §5.3 is the brief C15 mention; §5.4 is limitations.
5. **§1 Introduction.** Easier to write last because by now the paper's actual scope and emphasis are concrete.
6. **§6 Conclusion.** Last.
7. **Abstract.** Update once Phase C4 lands; tighten after §1 is in place.
8. **Appendices.** As supporting material is finalized.

For each section, work in `paper/sections/NN_section_name.md`. Stitch into `paper/main.md` (or `paper/main.tex` if going LaTeX) after individual sections are stable. Keep figures in `paper/figures/`.

Don't draft beyond what's specified here without checking back. Framing changes go through the source-of-truth chat, not unilaterally.
