# Paper 1 Plan: *Probing Accuracy Saturates; Fragility Doesn't*

**Status:** Framing locked in. Drafting can begin. Compositional probe (Phase C4) ablation is in flight; abstract has a placeholder sentence pending those results.

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

## Abstract (draft v1)

> Moral / neutral semantic distinctions are linearly decodable from base LLM hidden states within the first ~5 % of pre-training. But standard probing accuracy saturates within the first 4K of 36K early-training steps and provides no resolution for the remaining 95 % of training. We introduce *fragility* — the activation-noise level at which probe accuracy collapses — as a metric that continues evolving long after accuracy plateaus and reveals dynamics invisible to accuracy alone. We use it to establish three findings on the OLMo open-weight model family. (1) Moralized lexical distinctions emerge as a sharp phase transition before sentiment polarity (1K vs. 2K steps) and far before syntactic competence (6K steps), with semantic features showing phase-transition dynamics while syntactic features emerge gradually with no inflection point. **[PLACEHOLDER, awaiting Phase C4: compositional moral probe trajectory result; sentence will state where on the lexical→compositional gradient the result lands and what it tells us about whether the step-1K onset is purely lexical.]** (2) A layer-depth robustness gradient develops monotonically over training: late layers become maximally robust while early layers grow increasingly fragile, a pattern entirely invisible to probing accuracy. (3) LoRA fine-tuning on three matched corpora produces identical probing accuracy (~80 % across narrative-moral, declarative-moral, and general-text conditions) but distinct fragility profiles, demonstrating that data curation operates on representational structure rather than representational content. The methodological claim generalizes: in every comparison where probing accuracy returns a flat answer, fragility returns a structured one.

The placeholder sentence will read approximately:
- *Outcome A (compositional probe onsets ≤ step 2K):* "A compositional moral probe whose minimal pairs require multi-word integration to compute moral valence reaches the same step-1K onset, indicating that what we measure is not purely lexical accessibility."
- *Outcome B (compositional probe onsets between sentiment and syntax):* "A compositional moral probe whose minimal pairs require multi-word integration to compute moral valence onsets later than the lexical probe (between sentiment and syntax), mapping a finer-grained gradient from lexically-accessible to compositionally-integrated semantic features."
- *Outcome C (compositional probe never reaches 70 %):* "A compositional moral probe whose minimal pairs require multi-word integration to compute moral valence does not reach above-chance accuracy at 1B scale, bounding the standard probe's claim to lexical accessibility and motivating replication at larger scale."

Lock the appropriate variant once Phase C4 lands.

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

### 3. Methodology (~1.5 pages)

- **Minimal-pair datasets.** Moral / neutral pairs (240, balanced across six MFT foundations); sentiment pairs (210); syntax pairs (210); compositional moral pairs (200, Phase C4). Construction: matched syntactic skeleton, single-token swap for moral / sentiment / syntax probes, multi-token integrated swap for compositional probe. Validation gates: length ratio, word overlap, TF-IDF baseline, deterministic test split. All datasets API-free, deterministic, included in the toolkit.
- **`LayerWiseMoralProbe` and variants.** Binary linear probing trained at every transformer layer; mean-pooled hidden states; per-layer accuracy curve as the per-layer readout; onset layer / peak layer / encoding depth / encoding breadth as summary statistics.
- **`MoralFragilityTest`.** Inject Gaussian noise of magnitude σ into target-layer activations during the probe forward pass; record the σ at which probe accuracy drops below the fragility threshold (mean(chance, peak)/2, or chance + 0.1, whichever is larger). The minimum σ across a logarithmic sweep is the layer's *critical noise*. Per-layer critical noise across all transformer layers gives the *fragility profile*.
- **Target models and checkpoints.** OLMo-2 1B with 37 early-training checkpoints (steps 0–36K at 1K intervals); OLMo-3 7B with 20 stage-1 checkpoints. All on a single MacBook Pro M4 Pro (24 GB unified memory, MPS, fp16).
- **Required moral-probe validity controls (added before submission per RESEARCH_PLAN.md line 2253):** leave-lexeme-out splits, paraphrase transfer, adversarial lexical swap. Persona probe already has these; moral probe parity is mandatory before submission. ~4–6 hours of additional work; document the protocol and report results in the appendix or a dedicated subsection.

### 4. Results (~3 pages)

#### 4.1 Emergence ordering: lexical moral → sentiment → syntax (Phase C2)

Headline figure: four-curve onset overlay (moral, sentiment, syntax, compositional moral). Onset steps: 1K / 2K / 6K / [Phase C4 result]. Plateau accuracies: ~96 % / ~98 % / ~78 % / [Phase C4 result]. The semantic / structural dichotomy is the substantive scientific finding: semantic features (moral, sentiment) show phase-transition dynamics; syntactic features rise gradually with no inflection point. This dichotomy parallels grokking literature (Power et al. 2022) and frames the paper's contribution about *types* of representational learning, not just timing.

Numbers source: `outputs/phase_c2/RESULTS.md`.

#### 4.2 Probing accuracy saturates; fragility doesn't (Phase B5 + C1)

The figure that does the most work for the methodological thesis: two-panel comparison, shared training-step x-axis. Top panel: mean probing accuracy over training — sharp sigmoid 0 → ~95 % between steps 0 and 4K, then completely flat for the remaining ~33K steps. Bottom panel: fragility — initial jump alongside accuracy, then continued downward drift in early layers through step 36K, with late layers holding at maximum robustness. Visually self-evident: top panel is "ceiling reached, no information"; bottom panel is "still moving."

Layer-depth heatmap as Figure 2 companion: layers × training step, with two heatmaps stacked — top is probing accuracy (uniformly green after step 4K), bottom is critical noise (clear gradient developing, with late layers holding red while early layers fade toward blue).

Numbers source: `outputs/phase_c1/RESULTS.md` (1B trajectory, 37 checkpoints, dense). 7B fragility evolution from `outputs/phase_b/` is supporting evidence — same pattern at the 7B headline scale, less dense checkpoint sampling.

#### 4.3 Data curation reshapes structure, not content (Phase C3)

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

- **Lexical accessibility framing.** What we measure with moralized minimal-pair probes is closer to "moralized vocabulary becomes linearly separable from neutral vocabulary" than to "moral reasoning emerges." Phase C4's compositional probe explicitly maps where the lexical framing breaks down. State this honestly; don't soften.
- **Probe methodology.** Linear probes on mean-pooled hidden states are well-suited for lexical / semantic features and poorly suited for structural features (the syntax 78 % ceiling reflects probe limitations as much as representation quality). More sophisticated probes (attention-based, position-sensitive) would likely produce different syntactic emergence curves.
- **Single model family.** All findings are on OLMo-2 1B and OLMo-3 7B. Generalization to other architectures and training recipes is open. The Phase E plan addresses 7B replication on a Deep-Ignorance-style base.
- **Single language.** English pretraining text only.
- **Foundation-specific scope.** Liberty / oppression never fully stabilizes at either 1B or 7B; this cross-scale pattern is documented in the appendix but not in the main thesis.

### 6. Conclusion (~0.5 page)

Restate the thesis. Position fragility as a methodology contribution that the alignment-during-pre-training research program can adopt for any probing-based question, not just moral representations. End on the open question Phase C4 leaves: what does compositional moral integration look like at scale, and is fragility the right instrument there too?

### Appendices

- **A. Foundation emergence (Phase B3 / C1 finding 3).** Staggered emergence of the six MFT foundations, with liberty / oppression as the cross-scale outlier. Interesting but not load-bearing for the methodological thesis.
- **B. Causal-probing divergence (Phase B4).** Storage layers (probing peak ~10) and usage layers (causal peak ~0) diverge by ~10 layers at the 7B final checkpoint. Surprising and probably its own future paper rather than a body finding here.
- **C. Moral-probe validity controls.** Leave-lexeme-out splits, paraphrase transfer, adversarial lexical swap. Required for parity with the persona probe before submission.
- **D. Compositional probe construction.** Phase C4 dataset construction protocol, validation gates, full pair list (or representative samples).
- **E. Reproducibility appendix.** Hardware (MacBook Pro M4 Pro, 24 GB unified, MPS), seeds, software versions, command-line invocations for each result, output JSON schemas.

## Headline figures (in body order)

1. **Figure 1: Probing accuracy saturates; fragility doesn't.** Two-panel, shared training-step axis. Top: mean probing accuracy over training. Bottom: mean fragility (critical noise) over training. *This is the figure that does the most work; design it to be self-evident at two seconds of attention.*
2. **Figure 2: Layer-depth heatmaps.** Two stacked heatmaps (layers × steps): probing accuracy (uniformly green after step 4K) above, critical noise (gradient developing, late layers holding) below.
3. **Figure 3: Onset overlay.** Four curves (standard moral, sentiment, syntax, compositional moral) on shared step axis. *Updates once Phase C4 lands.*
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
| Compositional probe results | TBD | `outputs/phase_c4_compositional/RESULTS.md` |

Use these exact values; don't paraphrase or round inconsistently.

## Framing decisions locked in

- **Methodology-led, not science-led.** The thesis is "probing accuracy is the wrong instrument; fragility is right." The moral-emergence science earns the methodology its keep.
- **Lexical accessibility framing throughout.** Every sentence that could be read as "moral reasoning emerges" gets rephrased as "moralized lexical distinctions become decodable" or "moralized semantic distinctions become linearly accessible." The compositional probe (Phase C4) is the explicit ablation that takes this concern seriously.
- **C15 reframed gets one paragraph in §5.3 only.** No figure. Save for Paper 2.
- **Foundation emergence and causal-probing divergence go to appendix.** Not load-bearing for the methodological thesis.
- **Required moral-probe validity controls (leave-lexeme-out, paraphrase, adversarial swap) are mandatory before submission.** ~4–6 hours of work, runs against existing checkpoints.
- **Honest about scope.** Single model family, single language, lexical-accessibility framing for the standard probe, probe-methodology limitations for syntax. Stated in §5.4 without softening.

## Open items

- **Phase C4 compositional probe result.** Awaiting. Determines abstract sentence and adds Section 4.x or appendix material depending on outcome.
- **Validity controls.** Needs to be run before submission. Add as a Phase C5 task in RESEARCH_PLAN.md if not already there.
- **Final figure styling.** Paper-ready figures (consistent color palette, font sizing, error bars on per-checkpoint values where applicable). One day of polish work after the prose draft is stable.
- **Submission target final selection.** NeurIPS Safe Generative AI workshop deadline / ICLR R2-FM deadline / arXiv preprint date — pick after the prose draft is complete and the validity controls have run.

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
