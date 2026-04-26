# 4. Results

We organize the four results subsections to lead with the science
finding most useful for understanding the methodological gap (§4.1
emergence ordering), pause for the structural caveat that the gradient
finding makes visible (§4.2 plateau coincidence), then present the
methodological work itself (§4.3 accuracy saturates, fragility
doesn't), and end with the cleanest controlled comparison evidencing
the methodological claim (§4.4 data curation reshapes structure).

## 4.1 Emergence ordering: a lexical→compositional gradient

We train the four linear probes from §3.1-§3.2 on hidden states from
all 37 OLMo-2 1B early-training checkpoints (steps 0-36K at 1K-step
intervals). Onset is defined as the first checkpoint where mean probe
accuracy across all 16 layers reaches 0.70. **Figure 1** plots the
four mean-accuracy trajectories on a shared training-step axis;
**Table 1** summarizes the onset and plateau values.

| Probe | Construction | Onset step | Onset mean acc | Plateau mean acc (step 36K) |
|-------|--------------|-----------:|----------------:|-----------------------------:|
| Standard moral | single morally-loaded lexeme swap | 1,000 | 0.760 | 0.960 |
| Sentiment | single valenced adjective swap | 2,000 | 0.790 | 0.976 |
| **Compositional moral** | **multi-token integrated swap** | **4,000** | **0.721** | **0.774** |
| Syntax | structural well-formedness | 6,000 | 0.717 | 0.775 |

*Table 1: Probe onset and plateau by construction.*

Three findings, in order of importance for the paper's thesis.

**(1) The four probes resolve into a quantitative
lexical→compositional gradient.** The standard moral probe — whose
pairs differ in a single morally-loaded lexeme ("murdered" /
"greeted") — onsets at step 1K (mean 0.760). The compositional moral
probe — whose pairs hold the action verb constant and vary only
individually-mild tokens whose moral status flips in context
("protect" / "humiliate", "hungry" / "wealthy") — onsets at step 4K
(mean 0.721), a 3K-step lag. The standard probe's step-1K onset
therefore measures *how quickly moralized vocabulary becomes linearly
separable from neutral vocabulary*, not *how quickly moral valence
is encoded compositionally*. Both findings are true; they say
different things about what the model has learned. The strongest
reading of the C2 result — that moral concepts are encoded
compositionally from step 1K — is ruled out by the compositional
probe; the remaining reading — that lexically-marked moralized
vocabulary is decoded first, compositional moral integration second,
syntactic competence last — is supported.

The compositional probe's content-only floor (TF-IDF + LogReg, 5-fold
CV) is 0.113 overall and 0.14-0.20 per category (§3.2), so the
0.721 onset mean accuracy is +60.8 percentage points above
single-token bag-of-words separability. Whatever the linear probe is
recovering at step 4K must integrate multiple words. At the OLMo-2 1B
final checkpoint (~2.2T tokens), the compositional probe reaches 0.900
peak accuracy at layer 5 — +78.7 pp over the TF-IDF baseline,
satisfying the validation gate of §3.2 by a wide margin.

**(2) Phase-transition vs. gradual emergence dichotomy.** Standard
moral and sentiment probes show sharp sigmoidal phase transitions:
each crosses from ~55 % chance to its plateau within a single 1K-step
interval at onset, then flattens. Compositional moral and syntax
probes rise more gradually, with no equally sharp inflection — the
compositional probe takes ~3 checkpoints (steps 1K → 4K, mean acc
0.545 → 0.721) to cross the 0.70 threshold; syntax takes ~5K steps
(steps 1K → 6K, mean acc 0.552 → 0.717). This dichotomy parallels the
grokking literature on capability emergence (Power et al., 2022): some
capabilities emerge as sharp phase transitions while others develop
gradually. The semantic / structural split here suggests the
distinguishing factor is whether a capability can be acquired through
local lexical statistics (phase transition) or requires integrating
positional, attentional, or compositional structure (gradual
emergence). Compositional moral integration sits on the gradual side
of the split, with syntactic competence — closer to lexical statistics
than syntax usually is reputed to be at the lexeme-swap minimal-pair
granularity we use.

**(3) Plateau coincidence (developed in §4.2).** Standard moral and
sentiment plateau near 0.97; compositional moral and syntax plateau
near 0.77. The 0.20 plateau gap tracks the lexical / compositional
split: probes whose discriminative signal lives in single-token
statistics saturate higher than probes whose discriminative signal
requires multi-token integration, under our mean-pooled linear
probing methodology. We treat this as a structural caveat to the
gradient finding rather than a separate result, and develop it in
§4.2 as a single subsection that the four-curve overlay (Figure 1)
makes visually self-evident.

**Generalization to OLMo-3 7B.** We have not yet run the compositional
probe on the OLMo-3 7B trajectory; doing so is the cleanest
disambiguation of the §4.2 plateau-coincidence ambiguity (model
ceiling vs. probe ceiling) and is flagged as a Phase E experiment in
§5.4 limitations.

Numbers source: `outputs/phase_c2/c2_emergence_timing.json` (standard
moral + sentiment + syntax, 37 checkpoints) and
`outputs/phase_c4_compositional/c4_emergence_timing.json` (compositional
moral, 37 checkpoints; companion JSON with all four curves overlaid).
Validation source: `outputs/phase_c4_compositional/c4_validation.json`
(final-checkpoint validation gate on `allenai/OLMo-2-0425-1B`).

## 4.2 Plateau coincidence: compositional ≈ syntax under mean-pooled linear probing

The four-curve overlay (Figure 1) makes one structural finding
visually inescapable: probes whose signal lives in single-token
vocabulary statistics (standard moral, sentiment) plateau at 0.96 and
0.98, while probes whose signal requires multi-token structural or
compositional integration (compositional moral, syntax) plateau at
0.77 and 0.78. The 20-percentage-point ceiling gap is consistent
across the entire 0-36K trajectory and across both pairs of probes
that share a structural property (single-token vs. multi-token).

This is a probe-side property under our methodology, not necessarily
a model property. Two competing readings:

- **Model ceiling.** The OLMo-2 1B model genuinely encodes
  compositional moral valence at ≈0.77 accuracy and syntactic
  well-formedness at ≈0.78 accuracy at the granularity of our
  minimal-pair tasks; mean-pooled linear probing accurately recovers
  this representational state. Under this reading, both ceilings
  reflect model capacity.
- **Probe ceiling.** Mean-pooled linear probing on 1B-scale hidden
  states can only recover ≈0.77 of compositional / structural signals
  at this scale, regardless of whether the underlying representations
  are stronger. A non-linear probe (single hidden-layer MLP) or a
  position-sensitive probe (last-token rather than mean-pooled hidden
  states) would push the ceiling higher. Under this reading, both
  ceilings reflect probe limitations rather than model limitations,
  and the model encodes both compositional moral and syntactic
  features more accurately than our probe recovers.

We cannot distinguish the two at 1B with linear probing alone. The
cleanest disambiguation is repeating §4.1 at 7B and 32B in Phase E:
if compositional moral accuracy rises with model scale while syntax
accuracy does not, the bottleneck at 1B is the model not the probe;
if both ceilings track scale together, the probe is the bottleneck.
Either result is publishable and refines the gradient finding without
overturning it. We state both readings honestly in §5.4 limitations
rather than committing to one interpretation that the data does not
support.

## 4.3 Probing accuracy saturates; fragility doesn't

The figure that does the most work for the methodological thesis is
**Figure 2**: a two-panel comparison on a shared training-step axis,
top panel showing mean probing accuracy over training and bottom
panel showing mean fragility. Top panel is a sharp sigmoid from
chance (~55 %) to a plateau (~95 %) between steps 0 and 4K, then
completely flat for the remaining ~33K steps. Bottom panel shows the
initial rise alongside accuracy in the first 1K steps, then continued
movement throughout: early-layer fragility *declines* (early-layer
critical noise drops 10.0 → 1.7 between steps 1K and 36K) while
late-layer fragility holds at maximum (10.0 throughout). The visual
contrast is the methodological point: the top panel reaches a ceiling
and stops moving; the bottom panel keeps moving for the entire
remaining 95 % of the training trajectory.

**Phase C1 numbers (OLMo-2 1B, 37 checkpoints, dense).**

| Step | Mean acc | Peak acc | Mean critical noise | Late-layer crit | Mid-layer crit | Early-layer crit |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.557 | 0.635 | 0.18 | 0.1 | 0.1 | 0.3 |
| 1,000 | 0.768 | 0.823 | 10.0 | 10.0 | 10.0 | 10.0 |
| 2,000 | 0.883 | 0.917 | — | — | — | — |
| 3,000 | 0.921 | 0.948 | — | — | — | — |
| 4,000 | 0.938 | 0.958 | — | — | — | — |
| 10,000 | ~0.95 | ~0.96 | 7.0 | 10.0 | 10.0 | 7.7 |
| 20,000 | ~0.95 | ~0.97 | 7.5 | 10.0 | 10.0 | 6.5 |
| 36,000 | ~0.96 | ~0.98 | 5.3 | 10.0 | 5.8 | 1.7 |

*Table 2: Standard moral probe — accuracy plateaus by step 4K;
fragility evolves through step 36K with a layer-depth gradient that
develops monotonically (late > mid > early after step ~10K).*

**Generalization to OLMo-3 7B (Phase B5, 5 checkpoints, sparse).** The
same accuracy-saturates-fragility-doesn't pattern reproduces at the
7B headline scale. Mean critical noise rises 0.20 → 5.67 between
step 0 and step 353K (a ~28x increase), then plateaus at ~5.3 through
step 1.4M. The 7B layer-depth gradient is steeper than the 1B
gradient: late layers (22-31) hold critical noise ~10.0, mid layers
(11-21) ~5.5, early layers (0-10) ~2.0 — the same late-mid-early
ordering as 1B but with a wider gap. The most-robust layer drifts
deeper over 7B training (layer 2 → 7 → 11 → 10 → 15), evidencing the
"specialization" phase where the model progressively migrates robust
moral encoding into deeper layers.

We use the 7B trajectory as supporting evidence, not as the headline,
because the 1B 37-checkpoint trajectory has the dense sampling
required to resolve the saturation step (~4K) and the gradient
emergence rate. Both scales show the same qualitative pattern.

**Compositional probe fragility evolution (4-seed replication; the
methodological claim generalizes beyond the standard probe).** We
ran `MoralFragilityTest` (§3.4) on the compositional dataset across
all 37 OLMo-2 1B early-training checkpoints with four split seeds
(42, 43, 44, 45) — the original C4 trajectory plus a three-seed
replication ~50 min on the same MacBook Pro M4 Pro / MPS. **Table 3**
gives the 4-seed mean ± std at the diagnostic checkpoints, and
**Figure 2** (lower right inset, or a small fourth panel) plots the
full mean ± shaded std band against the C1 standard moral probe as
the comparison curve.

| Step | Compositional mean critical noise (4-seed mean ± std) | n |
|-----:|------------------------------------------------------:|---|
| 0 | 0.10 ± 0.00 | 4 |
| 1,000 | 0.14 ± 0.04 | 4 |
| 2,000 | 0.94 ± 0.17 | 4 |
| 3,000 | 3.47 ± 1.04 | 4 |
| 5,000 | **5.11 ± 0.95** (peak) | 4 |
| 7,000 | 4.65 ± 0.84 | 4 |
| 10,000 | 4.60 ± 0.48 | 4 |
| 20,000 | 3.07 ± 0.91 | 4 |
| 30,000 | 2.46 ± 0.28 | 4 |
| 36,000 | 2.49 ± 0.12 | 4 |

*Table 3: 4-seed compositional fragility evolution. The std collapses
from 1.57 (step 6K) to 0.12 (step 36K) — at the late plateau the
four seeds converge tightly.*

The compositional probe reproduces the qualitative
accuracy-saturates-fragility-doesn't pattern that anchors §4.3's
methodological thesis: probing accuracy plateaus by step ~5K (mean
acc 0.71 → 0.77 over the remaining 31K steps; Table 1) but mean
critical noise continues to evolve through step 36K. The
*direction* of the post-onset evolution differs from the standard
probe — compositional fragility rises through step 5K alongside
accuracy onset (4-seed mean 0.10 → 5.11), then declines steadily
through step 30K (5.11 → 2.46) and holds. To verify that the
post-step-7K decline is replicable rather than a single-seed
artifact, we apply a pre-registered decision rule (per
`paper/PAPER_PLAN.md` §4.3): the decline counts as real if 4-seed
mean critical noise drops by ≥ 1.0 between step 7K and step 30K,
*and* seed-to-seed std at both endpoints is smaller than the gap.
The realized values: gap = 4.65 − 2.46 = 2.19 (≥ 1.0 ✓), max std at
the two endpoints = 0.84 (< 2.19 ✓). Both conditions pass, with
substantial margin in both. The post-step-7K decline is a stable
property of the compositional probe across the four split seeds, not
seed-side noise.

Two readings of the diverging long-term trajectories. **(a) The
methodological claim generalizes both qualitatively and
quantitatively.** Both probes show fragility rising alongside
accuracy onset, peaking at or just after the onset step, and then
continuing to evolve while accuracy plateaus — establishing that
"fragility resolves what accuracy misses" is not specific to lexically-
marked moral encoding. **(b) The compositional probe's late-training
*direction* is its own finding.** Compositional fragility re-enters a
brittle state (mean critical noise drifting back toward 2-3 by step
30K) while standard moral fragility holds at its early peak across
mid- and late-layers throughout training (mean critical noise ≈ 5-10).
This divergence is consistent with two non-exclusive hypotheses,
either of which is testable in Phase E:

- **Mechanism hypothesis.** As the model continues training on natural
  text that does not specifically reinforce compositional moral
  integration, the compositional representation drifts from a state
  where it is accidentally robust (the early-training plateau) toward
  a state where it is encoded but fragile to noise. Standard-probe
  representations, in contrast, are continually reinforced by the
  density of moralized vocabulary in pre-training data.
- **Probe-ceiling hypothesis.** The compositional probe's final-training
  accuracy is structurally bounded at ~0.77 (§4.2 plateau coincidence)
  while the standard probe operates near 0.96. Fragility computed at
  the lower-accuracy operating point may have less headroom to
  saturate the noise sweep, producing a numerical difference that is
  partly an artifact of the accuracy ceiling rather than a
  representational property. A non-linear or position-sensitive probe
  that pushed compositional accuracy higher would test this directly.

We do not commit to either reading in the paper; we state both and
flag Phase E (7B / 32B compositional fragility replication) as the
cleanest disambiguation in §5.4.

Numbers source: `outputs/phase_c1/RESULTS.md` (1B standard probe),
`outputs/phase_b/` (7B corroboration),
`outputs/phase_c4_compositional/compositional_per_checkpoint.json`
(compositional seed 42),
`outputs/phase_c4_compositional/3seed/aggregate_per_checkpoint.json`
(4-seed mean ± std),
`outputs/phase_c4_compositional/3seed/decision.json` (decision rule
application),
`outputs/phase_c4_compositional/3seed/4seed_fragility_evolution.png`
(headline 4-seed plot).

## 4.4 Data curation reshapes structure, not content

The cleanest controlled comparison in the paper is Phase C3: LoRA
fine-tuning on three matched corpora with identical hyperparameters,
starting from the OLMo-2 1B step-1000 checkpoint (mid-transition,
~80 % peak probing accuracy). The three corpora are content-matched
where possible: a 247K-token narrative-moral corpus (Aesop's Fables,
Grimm's Fairy Tales, Andersen), a 500K-token declarative-moral corpus
(template-expanded `MORAL_SEEDS` covering the same six MFT foundations
in declarative form: "Stealing is wrong"), and a 420K-token general
non-moral corpus (Darwin's *Voyage of the Beagle* and *Origin of
Species*) as a content-free control. LoRA hyperparameters are
identical across conditions: rank 16, alpha 32, q\_proj + v\_proj
target modules, learning rate 2e-4, batch size 2, sequence length
1024, 1000 training steps. We evaluate the standard moral probe and
fragility test every 100 LoRA steps.

**Probing accuracy is identical across conditions.** Final peak probe
accuracy at LoRA step 1000: narrative 0.812, declarative 0.802,
general control 0.802. The three values are within 1 percentage point
across very different training data; the 80 % peak accuracy at the
base checkpoint (step 1000 of pre-training) does not budge under any
of the three LoRA conditions. The accuracy metric returns *no signal*
for which corpus produces what kind of representational change.

**Fragility profiles are condition-specific (the main result).** Final
mean critical noise at LoRA step 1000: narrative 10.0, declarative
9.42, general control 10.0. The mean-noise difference is small but
the per-layer breakdown is decisive. For the narrative and general
control conditions, every layer where the probe drops below threshold
under the noise sweep does so at maximum noise (10.0); the fragility
profile is uniformly robust. For the declarative condition, **layer 3
collapses to critical noise = 3.0** while every other layer holds at
10.0. This is a single, sharply localized fragility dip created by
LoRA training on declarative moral statements — a brittle shortcut at
exactly one layer that natural-text training (whether narrative-moral
or general-text) does not produce.

**Figure 4** plots the per-layer critical noise for all three
conditions on the same axis: narrative and general control are
indistinguishable flat lines at 10.0, declarative is the same flat
line with a single sharp dip at layer 3. The contrast is visually
self-evident. Probing accuracy as three identical bars across the top
of the figure makes the point inescapable: same accuracy, different
fragility.

**Training loss is decoupled from representational change.** Training
loss diverges across conditions in the opposite direction from
representational fragility: declarative loss drops from 5.6 to 1.0
(the model trivially memorizes the templated `MORAL_SEEDS` text),
narrative loss stays in the 5.0 → 3.9 range (diverse natural text),
control loss stays in the 5.0 → 4.5 range (no specific signal to
memorize). The condition with the deepest training loss reduction
(declarative, 4.6 nats of loss reduction) is the same condition that
produces the localized layer-3 fragility dip; the conditions with the
shallowest training loss reduction (narrative and control) leave
fragility profiles unchanged. **The model is learning the declarative
templates as surface text patterns without that learning translating
into either accuracy gains (the probe is unchanged) or robust
representational structure (the probe collapses at one layer).**

This is the cleanest single piece of evidence for the paper's
methodological thesis. Probing accuracy says "no difference between
narrative-moral and declarative-moral training"; fragility says
"declarative-moral training creates a brittle layer-3 shortcut that
natural-text training does not." The two metrics return *opposite*
answers to "does the kind of moral content matter" on the same data
with the same probe. The fragility answer is the actionable one for
data-curation decisions; the accuracy answer is the misleading one.

We treat this as a 1B-scale proof-of-concept and flag the natural
follow-up — replicating the C3 design with the compositional probe
rather than the standard moral probe, to ask whether data curation
reshapes compositional fragility the same way it reshapes lexical
fragility — as a Phase E experiment in §5.4.

Numbers source: `outputs/phase_c_tier2/c3/RESULTS.md` and
`outputs/phase_c_tier2/c3/{narrative,declarative,general}_moral.json`
(per-layer fragility for all three conditions).
