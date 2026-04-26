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

**Compositional probe fragility evolution (conditional on 3-seed
replication).** We ran `MoralFragilityTest` (§3.4) on the
compositional dataset across all 37 OLMo-2 1B early-training
checkpoints. Mean critical noise rises from 0.10 (step 0) to 5.69
(step 5K, peak), then drifts back down to 2.66 (step 36K) — *opposite*
to the standard moral probe's behavior in Phase C1, where mean
critical noise rises monotonically through step 1K and stays at the
maximum across mid- and late-layers throughout training. Compositional
fragility's post-step-7K decline could be one of two things:

- **Real and informative.** Compositional moral encoding becomes more
  brittle as the model continues training on natural text that does
  not specifically reinforce compositional moral integration; the
  representation drifts from a state where it is accidentally robust
  (sharp accuracy onset, all layers temporarily robust) toward a
  state where it is encoded but fragile to noise.
- **Probe-side noise.** A single train-test split (seed 42) at probe
  accuracy hovering around 0.75 is not enough resolution to claim a
  decline that small (5.7 → 2.7) is a real signal rather than
  seed-to-seed variation in probe initialization and split.

We have a 3-seed replication planned (split seeds 43, 44, 45;
identical pipeline; ~30 min compute) before locking the §4.3
structure. Two branches:

- **If the post-step-7K decline replicates** (mean critical noise
  drops by ≥ 1.0 between step 7K and step 30K, with seed-to-seed std
  smaller than the gap): we expand §4.3 by ~0.3 page with a
  "compositional fragility evolution" paragraph and a small companion
  panel in Figure 2; the fragility-resolves-what-accuracy-misses
  pattern then generalizes from the lexical to the compositional
  probe both *qualitatively* (the rise alongside accuracy reproduces)
  and *quantitatively* (a different long-term trajectory).
- **If the decline is within seed-noise:** we state the qualitative
  reproduction in one line — "compositional probe fragility rises
  alongside accuracy in the same 0-1K window; long-term trajectory
  noisy across seeds, see Appendix" — and Figure 2 stays as the
  standard-probe-only two-panel.

In either case, the *qualitative* generalization (compositional
fragility rises alongside compositional accuracy in the 0-1K window
and reaches its peak before accuracy plateaus, just as for the
standard probe) holds robustly: compositional mean critical noise
goes from 0.10 (step 0) to 1.19 (step 2K) to 3.71 (step 4K) to 5.69
(step 5K), reproducing the standard probe's pattern in this window.
The methodology generalizes; only the long-term shape is
seed-conditional.

Numbers source: `outputs/phase_c1/RESULTS.md` (1B), `outputs/phase_b/`
(7B), `outputs/phase_c4_compositional/compositional_per_checkpoint.json`
(compositional, 1-seed; 3-seed replication forthcoming at
`outputs/phase_c4_compositional/3seed/`).

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
