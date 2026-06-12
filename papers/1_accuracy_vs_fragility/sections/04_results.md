# 4. Results

## 4.1 Emergence ordering: a lexical→compositional gradient

We train the four linear probes from §3.1-§3.2 on hidden states from
all 37 OLMo-2 1B early-training checkpoints (steps 0-36K at 1K
intervals). Onset is the first checkpoint where mean probe accuracy
across all 16 layers reaches 0.70. **Figure 1** plots the four
mean-accuracy trajectories on a shared step axis.

| Probe | Construction | Onset step | Onset mean acc | Plateau mean acc (step 36K) |
|-------|--------------|-----------:|----------------:|-----------------------------:|
| Standard moral | single morally-loaded lexeme swap | 1,000 | 0.760 | 0.960 |
| Sentiment | single valenced adjective swap | 2,000 | 0.790 | 0.976 |
| **Compositional moral** | **multi-token integrated swap** | **5,000** | **0.709 ± 0.025** | **0.769 ± 0.030** |
| Syntax | structural well-formedness | 6,000 | 0.717 | 0.774 |

*Table 1: Probe onset and plateau by construction. Compositional
moral values are 4-seed mean ± std (split seeds 42 / 43 / 44 / 45).
Per-seed compositional onsets: 4K, 4K, 7K, 7K (substantial seed
variance, with the 4-seed mean curve crossing 0.70 at step 5K). The
single-seed standard moral / sentiment / syntax curves are reported
without std bands; their seed dependence is not characterized.*

**(1) The four probes resolve into a quantitative
lexical→compositional gradient.** The standard moral probe (single
morally-loaded lexeme swap, "betrayed" / "greeted") onsets at step
1K. The compositional moral probe (multi-token integrated swap;
contrast tokens "protect" / "humiliate", "hungry" / "wealthy" are
individually mild) onsets at step 5K under 4-seed averaging, a
4K-step lag, with per-seed onsets ranging 4K-7K and overall
trajectory always between sentiment (2K) and syntax (6K). The
standard probe's step-1K onset measures how quickly moralized
vocabulary becomes linearly separable, not how quickly moral
valence is encoded compositionally. Both findings are true; the
strongest single-token reading of the standard onset is ruled out,
while the gradient reading (lexically-marked moralized vocabulary
first, compositional moral integration second, syntactic competence
last) holds. Onset accuracy alone understates the case: the 0.709
onset sits only ~8 pp above the 0.63 lexical floor, and onset ordering
across all four datasets tracks lexical-floor height (unigram TF-IDF
floor: standard moral 0.86, sentiment 0.80, compositional 0.63, syntax
0.59; the more lexically separable a dataset, the earlier its onset and
the higher its plateau). Onset timing on its own therefore does not
separate compositional encoding from lexical difficulty. The evidence
that the probe recovers compositional rather than lexical signal comes
from transfer and lift (§3.2).

**(1b) Compositional encoding is real, not lexical lookup
(final-checkpoint evidence).** At the OLMo-2 1B final checkpoint
(~2.2T tokens), a probe trained on three construction categories and
tested on the held-out fourth transfers at **0.848** mean (0.80-0.91
across the four held-out constructions), essentially matching its
in-distribution pair-disjoint accuracy (**0.858** @ layer 7), while a
bag-of-words classifier doing the same leave-construction-out transfer
collapses to **0.598**. Within each construction the probe decodes
+0.20 to +0.28 above the unigram lexical floor. The decisive case is
*role_reversal*, where the same components appear on both sides and the
lexical floor is lowest (0.57): hidden states still decode moral
valence at 0.85 (lift +0.28) and held-out transfer reaches 0.81. A
probe reading contrast-token identity could not do this; the model
reads moral valence from context. This is the operational content of
"compositional" in this paper, and it is a positive result, not a thin
margin over a bag-of-words floor.

**(1c) Compositional encoding emerges in early pre-training, then
holds.** **Figure 5** plots the transfer-and-lift analysis across all
37 early-training checkpoints. At initialization the encoding is
absent: leave-construction-out transfer sits at chance (0.55) and lift
is ~0. It emerges over steps 2K-9K, crossing the bag-of-words transfer
floor (~0.60) by step 2K (0.667), passing 0.70 by step 3K, reaching
0.78 by step 5K, then plateauing by step ~9K at transfer ~0.82 and
lift ~+0.20 and holding there through step 36K (0.83 / +0.22). The
role_reversal construction, where lexical cues are scrambled by design,
follows the same curve (panel a). The encoding also localizes in
depth: once it appears, the most decodable layer settles into
mid-network (layers 8-10) and stays there for the rest of the
trajectory (panel b). Compositional moral encoding is therefore an
early-pre-training acquisition, emerging after lexical moral detection
(standard-probe onset at step 1K) and consistent with the
lexical→compositional ordering, that once acquired is stable across the
remaining ~30K steps we observe. Numbers source:
`outputs/phase_c4_compositional/b_traj/` (per-checkpoint) and
`b_traj_summary.json`.

**(2) Step-like vs. gradual emergence dichotomy.** Standard moral
and sentiment probes show sharp sigmoidal transitions (chance →
plateau within one 1K-step interval at onset, then flat).
Compositional moral and syntax rise more gradually (~3-5K steps
across the 0.70 threshold). At 1K sampling we resolve transitions
that are step-like at this resolution; we do not claim true
discontinuity. This parallels grokking-literature observations
(Power et al., 2022) that some capabilities emerge sharply and others
gradually; the within-run split here suggests the distinguishing
factor is whether the capability is acquirable from local lexical
statistics (sharp) or requires multi-token integration (gradual).
§5.1 develops.

**(3) Plateau coincidence.** The four-curve overlay (Figure 1) makes
a structural caveat visually inescapable: probes whose signal lives
in single-token vocabulary statistics (standard moral, sentiment)
plateau at 0.96 and 0.98, while probes whose signal requires
multi-token structural or compositional integration (compositional
moral, syntax) plateau at 0.77 and 0.77. The 20-percentage-point
ceiling gap is consistent across the entire 0-36K trajectory. This
may be a probe-side property under our methodology rather than a
model property: either the 1B model encodes both compositional moral
valence and syntactic well-formedness at ≈0.77 (model ceiling), or
mean-pooled linear probing on 1B hidden states bottoms out at ≈0.77
for multi-token integration regardless of underlying representational
quality (probe ceiling). The cleanest disambiguation is repeating
§4.1 at 7B and 32B: if compositional moral rises with scale while
syntax does not, the model is the bottleneck; otherwise the probe is.
We state both readings honestly in §5.3 and refine rather than
overturn the gradient finding.

**Generalization to OLMo-3 7B.** We have not yet run the compositional
probe on the OLMo-3 7B trajectory; doing so is the cleanest
disambiguation of the plateau-coincidence ambiguity and is flagged
as future work in §5.3.

Numbers source: `outputs/phase_c2/c2_emergence_timing.json` (standard
moral + sentiment + syntax, 37 checkpoints) and
`outputs/phase_c4_compositional/c4_emergence_timing.json` (compositional
moral, 37 checkpoints; companion JSON with all four curves overlaid).
Validation source: `outputs/phase_c4_compositional/c4_validation.json`
(final-checkpoint validation gate on `allenai/OLMo-2-0425-1B`).

## 4.2 Probing accuracy saturates; fragility doesn't

**Figure 2** provides the central comparison for the methodological
claim: a two-panel comparison on a shared step axis. Top panel:
mean probing accuracy, a sharp sigmoid from chance (~0.59) to a
plateau (~0.95) between steps 0 and 4K, then flat for the remaining
~33K steps. Bottom panel: mean fragility, an initial rise alongside
accuracy in the first few thousand steps, then continued movement throughout.
Top panel reaches a ceiling and stops; bottom panel keeps moving for
the entire remaining 90 % of training.

**OLMo-2 1B, 37 checkpoints, dense sampling.**

| Step | Mean acc | Mean critical noise | Late-layer crit | Mid-layer crit | Early-layer crit |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.590 | 0.77 | 0.1 | 0.6 | 1.6 |
| 1,000 | 0.728 | 7.81 | 10.0 | 8.8 | 4.4 |
| 4,000 | 0.941 | 10.0 | 10.0 | 10.0 | 10.0 |
| 10,000 | 0.943 | 7.90 | 10.0 | 8.8 | 7.2 |
| 15,000 | 0.954 | 7.50 | 10.0 | 10.0 | 5.0 |
| 20,000 | 0.950 | 7.40 | 10.0 | 10.0 | 2.2 |
| 36,000 | 0.954 | 6.12 | 10.0 | 6.5 | 1.8 |

*Table 2: Standard moral probe. Accuracy plateaus by step 4K;
fragility evolves through step 36K with a layer-depth gradient that
develops progressively (late > mid > early after step ~15K).*

**Figure 3** shows the same trajectory as two stacked layer-depth
heatmaps: probing accuracy (uniformly green after step 4K, no
remaining structure to resolve) above critical noise (gradient
emerging: late layers hold maximum noise tolerance throughout
while early layers grow progressively more brittle). Same data;
different metric; different visible structure.

The pattern reproduces at OLMo-3 7B (5 sparse checkpoints):
mean critical noise rises 2.68 → 5.14 between steps 0 and 353K,
then holds at ~5.3 through step 1.4M; layer-depth gradient is
steeper (late ~10.0 / mid ~6.2 / early ~2.0) and the most-robust
layer drifts deeper across training (layer 1 → 15 → 16 → 10 → 10).
The 1B trajectory is the headline because dense 1K-step sampling
resolves the saturation step (~4K) and gradient emergence rate.

**Compositional probe fragility evolution (4-seed replication; the
methodological claim generalizes beyond the standard probe).** We
ran `MoralFragilityTest` (§3.4) on the compositional dataset across
all 37 OLMo-2 1B early-training checkpoints with four split seeds
(42, 43, 44, 45), the original seed-42 trajectory plus a three-seed
replication ~50 min on the same MacBook Pro M4 Pro / MPS. **Table 3**
gives the 4-seed mean ± std at the diagnostic checkpoints; the
4-seed accuracy band on **Figure 1** carries the matching probing-
side trajectory.

| Step | Compositional mean critical noise (4-seed mean ± std) | n |
|-----:|------------------------------------------------------:|---|
| 0 | 0.10 ± 0.00 | 4 |
| 1,000 | 0.14 ± 0.04 | 4 |
| 2,000 | 0.94 ± 0.17 | 4 |
| 3,000 | 3.47 ± 1.04 | 4 |
| 5,000 | **5.11 ± 0.95** (peak) | 4 |
| 6,000 | 4.31 ± 1.57 | 4 |
| 7,000 | 4.65 ± 0.84 | 4 |
| 10,000 | 4.60 ± 0.48 | 4 |
| 20,000 | 3.07 ± 0.91 | 4 |
| 30,000 | 2.46 ± 0.28 | 4 |
| 36,000 | 2.49 ± 0.12 | 4 |

*Table 3: 4-seed compositional fragility evolution. The std collapses
from 1.57 (step 6K) to 0.12 (step 36K); at the late plateau the
four seeds converge tightly.*

The compositional probe reproduces the qualitative pattern (accuracy
plateaus by step ~5K; mean critical noise continues evolving through
step 36K) and shows its own quantitatively distinct long-term shape:
fragility rises through step 5K alongside accuracy onset (4-seed
mean 0.10 → 5.11), then declines through step 30K (5.11 → 2.46) and
holds. To verify the post-step-7K decline is replicable rather than
a single-seed artifact, we apply a pre-registered decision rule: the
decline counts as real if 4-seed mean critical noise drops by ≥ 1.0
between step 7K and step 30K *and* seed-to-seed std at both endpoints
is smaller than the gap. Realized values: gap = 4.65 − 2.46 = 2.19
(≥ 1.0 ✓), max endpoint std = 0.84 (< 2.19 ✓). Both pass with
substantial margin; the post-step-7K decline is a stable property
across the four split seeds.

Two non-exclusive readings of the diverging long-term direction
(7B / 32B replication disambiguates both): a *mechanism reading* (as training
continues on text that does not specifically reinforce compositional
moral integration, the compositional representation drifts toward
brittleness while standard-probe representations are continually
reinforced by moralized vocabulary density) and a *probe-ceiling
reading* (fragility at the 0.77 operating point has less headroom
than at 0.96, partly artifacting the difference). We state both in
§5.3 without commitment.

Numbers sources: `outputs/phase_c1/RESULTS.md` (1B standard probe),
`outputs/phase_b/` (7B corroboration),
`outputs/phase_c4_compositional/3seed/{aggregate_per_checkpoint,decision}.json`
(4-seed mean ± std and decision rule application),
`outputs/phase_c4_compositional/3seed/4seed_fragility_evolution.png`
(headline 4-seed plot).

## 4.3 Data curation reshapes probe robustness, not probe accuracy

LoRA (Hu et al., 2022) fine-tuning on three matched
corpora from the OLMo-2 1B step-1000 checkpoint (mid-transition, ~80 % peak probing
accuracy). Corpora: a 247K-token narrative-moral corpus (Aesop /
Grimm / Andersen), a 500K-token declarative-moral corpus
(template-expanded `MORAL_SEEDS`: "Stealing is wrong"), and a 420K-token
general non-moral control (Darwin). Identical LoRA hyperparameters
(rank 16, alpha 32, q\_proj + v\_proj, lr 2e-4, batch 2, seq 1024,
1000 steps); standard moral probe + fragility every 100 LoRA steps.

**Probing accuracy is identical across conditions.** Final peak
accuracy at LoRA step 1000: narrative 0.740, declarative 0.750,
general control 0.750, all within 1 pp across very different training
data. The accuracy metric returns no signal for which corpus produces
what kind of representational change.

**Fragility profiles are condition-specific (the main result).**
Final mean critical noise: narrative 7.38, declarative 5.63, general
control 6.94. The per-layer breakdown separates the conditions:
narrative and general control show fragility dips at 6 and 7 of 16
layers respectively; the declarative condition shows dips at **10 of
16 layers**, creating a broadly more fragile representation than
either natural-text condition. **Figure 4** plots all three per-layer
profiles plus the three identical accuracy bars: same accuracy,
different fragility.

**Training loss is decoupled from representational change.**
Declarative loss drops 5.5 → 1.0 (template memorization), narrative
4.3 → 4.0, control 5.0 → 4.4. The condition with the deepest loss
reduction is the same condition with the most diffuse fragility; the
two with the shallowest loss reductions retain more robust
representations. The model is learning declarative templates as
surface text patterns without that learning translating into either
accuracy gains or robust representational structure.

**Why diffuse fragility under declarative training?** The declarative
corpus consists of repeated syntactic templates ("X is wrong", "Y
is immoral") that the model memorizes easily (loss → 1.0). This
memorization creates narrow pattern-matching features across
multiple layers, features whose probe accuracy is maintained by
low-margin decision boundaries that collapse under small noise.
Narrative and general-control conditions produce fewer fragile
layers because natural text has no repeated syntactic template;
moral content in Aesop's fables is embedded in diverse narrative
structures, forcing the probe to rely on distributed features that
tolerate noise. The declarative fragility pattern is consistent
with the §4.2 finding that early layers grow progressively more
brittle over training; declarative LoRA produces a diffuse
version of a vulnerability pattern that pre-training produces as
a layer-depth gradient.

This is direct evidence for the methodological thesis in a
controlled setting: same data, same probe; accuracy returns no
signal, fragility separates the conditions.

**Why the fragility is diffuse rather than localized.** The diffuse
pattern, fragility across 10 of 16 layers (mean σ* = 5.63 vs.
7.38 / 6.94) rather than a single dramatic dip, has a straightforward
mechanistic explanation. When the probing dataset controls for
animacy and register confounds (§3.1), the probe detects actual
moral features at every layer rather than exploiting shortcuts like
"is this about a person or a circuit?" that survive Gaussian
perturbation easily. When declarative templates are memorized, the
model's moral representations throughout the network become
template-dependent: narrow-margin features that collapse under
noise at multiple layers rather than just one. Template memorization
does not corrupt one layer; it degrades the network's moral
representations broadly. This sensitivity to dataset quality
confirms the importance of the validation methodology described
in §3.1: probing datasets that contain animacy or register
shortcuts will systematically underestimate fragility at layers
where the probe exploits those shortcuts rather than moral content.

Numbers source: `outputs/phase_c_tier2/c3/RESULTS.md` and
`outputs/phase_c_tier2/c3/{narrative,declarative,general}_moral.json`
(per-layer fragility for all three conditions).
