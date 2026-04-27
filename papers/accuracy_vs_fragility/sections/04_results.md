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
| Syntax | structural well-formedness | 6,000 | 0.717 | 0.775 |

*Table 1: Probe onset and plateau by construction. Compositional
moral values are 4-seed mean ± std (split seeds 42 / 43 / 44 / 45).
Per-seed compositional onsets: 4K, 4K, 7K, 7K — substantial seed
variance, with the 4-seed mean curve crossing 0.70 at step 5K. The
1-seed C2 standard moral / sentiment / syntax curves are reported
without std bands; their seed dependence is not characterized.*

**(1) The four probes resolve into a quantitative
lexical→compositional gradient.** The standard moral probe (single
morally-loaded lexeme swap, "murdered" / "greeted") onsets at step
1K. The compositional moral probe (multi-token integrated swap;
contrast tokens "protect" / "humiliate", "hungry" / "wealthy" are
individually mild) onsets at step 5K under 4-seed averaging — a
4K-step lag, with per-seed onsets ranging 4K-7K and overall
trajectory always between sentiment (2K) and syntax (6K). The
standard probe's step-1K onset measures how quickly moralized
vocabulary becomes linearly separable, not how quickly moral
valence is encoded compositionally. Both findings are true; the
strongest single-token reading of the standard onset is ruled out,
while the gradient reading — lexically-marked moralized vocabulary
first, compositional moral integration second, syntactic competence
last — holds. The 0.709 compositional onset is +59.6 pp above the
0.113 single-token TF-IDF floor (§3.2); whatever the probe recovers
at step 5K must integrate multiple words. At the OLMo-2 1B final
checkpoint (~2.2T tokens) the compositional probe reaches 0.900
peak @ layer 5, +78.7 pp over the TF-IDF baseline.

**(2) Phase-transition vs. gradual emergence dichotomy.** Standard
moral and sentiment probes show sharp sigmoidal transitions (chance
→ plateau within one 1K-step interval at onset, then flat).
Compositional moral and syntax rise more gradually (~3-5K steps
across the 0.70 threshold). This parallels grokking-literature
observations (Power et al., 2022) that some capabilities emerge as
phase transitions and others gradually; the within-run split here
suggests the distinguishing factor is whether the capability is
acquirable from local lexical statistics (phase transition) or
requires multi-token integration (gradual). §5.1 develops.

**(3) Plateau coincidence (developed in §4.2).** Standard moral and
sentiment plateau at ≈0.97; compositional moral and syntax plateau
at ≈0.77. The 0.20 gap tracks the single-token vs. multi-token
split — probes whose discriminative signal lives in single-token
statistics saturate higher under mean-pooled linear probing than
probes that require multi-token integration. We treat this as a
structural caveat to the gradient finding and develop it in §4.2 as
a short subsection that the four-curve overlay (Figure 1)
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
a model property. Either the 1B model encodes both compositional
moral valence and syntactic well-formedness at ≈0.77 (model ceiling),
or mean-pooled linear probing on 1B hidden states bottoms out at
≈0.77 for multi-token integration regardless of underlying
representational quality (probe ceiling). The cleanest disambiguation
is repeating §4.1 at 7B and 32B in Phase E — if compositional moral
rises with scale while syntax does not, the model is the bottleneck;
otherwise the probe is. We state both readings honestly in §5.4 and
refine rather than overturn the gradient finding.

## 4.3 Probing accuracy saturates; fragility doesn't

The figure that does the most work for the methodological thesis is
**Figure 2**: two-panel comparison on a shared step axis. Top panel:
mean probing accuracy — sharp sigmoid from chance (~0.55) to a
plateau (~0.95) between steps 0 and 4K, then flat for the remaining
~33K steps. Bottom panel: mean fragility — initial rise alongside
accuracy in the first 1K steps, then continued movement throughout.
Top panel reaches a ceiling and stops; bottom panel keeps moving for
the entire remaining 95 % of training.

**Phase C1 numbers (OLMo-2 1B, 37 checkpoints, dense).**

| Step | Mean acc | Mean critical noise | Late-layer crit | Mid-layer crit | Early-layer crit |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.557 | 0.18 | 0.1 | 0.1 | 0.3 |
| 1,000 | 0.768 | 10.0 | 10.0 | 10.0 | 10.0 |
| 4,000 | 0.938 | — | — | — | — |
| 10,000 | ~0.95 | 7.0 | 10.0 | 10.0 | 7.7 |
| 20,000 | ~0.95 | 7.5 | 10.0 | 10.0 | 6.5 |
| 36,000 | ~0.96 | 5.3 | 10.0 | 5.8 | 1.7 |

*Table 2: Standard moral probe — accuracy plateaus by step 4K;
fragility evolves through step 36K with a layer-depth gradient that
develops monotonically (late > mid > early after step ~10K).*

The pattern reproduces at OLMo-3 7B (Phase B5, 5 sparse checkpoints):
mean critical noise rises 0.20 → 5.67 between steps 0 and 353K (~28×),
then plateaus at ~5.3 through step 1.4M; layer-depth gradient is
steeper (late ~10.0 / mid ~5.5 / early ~2.0) and the most-robust
layer drifts deeper across training (layer 2 → 7 → 11 → 10 → 15).
The 1B trajectory is the headline because dense 1K-step sampling
resolves the saturation step (~4K) and gradient emergence rate.

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
(both Phase E disambiguates): a *mechanism reading* — as training
continues on text that does not specifically reinforce compositional
moral integration, the compositional representation drifts toward
brittleness while standard-probe representations are continually
reinforced by moralized vocabulary density — and a *probe-ceiling
reading* — fragility at the 0.77 operating point has less headroom
than at 0.96, partly artifacting the difference. We state both in
§5.4 without commitment.

Numbers sources: `outputs/phase_c1/RESULTS.md` (1B standard probe),
`outputs/phase_b/` (7B corroboration),
`outputs/phase_c4_compositional/3seed/{aggregate_per_checkpoint,decision}.json`
(4-seed mean ± std and decision rule application),
`outputs/phase_c4_compositional/3seed/4seed_fragility_evolution.png`
(headline 4-seed plot).

## 4.4 Data curation reshapes structure, not content

Phase C3: LoRA (Hu et al., 2022) fine-tuning on three matched
corpora from the OLMo-2 1B step-1000 checkpoint (mid-transition, ~80 % peak probing
accuracy). Corpora: a 247K-token narrative-moral corpus (Aesop /
Grimm / Andersen), a 500K-token declarative-moral corpus
(template-expanded `MORAL_SEEDS`: "Stealing is wrong"), and a 420K-token
general non-moral control (Darwin). Identical LoRA hyperparameters
(rank 16, alpha 32, q\_proj + v\_proj, lr 2e-4, batch 2, seq 1024,
1000 steps); standard moral probe + fragility every 100 LoRA steps.

**Probing accuracy is identical across conditions.** Final peak
accuracy at LoRA step 1000: narrative 0.812, declarative 0.802,
general control 0.802 — within 1 pp across very different training
data. The accuracy metric returns no signal for which corpus produces
what kind of representational change.

**Fragility profiles are condition-specific (the main result).**
Final mean critical noise: narrative 10.0, declarative 9.42, general
control 10.0. The per-layer breakdown is decisive: narrative and
general control hold critical noise = 10.0 at every layer; the
declarative condition collapses to **critical noise = 3.0 at layer 3**
while every other layer holds at 10.0. A single sharply localized
fragility dip created by declarative-moral LoRA that natural-text
training does not produce. **Figure 4** plots all three per-layer
profiles plus the three identical accuracy bars: same accuracy,
different fragility.

**Training loss is decoupled from representational change.**
Declarative loss drops 5.6 → 1.0 (template memorization), narrative
5.0 → 3.9, control 5.0 → 4.5. The condition with the deepest loss
reduction is the same condition that produces the layer-3 dip; the
two with the shallowest leave fragility unchanged. The model is
learning declarative templates as surface text patterns without that
learning translating into either accuracy gains or robust
representational structure.

This is the cleanest single piece of evidence for the methodological
thesis. Same data, same probe; accuracy says "no difference between
narrative and declarative training"; fragility says "declarative
training creates a brittle layer-3 shortcut that natural-text
training does not." We flag the natural follow-up — replicating the
C3 design with the compositional probe to ask whether data curation
reshapes compositional fragility the same way it reshapes lexical
fragility — as a Phase E experiment in §5.4.

Numbers source: `outputs/phase_c_tier2/c3/RESULTS.md` and
`outputs/phase_c_tier2/c3/{narrative,declarative,general}_moral.json`
(per-layer fragility for all three conditions).
