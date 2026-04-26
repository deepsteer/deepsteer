# 3. Methodology

We measure moralized representational dynamics during pre-training using
linear probing classifiers trained on matched minimal-pair datasets.
The four datasets share construction philosophy but vary in what
discriminative signal each pair allows: three (moral, sentiment,
syntax) hold the syntactic skeleton constant and swap a single token,
and the fourth — our methodological contribution — holds the action
verb constant and varies only individually-mild tokens whose moral
status flips in context. We apply two probe families to all four
datasets: a standard layer-wise linear probe
(`LayerWiseMoralProbe`) and a fragility test
(`MoralFragilityTest`) that injects Gaussian noise into cached
activations and records the noise level at which probe accuracy
collapses. All experiments target two OLMo open-weight base models
(OLMo-2 1B and OLMo-3 7B) on a single MacBook Pro M4 Pro with MPS
acceleration. Code, datasets, and per-checkpoint outputs are released
with the paper.

## 3.1 Standard minimal-pair datasets

The three single-token-swap datasets mirror established
minimal-pair-probing practice (Belinkov, 2022) and isolate the lexical
property of interest by holding all other surface features constant.

**Moral / neutral pairs (240).** Each pair holds the syntactic
skeleton constant and swaps a single morally-loaded lexeme for a
neutral one (e.g. "She murdered the woman" / "She greeted the
woman"). Pairs are balanced across the six Moral Foundations Theory
(Haidt, 2012; Graham et al., 2013) categories at 40 pairs per
foundation. Validation gates: word-count ratio ≤ 1.5 between paired
sentences; deterministic test split at seed 42 with 20% held out.

**Sentiment pairs (210).** Each pair swaps a single valenced adjective
between positive and negative sentiment (e.g. "The restaurant served
excellent food tonight" / "...terrible food..."). Pairs are
distributed across ten domains (food, work, weather, products,
relationships, travel, health, entertainment, education, home) at
~21 per domain. Same length-ratio gate and seed-42 split.

**Syntax pairs (210).** Each pair swaps a single token to flip
grammaticality while preserving lexical content (e.g. "She handed the
finished report to her manager today" / "She handed finished the
report to her manager today"). Construction targets local
well-formedness violations — word-order swaps, agreement violations,
auxiliary misplacements — that mean-pooled linear probing can in
principle detect. Same length-ratio gate and seed-42 split.

These three datasets establish the baseline emergence ordering
reported in §4.1: moral onset at step 1K, sentiment at step 2K, syntax
at step 6K (Phase C2 results, `outputs/phase_c2/RESULTS.md`).

## 3.2 Compositional moral probing dataset

The standard moral probe's pairs differ in a single morally-loaded
lexeme, which a linear probe can separate from single-token
distributional statistics alone. The standard probe's step-1K onset
therefore measures *how quickly moralized vocabulary becomes
linearly separable from neutral vocabulary*, not *how quickly moral
valence is encoded compositionally*. To bound this lexical-accessibility
framing empirically, we construct a compositional moral minimal-pair
dataset whose pairs require multi-word integration to determine moral
valence. This dataset is a primary methodological contribution of the
paper.

**Construction.** Each pair holds the action verb constant and varies
1–2 individually-mild tokens whose moral status only resolves in the
surrounding action context. Four 50-pair categories (200 pairs total),
all hand-curated:

- *action_motive:* same action verb, motive determines valence —
  "She lied to her parents to **protect** her younger brother today" /
  "...to **humiliate**..."
- *action_target:* same action, target descriptor determines moral
  relevance — "He gave the last loaf of bread to the **hungry** stranger
  at the door" / "...the **wealthy** stranger..."
- *action_consequence:* same action, consequence framing determines
  valence — "He kept the secret about the surprise to keep his sister
  **safe** today" / "...to keep his sister **hurt** today"
- *role_reversal:* same components, role / target / context
  determines valence — "The judge accepted the gift to free the
  **innocent** prisoner from prison" / "...the **guilty**
  prisoner..."

Contrast tokens are individually mild: `protect`, `humiliate`,
`hungry`, `wealthy`, `safe`, `hidden`, `innocent`, `guilty`. None
appear in the strong-valence blocklist of 47 tokens (`murder`,
`torture`, `stole`, `rape`, `assault`, etc.) we use as a construction
gate. Each token is unmarked or only weakly valenced in isolation;
the moral signal in each pair lives in the integration of the
contrast token with the surrounding action context.

**Validation gates.** All 200 pairs pass five gates simultaneously
(`validate_compositional_dataset`):

1. *Length difference:* per-pair absolute difference ≤ 2 alphabetic
   tokens.
2. *Length band:* both halves in the 8-20-word band.
3. *Content-word overlap:* per-pair ≥ 0.60 with stopwords removed
   (`|shared| / max(|moral|, |immoral|)`). This metric matches
   `deepsteer.datasets.validation.validate_pairs` so the 0.60
   threshold here is directly comparable to the standard moral probe
   gate.
4. *Strong-valence blocklist:* zero tokens from the 47-word blocklist
   on either side.
5. *No exact duplicates* of either side across the 200 pairs.

**Compositional gate (the operational check).** A TF-IDF + logistic
regression classifier on bag-of-words unigram features (5-fold
stratified CV) achieves 0.113 mean accuracy overall and 0.14-0.20
per-category — well below the 0.65 design ceiling. Single-word
features cannot separate the classes; anything the linear probe
achieves on hidden states above this floor must integrate multiple
words. This is the operational definition of "compositional" in our
experiments. (The construction iterated through ~5 rewriting passes
to satisfy the 0.60 content-overlap gate alongside the multi-word
contrast requirement — the two constraints are in genuine tension;
see Appendix D.)

**Train / test split.** 160 / 40, stratified by category (40 train +
10 test per category), seed = 42. The dataset and validation gates
are deterministic, API-free, and included in the toolkit at
`deepsteer/datasets/compositional_moral_pairs.py`.

## 3.3 Linear probing

We use a single probing methodology across all four datasets, varying
only the input data. This is methodologically important: when we
report that the compositional probe lags the standard moral probe by
3K training steps (§4.1), the only experimental variable separating
them is the dataset, not the probe.

For each dataset and each transformer layer ℓ, we collect mean-pooled
hidden states from a single forward pass per text:

```
h_ℓ(x) = mean_t ( H_ℓ[t] )         where H_ℓ ∈ ℝ^(seq_len × hidden_dim)
```

We train a binary linear classifier `y_ℓ = σ(W_ℓ · h_ℓ + b_ℓ)` to
distinguish the two sides of each minimal pair (1 = moral / positive
sentiment / grammatical / compositional-moral; 0 = neutral / negative
sentiment / ungrammatical / compositional-immoral). Optimization:
binary cross-entropy with logits, Adam (lr = 1e-2), 50 epochs, no
weight decay, no early stopping. Probes run in fp32 on the activation
cache; the underlying model is fp16 on MPS. We report per-layer
test-set accuracy and four summary statistics:

- *onset layer:* first layer with accuracy ≥ 0.6
- *peak layer:* layer with maximum accuracy
- *encoding depth:* `onset_layer / n_layers` (lower = encoded earlier
  in the network)
- *encoding breadth:* fraction of layers with accuracy ≥ 0.6

For the compositional probe we additionally track the *content-only
TF-IDF baseline* per checkpoint as the bag-of-words floor that hidden-
state probing must exceed by ≥10 pp to count as a meaningful signal
(the validation gate from §3.2). Implementation:
`LayerWiseMoralProbe` for the standard moral and the compositional
probe (the latter via a `CompositionalMoralProbe` subclass that
overrides only the dataset path), `GeneralLinearProbe` for sentiment
and syntax. All four use the identical training loop.

Probing accuracy at the final OLMo-2 1B checkpoint (~2.2T tokens):
the compositional probe reaches 0.900 peak accuracy at layer 5 — a
+78.7 pp improvement over the 0.113 TF-IDF baseline, passing both
validation gates (≥+10 pp delta and ≥0.65 absolute) by wide margins.

## 3.4 `MoralFragilityTest`

The fragility test asks how robust a trained linear probe is to
activation noise. The procedure: (1) train per-layer linear probes
on clean training-set activations as in §3.3; (2) for each layer ℓ,
add Gaussian noise N(0, σ²) to the cached test-set activations and
re-evaluate the trained probe; (3) record the smallest σ in a
logarithmic sweep at which probe accuracy drops below the *fragility
threshold* (0.6 by default — chance + 0.1 on a binary task). Default
sweep: σ ∈ {0.1, 0.3, 1.0, 3.0, 10.0}. The smallest σ at which
accuracy crosses below threshold is the layer's *critical noise*; if
no noise level in the sweep brings the probe below threshold, critical
noise is reported as the maximum sweep value.

Per-layer critical noise across all transformer layers gives the
*fragility profile*; its mean across layers is `mean_critical_noise`
as a scalar summary. The fragility threshold and noise sweep range
are hyperparameters: we use 0.6 throughout and report sweep range with
each result. The same `MoralFragilityTest` infrastructure runs against
both the standard moral dataset (Phase C1, `outputs/phase_c1/`) and
the compositional dataset (Phase C4, `outputs/phase_c4_compositional/`)
— methodology generality is established by reuse, not reimplementation.

Critically, fragility is a representation-geometry property that
survives accuracy saturation. A probe with 95% accuracy and a probe
with 100% accuracy are indistinguishable on the accuracy metric; the
same two probes can have very different fragility (the 100%-accuracy
probe might collapse below threshold at σ = 0.1 while the 95%-accuracy
probe holds at σ = 10). §4.3 demonstrates this empirically.

## 3.5 Target models and checkpoints

Three OLMo (Groeneveld et al., 2024) base models. **OLMo-2 1B
early-training** (`allenai/OLMo-2-0425-1B-early-training`), 37
checkpoints at 1K-step intervals from step 0 to step 36K (~76B
tokens) — the primary data source for §4.1 onsets and §4.3
fragility. **OLMo-3 7B stage 1** (`allenai/OLMo-3-7B`), 20
checkpoints through ~1.4M steps (~10T tokens) — §4.3 7B
corroboration and Appendix B causal tracing. **OLMo-2 1B final**
(`allenai/OLMo-2-0425-1B`, ~2.2T tokens), used only for the
compositional probe validation gate (§3.2). All on a single MacBook
Pro M4 Pro / 24 GB unified / MPS, fp16, with activation collection
via PyTorch forward hooks on the underlying HuggingFace `model.layers[ℓ]`
modules; ~6 hours of MPS time across the full §4 experimental record.

## 3.6 Validity controls

Three controls standard for linear-probing studies — leave-lexeme-out
splits, paraphrase transfer, adversarial lexical swap — are reported
in Appendix C for parity with the persona probe in companion work.
The compositional probe (§3.2) addresses the strongest version of
"your probe is just reading vocabulary" by construction (TF-IDF
baseline 0.113 ≪ 0.65) and is a strictly stronger ablation than
those three controls combined for the relevant question.
