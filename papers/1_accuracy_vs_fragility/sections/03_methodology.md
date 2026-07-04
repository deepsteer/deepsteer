# 3. Methodology

We apply linear probing classifiers on four matched minimal-pair
datasets to all 37 OLMo-2 1B early-training checkpoints, 20 OLMo-3
7B stage-1 checkpoints, and the OLMo-2 1B final checkpoint. Two
probe families: `LayerWiseMoralProbe` (per-layer accuracy) and
`MoralFragilityTest` (per-layer noise robustness). All experiments
run on a single MacBook Pro M4 Pro / MPS; code, datasets, and
per-checkpoint outputs are released with the paper.

## 3.1 Standard minimal-pair datasets

Three single-token-swap datasets mirror established minimal-pair
probing practice (Belinkov, 2022) by holding the syntactic skeleton
constant and swapping a single token: **moral / neutral** (240 pairs,
40 per Moral Foundations Theory category (care/harm,
fairness/cheating, loyalty/betrayal, authority/subversion,
sanctity/degradation, liberty/oppression; Haidt, 2012; Graham et
al., 2013; e.g. "She betrayed the woman" / "She greeted the
woman"). The moral pairs are drawn from a 1,200-pair dataset
constructed per published quality guidelines with LLM-assisted
filtering for naturalness and moral neutrality of neutral-side
sentences (see `DATASET_GUIDELINES.md`); 240 pairs (40 per
foundation) are subsampled with a deterministic seed.
**Sentiment** (210 pairs across ten domains; positive /
negative adjective swap, e.g. "...excellent food..." / "...terrible
food..."); **syntax** (210 pairs targeting local grammaticality
violations (word-order swaps, agreement, auxiliary misplacement),
e.g. "She handed the finished report..." / "She handed finished the
report..."). All three use a length-ratio gate ≤ 1.5 and a
deterministic seed-42 train / test split. They establish the
baseline emergence ordering in §4.1.

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

- *action_motive:* same action verb, motive determines valence.
  "She lied to her parents to **protect** her younger brother today" /
  "...to **humiliate**..."
- *action_target:* same action, target descriptor determines moral
  relevance. "He gave the last loaf of bread to the **hungry** stranger
  at the door" / "...the **wealthy** stranger..."
- *action_consequence:* same action, consequence framing determines
  valence. "He kept the secret about the surprise to keep his sister
  **safe** today" / "...to keep his sister **hurt** today"
- *role_reversal:* same components, role / target / context
  determines valence. "The judge accepted the gift to free the
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

**Operationalizing "compositional": transfer and lift, not a
bag-of-words gate.** A minimal pair that flips its label by swapping a
single contrast token makes that token a lexical cue *by construction*:
a bag-of-words classifier separates the seen pairs no matter how mild
the token is in isolation. "Can unigrams separate the data" is
therefore the wrong question (they can: a pair-disjoint,
orientation-invariant unigram TF-IDF classifier reaches 0.63 overall
and 0.57-0.64 per construction, where pair-disjoint means `GroupKFold`
keyed on pair so neither half of a pair leaks across the split, and
orientation-invariant means scoring `max(acc, 1 - acc)`). The right
question is whether the hidden-state representation encodes moral
valence in a way that generalizes *beyond the specific contrast
tokens*. We operationalize compositional encoding through two tests,
both measured against this unigram TF-IDF baseline (the *lexical
floor*):

1. *Leave-construction-out transfer.* Train the probe on three of the
   four construction categories and test on the held-out fourth, which
   shares almost no contrast tokens with the training categories. A
   probe relying on token identity cannot transfer; a
   construction-general representation can.
2. *Lift over the lexical floor.* Within each construction, compare
   hidden-state decodability (pair-disjoint CV) to that construction's
   unigram lexical floor. The gap is signal the bag-of-words baseline
   cannot reach.

§4.1 reports both, at the final checkpoint and across training. (The
construction iterated through ~5 rewriting passes to satisfy the 0.60
content-overlap gate alongside the multi-word contrast requirement;
the two constraints are in genuine tension; see Appendix D.)

**Train / test split.** 160 / 40, stratified by category (40 train +
10 test per category), seed = 42. The dataset and validation gates
are deterministic, API-free, and included in the toolkit at
`deepsteer/datasets/compositional_moral_pairs.py`.

## 3.3 Linear probing

Identical probing methodology across all four datasets: when we
report a 4K-step gap between standard and compositional moral
onsets (§4.1), the only experimental variable is the dataset.

For each transformer layer ℓ we mean-pool hidden states across the
sequence dimension and train a binary linear classifier
(`nn.Linear(hidden_dim, 1)`) to distinguish the two sides of each
minimal pair, with BCE-with-logits loss, Adam (lr = 1e-2), 50
epochs, no weight decay or early stopping; probes run in fp32 on
fp16 activation caches collected via PyTorch forward hooks on the
HuggingFace `model.layers[ℓ]` modules. We report per-layer test-set
accuracy and four summary statistics: onset layer (first ≥ 0.6),
peak layer, encoding depth (`onset_layer / n_layers`), and encoding
breadth (fraction of layers ≥ 0.6). The compositional probe
additionally tracks a TF-IDF content-only floor per checkpoint that
hidden-state probing must beat by ≥10 pp. Implementation:
`LayerWiseMoralProbe` for standard and compositional moral (the
latter a subclass that overrides only the dataset path),
`GeneralLinearProbe` for sentiment and syntax (same training loop).

## 3.4 Fragility testing

Train per-layer linear probes on clean activations as in §3.3; for
each layer ℓ, add Gaussian noise N(0, σ²) to the cached test-set
activations and re-evaluate the trained probe across a logarithmic
sweep σ ∈ {0.1, 0.3, 1.0, 3.0, 10.0}. The smallest σ at which
accuracy drops below the fragility threshold (0.6, i.e. chance + 0.1
on a binary task) is the layer's *critical noise*; if no σ in the
sweep brings the probe below threshold, critical noise is reported
as the maximum (10.0), and never-fragile layers are censored at that
maximum (not dropped) when averaging, so the mean over all layers is
`mean_critical_noise` as a scalar summary. The
same `MoralFragilityTest` runs against both the standard and
compositional datasets; methodology generality is established by
reuse, not reimplementation.

Two properties of this estimator bound its interpretation. First,
σ* is **floor-dominated when baseline accuracy is below τ=0.6**: if
the clean probe already sits below threshold (step 0, and the
compositional probe for its first ~2K steps), σ*=0.1 by construction,
so part of the early "fragility rise" tracks accuracy crossing
threshold rather than a change in noise tolerance. Second, σ* is
**right-censored at the grid maximum**: layers pinned at 10.0 for the
whole trajectory could have larger true critical noise. Where
censoring is heavy (late layers throughout, and per-foundation 7B
fragility), we lift the cap with an **extended grid**
σ ∈ {0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0}.

## 3.5 Target models and checkpoints

Three OLMo (Groeneveld et al., 2024; Team OLMo, 2025a; Team OLMo, 2025b) base models.
**OLMo-2 1B early-training** (`allenai/OLMo-2-0425-1B-early-training`; Team OLMo, 2025a), 37
checkpoints at 1K-step intervals from step 0 to step 36K (~76B
tokens), the primary data source for §4.1 onsets and §4.2
fragility. **OLMo-3 7B stage 1** (`allenai/Olmo-3-1025-7B`; Team OLMo, 2025b):
probing on 20 stage-1 checkpoints through ~1.4M steps (~10T tokens), and
fragility on 5 of those checkpoints (steps 0, 353K, 705K, 1,059K, 1,413,814);
§4.2 7B corroboration and Appendix B causal tracing. **OLMo-2 1B final**
(`allenai/OLMo-2-0425-1B`, ~2.2T tokens), used only for the
compositional probe validation gate (§3.2). All loaded in fp16 on MPS;
~6 hours of MPS time across the full §4 experimental record.

## 3.6 Validity controls

Three controls standard for linear-probing studies (leave-lexeme-out
splits, paraphrase transfer, adversarial lexical swap) are reported
in Appendix C. The compositional probe (§3.2) addresses the strongest
version of "your probe is just reading vocabulary": its compositional
encoding is established by leave-construction-out transfer and lift over
the unigram lexical floor (§3.2, §4.1), not by the lexical floor alone,
and it is a strictly stronger ablation than those three controls
combined for the relevant question.
