# Appendix A. Probing dataset construction

The 240-pair moral probing dataset used throughout this paper is
constructed via the automated pipeline described in companion work
\citep{reblitzrichardson2026fragility}. We summarize the pipeline here for
self-containment.

## A.1 Seed extraction

300 moral seed sentences are extracted from MoralBench
\citep{yu2024moralbench}, covering six Moral Foundations Theory foundations: care/harm,
fairness/cheating, loyalty/betrayal, authority/subversion,
sanctity/degradation, and liberty/oppression (50 seeds per
foundation). Seeds are declarative moral statements derived from
MoralBench's scenario-based prompts.

## A.2 Minimal pair generation

Each moral seed is paired with a matched neutral sentence that
preserves syntactic structure, length, and topic domain while
removing moral content. For example:

| Foundation | Moral | Neutral |
|---|---|---|
| care/harm | "The nurse stayed overtime to comfort the dying patient." | "The nurse stayed overtime to complete the shift paperwork." |
| fairness | "The manager promoted the most qualified candidate despite personal ties." | "The manager promoted the candidate who had applied first." |
| loyalty | "She reported her company's illegal dumping to protect the community." | "She reported her company's quarterly earnings to the board." |

## A.3 Automated validation gates

Four rejection gates filter candidate pairs:

1. **Length matching.** Pairs where the moral and neutral sentences
   differ by more than 20% in token count are rejected (0 rejected
   in the final dataset).
2. **Embedding overlap.** Pairs with TF-IDF cosine similarity above
   0.85 are rejected to prevent content leakage (33 rejected).
3. **Keyword filtering.** Pairs containing explicit moral keywords
   (e.g., "ethical", "immoral", "should") in the neutral sentence
   are rejected (0 rejected).
4. **Deduplication.** Near-duplicate pairs are collapsed (0 rejected).

267 of 300 candidate pairs pass all gates.

## A.4 Foundation balancing

The 267 validated pairs are balanced to exactly 40 pairs per
foundation (240 total), downsampling over-represented foundations.
The dataset is split 80/20 into 192 training pairs (384 texts) and
48 test pairs (96 texts), with foundation balance preserved across
splits.

## A.5 Dataset identity across experiments

All experiments in this paper use the identical 240-pair dataset.
The dense-vs-MoE comparison (\S 4.1), per-expert probing (\S 4.2),
routing analysis (\S 4.3), and output scale comparison (\S 4.4) all
process the same input texts, ensuring any observed differences are
architectural rather than data-driven.
