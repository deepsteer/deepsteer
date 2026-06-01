# Appendix A. Probing dataset construction

The 240-pair moral probing dataset used throughout this paper is
a deterministic subsample of a 1,200-pair dataset constructed per
published quality guidelines (`DATASET_GUIDELINES.md`) with
LLM-assisted filtering. The pipeline and full dataset are described
in companion work \citep{reblitzrichardson2026fragility}; we summarize
the construction here for self-containment.

## A.1 Seed extraction and pair generation

The 1,200-pair parent dataset covers six Moral Foundations Theory
foundations \citep{haidt2012righteous,graham2013mft}: care/harm,
fairness/cheating, loyalty/betrayal, authority/subversion,
sanctity/degradation, and liberty/oppression (200 pairs per
foundation). Each pair consists of a moral sentence and a matched
neutral sentence that preserves syntactic structure and topic domain
while removing moral content. For example:

| Foundation | Moral | Neutral |
|---|---|---|
| care/harm | "Offering shelter to someone stranded in a storm, everyone helped." | "Offering directions to someone lost in a new city, everyone helped." |
| fairness | "The manager promoted the most qualified candidate despite personal ties." | "The manager promoted the candidate who had applied first." |
| loyalty | "She reported her company's illegal dumping to protect the community." | "She reported her company's quarterly earnings to the board." |

Neutral sentences are generated with LLM assistance and filtered
for naturalness and moral neutrality of the neutral side.

## A.2 Automated validation gates

Pairs pass length-ratio gates ($\leq$ 1.5 ratio), keyword filtering
(no explicit moral keywords in neutral sentences), and
deduplication. The 1,200-pair dataset is released alongside the
companion paper.

## A.3 Subsampling

240 pairs (40 per foundation) are subsampled from the 1,200-pair
parent with a deterministic seed (42). The subsample is split
80/20 into 192 training pairs (384 texts) and 48 test pairs (96
texts), with foundation balance preserved across splits.

## A.4 Dataset identity across experiments

All experiments in this paper use the identical 240-pair subsample.
The dense-vs-MoE comparison (\S 4.1), per-expert probing (\S 4.2),
routing analysis (\S 4.3), and output scale comparison (\S 4.4) all
process the same input texts, ensuring any observed differences are
architectural rather than data-driven.
