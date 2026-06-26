# Direction 1 — Held-out paraphrase protocol (G2 instrument)

**Date:** 2026-06-26 · **Scope:** eval-source pairs ONLY · **Frozen before any
paraphrasing.** Thresholds below are pre-registered; changing one is a dated amendment.

Companion: `PREREGISTRATION.md` (G2 definition), `partition_manifest.json` (the
eval-source pools this protocol draws from).

## Why this exists

G2 asks: does `V_moral` read *moral structure* or *memorized benchmark surface text*?
All three primary sources are 2020–2025 and near-certain to be in OLMo-3's pretraining.
The eval-source paraphrase set is the only instrument that separates the two readings:
for every eval-source moral judgment we build a **1:1 paraphrase** that preserves the
moral content and judgment but breaks the surface form. G2 compares `acc_surf`
(original eval surface) against `acc_para` (paraphrases). The protocol must guarantee
each paraphrase is:

- **genuinely surface-divergent** — else a memorizing probe keeps its verbatim foothold,
  `acc_para` stays inflated, and G2 falsely passes; and
- **judgment-faithful** — else the held-out labels are corrupted and G2 measures noise
  (a paraphrase that flips or muddies the morality is *worse than no paraphrase*).

Paraphrases are built **only** from the eval-source pools. Train-source pairs are never
paraphrased.

## C1 — Surface-divergence floor (mechanical, auto-enforced)

Removes verbatim / near-verbatim memorization footholds. A paraphrase must satisfy **all**:

| Check | Threshold | Kills |
|---|---|---|
| Longest shared contiguous word n-gram | **no shared n-gram of length ≥ 5** | verbatim span recall |
| Content-word Jaccard (stopword-stripped, lemmatized) | **≤ 0.50** | near-verbatim one-word swaps |
| ROUGE-L F1 (original vs paraphrase) | **≤ 0.60** | high-overlap reorderings |

Failing any → auto-reject and regenerate. **Grounding:** Paper 1's bag-of-words probe
collapsed to 0.598 on leave-construction-out transfer once surface was held out. This
floor makes a paraphrase as un-memorizable as a held-out construction, so a probe that
scored high on the original via recall drops on the paraphrase — which is exactly the
signal G2 needs.

## C2 — Judgment & meaning preservation (LLM-judge + mechanical backstop)

| Check | Method | Reject if |
|---|---|---|
| **Judgment identity** (primary) | LLM-judge (Claude Sonnet 4.6) is shown source + paraphrase and asked whether the moral judgment is unchanged | judgment differs from the source label (MORABLES: same correct moral; Moral Stories: same moral-vs-immoral polarity; ETHICS: same acceptable/unacceptable) |
| **No new moral content** | same judge, §1.5 spirit | paraphrase introduces a foundation/charge the original lacked |
| **Meaning floor** (mechanical backstop) | sentence-embedding cosine (e.g. `all-MiniLM-L6-v2`), original vs paraphrase | cosine **< 0.70** (C1 divergence destroyed the meaning) |

## C3 — Rewrite-aggressiveness target

Aim for **maximal surface change consistent with C2.** Operational target band:
content-word Jaccard in **[0.20, 0.50]** with **zero shared 5-grams**, while embedding
cosine **≥ 0.70**. Below the band (lazy one-word swaps) is rejected by C1; above it
(meaning drift) is rejected by C2. The window between them is the protocol's target.

## Generation procedure

1. **Generator:** Claude (Sonnet 4.6 or stronger). Prompt: rewrite preserving the moral
   judgment, change surface maximally, keep length within the §4.1 ratio (1.4:1) and the
   same register.
2. **Up to 3 attempts per item;** keep the first that clears C1 **and** C2.
3. If no attempt clears in 3 tries → **flag for manual review, do not silently drop**
   (silent drops unbalance the eval set per source/register and would reintroduce the
   source-shift confound the partition was built to prevent).

## Exit criterion (Phase 1)

Every eval paraphrase clears C1 (mechanical divergence floor) and C2 (judgment
preservation). The run reports, per source and per register: pass rate, the realized
divergence distribution (longest shared n-gram, Jaccard, ROUGE-L), and the embedding-cosine
distribution — so the held-out set's surface-divergence is auditable, not asserted.

## What this buys G2

C1 collapses the partial-memorization regime toward structural reading: with no verbatim
foothold, a probe whose original-surface accuracy came from recall drops on the
paraphrases, widening `acc_surf − acc_para` precisely when memorization drove the original
score. That is what turns the pre-registered **0.10** G2 gap band into a clean test rather
than a memorization-contaminated one.

### Amendments
*(none)*
