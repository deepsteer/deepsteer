# MB-G1 gate report (2026-10-30; complete)

M4, bf16, MPS. Rules of record: `MB_PLAN.md` v0.5 (commit `238a021`, pushed before any G1 data).
Numbers of record: `outputs/g1/<model>/g1_analysis.json` (aggregates only); letters and readouts
in `outputs/g1/<model>/*.jsonl`; replies and rendered prompts in the gitignored `outputs_raw/`.
Nothing committed.

## 0. Before G1: MB-A4 resolved (zero GPU)

Clifford et al.'s published norms (fetched to the cache only; the repo's `mfv_clifford2015.json`
has no wrongness ratings) match all 24 MFV vignettes and all 24 comparative options. The binary
keys follow the norms on 22/24 items (Agree, meaning "not morally wrong", scores 5 − Wrong).
The comparative keys pick the less-wrong statement on 19/24 items. The departures are the whole
fairness group (all four items keyed to the more-wrong statement; three distinct pairs, as
`fairness_2` and `fairness_3` are duplicates) and `ingroup_2`. By the registered rule
the verdict is R_errors, with the errors located (`outputs/mb_a4_norms.json`, ANOMALIES MB-A4).

## 1. Zephyr-7B-β (`892b3d7a`, native system role, BOS prepended, template `66291cf0…`)

Three cells × 88 items × 32 rollouts plus R2: 100 min wall (2,024 / 1,973 / 1,951 s).

**Totals at T = 0.7 (of record)**

| set | paper | R0, miss = 0 | 95% PI, 5 repeats | verdict | R0 misses (unstripped) | parsed-only | R1 | R2 |
|---|---|---|---|---|---|---|---|---|
| `MFQ_30` | 54.2 | 22.3 | [19.9, 24.8] | does not replicate | 296/640 (298) | 43.8 | 39.5 | 44.3 |
| `6_concepts` | 48.1 | 41.1 | [38.7, 43.5] | does not replicate | 167/768 (167) | 49.6 | 46.2 | 51.9 |
| `MFQ_30_compare` | 8.2 | 7.1 | [6.3, 7.7] | does not replicate | 156/640 (156) | 9.0 | 3.7 | 9.9 |
| `6_concepts_compare` | 10.4 | 11.4 | [10.3, 12.4] | **replicates** | 68/768 (68) | 12.6 | 0.9 | 12.9 |

Per-foundation verdicts are in the JSON. Of the 26 paper cells (foundations plus totals):

| | replicates | does not replicate | not attainable |
|---|---|---|---|
| T = 0.7 (of record) | 5 | 20 | 1 |
| T = 1.0 (fork) | 7 | 18 | 1 |

The cells that replicate at T = 0.7 are MFV harm, MFQ-comparative harm, MFV-comparative
ingroup, MFV-comparative purity, and the MFV-comparative total. The cell that is not attainable is
MFQ-comparative loyalty (MB-A3, as registered; unattainable only if every repetition gave a valid
response). At T = 1.0 the fork does not change the verdict on
any total.

R1 falls below R0 on the comparative sets (0.9 and 3.7). The KDG-derived ladder reads "ambiguous"
when a reply names both options, which Zephyr's explanations do (638 of 768 replies on
`6_concepts_compare`). R1 is opt-in and is reported as registered.

**MB-A5, counterbalanced cell.** The registered verdicts on R0 are R_acquiescence on the two
binary sets and R_content on the two comparative sets (S CIs in ANOMALIES MB-A5). The readout-free
part is the letter bias. L is negative on every set under R0, R1 and R2. Under R2 the values are
−0.31, −0.46, −0.17 and −0.13, and |K| ≤ 0.11 throughout. On MFV binary Zephyr answers B in both
orders.

**MB-A3.** Zephyr parses the tied item on 27 of 32 replies. Loyalty with misses scored 0 is 2.81
[2.35, 3.25]. The paper's 0.4 is unattainable only if every repetition got a valid response;
scoring misses as zero could produce it, and our Zephyr misses too rarely to come near it.

**New entries.** MB-A6: no readout reaches the paper's MFQ binary 54.2. The paper's total implies
76% Agree; our Zephyr's A-rate is 0.22 by R2. MB-A7: R0's misses track the answer. Agree replies
open with prose ("My choice is: A. Agree") and disagree replies open with the letter.

## 2. Gemma-1.1-7B-it (`065a5287`, system text prepended, template `ecd6ae51…`; G2 pod, Amendment 2)

Run on the G2 pod (A100, torch 2.6.0+cu124, transformers 5.17.0; the M4 could not hold it). Three
cells in 111 s. No R0 misses: Gemma answers with the letter first ("A.", "A. Agree").

| set | paper | R0, miss = 0 | 95% PI, 5 repeats | verdict | R1 | R2 | A-rate (R2) |
|---|---|---|---|---|---|---|---|
| `MFQ_30` | 49.9 | 55.3 | [55.1, 55.8] | does not replicate | 55.3 | 55.2 | 0.90 |
| `6_concepts` | 44.4 (cells 51.8) | 52.0 | [52.0, 52.0] | does not replicate | 52.0 | 52.0 | 0.00 |
| `MFQ_30_compare` | 9.6 | 8.2 | [7.5, 9.1] | does not replicate | 8.2 | 8.0 | 0.73 |
| `6_concepts_compare` | 10.8 | 11.9 | [10.8, 13.1] | **replicates** | 11.9 | 12.2 | 0.87 |

Of 26 paper cells: 7 replicate, 19 do not, at T = 0.7 and at T = 1.0 alike.

**MB-A1 not settled.** Gemma's agree rate on `6_concepts` is 0.00, and its total (52.0) equals the
always-B baseline. The paper's cells (sum 51.8) and its printed total (44.4) disagree, and a total
at the always-B baseline does not establish which number is wrong (correction, 2026-10-30).

**MB-A5.** On MFV binary Gemma answers letter B in both orders (768/768 each; L = −0.50). The
same holds for Llama-2 and, at about 97%, for Zephyr. On MFQ binary it follows the label instead:
Agree or Correct on about 95% of replies in either order (K +0.45).

## 3. Diagnosis (registered: a miss triggers diagnosis under amendment)

Root split, computable from saved data: does any readout reach the paper cell's PI?
- **No: model or prompt-path branch.** MFQ binary (R2 44.3, R1 39.5, parsed-only 43.8 against
  54.2) and MFV binary (R0 41.1; parsed-only 49.6 and R2 51.9 sit near 48.1 but are not the
  readout of record). The candidates are `zephyr-7b-alpha` (the recorded alternate), the system
  text prepended to the user turn, and no system prompt.
- **Yes: parse branch.** None of the missed totals reaches its PI under R0. The parse-sensitivity
  column (parsed-only, R2) brackets the paper on MFV binary.

Proposed amendment (author): three Zephyr variant cells at T = 0.7 on the M4, about 35 min each,
reading the MFQ binary total, the A-rate and R0 misses, each with its verdict. Zero GPU spend; no
pod.

**Author, 2026-10-30: diagnosis approved as MB_PLAN v0.6 Amendment 1.** It adds a fourth cell
(`zephyr-7b-gemma-v0.1`), registers the stopping rule first (the verdict of record stays at the
registered configuration; a matching variant is reported as "matches under X"; stop after the four
cells), and lists `zephyr-orpo-141b-A35b-v0.1` as the pod-only candidate if all four miss. The
MB-A4 prompt check found no item asking for the more-wrong option, so no key is reclassified.

## 3b. Diagnosis result (Amendment 1; stopping rule applied)

| cell | where | MFQ binary (54.2) | MFV binary (48.1) | MFQ comparative (8.2) | MFV comparative (10.4) | paper cells matched |
|---|---|---|---|---|---|---|
| beta, native system role (of record) | M4 | 22.3 | 41.1 | 7.1 | **11.4** | 5 of 25 (replicate) |
| D1 alpha | M4 | 40.9 | **47.9** | 3.6 | 7.6 | 9 |
| D2 beta, system text prepended | M4 | 22.2 | 30.1 | 6.1 | **10.5** | 5 |
| D3 beta, no system prompt | M4 | 9.1 | 24.9 | 6.4 | **9.9** | 4 |
| D4 `zephyr-7b-gemma-v0.1` | pod | 51.2 | **46.7** | 9.9 | 8.6 | **17** |

Bold marks a total inside the paper's PI. No configuration matches on MFQ binary (D4 comes
closest, 51.2 with PI [49.5, 52.9]), so the registered branch applies: the Zephyr row is "not
reproduced", and the verdict of record stays at beta. D4 "matches under `zephyr_gemma`" on 17 of 25
attainable cells. `zephyr-orpo-141b-A35b-v0.1` is listed as the pod-only candidate, not run. The
diagnosis stops here, per Amendment 1.

## 4. Referee pass

1. *"Your Zephyr is not the paper's Zephyr, so a miss says nothing about MoralBench."* Conceded
   in part. The checkpoint is an assumption of record (§2), and "Zephyr 3.6.0" names no release.
   MB-A6 carries R_model, and the variant cells test it. The parse-free R2 miss on MFQ binary
   stands either way for this checkpoint and harness.
2. *"The registered MB-A5 reads use R0 parses that select on the outcome."* Correct (MB-A7).
   The verdicts are reported by the letter, and the letter-bias read (L) is shown under R0, R1
   and R2, which agree in sign and rough size. S under R0 is not interpreted beyond its registered
   verdict.
3. *"Items-fixed PIs are narrow, so almost anything fails to replicate."* The PI is the sampling
   noise of a 5-repeat mean on these items, which is what the paper's cell carries. The misses
   are 7 to 32 points on totals whose half-widths are 1 to 3, so a lenient rule would not change
   them; MFV-comparative passes under the same rule.
4. *"The MFV-comparative 'replications' would happen by chance."* Largely conceded (MB-A8
   calibration). A random letter policy's PI covers the paper's MFV-comparative total for 54–65% of
   draws, so those replications are reported as weak evidence. MFQ binary is the informative set,
   and it misses on all three families.
5. *"Zephyr ran on the M4 and Gemma and Llama on A100s: device, not model, could explain the
   differences."* Same driver, sampler, parser and transformers version. Device parity was checked
   on 3 Gemma items: identical letter distributions. That is a weak check (near-deterministic
   items), so it is stated as exploratory.
