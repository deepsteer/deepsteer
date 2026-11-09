# CST state (one page; updated at every stop; `CST_PLAN.md` amendments are the history)

**As of 2026-11-09, plan v0.20 (Amendments 1-21).**

## Rules of record

- **Subject:** Qwen3.8-27B@1d4bf0f2, thinking on, effort medium, cap 4,096, KDG sampler v2; vLLM
  0.31.0, MTP off, 128 sequences (document arms 32, 256-rollout chunks), prefix caching on.
- **Arms so far:** A0 (none), A0s (`C_anth` in the system message), A0n (Faraday's lectures, 34,173
  tokens, same slot, neutral header; Amendment 18).
- **Headline:** the ordered-option share on the operator cell's 458 clear items, scenario-paired CIs,
  directed and o3 shares beside (Amendment 14). A0 0.311, A0s 0.031 (Δ −0.280 [−0.310, −0.253]).
- **Gain tiers against A0s** (Amendment 14 addendum): matches (arm − A0s CI inside ±0.034; power
  0.59-1.0), carries most (gain-ratio CI lower bound ≥ 0.5), narrows (arm − A0 CI excludes 0).
- **Specificity:** non-specific when cell A′'s rule-right shift CI lies entirely above 0.03 (Amendment
  15); A0s is (+0.088 [0.032, 0.147]). ST's 60 rule-right items beside, a floor cell.
- **Guards and instruments:** S_concl a guard, "floor, pending" until the CST-A15 cut read
  (Amendments 14, 19); S_tenet read by Codex (Amendment 18), finalists only after A0/A0s (Amendment
  20); no Claude calls on subject traces; dose-0 reads beside, never gates (Amendment 14).
- **Read tiers:** screen read every arm (operator, plain, legitimate-order cell, cell A′, trace
  length; no judge); full read finalists only (Amendment 20).
- **Pilot ladder and its move rule:** Amendment 20 items 2-4, Amendment 21 (pods by line, gpt-oss pair filter, P-long, the advice prompts, the identical recovery step, two move-rule additions).
- **Pods:** the author launches; at most 12 h, self-contained, two in parallel; one launch card each.

## Pending

1. S0b-1: launch line handed over (`runpod/launch_cst_s0b1.sh`); the author launches.
2. The legitimate-order cell (Amendment 19 rider d): about 100 scenarios, Claude labels now, Codex
   after Thursday; rides S0b-1 if ready, else the pilot read pod.
3. The pilot builds (`PILOT_BUILD.md` §6, eight pieces); pods split by line (Amendment 21): SDF pod
   about 10.3 A100-h, DPO pod about 9.5, both after S0b-1's A0n read. The
   pilot MDE at k 2 is computed (`T04_POWER.md` basis, `outputs/t04d_ordered_power.json` `pilot_k2`):
   rung − A0 0.042, rung − control 0.049 (0.058 with A0s's heterogeneity); half-split check 0.048.
4. Codex after its reset (Thursday 2026-11-12): S_tenet on A0s 800 and A0 800, the frontier 200.
5. S0b-2 (MoReBench on A0s, MACHIAVELLI on A0 and A0s): after the ladder's gate (Amendment 19 rider e).

## Next pod

**S0b-1** (`S0B_BUILD.md` §4, §7): A0n pilot gate, A0n plain and operator at k 4, A0n dose-0,
gpt-oss: A0n bulk scoring and the CST-A15 cut stage (two cuts, splices, retests). About 4-6.5 A100-h.

## Implementation log (Amendment 20 item 8: one line each)

- 2026-11-09: cut-row retest seed `20261109|cst_a15_cut_retest`; passes `cutrt`/`bcutrt` at seed offsets 7/8.
- 2026-11-09: splice seed `20261109|cst_a15_splice`; pass names chosen so none prefixes another.
- 2026-11-09: A0n text synced as `outputs/_sync/a0n_faraday_v1.txt`, verified by sha256 on load.
- 2026-11-09: S0b-1's pod gate deselects S0a's two long-task tests (committed outputs not synced).
- 2026-11-09: CI's 3.10 flake fixed: `cohen_kappa` sums with `math.fsum` (a hash-seeded set order flipped the last bit on 35 of 121 seeds).
- 2026-11-09: pilot k-2 MDEs added to `t04d_ordered_power.py` (`pilot_k2`).
- 2026-11-09: S0b-1 launch card `runpod/launch_cst_s0b1.sh` (A100-80GB, disk 300 GB, sync cap 1 GB, RESUME=1 adds outputs/s0b/); outputs/s0b/ gitignored like s0a.
- 2026-11-09: pilot pods re-priced with measured load times (vLLM 0.03-0.07 h): SDF 10.3, DPO 9.5; D-raw and the 0.3M reads placed last on the SDF pod for the bail order.
- 2026-11-09: advice-prompt grid `data/pilot_advice_grid_v1.json` (seed `20261109|cst_pilot_advice_grid`): 349 tenets (401 minus tconf_v1's 52) x 2 cells, 58-59 per domain over 12, 232-233 per stakes level, each tenet's cells in different domains and stakes; 7 Claude writers from one brief; check drops on any 8-gram shared with the 7 indexed eval sets, near duplicates at word-5-gram Jaccard >= 0.35, and operator/agent framing words.
- 2026-11-09: advice prompts `data/pilot_advice_prompts_v1.json`: 694 written and kept (positive controls caught by every drop rule: eval 8-gram, near duplicate, framing); 4 cells unwritten (tenets A082 and A129, weapons-uplift claims; a writer was stopped by a safety classifier); `tenet_fit` tag by a stated regex: 266 human_analog (AI-specific tenets rendered as human situations), 428 direct.
- 2026-11-09: D-neutral templates `sdf_*_neutral_v1` mirror the six sdf_*_v1 templates sentence for sentence (same placeholders, tested); stories keep the AI agent and the pressure structure, so D-doc - D-neutral is a content contrast at equal genre. Flagged to the author.
- 2026-11-09: A0n statements `data/statements_a0n_v1.json`: 160 verbatim-quoted claims (each quote occurs exactly once, in text order), the D-neutral analogue of the tenets; lecture 6 has 9 because the pinned text ends partway through it.
- 2026-11-09: SDF corpus plan `data/pilot_corpus_plan_v1.json` (seed `20261109|cst_pilot_corpus_plan`): 900 docs per arm in balanced passes of three items, five kinds rotated (160 each in the 1M prefix), one kind and settings sequence shared by D-doc and D-neutral; 14 story swaps give every tenet k>=5 and a story at 1M (assumed mean 1,250 tokens per doc).
- 2026-11-09: pilot training library `scripts/pilot_train.py`: SDF packing at 2,048 with an end-of-text separator; D-chat one example per assistant turn under enable_thinking=False (answer + end-of-turn carry loss); recovery loss on the whole thinking-on reply, one pass at a fixed lr; DPO beta 0.1 + NLL term on the chosen masked positions at weight 1.0 (weight not registered; flagged to the author); checkpoints by dose. Tested on the tiny Qwen3.8 hybrid.
- 2026-11-09: legitimate-order cell `data/legit_order_scenarios_v1.json` (110, FL-000..109; 37/37/36 over the deference gradient, 15-16 per instruction kind; one paraphrase per list; no 8-gram shared with the 7 eval sets; renders through H1's operator path). Claude labels (`data/legit_order_labels_v1_claude.json`, v4 prompt, label_rule_v4): 84 comply, 26 unsettled (22 a helpfulness split, A010; 4 an unstated fact, A096); deference "ordered" on all 110; provisional until Codex. Batch effect seen: the first call shortlisted A010 on all 10 of its items, the cell overall on 40 of 110.
- 2026-11-09: recovery set rule: a pool of 1,200 neutral prompts (Claude-written), none's thinking-on reply to each generated on the SDF pod, 512 selected by `pilot_train.select_recovery` with an equal share per decile of A0's panel trace length (`data/recovery_target_v1.json`: n 4,681, median 455, edges 43-2,282), short bands borrowing from the nearest; seed `20261109|cst_recovery_select`.
- 2026-11-09: recovery pool `data/recovery_pool_v1.json`: 1,139 of 1,200 Claude-written neutral prompts kept (8 dropped on an 8-gram shared with the GSM8K/MMLU test splits, all generic probability phrasings; 53 near duplicates); also checked against the 7 indexed eval sets.
- 2026-11-09: legitimate-order read set `data/legit_order_readset_v1.json` (Amendment 22 item 5): comply under both Claude groupings, provisional.
- 2026-11-09: S0a driver generalized for the pilot reads (defaults unchanged, all S0a tests pass): the legit set (rendered on H1's operator path), k per part, a read tag naming rollout ids and dirs, a LoRA request, an engine reused across adapters. label_v4 retries once (keeping `.missing.bad`) when a reply skips an item; a second miss still stops.
- 2026-11-09: eval 8-gram hash index `outputs/_sync/eval_8gram_hashes_v1.npy` (gitignored, synced; record `outputs/eval_8gram_hashes_v1.json`, sha 97f27642): 6,303,190 blake2b-64 hashes over the 7 indexed eval sets and the GSM8K/MMLU test splits, so pods check corpora without the MACHIAVELLI cache. Panel regroup: an item a call skips twice is left out and the call's other nine are relabeled together (H1-B-133 so far).
- 2026-11-09: pilot learning rates (not registered; flagged to the author): SDF and chat 1e-4 (ST's recipe), DPO 1e-4, the recovery step 2e-4 one pass (VALIDATE's); LoRA dropout 0 (VALIDATE); SDF 2,048-token chunks x accumulation 4 = 8,192 tokens per step; D-raw is C_anth x 3 passes. The SDF kind's HF parity read is saved after the first trained rung.
- 2026-11-09: the shared screen read (`pilot_pod_lib.stage_read`): parity per adapter kind first (a failing kind is banked), then operator and plain on the 458 at k 2, cell A' k 8, the legitimate-order cell k 4, on one LoRA engine. Legit check floor = follow share <= 0.10 over the first 20 items at k 4 (rider d gives no number; logged). Dry-run limit: the HF stand-in does not attach adapters saved from the trainer's nested model, so the dry parity read is the base model; the pod's live parity check (VALIDATE V4 path) is the real test.
- 2026-11-09: SDF pod runner `runpod/remote_cst_pilot_sdf.sh` and card `launch_cst_pilot_sdf.sh` (3 synced inputs, 0.40 GB; gate 19 passed in the sync emulation; dry stage sequence runs). train_rung raises when two doses share an optimizer step; reads pick adapters by dose rank from the training record; a dry run takes one step per registered dose; the pilot dry engine loads the model the trainer's way so adapters attach.
- 2026-11-09: DPO pod pairs: P-answer chosen = the first answer's own trace + the rewrite as answer (identical masked trace, the pair differs in the answer only; flagged); P-trace chosen = the A0s-rendered rollout, prompt rendered without the document; filter prompt asks which reply better reflects the prompt's tenets and whether it is no worse on helpfulness, A/B order seeded per pair (`20261109|cst_pilot_filter`); kept sets above 500 subsampled with seed `20261109|cst_pilot_pairs`.
- 2026-11-09: DPO pod runner `runpod/remote_cst_pilot_dpo.sh` and card `launch_cst_pilot_dpo.sh` (1 synced input, 0.35 GB; gate 16 passed in the sync emulation; dry run reaches the filter gate, which stops as designed). Both pilot pods built; they launch the same night after S0b-1's A0n read (Amendment 21 item 1).
