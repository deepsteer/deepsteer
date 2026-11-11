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

## Spend ledger (Amendment 29; the author, 2026-11-11)

A100-80GB hours per pod from each session log's own clock, plus about 0.15 h per pod for provisioning, sync and download that the logs do not capture (estimated). The re-read pod is billed to its teardown (it sat idle 2.6 h after its runner died). Dollars at the plan's assumed $1.6 and $2.0 per A100-80GB hour (`CST_PLAN.md` §8: "author to confirm the current rate"); a confirmed rate replaces both columns.

| date | pod | session h | billed h | $ at 1.6 | $ at 2.0 | running h |
|---|---|---|---|---|---|---|
| 11-07 | VALIDATE run 1 (validate_run1/session_validate.log) | 0.86 | 1.01 | 2 | 2 | 1.0 |
| 11-07 | VALIDATE (validate/session_validate.log) | 1.93 | 2.08 | 3 | 4 | 3.1 |
| 11-07 | probe pod (probe/session_probe.log) | 1.25 | 1.40 | 2 | 3 | 4.5 |
| 11-07 | scorer (effort high) (scorer_high/session_scorer_high.log) | 1.48 | 1.63 | 3 | 3 | 6.1 |
| 11-08 | S0a session 1 | 3.71 | 3.86 | 6 | 8 | 10.0 |
| 11-08 | S0a session 2, first pod (A0s saturated, stopped) | 1.25 | 1.40 | 2 | 3 | 11.4 |
| 11-08 | S0a session 2, relaunch | 8.30 | 8.45 | 14 | 17 | 19.8 |
| 11-08 | S0a session 3 | 4.53 | 4.68 | 7 | 9 | 24.5 |
| 11-08 | S0a session 4 | 1.48 | 1.63 | 3 | 3 | 26.1 |
| 11-09 | S0b-1 | 5.11 | 5.26 | 8 | 11 | 31.4 |
| 11-10 | pilot SDF, first (D-doc gate bail) (Amendment 24) | 0.38 | 0.53 | 1 | 1 | 31.9 |
| 11-10 | pilot DPO, first (engine bail) (one engine per process) | 1.76 | 1.91 | 3 | 4 | 33.8 |
| 11-10 | pilot SDF, relaunch (base-model reads, CST-A22) | 11.26 | 11.41 | 18 | 23 | 45.3 |
| 11-10 | pilot DPO, relaunch (filter gate bail) (CST-A21) | 2.30 | 2.45 | 4 | 5 | 47.7 |
| 11-10 | parity mini-pod (CST-A22) | 0.23 | 0.38 | 1 | 1 | 48.1 |
| 11-10 | pressure 2-roll (the action read failed at 6.2 h) | 6.20 | 6.35 | 10 | 13 | 54.4 |
| 11-10 | SDF re-read (pod 1) (runner died at 10.0 h; idle until teardown 05:59Z) | 10.03 | 12.62 | 20 | 25 | 67.1 |
| 11-11 | pressure 2-act | 0.34 | 0.49 | 1 | 1 | 67.5 |
| 11-11 | pressure 2-ctx, first (smoke bail) (the top-20 letters) | 0.28 | 0.43 | 1 | 1 | 68.0 |
| 11-11 | pressure 2-dpo, first (stopped minutes in) (no log downloaded; estimated) | 0.10 | 0.25 | 0 | 0 | 68.2 |
| 11-11 | pressure 2-dpo, relaunch | 3.64 | 3.79 | 6 | 8 | 72.0 |

**Total to 2026-11-11 (before 2-ctx and SDF pod 2): 72.0 A100-h, $115 to $144.** Envelope of record 150 A100-h ($500 to $1,000 cap): **78.0 A100-h remain**, $356 to $885 under the dollar cap. Today's two pods are priced at about 6.5 (2-ctx) and 10.3 (SDF pod 2) A100-h, which would leave about 61.2.

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
- 2026-11-09: Amendment 24 build: `annotate` keeps a document only if its final did not hit the cap and shares no 8-gram with the eval index (D11's drop was counted at the gate but never applied); the gate records each template's median words; `train_<rung>.json` carries `corpus` (kept count, document-equivalents per dose) and every read writes `read_docs.json`.
- 2026-11-09: the first DPO pod failed its candidate stage at 1.76 h: one process built A0's engine (the none candidates, 2,776 rows, banked) and then A0s's, which could not start with 5 of 79 GB free. Every stage process now builds at most one engine: the runner calls `candidates --part none|doc` and `legit --arm a0|a0s|a0n` as separate processes; A0's legit check and read share one engine (both pods). The DPO relaunch resumes from the banked none candidates (RESUME=1). The running SDF pod keeps the old legit stage, which may fail the same way; its A0 legit read is duplicated by the DPO pod's.
- 2026-11-10: CST-A22's parity mini-pod `scripts/cst_parity_pod.py` (prep, hf, served, gate; runner `remote_cst_parity_a22.sh`, card `launch_cst_parity_a22.sh`): Amendment 25 A4's gate in `pilot_pod_lib.parity_record` with write-once records; the renamed-path probe moves adapter keys from `model.layers` to `model.language_model.layers`; the dry run serves through VALIDATE's stand-in (the served class) and reads a trainer-class adapter as the base. The card syncs the four adapter files by name: directory entries are filtered after rsync_exclude.txt, which drops *.safetensors.
- 2026-11-10: Amendment 25 A4 and Amendment 26 items 1, 3, 4 in the pilot library: `hf_letters` (a `parity_hf` stage per driver, its own process) and a write-once parity record per read tag; every adapter served from `served_copy` (keys on the served class's path, under `served_adapters/`, outside the downloaded `adapters/`); the dry stand-in is VALIDATE's `HFStandIn` (the served class; `DryEngine` removed); `stage_read` reads `base_session` first on the same engine at READ_K; the corpus gate's band is a 5% bar.
- 2026-11-10: pod 1 (Amendment 26 item 5) `runpod/remote_cst_sdf_reread.sh`, card `launch_cst_sdf_reread.sh` (CST_SESSION=2, about 8.3 h, watchdog 10.5 h): seven saved adapters, four training records and the recovery set synced by name (3.5 GB, emulated; gate tests 25 passed in the copy). Last night's reads, the first D-neutral attempt and the A0 legit reads moved to `outputs/pilot_sdf_base_reads/` (gitignored) so the download cannot overwrite them; the card refuses while `outputs/pilot_sdf/gen` exists.
- 2026-11-10: Part C's writer: the first batch of 20 cells was declined after a safety classifier stop and is not retried in any form; its cells are recorded as not written. `PRESSURE_BUILD.md` prices the design at about 13 A100-h, over C8's 10.5 h; two fits to the author.
- 2026-11-10: Amendment 27 build: Claude's writer stopped after 5 finished and 2 declined batches; grid v2 (600 cells: 92 Claude, 508 subject) drops the declined cells for every writer and `no_caveats_research`; Claude's 92 checked locally, 90 kept (2 eval 8-gram drops go to the subject). 2-roll: `scripts/cst_pressure_pod.py` (write, roll per arm, act, contrast), runner `remote_cst_pressure_roll.sh`, card `launch_cst_pressure_roll.sh` (about 4.2 h, watchdog 6.0 h); dry session end to end in a test; sync emulated (the three _sync inputs), gate tests 8 passed in the copy.
- 2026-11-10: the re-read pod's first read stage failed at 1.21 h before any parity record: `served_copy` built the model classes to find the decoder paths in a process that had loaded vLLM, where the 27B config raises on `vocab_size` (the mini-pod's prep ran without vLLM; the dry tests use the tiny model without vLLM). Fix: `hf_letters` writes `parity/prefixes.json` in its HF process and `served_copy` reads it, else computes the paths in a fresh process; the SDF driver's `catch_up` reads, first, earlier tags whose read stage failed before its parity record. Tests name both.
- 2026-11-10: 2-ctx and 2-dpo built (`scripts/cst_pressure_train_pod.py`, runners and cards `*_cst_pressure_ctx.sh`, `*_cst_pressure_dpo.sh`): the smoke pass (data builders, one step per loss, a random adapter scaled until TV(base, attached) >= 0.15, saved, renamed, served, through the gate, read on 2 prompts) as the first two stages; `train_dpo` takes NLL-only retain items; capped or answerless rollouts are not trained on; a dry session of both pods runs end to end in a test. Prices are recomputed from 2-roll's measured rollout lengths before the cards go out (A0's pressure rollouts average 1,560 tokens, 3.7 times the operator cell's).
- 2026-11-10: 2-roll projected to about 7.5 h against its 6.0 h watchdog (writing 1.55 h against 0.3 priced; A0 rollouts 1,560 tokens on average, 3.7 times the operator cell's). With the author's approval (option 1) the watchdog subshell alone was sent TERM at about 3.1 h (its trap killed its sleep; no pkill ran); the runner and the A0s stage were untouched. The session now has no watchdog; the session ends on its own after the contrast stage.
- 2026-11-10: the re-read pod's sshd died at about 2.2 h (11:36 PST, the moment the lead's first scp of the patch failed; the cause is not established); the runner shell went with it, the D-neutral corpus stage finished alone at 12:59 PST, and nothing started the next stage. With the author's approval: `service ssh start` restored the direct endpoint, and `runpod/remote_cst_sdf_reread_cont.sh` (the runner's remaining lines unchanged, the original clock, the original watchdog's 10.5 h cap, the same EXIT trap) was run locally with a stub and started detached (setsid nohup).
- 2026-11-10: 2-roll's action read failed at 6.2 h on Claude's scenarios (no `standing_instruction` field: Claude's brief did not ask for one; the subject's did), so the contrast had no labels. The writing and all rollouts are complete and local (A0 2,356; A0s 1,178; A0n 1,178). The dry action read used fixed labels and never built the prompt; it now builds every prompt, `act_prompt` points the judge at the system prompt when the field is absent (a test names it), and 2-act (`remote_cst_pressure_act.sh`, card `launch_cst_pressure_act.sh`, about 0.7 A100-h) reruns the action read and the contrast after a smoke read of two rollouts per writer.
- 2026-11-10: 2-ctx and 2-dpo both wrote to outputs/pressure, so the second launcher's download would have overwritten the first session's same-session base and parity files; at the author's choice 2-dpo was stopped minutes in and relaunched with its own folder (`outputs/pressure_dpo`: the runner copies 2-roll's synced inputs in; the card downloads only that folder).
- 2026-11-11: 2-ctx's smoke pass stopped the pod at 0.28 h, before training: its random adapter (TV 0.41) pushed every letter out of vLLM's top 20 log-probs on some prompt, the letters read -1e9 and `tv` divided by zero (2-dpo's smoke adapter, with different A matrices, passed). `tv` now counts such a prompt as 1.0 and the record counts them (a real read banks instead of crashing); the smoke scale search keeps every letter in the served top 20 on the real model. The bailed session's files are in `outputs/pressure_ctx_bailed/`.
- 2026-11-11: Amendment 29 build for 2-ctx: the floor-aware parity gate with the resolution clause (`parity_record`'s `read` and `parity_unresolved`; P-long's numbers in a test); the held-out cell (`data/pressure_heldout_v1.json`, 177 of 589, stratified by kind and writer; P-ctx and P-ctx-n exclude it); `stage_heldout` and `stage_heldout_act` (the action read factored into `cst_pressure_pod.judge_rows`); 2-ctx reads P-long on its shared base with P-long's adapter and record synced from 2-dpo; re-priced at about 6.5 h; sync emulated (22 files, 0.9 GB), gate tests 25 passed in the copy.
- 2026-11-11: SDF pod 2 (Amendment 29 items 5-7, 10): `scripts/cst_sdf_norec_pod.py` (smoke pass via `pilot_pod_lib.smoke_random_adapter`/`smoke_serve`; reads without recovery on one base; capability at n 200 on a seeded half-GSM8K half-MMLU subset with the adapter served; the held-out cell; the injected-order guard cell, `cst_pressure_pod.injected_msgs`, 120 held-out subject scenarios; the second D-doc draw, seed 1 and a new chunk order), runner `remote_cst_sdf_norec.sh`, card `launch_cst_sdf_norec.sh`, its own folder `outputs/pilot_sdf2`; about 7.2 h; a dry session of every stage in a test; sync emulated (2.6 GB), gate tests 23 passed in the copy.
- 2026-11-11: P-ctx's constitution-reference audit (`scripts/pctx_reference_check.py`, `outputs/pctx_reference_check.json`): the pointing pattern read 0.216 (172 of 797), but its matches are references to the operator's order, which is in the student's context ("the system prompt says"; A0, with no constitution, 0.206): the lead's example pattern was too broad. Genuine document pointing 0.003. The author ruled: 2-ctx launches unchanged, no exclusions; possessive references (0.381; away 0.668) are reported beside P-ctx's read by scenario kind.
- 2026-11-11: the D-doc plan extended to 2,600 documents (`data/pilot_corpus_plan_v2.json`, `pilot_corpus_plan.extend`: v1's 900 exactly, then the same rule's deal; every tenet at k >= 5 and in a story within 3M); pod 2's last stage continues D-doc's corpus under it and the same gate until 10.3 h.
- 2026-11-09: legitimate-order read set `data/legit_order_readset_v1.json` (Amendment 22 item 5): comply under both Claude groupings, provisional.
- 2026-11-09: S0a driver generalized for the pilot reads (defaults unchanged, all S0a tests pass): the legit set (rendered on H1's operator path), k per part, a read tag naming rollout ids and dirs, a LoRA request, an engine reused across adapters. label_v4 retries once (keeping `.missing.bad`) when a reply skips an item; a second miss still stops.
- 2026-11-09: eval 8-gram hash index `outputs/_sync/eval_8gram_hashes_v1.npy` (gitignored, synced; record `outputs/eval_8gram_hashes_v1.json`, sha 97f27642): 6,303,190 blake2b-64 hashes over the 7 indexed eval sets and the GSM8K/MMLU test splits, so pods check corpora without the MACHIAVELLI cache. Panel regroup: an item a call skips twice is left out and the call's other nine are relabeled together (H1-B-133 so far).
- 2026-11-09: pilot learning rates (not registered; flagged to the author): SDF and chat 1e-4 (ST's recipe), DPO 1e-4, the recovery step 2e-4 one pass (VALIDATE's); LoRA dropout 0 (VALIDATE); SDF 2,048-token chunks x accumulation 4 = 8,192 tokens per step; D-raw is C_anth x 3 passes. The SDF kind's HF parity read is saved after the first trained rung.
- 2026-11-09: the shared screen read (`pilot_pod_lib.stage_read`): parity per adapter kind first (a failing kind is banked), then operator and plain on the 458 at k 2, cell A' k 8, the legitimate-order cell k 4, on one LoRA engine. Legit check floor = follow share <= 0.10 over the first 20 items at k 4 (rider d gives no number; logged). Dry-run limit: the HF stand-in does not attach adapters saved from the trainer's nested model, so the dry parity read is the base model; the pod's live parity check (VALIDATE V4 path) is the real test.
- 2026-11-09: SDF pod runner `runpod/remote_cst_pilot_sdf.sh` and card `launch_cst_pilot_sdf.sh` (3 synced inputs, 0.40 GB; gate 19 passed in the sync emulation; dry stage sequence runs). train_rung raises when two doses share an optimizer step; reads pick adapters by dose rank from the training record; a dry run takes one step per registered dose; the pilot dry engine loads the model the trainer's way so adapters attach.
- 2026-11-09: DPO pod pairs: P-answer chosen = the first answer's own trace + the rewrite as answer (identical masked trace, the pair differs in the answer only; flagged); P-trace chosen = the A0s-rendered rollout, prompt rendered without the document; filter prompt asks which reply better reflects the prompt's tenets and whether it is no worse on helpfulness, A/B order seeded per pair (`20261109|cst_pilot_filter`); kept sets above 500 subsampled with seed `20261109|cst_pilot_pairs`.
- 2026-11-09: DPO pod runner `runpod/remote_cst_pilot_dpo.sh` and card `launch_cst_pilot_dpo.sh` (1 synced input, 0.35 GB; gate 16 passed in the sync emulation; dry run reaches the filter gate, which stops as designed). Both pilot pods built; they launch the same night after S0b-1's A0n read (Amendment 21 item 1).
- 2026-11-09: Amendment 22 items 1-3 read: panel agreement 0.908, 441/456 clear stay clear, stable Delta -0.277 inside the null [-0.286, -0.275] (CST-A19 closes for the headline); H1 agreement 0.576, 48/82 stay clear, Delta-P stable +0.128 plain / +0.281 operator. The null uses point deltas only (a full bootstrap per draw took hours).
- 2026-11-09: Amendment 23 items 1-5 read and built: A0n log-odds ratio 0.343, reductions 0.51 / 0.80, content band -0.123 / -0.238 (odds at A0); injection-talk pattern (registered, byte-identical) 0.41 / 0.62 / 0.05 with the ordered split; the DPO pod reads A0n on the legitimate-order cell (card syncs the A0n text, 0.36 GB).
- 2026-11-09: R_injection share of A0n's drop stated as a range, 0.31-0.53 (composition at A0n's and at A0's conditional rates; `injection_decomposition`); the earlier whole-drop reading is superseded.
- 2026-11-09: S0b-1 done (5.11 A100-h): CST_S0B_REPORT.md; CST-A15 resolved (R_conclusion under both cuts); CST-A16 read (neither registered branch exactly; the dose-0 shift is content-specific); A0n partial (CST-A20).
