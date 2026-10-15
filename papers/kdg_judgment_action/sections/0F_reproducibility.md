# Appendix F. Reproducibility {#app:repro}

The repository (<https://github.com/deepsteer/deepsteer/>) supports three levels of reproduction, and they
need different things.

1. **Figure regeneration** (seconds, no GPU, repository only).
   `papers/kdg_judgment_action/figure_data/regen_kdg_figures.py` redraws every figure from the committed
   analysis JSON and per-scenario tables and writes one CSV per figure with the plotted values.
2. **Statistic recomputation** (minutes, no GPU). Every number in the paper is recomputed by an analysis
   script from per-rollout data, with the bootstrap seed and draw count fixed in the script. The scripts read
   the pods' per-rollout arrays (about 133 GB across all pods, not in the repository); the released
   per-rollout table below carries what the letter-based statistics need, so most of them can be recomputed
   from it alone.
3. **Model regeneration** (GPU). Regenerating the arrays means rerunning the pod drivers on the pinned model
   revisions. Forward readouts reproduce to the precision checked in each pod's validation step; sampled
   cells were generated under sampler version 1 (one seed stream per rollout index, each model's default
   nucleus; \Cref{instruments}), so a rerun under the corrected harness reproduces their distribution, not
   their exact text.

- **Specifications and amendments**: `papers/KDG_PANEL_SPEC.md` (v0.4 plus A1 to A17),
  `papers/KDG_PHASE1_SPEC.md` (P1-A1 to P1-A17) and `papers/KDG_GPTOSS_SPEC.md` (G-A1 to G-A29)
  (\Cref{app:prereg}).
- **Scenarios of record**: `papers/kdg_panel/data/{pilot,panel,round2}_scenarios_*.json` and
  `swap_scenarios_F4_*.json` (440 scenarios; both frames, paraphrases, twins, covariate tags,
  external labels, construction flags; generator, prompt version, and system-prompt hash in the
  metadata); `round3_scenarios_*.json` adds the 192 scenarios that, with the panel, make the 586-scenario
  union read in \Cref{base}, \Cref{recipe} and \Cref{deliberation}; `data/harm_divergence/` holds the
  rule-harms set of \Cref{rule-harm} (all 270 generated scenarios, both raters' labels, the kept ids and
  the author's spot-check sheet). \Cref{tab:flow} traces every set to these files.
- **Library**: `deepsteer/kdg/` (schema and frame templates 1.0.0, harness 1.0.0, breadth
  rubric, statistics, calibration), with tests that each assert a named failure mode.
- **Pod driver and runner**: `papers/kdg_panel/scripts/pod_kdg_pilot.py` (stub dry run without a
  model; per-rollout saves; manifest with SHA-256 per artifact and the resolved model commit) and
  `runpod/remote_kdg_pilot.sh`; for the multi-model sessions, `scripts/pod_kdg_phase1.py` and
  `runpod/remote_kdg_phase1.sh`, with every model's repository, revision and template hash pinned in
  `papers/kdg_panel/models.yaml`.
- **Analysis**: `analyze_pilot.py` (screen, gates, ladder, strictness levels, argmax three-cell,
  swap; unions runs), `analyze_continuous.py` (the continuous readout), `analyze_paper8.py`
  (amendment A17: the raw-frame three-cell with paired CIs, the exploratory items, and the
  per-scenario tables), `analyze_anomalies.py`; for the multi-model sessions (specification `papers/KDG_PHASE1_SPEC.md`,
  amendments P1-A1 to P1-A17), `analyze_phase1_session_{a,b,c}.py`, `analyze_kdg_a8.py`,
  `analyze_kdg_a12.py`, `analyze_screen_rates.py`, `analyze_own_screen.py` and `analyze_dose_twin.py`;
  for GPT-OSS-20B, `analyze_gptoss*.py`; for the rule-harms set, `generate_harm_divergence.py` and
  `analyze_h1_*.py`; for the v2 corrections, `analyze_kdg_a23_twoway*.py`, `analyze_gap_twoway.py`,
  `analyze_v2_dose_twoway.py`, `analyze_v2_overlap_cluster.py`, `analyze_v2_crossing_audit.py`,
  `analyze_v2_slice_check.py`, `analyze_meta_vs_tulu_paired.py`, `analyze_h1_three_option.py` and `analyze_brake_panel_vs_h1.py`;
  the analyses of record are copied to
  `papers/kdg_panel/data/analysis_*.json` with their manifests.
- **Per-scenario tables**: `papers/kdg_panel/data/per_scenario_union.csv` (chat cells: screen
  outcome, strictness level, binary gap, $p_D$, $p_J$, and their pressure-removed values) and
  `per_scenario_raw_union.csv` (raw-frame cells on both models).
- **Calibration**: `data/calibration_set_v1.json` and `data/calibration_set_v{2,3,4}_real.json`, judge
  label files, and stage-2 reports.
- **Results documents**: `papers/kdg_panel/KDG_RESULTS.md` (sections 1 through 43, each with a
  referee pass), `SCREEN_*.md`, the blind-read packet and scored answers.
- **Models**: `allenai/Olmo-3-7B-Instruct` and `allenai/Olmo-3-1025-7B` for the panel, and for the
  multi-model sessions the OLMo-3 SFT, DPO and RL-step checkpoints, Llama-3.1-8B and
  Llama-3.1-8B-Instruct, Tulu 3 (SFT, DPO, final), Qwen2.5-7B and Qwen2.5-7B-Instruct, and
  `openai/gpt-oss-20b`, each at the
  revision in `models.yaml` and the commit recorded in each pod manifest; `transformers` 5.12.1 on the
  pods; temperature 0.7 for sampled cells; seeds fixed and logged; bootstrap seed 0, 2,000 draws for the
  panel analyses and 10,000 for the multi-model sessions.

**Per-rollout release.** `kdg_rollout_option_probs.csv.gz`, written by
`papers/kdg_panel/scripts/export_rollout_option_probs.py` from every pod of record, has one row per
rollout: pod, model, condition (cell), frame (primary or pressure-removed), scenario id with its role and
its primary's id for harm twins, rollout and seed index, the letter-to-option order, the parsed option, and
the displayed options' probabilities at the decision position, renormalized over the saved letter tokens
and, for the forced-answer arms, over each letter's bare and space-prefixed tokens together, as the dose
analyses read them. A manifest gives row counts per source file and the table's SHA-256. The table is
published with v2 at \textbf{[release location: author to confirm]}; the full arrays (next-token
log-probabilities over the vocabulary, residual-stream saves) are available on request.
