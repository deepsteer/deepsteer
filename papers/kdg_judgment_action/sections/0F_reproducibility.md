# Appendix F. Reproducibility {#app:repro}

Everything in this paper is reproducible from files in the repository without a GPU or an API
key, except the generation of the arrays themselves.

- **Specification and amendments**: `papers/KDG_PANEL_SPEC.md` (v0.4 plus A1 to A17).
- **Scenarios of record**: `papers/kdg_panel/data/{pilot,panel,round2}_scenarios_*.json` and
  `swap_scenarios_F4_*.json` (440 scenarios; both frames, paraphrases, twins, covariate tags,
  external labels, construction flags; generator, prompt version, and system-prompt hash in the
  metadata).
- **Library**: `deepsteer/kdg/` (schema and frame templates 1.0.0, harness 1.0.0, breadth
  rubric, statistics, calibration), with tests that each assert a named failure mode.
- **Pod driver and runner**: `papers/kdg_panel/scripts/pod_kdg_pilot.py` (stub dry run without a
  model; per-rollout saves; manifest with SHA-256 per artifact and the resolved model commit) and
  `runpod/remote_kdg_pilot.sh`.
- **Analysis**: `analyze_pilot.py` (screen, gates, ladder, strictness levels, argmax three-cell,
  swap; unions runs), `analyze_continuous.py` (the continuous readout), `analyze_paper8.py`
  (amendment A17: the raw-frame three-cell with paired CIs, the exploratory items, and the
  per-scenario tables), `analyze_anomalies.py`; the analyses of record are copied to
  `papers/kdg_panel/data/analysis_*.json` with their manifests.
- **Per-scenario tables**: `papers/kdg_panel/data/per_scenario_union.csv` (chat cells: screen
  outcome, strictness level, binary gap, $p_D$, $p_J$, and their pressure-removed values) and
  `per_scenario_raw_union.csv` (raw-frame cells on both models).
- **Calibration**: `data/calibration_set_v{1,2,3,4}_real.json`, judge label files, and stage-2
  reports.
- **Results documents**: `papers/kdg_panel/KDG_RESULTS.md` (sections 1 through 13, each with a
  referee pass), `SCREEN_*.md`, the blind-read packet and scored answers.
- **Figures**: `papers/kdg_judgment_action/figure_data/regen_kdg_figures.py` regenerates every
  figure from the analysis JSON and the per-scenario tables and writes a CSV per figure.
- **Models**: `allenai/Olmo-3-7B-Instruct` and `allenai/Olmo-3-1025-7B` at the commits recorded in
  each pod manifest; `transformers` 5.12.1 on the pods; temperature 0.7 for sampled cells; seeds
  fixed and logged; bootstrap seed 0, 2000 draws.

Per-rollout arrays (15 GB) are not in the repository; they are regenerable from the scenario
files with the driver and are available on request.
