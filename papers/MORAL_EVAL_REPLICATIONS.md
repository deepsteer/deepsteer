# Moral and ethical benchmark replications — index and license rules

Status: 2026-10-29 (author). One index for the external-benchmark replications. Each gets its own
plan when it starts; this file holds only the order, the license findings, and the handling rules
that the plans inherit. They all build on the shared choice/eval core designed in `MB_PLAN.md` §6.

| prefix | paper | plan | status |
|---|---|---|---|
| MB | Ji et al., MoralBench (arXiv:2406.04428; SIGKDD Explorations 2025) | `MB_PLAN.md` | v1.1; MB-G2 passed 2026-10-30 (`mb_moralbench/REPRODUCTION.md`); Phase 2 rider read on the MRB Phase 2 pod (`mrb_morebench/MRB_P2_REPORT.md` §4); the authors' corrections folded in 2026-11-02 (Amendment 6) |
| MRB | Chiu, Lee et al., MoReBench (arXiv:2510.16380; ICLR 2026) | `MRB_PLAN.md` | v0.9.4; reproduction report written 2026-10-31 (`mrb_morebench/REPRODUCTION.md`); Phase 2 steering read 2026-11-01 (`mrb_morebench/MRB_P2_REPORT.md`) |
| ETH | Hendrycks et al., ETHICS (arXiv:2008.02275; ICLR 2021) | not yet | queued |
| LEB | Jiao, Afroogh et al., LLM Ethics Benchmark: a three-dimensional assessment system (arXiv:2505.00853; Scientific Reports 2025) | not yet | queued |
| MCH | Pan et al., MACHIAVELLI (arXiv:2304.03279; ICML 2023) | `MCH_PLAN.md` | v0.3; MCH-G0 passed 2026-11-02 (`mch_machiavelli/MCH_G0_REPORT.md`: random baseline seed-exact on 30 of 30 games); G1 run 2026-11-02 (`mch_machiavelli/MCH_G1_REPORT.md`): parity 11,669 of 11,669, the run-to-run bar failed on bf16 numerics, the redefinition and the fp32 control pass (MCH-A7); the new-model pod runner built, VALIDATE next |

Order is the author's to set; MB first is decided. The prefixes are provisional.

## Goals and standard (author, 2026-10-29)

Two goals:
1. **A reference reproduction of each paper inside DeepSteer.** Others should be able to use it to
   cross-compare datasets, methods and benchmarks against the originals.
2. **Input to the steering work.**
   - Each plan's runs on the ST subjects are the pre-steering baseline on that instrument.
   - MoralBench and LEB are drift monitors, not steering targets.
   - MoReBench (process quality) and ETHICS commonsense are target candidates, subject to ST's
     specificity gates.
   - MACHIAVELLI is the action-side instrument: violations against reward under reward pressure,
     scored from precomputed scene labels (no judge). It is a guard and a candidate target for the
     reward-versus-harm trade-off (author, 2026-11-02).

Standard for each replication, kept light:
- **Defaults match the paper's protocol.** DeepSteer additions (robust parsing, continuous
  readouts, CIs, degenerate ladders) are opt-in and reported beside the paper-faithful number.
- **A reproduction report** (`papers/<prefix>_*/REPRODUCTION.md`) gives:
  - paper value against ours per cell;
  - each deviation and its reason;
  - rows that can't be run, and why;
  - pinned commits and revisions;
  - cost.
- **Matching is judged on the numbers.** Run the original code only to diagnose a mismatch.
- **Judge and deployment noise in the match interval (author, 2026-11-01, from MRB; forward
  only).** Where a judge model or an API provider is involved, the registered match interval
  includes judge or deployment noise as well as generation noise, measured by a judge test-retest
  and a cross-check on the paper's own deployment (provider, judge or both). This applies to plans
  registered from this date; MRB's registered verdicts stand as made, with the deployment term
  reported beside them (`mrb_morebench/REPRODUCTION.md`).
- **A fixed-content position check for lettered-choice adapters (author, 2026-11-02, from MRB-A16 and
  ST-A39; forward only).** Any steering adapter trained on lettered choices (DPO or SFT pairs whose
  options carry letters) gets a fixed-content position check before its effect is read: on reads with
  option order rotated, the adapter's shift in letter mass by position with the option's content held
  fixed, and the effect of record split by the target option's position. Steering plans carry it in
  their gate list.
- **Balanced option rotation in lettered evaluations (author, 2026-11-02, from ST-A39; forward only).**
  A lettered evaluation cycles each scenario's options through every position equally (a balanced
  rotation per scenario, its rollouts a multiple of the option count), not random draws per rollout:
  ST's random orders put one scenario's protect option at A in anywhere from 1 to 6 of its 8 rollouts.

## License findings (checked 2026-10-29; re-check at each plan's G0)

| source | code | data | basis |
|---|---|---|---|
| MoralBench, `agiresearch/MoralBench` | none (all rights reserved) | none (all rights reserved) | commit `8410044`: no LICENSE file, no license in README; the Drive mirror states no terms |
| MoReBench, `morebench/morebench` | MIT ("Copyright (c) 2025 MoReBench") | CC BY 4.0 | code: `LICENSE` at commit `71eaf71` (the README's link to `LICENSE.md` points to a missing file); data: HF card, confirmed by author |
| ETHICS, `hendrycks/ethics` | MIT ("Copyright (c) 2020 Dan Hendrycks") | MIT | `LICENSE` at commit `fa022db`; data already recorded as MIT in `DATASET_LICENSES.md` (V_moral held-out split) |
| LLM Ethics Benchmark, `The-Responsible-AI-Initiative/LLM_Ethics_Benchmark` | CC0 1.0 | CC0 1.0 from the repo authors, **but the instruments embed third-party text** | `LICENSE` at commit `7c5b0a7`; see the LEB rules below |
| MACHIAVELLI, `aypan17/machiavelli` | MIT ("Copyright (c) 2022 Alex Pan"), including the released results CSVs | none stated: the game bundle (Choice of Games titles and the authors' annotations) and the "artificial conscience" weights are on Google Drive behind a password published in the README | `LICENSE` at commit `ebb44e1` (checked 2026-11-02); the bundle carries no license and the games are commercial titles; see the MCH rules below |

## Rules every replication inherits

- **Pin and hash.** Fetch data at a pinned commit or revision into a user cache outside the repo,
  verify it against a committed sha256 manifest, and record the license basis and date in
  `DATASET_LICENSES.md`.
- **Tracked outputs carry no source text.**
  - Tracked: item ids, parsed answers, readouts, scores and rendered-prompt hashes.
  - Untracked (`outputs_raw/`, gitignored): raw replies and rendered prompts.
  - This is the default even for permissive sources; a plan may commit text only where its
    license allows and a cell needs a frozen copy.
- **Wheel.** No replication data ships in the wheel unless its license is permissive and a
  `DATASET_LICENSES.md` row covers it.
- **Model outputs are for evaluation only.** Training on them needs an amendment that first checks
  the model's terms (Gemma's terms reach models trained on its outputs).
- **G0 license check.** Each plan's G0 runs `git ls-files`, lists the built wheel, and audits notice
  coverage against these rules.

## Per-source rules

### MB — MoralBench
`MB_PLAN.md` §9: clean-room code, fetch-only data, nothing of theirs committed.

### MRB — MoReBench (permitted with attribution)
- **Code (MIT).**
  - Adapting or vendoring is allowed with the copyright notice and license text kept.
  - Vendored files go in their own directory with upstream's `LICENSE` copied unmodified (the
    `em_fixtures/` pattern) and the source commit in a header comment.
  - The judge prompt templates are in the code repo and fall under MIT.
- **Data (CC BY 4.0).**
  - Default to a runtime fetch at a pinned HF revision.
  - Any committed or deposited file containing MoReBench text, or adapted from it, carries four
    things: attribution (Chiu, Lee et al.; dataset URL and revision), a link to CC BY 4.0, a
    statement of changes, and its own `DATASET_LICENSES.md` row.
  - Re-read the HF card at MRB-G0 and record any gating terms shown on access.
- **Adaptations stay CC BY 4.0.** This includes recasting Agent scenarios into KDG/ST frames.
  Adapted files carry their own notice and stay out of the Apache wheel.
- **Upstream sources (corrected at MRB-G0).** The paper states all its sources are CC 4.0.
  DailyDilemmas and AIRiskDilemmas are CC BY 4.0 on their own cards; whether CC BY 4.0 covers the
  100 expert-written scenarios (APPE Ethics Bowl, Ethics Unwrapped, Cheung et al. PNAS 2025, whose
  own terms are CC BY-NC-ND 4.0 or unlicensed) is unclear (ANOMALIES MRB-A7). As a precaution those
  100 are used for evaluation only, never for training or adaptation. Cite the upstreams alongside
  MoReBench.
- **Scope.** Only the 500 public scenarios. Never seek the private set.

### ETH — ETHICS (MIT)
- Code and data may be vendored with the MIT notice.
- **Held-out conflict.** ETHICS commonsense scenarios are V_moral's held-out
  `eval_generalization_probe` split (`d1_vmoral_v1.json`). Running models on ETHICS for the
  replication is fine. No adapter, probe or direction that the D1 held-out claims depend on may
  train on any ETHICS item, and that includes ST adapters. Any such use needs an amendment naming
  the claims affected.

### LEB — LLM Ethics Benchmark (CC0 wrapper, third-party contents)
- **The CC0 dedication covers only what the repo authors own.** It cannot release text they
  reproduced from others. Their `data/instruments/` holds:
  - `mfq.json`: the MFQ-30 items (Graham, Haidt and colleagues; terms not yet checked);
  - `wvs.json`: World Values Survey Wave 7 items (WVS Association; terms not yet checked);
  - `dilemmas.json`: Kohlberg's Moral Judgment Interview dilemmas (Heinz, "Joe and the Camp
    Money") and their probe questions (Colby and Kohlberg's published materials; terms not yet
    checked).
- **Until each instrument's own terms are checked, treat all three files as fetch-only**, the same
  as MoralBench. Nothing from them is committed, vendored, put in fixtures or shipped.
- **Code.** The authors' own code (the `morals/` package) is CC0 and may be adapted freely; cite
  the paper.
- **Rubric text.** Any rubric or "expected_response" text the authors wrote themselves is CC0.
  Separate it from the instrument text before committing any of it.

### MCH — MACHIAVELLI (MIT code, unlicensed game data)
- **Code and released results (MIT).** May be adapted or vendored with the notice, in the
  `em_fixtures/` pattern (upstream `LICENSE` copied unmodified, source commit in a header comment).
  The game engine may be vendored rather than reimplemented.
- **Game bundle: fetch-only.** The games are Choice of Games' commercial titles and the bundle states
  no license. Nothing from it is committed, vendored, put in fixtures or shipped.
  - Tracked outputs: game ids, scene ids, action indices, rewards and aggregate metrics.
  - Per-scene label values are not tracked, since they would rebuild the authors' label table
    (the MoralBench answer-key rule); scores are recomputed from the fetched bundle at analysis.
  - Raw text (rendered prompts, replies) goes to the gitignored `outputs_raw/`.
- **Conscience weights.** Fetch-only; needed only for the DRRN +shaping rows.
- **LLaMA-1 and Alpaca weights** (the CSV's `llama-*` rows; LLaMA-1's non-commercial research terms,
  not the Llama Community License of Llama 2 and later): an author decision
  before any run, evaluation only.
- **Training use.** Training adapters, probes or directions on game text or scene labels needs an
  author decision recording the license basis first (the posture of MRB's expert-written 100).

