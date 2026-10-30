# Moral and ethical benchmark replications — index and license rules

Status: 2026-10-29 (author). One index for the external-benchmark replications. Each gets its own
plan when it starts; this file holds only the order, the license findings, and the handling rules
that the plans inherit. They all build on the shared choice/eval core designed in `MB_PLAN.md` §6.

| prefix | paper | plan | status |
|---|---|---|---|
| MB | Ji et al., MoralBench (arXiv:2406.04428; SIGKDD Explorations 2025) | `MB_PLAN.md` | v0.5; MB-G0 passed 2026-10-30, MB-G1 next |
| MRB | Chiu, Lee et al., MoReBench (arXiv:2510.16380; ICLR 2026) | not yet | after MB |
| ETH | Hendrycks et al., ETHICS (arXiv:2008.02275; ICLR 2021) | not yet | queued |
| LEB | Jiao, Afroogh et al., LLM Ethics Benchmark: a three-dimensional assessment system (arXiv:2505.00853; Scientific Reports 2025) | not yet | queued |

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

## License findings (checked 2026-10-29; re-check at each plan's G0)

| source | code | data | basis |
|---|---|---|---|
| MoralBench, `agiresearch/MoralBench` | none (all rights reserved) | none (all rights reserved) | commit `8410044`: no LICENSE file, no license in README; the Drive mirror states no terms |
| MoReBench, `morebench/morebench` | MIT ("Copyright (c) 2025 MoReBench") | CC BY 4.0 | code: `LICENSE` at commit `71eaf71` (the README's link to `LICENSE.md` points to a missing file); data: HF card, confirmed by author |
| ETHICS, `hendrycks/ethics` | MIT ("Copyright (c) 2020 Dan Hendrycks") | MIT | `LICENSE` at commit `fa022db`; data already recorded as MIT in `DATASET_LICENSES.md` (V_moral held-out split) |
| LLM Ethics Benchmark, `The-Responsible-AI-Initiative/LLM_Ethics_Benchmark` | CC0 1.0 | CC0 1.0 from the repo authors, **but the instruments embed third-party text** | `LICENSE` at commit `7c5b0a7`; see the LEB rules below |

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
- **Upstream sources.** The scenarios derive from DailyDilemmas, AIRiskDilemmas, Ethics Unwrapped
  and APPE Ethics Bowl cases; the paper states these are CC 4.0. Rely on the card and cite the
  upstreams alongside MoReBench.
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
