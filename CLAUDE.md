# CLAUDE.md — DeepSteer Project Instructions

## What This Project Is

DeepSteer is a PyTorch-native toolkit for evaluating and steering alignment depth in LLM pre-training. It measures how deeply moral reasoning is embedded in language models, distinguishing shallow post-hoc alignment (RLHF, Constitutional AI) from deep pre-training alignment. The primary focus is **base (non-instruct) models** — representational probing reveals what models learn during pre-training, before instruction tuning modifies their representations.

The library targets three model access tiers:
- **API** (Claude, GPT): behavioral evaluations only (requires instruction-tuned models)
- **Weights** (OLMo, Llama): representational probing (base models preferred) + behavioral (instruct only)
- **Checkpoints** (OLMo): + training trajectory analysis (base models)

## Architecture Overview

```
deepsteer/
├── core/                  # Types, model interface, benchmark runner
│   ├── model_interface.py # WhiteBoxModel, APIModel, ModelFamily, architecture detection
│   └── moe_model.py       # MoEWhiteBoxModel for OLMoE expert/router analysis
├── foundations.py         # Canonical MFT constants (FOUNDATION_ORDER, groups)
├── directions/            # Direction extraction (mean-diff, LEACE, probe-weight)
├── geometry/              # Geometric analysis (cosine, clustering, subspace)
├── causal/                # Causal validation (ablation, steering, behavioral)
├── benchmarks/            # Evaluation implementations
│   ├── moral_reasoning/   # MoralFoundationsProbe (API tier)
│   ├── compliance_gap/    # ComplianceGapDetector (API tier)
│   └── representational/  # LayerWiseMoralProbe (weights tier)
├── datasets/              # Probing datasets and generation pipeline
├── viz/                   # Matplotlib visualization functions
├── steering/              # Training-time intervention tools
└── outputs/               # Untracked output viz and matching JSON
```

Read `ARCHITECTURE.md` for the full design rationale.
Read `deepsteer/datasets/PROBING_PIPELINE_DESIGN.md` for the probing dataset pipeline.

## Methodology Skills

[#methodology-skills](#methodology-skills)

Validity-gating skills live in `.claude/skills/`. Each is a `<name>/SKILL.md`
with trigger conditions in its frontmatter; Claude Code auto-discovers them.
Consult the relevant skill *before* the work it governs. These are not optional
for the situations they name — a silently skipped calibration check is worse
than a stall.

| Skill                   | Consult before                                                                                          | Read first? |
| ----------------------- | ------------------------------------------------------------------------------------------------------ | ----------- |
| `construct-audit`       | comparing any two directions/subspaces: cosine, projection fraction, principal angle, CKA, steering, ablation; any "X is orthogonal to Y" claim | For direction/geometry work |
| `instrument-calibration`| reporting, accepting, or gating on any null / orthogonality / below-threshold result                   | For any NULL verdict |
| `intervention-validity` | building or interpreting any patching, interchange, ablation, steering, or noise-injection experiment  | For causal work |
| `estimator-traps`       | any CI, bootstrap, threshold, extremum, or evidence-combination verdict                                | For any stats verdict |
| `compute-ordering`      | planning any experiment sequence, phase plan, or A100 session — enforces zero-GPU-first ordering       | When planning runs |
| `anomaly-triage`        | writing any results/discussion section; at every phase end and human gate — maintains `ANOMALIES.md`   | For write-ups |
| `program-thesis`        | committing any results, README, abstract, framing, or verdict wording — maintains `SYNTHESIS.md`       | For framing/prose |

If a skill's `SKILL.md` is missing or its trigger is ambiguous, stop and flag it
rather than proceeding without the check.

## Code Conventions

### Style
- Commits should avoid the "Co-authored by Claude line"
- Python 3.10+, type hints everywhere
- Use `from __future__ import annotations` in every file
- Line length: 100 chars (set in pyproject.toml via ruff)
- Docstrings: Google style, on all public functions and classes
- Logging: use `logging.getLogger(__name__)`, never `print()` in library code
- `print()` is fine in example scripts and CLI entrypoints

### Prose Style (papers and documentation)
- Minimize em-dashes. Use commas, semicolons, colons, parentheses, or separate sentences instead. One or two per section is fine; five is too many.
- Avoid these AI-tell words entirely: "underscores", "landscape", "paradigm", "nuanced", "multifaceted", "tapestry", "holistic", "cutting-edge", "at the forefront", "a testament to"
- Prefer "shows" or "finds" over "reveals"; "use" over "leverage" or "utilize"; "important" over "crucial" or "critical" (unless literally about criticality)
- Replace vague intensifiers ("meaningful", "key insight", "key takeaway") with specific claims
- Don't hedge with "This suggests" or "It is worth noting that"; state the conclusion directly
- Don't open sentences with "Importantly,", "Notably,", "Crucially,", "Interestingly,", or "Specifically,"
- Write like a careful human researcher, not a summarizer. Prefer short direct sentences over long ones joined by dashes.

### Design Principles
- **Hooks on real models, not reimplementations.** Never write a custom transformer. Use PyTorch forward hooks on actual model modules from HuggingFace.
- **Benchmark-first, infrastructure-second.** A working evaluation that produces a novel number is more valuable than a beautiful abstraction with no results.
- **Accelerate everything.** Algorithms should be easily accelerated on GPU and NPU including MPS for Apple Silicon and CUDA for Nvidia GPUs through PyTorch, including torch.compile in the future.
- **Tiered graceful degradation.** Every benchmark declares `min_access_tier`. The `BenchmarkSuite` automatically skips benchmarks the model can't support. Never error on a tier mismatch — skip and log.
- **Reproducible and publishable.** Every evaluation produces structured JSON with full metadata (model name, timestamp, hyperparameters, dataset version). Someone should be able to reproduce results from the JSON alone.
- **Composable.** Benchmarks are independent units. Never create dependencies between benchmarks. A researcher should be able to run just one benchmark.
- **Tool and AI-friendly output.** Visualization output should always be 1:1 matched with JSON output easily ingested into other tools. See "reproducible" above.

### Dependencies
- Core: `torch`, `transformers`, `datasets`, `numpy`, `matplotlib`, `seaborn`, `pandas`, `tqdm`
- API: `anthropic`, `openai` (optional, only needed for API tier)
- Always use `pip install --break-system-packages` in this environment
- Minimize new dependencies. Prefer stdlib, torch, and numpy over heavy libraries.

### File Organization
- One concept per file. If a file exceeds ~400 lines, consider splitting.
- Every directory has an `__init__.py` that re-exports public symbols.
- Tests mirror source structure: `tests/benchmarks/test_foundations.py` tests `deepsteer/benchmarks/moral_reasoning/foundations.py`

### Testing
- Use `pytest` with fixtures for mock models
- For white-box tests that need a real model, use a tiny model (OLMo-1B or a 125M param model) and mark with `@pytest.mark.slow`
- For API tests, mock the API client — never make real API calls in CI
- Every benchmark should have at least one test that runs on synthetic data and verifies the output schema (correct keys, reasonable value ranges)
- The hand-written probing dataset in `_build_probing_dataset()` serves as the test fixture for probing tests

### What NOT To Do
- Do not reimplement transformer architectures. Use HuggingFace models with hooks.
- Do not add `torch.compile` or mixed precision — keep it simple for now. Optimization later.
- Do not create a web UI or dashboard. Matplotlib plots saved to disk are sufficient for Phase 1-4.
- Do not add MLflow, wandb, or other experiment tracking. JSON files are the tracking system.
- Do not train or fine-tune models. DeepSteer is an evaluation and analysis toolkit. Steering happens through data curation and curriculum design, not through gradient updates within this library.
- Do not hardcode model paths or API keys. Use constructor arguments and environment variables.
- Do not write overly defensive code with excessive try/except blocks. Let errors propagate with clear messages. The user is a researcher, not an end consumer.

## Key Files to Understand

| File | Purpose | Read first? |
|---|---|---|
| `core/types.py` | All dataclasses and enums | Yes |
| `core/model_interface.py` | Model abstraction, ModelFamily, architecture detection | Yes |
| `core/moe_model.py` | MoEWhiteBoxModel for OLMoE expert/router analysis | For MoE work |
| `core/benchmark_suite.py` | Benchmark base class + suite runner | Yes |
| `foundations.py` | Canonical MFT constants (FOUNDATION_ORDER, groups) | Yes |
| `directions/` | Direction extraction (mean-diff, LEACE, probe-weight) | For direction work |
| `geometry/` | Geometric analysis (cosine, clustering, subspace) | For geometry work |
| `causal/` | Causal validation (ablation, steering, behavioral) | For causal work |
| `benchmarks/moral_reasoning/foundations.py` | MoralFoundationsProbe implementation | For behavioral work |
| `benchmarks/compliance_gap/greenblatt.py` | ComplianceGapDetector implementation | For behavioral work |
| `benchmarks/representational/probing.py` | LayerWiseMoralProbe implementation | For white-box work |
| `viz/__init__.py` | All visualization functions | When producing plots |
| `datasets/PROBING_PIPELINE_DESIGN.md` | Full design for dataset generation | For Phase 1 |

## Running Things

```bash
# Install in dev mode
pip install -e ".[all]"

# Run tests
pytest tests/ -v

# Representational probing on OLMo base (primary use case)
python scripts/run_evaluation.py --model olmo --output-dir outputs/

# Representational probing on Llama base
python scripts/run_evaluation.py --model llama --output-dir outputs/

# Include behavioral evals (requires instruction-tuned model)
python scripts/run_evaluation.py --model olmo --behavioral \
    --weights allenai/OLMo-7B-Instruct-hf --output-dir outputs/

# Behavioral evals on an API model
python scripts/run_evaluation.py --model claude --model-id claude-sonnet-4-20250514

# Checkpoint trajectory analysis
python scripts/run_evaluation.py --model olmo --weights allenai/OLMo-7B-hf \
    --checkpoint-revisions step1000-tokens4B step5000-tokens21B
```

## Context: Research Goals

This toolkit supports research into whether embedding moral reasoning during LLM pre-training produces deeper, more robust alignment than post-hoc methods (RLHF, Constitutional AI). The key hypothesis: pre-training aligned models will show:

1. Moral concepts decodable from **earlier layers** (lower `moral_encoding_depth`)
2. Moral encoding **distributed** across the network (higher `moral_encoding_breadth`)
3. **Zero compliance gap** between monitored and unmonitored conditions
4. **Greater resistance** to adversarial moral scenarios (smaller `depth_gradient`)

OLMo is the primary target because Ai2 publishes intermediate training checkpoints, enabling trajectory analysis that no other frontier-adjacent model allows.


## Research boot sequence (before any planning, code, or answer)

1. Read: `papers/SYNTHESIS.md` (thesis + standing claims), `papers/ANOMALIES.md` (open
   entries), `MISSING_ARTIFACTS.md`, and the last ~10 commits (`git log --oneline -10`)
   plus the most recent RESULTS or amendment delta.
2. Open the session with a five-line **program state**: current thesis sentence; verdicts
   pending; top open anomalies with their cheapest discriminators; what the last session
   banked; what today's work gates.
3. Any statement in (2) that cannot be grounded in a file gets flagged as ungrounded —
   never silently reconstructed from memory.

Rationale: sessions are stateless; the external reviewer's advantage is accumulated
context. This recreates it mechanically, every time.

## Skill consultation points (mandatory, consult = apply the required artifact)

- Any experiment/phase/session plan → `compute-ordering`.
- Creating or comparing any direction/subspace/probe → `construct-audit` (type block with
  `participation_ratio` + `outcome_variable` is required metadata).
- Any patch / ablation / steering / attribution cell → `intervention-validity` (spec block
  committed in the prereg before the pod).
- Any CI, threshold verdict, bootstrap, or estimate comparison → `estimator-traps`.
- Any NULL/negative/orthogonality verdict → `instrument-calibration` (no NULL without a
  ladder, a positive control, and position validity).
- Anything "unexpected / interesting / caveat / exception" → `anomaly-triage` (ledger
  entry before the caveat sentence).
- Every human gate and every RESULTS commit → `program-thesis` (referee pass + SYNTHESIS
  update in the same commit).

## Standing inference moves (apply continuously — these are the job, not flourishes)

1. **SECOND DERIVATION.** Every headline scalar gets one independent estimate — closed
   form, a different artifact, or dimensional analysis — reported next to it with
   agree/disagree. (Canonical: sqrt(3/PR) predicting the rank-3 null median; sqrt(2/π·d)
   channel chance closing the Qwen anomaly; the 34% + 76% ≈ 110% additivity read.)
2. **RIVAL READING.** No verdict ships without the strongest alternative reading that fits
   the same data, written down, plus the cell that separates them. If no separating cell
   exists, the verdict is downgraded to a reading. (Canonical rivals that each changed a
   verdict: dose/under-transfer vs reads-elsewhere; richer-harm-percept vs reads-broad;
   saturation vs content-robustness.)
3. **BLAST RADIUS.** When any instrument, control, or assumption is invalidated, enumerate
   every prior verdict it ever gated *before doing anything else*, and propagate
   voids/scope notes to RESULTS + SYNTHESIS in the same commit. (Canonical: latched cells
   → voiding the 0.44/0.14 directional hint; the persona-control audit reopening G3
   wording.)
4. **COINCIDENCE INTERROGATION.** Any reported ratio, near-equality, sign flip, or
   point-outside-its-CI gets asked: what does the null / simplest model predict here?
   (Canonical: the 26% restricted/full proportionality → ratio-of-ratios; band-min 0.65 ∉
   [0.47, 0.64] → attenuation.)
5. **REFRAME BEFORE CAVEAT.** When something reads "degenerate / broken / limited," first
   attempt one honest paragraph where it is a mechanism or finding; keep whichever
   survives. (Canonical: degenerate position → measured decision-site bottleneck; failed
   Llama cells → A5 dynamic-range finding; A1 saturation → methods-note spine.)
6. **ONE-ROOT DIAGNOSIS TREE.** Any "why did X fail" becomes a pre-registered decision
   tree whose root split is one number computable from saved data — never a grab-bag of
   probes. (Canonical: Amendment 7 — judgment-delta coherence as the root.)
7. **POSITIVE VOICE AT SYNTHESIS.** After any result lands, write the program's thesis
   sentence in positive voice before writing limitations. Nulls accumulate; someone must
   keep stating what the program now claims.

## Hard gates

**Pod-boundary (no GPU session without):** power table (MDE at candidate n from *measured*
variance — power is computed, not guessed); both branch framings pre-registered and
publishable; an explicit bail condition (screen → gate → proceed/bank); per-unit artifact
save list; dependency check (data the session needs exists and is committed).

**Commit-boundary (no RESULTS commit without):** referee pass (3 damaging objections,
answered or conceded) and SYNTHESIS.md update in the same commit; verdict sentences that
carry the ladder; null wording that embeds detection bars ("no coupling detectable at
|cos| ≳ 0.5", never bare "dissociation"); anchored adjectives only.

**Verdict-boundary (no verdict without):** a positive control on the same instrument, same
model; the rival reading (move 2); type blocks on every object involved; position validity
(band-below-null check, PR recorded).

**Amendment discipline:** dated, committed, and pushed BEFORE any recompute it licenses.
Post-hoc analysis-choice changes are forks: amendment + verdict under both choices, never
in the same commit as the results they affect.

**Local tests:** every harness local test asserts its single most-probable failure mode by
name in a comment (e.g., "assert chat null consumes chat act_samples"), not just that the
script runs.

## Pre-review protocol (the load-bearing habit)

Before presenting ANY gate, decision menu, results summary, or plan to Orion: draft the
**anticipated review** — the 3–5 riders a hostile expert reviewer would attach. Implement
the zero-GPU ones immediately; attach the rest with costs. Never present a bare option
list.

```
Anticipated review:
1. [observation in the data] → [rule/move it triggers] → [concrete change] (cost)
2. ...
Implemented now: ...
Open (with costs): ...
Question behind the question: [what the decision actually depends on that wasn't asked]
```

Quality bar: at least one rider should be something you did not already plan to do. If
every rider is a restatement of the existing plan, the review pass failed — run it again
against a different attack surface (instrument validity, statistics, framing, sequencing).

## Escalate to the author — do not decide alone

Thesis-level reframes; voiding or scoping any committed/published claim; packaging and
publication decisions; anything adjacent to the safety scope (removability, coupling
robustness as an optimization target); cross-paper restructuring; any amendment changing a
pre-registered PRIMARY. When unsure whether a decision is in this class, it is.

## Numerical hygiene defaults

Bootstrap CI with every headline number; per-pair / per-rollout / per-head arrays saved by
default; seeds fixed and logged; harness + classifier versions pinned in artifact
metadata; σ provenance typed (format + position class); reconstruction and specificity
reported separately for any decomposition.

## Exemplar pointers (templates, by artifact)

- Diagnosis tree: d3 `PREREGISTRATION.md` Amendment 7 (root split → branches → escalation
  menu gated on branch).
- Bug-report-as-finding: `papers/ANOMALIES.md` A5 (and A1, A2).
- Calibrated ladder + position validity: `papers/d1_moral_subspace/CALIBRATION_RESULTS.md`.
- Verdict reclassification done right: the ratio-of-ratios commit (under_transfer
  reclassification) — caught, named ("MDE tightened past a ~constant effect"), retracted,
  panel held.
- Futility catch: the Amendment 6 power table (a pod prevented by an afternoon of saved-
  array work).
- Both-branches prereg: Amendment 6 §4 (three shape verdicts, all publishable).

## Honest scope

This layer plus the skills covers: context recovery, the named inference moves, gate
discipline, instrument validity, claim language, and sequencing. It does not cover:
strategic packaging judgment, thesis synthesis across long horizons, or statistical traps
outside the named patterns. Those remain scheduled external review (Orion + the review
layer). The pre-review protocol narrows that residue; it does not close it. When a
pre-review pass produces nothing surprising twice in a row on high-stakes decisions, say
so — that is a signal to request external review, not evidence it isn't needed.