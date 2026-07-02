# DeepSteer Research Skills — v2 (2026-07-02)

Seven skills encoding review-level checks for the DeepSteer program, for Claude Code
(Opus 4.8) and the Claude desktop app. v1 was distilled from a Fable-5 program review;
v2 folds in four live sessions of usage — every rule added below was earned by a real
catch or a real miss, with the numbers preserved as case studies. Design premise
unchanged: the observed gaps are not knowledge gaps but **failures to invoke a known
check unprompted at the right moment**, so each skill couples pushy trigger descriptions
with **required artifacts** (ladders, type blocks, ledgers, spec blocks) whose absence is
mechanically visible in review.

## What's in v2

| skill | v2 delta | earned by |
|---|---|---|
| `instrument-calibration` | band-below-null position-validity tell; PR beside every ladder; rung co-location; intervention-instrument calibration | the cross-model decision-site bottleneck (PR 14.7/8.6/10.2) caught by the positive-control band |
| `construct-audit` | type block gains `participation_ratio` + `outcome_variable`; stimulus–outcome matching; co-location well-posedness | narrative-vs-request twins; R2/G3 not well-posed across positions |
| `estimator-traps` | traps 9–11: massive-activation null saturation (+ σ provenance, invariance proof); channel-relative chance (E\|cos\| ≈ sqrt(2/πd)); named-reason rule for null drift | Qwen/Llama saturation (59%/32% single dims); Qwen R3 closure at channel-chance |
| `anomaly-triage` | archetype 6 (bug-report-as-finding); `resolution_type` incl. calibration closures; merge rule for twice-flagged entities | two degeneracies became a methods note; Qwen 0.32 closed by chance model |
| `program-thesis` | rules 7–11: post-hoc fork amendments, commit-boundary ship-blockers, intervenability check on immutability words, detection-bar verdict wording, citation verification; packaging principle | the mean_content preference fork; "architecturally guaranteed" reframe; the C1 lit pass |
| `compute-ordering` | novelty-gating lit pass in the zero-GPU layer; clean-checkpoint session discipline | the pre-build Sahara catch |
| `intervention-validity` | **NEW** — spec block for causal cells: baseline discrimination, transport positive controls, channel-matched attribution specificity, ablation semantics, alignment rules, harness parity | the C1 twin-patch mismatch + degenerate negative branch fell between the existing skills |

## Install — Claude Code (project-scoped, versioned with the repo)

```bash
unzip deepsteer-opus-skills-v2.zip -d /path/to/deepsteer/   # replaces .claude/skills/
git add .claude/skills && git commit -m "Research skills v2 (+intervention-validity)"
```

Global alternative: place the seven folders in `~/.claude/skills/`. Verify with `/skills`.

## Install — Claude desktop app (macOS) / claude.ai

Each skill is also packaged as a `.skill` file. In a chat, the presented `.skill` card
shows a **Save skill** button — clicking it installs the skill to your profile, where it
is available to Claude in the desktop app and on the web (Settings → Capabilities to
manage). Install at minimum the review-layer four for the conversation surface where you
paste Claude Code results for review: `program-thesis`, `anomaly-triage`,
`estimator-traps`, `instrument-calibration`. The execution-layer three
(`construct-audit`, `intervention-validity`, `compute-ordering`) matter most inside the
repo, but installing all seven keeps the two surfaces consistent.

## Wiring into the workflow (CLAUDE.md block — updated for v2)

```markdown
## Research methodology skills (mandatory consultation points)
- Before writing any experiment/phase/session plan → `compute-ordering`.
- Before extracting or comparing any direction/subspace → `construct-audit`
  (type blocks with participation_ratio + outcome_variable are required metadata).
- Before designing or running ANY patch/ablation/steering/attribution cell →
  `intervention-validity` (spec block committed in the prereg before the pod).
- Before any CI, threshold verdict, or bootstrap → `estimator-traps`
  (top-dim variance share + channel-chance line are mandatory).
- Before writing any NULL/negative verdict → `instrument-calibration`
  (no NULL ships without a ladder, a positive control, and position validity).
- When any result is "unexpected", an exception, or a near-miss → `anomaly-triage`.
- At every human gate and in the SAME COMMIT as any RESULTS document →
  `program-thesis` (referee pass + SYNTHESIS.md update are commit-boundary blockers).
```

## Interlock

plan (`compute-ordering`) → build objects (`construct-audit`) → design causal cells
(`intervention-validity`) → measure and judge (`instrument-calibration`,
`estimator-traps`) → interpret (`anomaly-triage`) → frame and gate (`program-thesis`) →
promoted discriminators feed the next plan.

## Maintenance

Living documents: when a new gap class appears, add it as a numbered rule/trap/archetype
**with the concrete instance and numbers** — the case studies are what make a check
recognizable in-context rather than merely recitable. The v1→v2 delta is the loop working:
the band caught the bottleneck (skill fired), the twin-patch mismatch fell between skills
(coverage hole → new skill). In Claude Code, `skill-creator`'s description-optimization
loop can tune trigger rates once usage accumulates.

## Honest scope note

What transfers: calibration protocol, object typing, causal-cell design, estimator
hygiene, anomaly promotion, claim discipline, sequencing. What only partially transfers:
cross-context conjunction-spotting and design-validity judgment at interfaces no schema
yet names. The v2 additions narrow that residue; they do not close it. Periodic
whole-program review by a stronger reviewer remains a scheduled function, not a skill.
