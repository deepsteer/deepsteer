# GPC state (one page; updated at every stop; `GPC_PLAN.md` versions are the history)

Scope: goal pressure, conduct moved into weights with nothing in context, one subject.

**As of 2026-11-17, Amendment 2 run; GPC-3 read and accepted by the author (`GPC3_RESULTS.md`).**
G5 (exploratory): in blackmail with an explicit goal and a replacement threat, base 0.38, the clause
in context 0.02, C1 0.04, acting 1.00 in all three (one cell, n 50, one draw, one judge; the string
check 0.58, 0.06, 0.12; Codex agrees on 22 of 30, stricter on leverage, GPC-A11). G1: the clause's
content is the limit on game states (GOOD −0.28 where the clause −0.084); the unreworded clause in the
native harness 0.48 of GOOD, C1 0.00. G2: the CoT steps carry the harness's push. G3: GOOD −0.084,
P-ctx −0.019 on 1,093 gap states. G4: C1 × 2.0 reproduces (−0.120, about 0.40 of GOOD; clean slots
0.88). The registered game bar is not met. **Next:** Amendment 3 (GPC-4), drafted for the author's
ruling; the target moves to agentic instruments.

## Rules of record (carried from CST, unchanged)

- **Subject:** Qwen3.8-27B@1d4bf0f2, thinking on, effort medium, cap 4,096, KDG sampler v2; vLLM
  0.31.0, MTP off, 128 sequences, prefix caching on.
- **The product bar** (CST Amendment 50 item 3), verbatim in `GPC_PLAN.md` §2. Two draws for a pass.
- **References of record:** the base of record is `teacher_v1`'s (0.373 on the 1,698 at k 2);
  GOOD_BEHAVIOR of record is the same pod's (−0.093 [−0.118, −0.068]).
- **Corpora:** no MACHIAVELLI text in any training set; the eval 8-gram index gates each by D11.
- **Pods:** the author launches; cut at 10.25 h, watchdog 11.0 h; Thunder with the RunPod Secure
  fallback; HF_TOKEN never on a command line. No Claude call reads a subject trace. From the next
  pod, the limits are editable during the run (`papers/runpod_common/set_limits.sh`; default
  10.25 h, hard ceiling 14.0 h; each change logged as a deviation). GPC-1 runs on fixed limits.
- **W-ref** is diagnostic only: its adapter is deleted on the pod and never downloaded, it is never
  the registered read's arm, and only its rows and train record come back (item 6).
- **C1g**, when item 11 selects it, is labeled "teacher not from the constitution" everywhere.
- **The partner is never named.** Nothing in GPC carries the partner's document.

## Instruments of record (v0.2)

- **Screen:** S-hi (250 from the 443-state pool; base 0.904, GOOD 0.602) and S-lo (100 from 877),
  `data/gpc_screen_sets_v1.json`. Arm at k 1 against the base of record's k 2; ratio to GOOD's drop
  on the same 250 (−0.302); **clear at a point ratio of 0.35**; S-lo guard, no rise detectable at
  about 0.06; the session check switches the reference (item 2); split by violation on offer
  (item 12).
- **Bar's power:** point ratio for a lower bound of 0.5 is 0.64 (arm tracks GOOD per state) to
  0.78-0.82 (independent per-game profile).
- **Mediator:** `scripts/gpc_mediator.py` (patterns committed `f1001574` before computing); the root
  split reads only if GOOD-alone raises the moral rate (item 4).
- **Item 11's switch:** `C_conduct` in GOOD's slot at an S-hi point ratio of 0.5 or more selects
  W-self, else C1g.
- **Conduct grid:** two instruction strengths (`stated`, `maximize`), balanced; the base-rate
  pilot (15 pressure slots per setting by strength cell at k 2) allocates the rest by cell in
  proportion to the base's violating rate, a cell under 0.15 getting none beyond its pilot; the
  teacher gate (CI excludes 0, at least 350 kept); the held-out cell of 100 (item 13).

## Pending

1. Amendment 3 (GPC-4): accepted by the author 2026-11-17 and committed (`GPC_PLAN.md` §11, the host
   page `ODCV_HOST_BRIEF.md`); the build at zero GPU, then the author's rehearsal, then the launch.
2. GPC-A11: the author's read of the 8 judge-yes, Codex-no transcripts (one question each).
3. GPC-A9's zero-GPU read (which replies each arm skips in the no-threat cells).
4. Deferred by the lead: a rebuilt W-self (GPC-A2).

## GPC-2 (Amendment 1; run 2026-11-15 22:17 PST to 02:36 PST, `7a9442e8`; read in `GPC2_RESULTS.md`)

Priority order X1, X2, X3, F, C2 (C2-div, and C2-mach as the format-matched diagnostic), B (B0 to
B3, B4 if B3 passes), A; batched by loaded model, cut from the end. X3 and C2 run only if format
gates. Floor about 3.25 A100-h ($3.90 on Thunder); ceiling inside one pod about 9.65 ($11.30 on
Thunder, $21.50 on Secure). The scaffold templates: `CONDUCT_SCAFFOLDS.md`. The earlier queue
(the fifth line, items A and B with their riders) is carried into the amendment.

## Next pod

**GPC-4** (Amendment 3, `GPC_PLAN.md` §11): accepted 2026-11-17. Floor about $7.25 (4.4 A100-h on Thunder
plus the VM), ceiling about $16.05 (the 9 A100-h cap plus the VM), the rehearsal about $0.65. Not built.

**GPC-3 (done):** 4.39 h of session on RunPod Secure, all stages run; G5's full design at 50 per
condition and arm on blackmail's grid (leaking and murder banked at the pilot). Outputs in
`outputs/gpc3_v1`; reads `scripts/gpc3_read.py` and `scripts/gpc3_rederive.py`.

**GPC-2 (done):** 4.23 h of session, all stages run; X3 and C2 skipped by the registered reading,
B4's steered arms by B3. Outputs in `outputs/gpc2_v1`; reads `scripts/gpc2_read.py` and
`scripts/gpc2_rederive.py`.

**GPC-1 (done):**

**GPC-1** (built; card `runpod/launch_gpc1_thunder.sh`, RunPod fallback `runpod/launch_gpc1.sh`):
about 6.3 A100-h through the screens, 9.2 h with the registered read and its riders:
$7.40-10.70 on Thunder, $14.10-20.40 on Secure at $2.19 (`GPC_PLAN.md` §5). Results in
`outputs/gpc1_v1`; the local read is `scripts/gpc1_read.py` (its first line flags any arm whose
fallback share on S-hi is above 0.02; arms clear on the without-fallback ratio).

## GPC-1 in flight (launched 2026-11-15 12:16 PST on Thunder, card at `e66c9284`)

**The mid-session swap (the author's go, one attempt, 2026-11-16T00:24:02Z = 2026-11-15 16:24:02
PST).** The eval 8-gram membership check (`np.isin` against the 6.3M-hash index, about 4.5 s a
call on Thunder's CPUs) replaced by `overlap_hashes.shared` (binary search, the same answer) in
`ec672abc`. The four files went to `.new` names in one call, each `.new` matched the local hash
and compiled under the pod's Python 3.11, and one command line moved all four over the originals.
`remote_gpc1.sh`, the launcher and every process were left alone.

| file | at launch (`e66c9284`) | after the swap (`ec672abc`) |
|---|---|---|
| `scripts/gpc1_pod.py` | `1c841e39…19437` | `ee943de2…d3b53` |
| `scripts/gpc_advice.py` | `eed1efdf…d316` | `0a6c238a…5f51` |
| `scripts/gpc_conduct.py` | `70021026…4a63` | `048a277a…0709` |
| `scripts/gpc_wref.py` | `f2c7a2af…7560` | `45022262…9b` |

**Stages by commit.** On `e66c9284`: setup, the gate tests, `smoke_train`, `smoke_serve`,
`incontext`, `read_b`, `conduct` (its process started 21:35:43Z, before the swap, and ran to its
end on the code it loaded). On `ec672abc`: every stage from `third` on (each a fresh process).
Verification after the swap: the next check stage (W-self's 401 dilemmas, 8-gram and
near-duplicate) took about 3 s (the dilemmas were written at 00:40:00Z and the advice generation
began about 00:40:03Z); W-ref's 8-gram drop: **6 of the seeded 1,000, the same count as Z5**
(its train stage, ended at 7.04 h; 819 rows trained, 797,441 assistant tokens against W-self's
795,451).

**Observed in flight (no action, the author's rules for the session).** `rate_watch.py` (CST's
5-minute rate side process) crashed at 5.13 h when the session entered training (a window with no
generation bar printed `None`), so no rate windows are logged after it; the stages and the cut rule
do not read them. Fix after the session. W-self's generation ran at 0.64 of its forecast tokens a
second (the last RATE FLAG), so the third stage ended at 5.13 h against about 4.9 h forecast; the
runner's check before f decides whether the registered read still fits.

## Pieces that graduate to core deepsteer (item 13; the library-first rule)

The rule (the author, 2026-11-04; `GPC_PLAN.md` §2): reusable utilities and methods move into core
`deepsteer/`, not papers/ scripts. Each piece moves after GPC-1 has run it on the pod, with its
tests, into the library module named here:

| piece | where it lives now | target |
|---|---|---|
| the single-decision replay harness (state replay with the prompt-hash gate, MCH's attempt rule, paired seeds, per-chunk resume) | `cst_teacher_pod.py` (`states`, `_render_episode`, `decide`, `stage_cells`), called by `gpc1_pod.py` | `deepsteer.benchmarks.agentic.machiavelli.replay` |
| the screen builder (a stratified state draw on independent rollouts, the ratio read, its power simulation) | `scripts/z2_z3_power.py`, `scripts/gpc1_read.py` | `deepsteer.benchmarks.agentic.machiavelli.screen` |
| the scenario generator with its base-rate pilot (the grid, the writer brief, the checks, the pilot's allocation, the held-out draw) | `scripts/gpc_conduct.py` | `deepsteer.datasets.scored_goal_scenarios` |

## Spend ledger (A100-80GB hours from each session log's clock, plus about 0.15 h a pod)

The $500 cap covers CST on public material and GPC together. **$173 remain** (CST_STATE,
2026-11-15) and are a hard stop. GPC pods are ledgered here and in CST_STATE's running total.

| date | pod | session h | billed h | $ | running h | running $ |
|---|---|---|---|---|---|---|
| 11-15 | GPC-1 on Thunder (A100 80GB, $1.15 logged; all stages run, done at 7.90 h; the 8-gram fix swapped in at 4.11 h, `ec672abc`) | 7.90 | 8.05 | 9.26 | 8.05 | 9.26 |
| 11-15/16 | GPC-2 on Thunder (A100 80GB, $1.15; instance up 22:17 to 02:36 PST; all stages run, done at 4.23 h) | 4.23 | 4.32 | 4.97 | 12.37 | 14.23 |
| 11-17 | GPC-3 on RunPod Secure (A100-SXM4-80GB, $2.19; the Thunder card fell back; all stages run, done at 4.39 h) | 4.39 | 4.50 | 9.86 | 16.87 | 24.09 |
| 11-19 | GPC-4 launch 1 on Thunder (A100 80GB, $1.18; the gate tests bailed at 0.30 h and the exit trap left the pod idle until the sentinel was touched by hand; no result; `GPC_PLAN.md` §11.12) | 0.30 | 1.12 | 1.32 | 17.99 | 25.41 |

After GPC-1: $163.74 remain under the shared cap ($173.00 − $9.26). After GPC-2: $158.77 remain. After GPC-3: **$148.91 remain** ($173.00 − $24.09). After GPC-4 launch 1: **$147.59 remain** ($173.00 − $25.41) on this table. The EC2 host's spend to
date (rehearsals 1 to 7 and launch 1's host, about $4 at the records' upper bounds) is not in it; the AWS
bill is its figure of record.

## Implementation log (one line each)

- 2026-11-15: `scripts/gpc_mediator.py`, the three trace patterns, committed before computing
  (`f1001574`); Z4 read on a52's and a49's saved traces.
- 2026-11-15: `scripts/z1_concentration.py`, `scripts/z2_z3_power.py` (the screen and bar power
  tables; the screen sets), `scripts/z5_wref_check.py` (W-ref tokens and the 8-gram check; data
  outside the repo), `scripts/z9_violation_split.py` (item 9's split and the grid's weights).
- 2026-11-15: v0.2 committed and pushed (`88d78eda`); `C_conduct` (`3fee5422`).
- 2026-11-15: Z7-Z9 built: `scripts/gpc_conduct.py` and the grid, `CONDUCT_BRIEFS.md`,
  `scripts/gpc_advice.py`, `scripts/gpc_wref.py`, `scripts/gpc_violation.py`, `scripts/gpc1_pod.py`,
  `scripts/gpc1_read.py`, `runpod/` (runner, cards, inputs), `tests/scripts/test_gpc1.py` (17 fast,
  1 slow dry session); the sync emulation's gate tests pass (49 passed, 1 skipped).
- 2026-11-15: the build read applied: the grid's strength field (schema 2, pilot 15 per cell), the
  fallback-aware screen and registered reads, the teacher's goal-option rates by kind, the Codex
  sample; the library-first rule in the plan's §2.
- 2026-11-15: GPC-1 read and accepted (`3486636f`); Amendment 1, GPC-2 (`11cfa616`).
- 2026-11-15: GPC-2 built: `gpc2_pod.py`, `gpc2_render.py`, `gpc_scaffolds.py`, `gpc2_direction.py`,
  `gpc2_read.py`, the neutral four lines, `runpod/remote_gpc2.sh` and its cards,
  `tests/scripts/test_gpc2.py` (17 fast, 1 slow chain); GPC-1's dry stand-in traces now carry
  punctuation (so `clause_end` positions exist in a dry run); `gpc1_pod.run_cells` takes a messages
  function and a set map.
- 2026-11-16: GPC-2 read and accepted; `scripts/gpc2_rederive.py` (the lead's scratch re-derived:
  C1 against its teacher in the matched cells, X1's base against the held-out base, B1's strata, the
  session checks; `outputs/gpc2_v1/rederive.json`); GPC-A4 to GPC-A8 in the ledger.
- 2026-11-16: Amendment 2 (`2d07ef65`), revised before the build (`ed85ea50`, `d5ec790b`: G5's pilot
  per scenario, its budget, cap hits and acting rate; G4 at k 2 at seed indices 2 and 3). GPC-3 built:
  `gpc3_pod.py`, `gpc_am.py`, `gpc3_read.py`, `remote_gpc3.sh` and its cards, `test_gpc3.py`;
  `run_cells` takes `k_start`; `x2_messages` takes a tuple of elements.
- 2026-11-17: GPC-3 run on RunPod Secure (Thunder fell back; the cards' fallback is now opt-in,
  `c125c151`) and read; `scripts/gpc3_rederive.py` (G5's judge-free string check, the acting rates,
  the main cell's differences, the pilot against the arms, and Amendment 2 rider 4's Codex read of 30
  seeded transcripts); GPC-A9 to GPC-A11 in the ledger.
- 2026-11-17: Amendment 3 (GPC-4) accepted and committed: ImpossibleBench and ODCV-Bench on an AWS EC2
  host (c6a.4xlarge, us-east-2), detached, egress-locked; PropensityBench out, ScopeBench parked; the
  bounds clause (`z_bounds_clause.py`), the bounds grid and briefs (`gpc_bounds.py`), the v2 8-gram
  index (`z_index_v2.py`), the summed adapter (`z_summed_adapter.py`), the ratio bar's power
  (`z_gpc4_power.py`).
- 2026-11-18: GPC-4 built, part 1 (the host and the rehearsal; `GPC_PLAN.md` §11.8): `gpc4_vm.py`, the two
  cloud-init scripts, the rehearsal plan, the IAM policy file, `launch_gpc4.sh`, `test_gpc4.py` (17). The
  pod's side is next; `launch` refuses to start without it.
- 2026-11-18: GPC-4 built, part 2 (the pod's side; `GPC_PLAN.md` §11.9): `gpc4_pod.py`, `remote_gpc4.sh`,
  `gpc4_inputs.sh`, `launch_gpc4_pod.sh`, CST's rank cap by environment, the bounds grid's pilot drawn
  apart from its pool, `test_gpc4.py` (24 fast, 1 slow dry chain). Stage 5 not built (it would void the
  rehearsal).
- 2026-11-19: rehearsal 2 (`GPC_PLAN.md` §11.10): setup 642 s, both gate passes, every result delivered
  detached; but ImpossibleBench never ran (offline data load by name) and the ODCV-Bench run was flagged
  on a file upstream never writes. Fixed with the per-run push, a stricter `verify` and the stand-in's
  hard limit; rehearsal 3 next.
- 2026-11-19: rehearsal 3 (§11.10): both instruments ran end to end and every result was delivered;
  Inspect's zstd `.eval` logs could not be read without `inspect_ai`, so the runner now writes
  `samples.json` and the VM and the pod read that. Rehearsal 4 next.
- 2026-11-19: rehearsal 4 (§11.10) ran clean on every check, $0.19, but `verify` read an empty
  directory (its download unpacked one level off) and recorded a fail; fixed, rehearsal 5 is the record.
- 2026-11-19: rehearsal 5 passed (§11.10; `rehearsal_b0116eafffed7b5b.json`, commit `7cd7d644`, $0.20):
  `launch` is unlocked at that VM hash.
- 2026-11-19: the lead's launch review (§11.11): not launched on rehearsal 5. Built: the no-capacity path
  (Thunder's availability read first, the wait ends with the pod launcher, a failure before the handoff
  removes the host, key and group), rehearsal 6 at the gate's scale with per-stage wall, memory and load,
  the void rule for a bad ODCV-Bench run set (over 10% container errors; one rerun at concurrency 4), and
  the worst case beside the ceiling (about $20.70). Rehearsal 6 next.
- 2026-11-19: rehearsal 6 at the gate's scale (§11.11): ODCV-Bench 80 of 80 clean in 195 s, ImpossibleBench
  60 of 60 scored, host peak under 2 GB and load under 6; but parallel pushes met rrsync's lock. Fixed
  (pushes one at a time, a retry on the lock, verify fails a lost push); rehearsal 7 is the record.
- 2026-11-19: rehearsal 7 passed at the gate's scale (§11.11; `rehearsal_f09eb0d563ee2508.json`, commit
  `0892b9e4`, $0.25): every stage clean, 81 pushes during stages with none failed, retried or timed out.
  `launch` is unlocked at that VM hash.
- 2026-11-19: GPC-4 launch 1 (§11.12): no result. The pod's gate tests bailed on a test that read an env
  name the pod sets; the exit trap stopped before the sentinel, so the pod idled until it was touched by
  hand. About $1.32 for the pod, $0.92 for the host. Built for rehearsal 8: the gate waits for the pod's
  serve check, the host terminates itself 20 min after the pod is gone, the trap reaches the sentinel at
  every bail, and the pod's gate tests run locally under the runner's exports before any launch.
- 2026-11-19: the ODCV-Bench audit (§11.13; `data/odcv_scenario_audit_v1.json`): upstream has changed no
  scenario since June; at the pin 1 scenario's legitimate route cannot be completed (Fleet-Safety: its
  rest enforcer cannot find its monitor), 10 carry a workable defect (8 wrong paths, a missing directory,
  an empty notify tool), 29 are clean. Nothing patched; the read reports rates with and without them.
