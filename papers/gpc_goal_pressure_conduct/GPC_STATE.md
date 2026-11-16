# GPC state (one page; updated at every stop; `GPC_PLAN.md` versions are the history)

Scope: goal pressure, conduct moved into weights with nothing in context, one subject.

**As of 2026-11-15, plan v0.2 with the build read; GPC-1 read and accepted by the author
(`GPC1_RESULTS.md`).** No arm clears the screen (C1 0.00, W-self 0.05, W-ref 0.04 on S-hi; every
upper bound 0.19 or less); registered branch 3, the root split at the base's level for every
arm (the training does not fire in games). C1 carries its in-context teacher's full effect to the
held-out conduct cell (−0.39 [−0.51, −0.28]), the transfer gap GPC-A1 that GPC-2 opens on. W-self's
targets carried a rewrite frame (GPC-A2), so advice data from the tenets has not had a clean read.
Done-when: two of three routes read at pilot dose, none at 0.5. **Next:** GPC-2, drafted as
`GPC_PLAN.md` Amendment 1.

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

1. GPC-2 (Amendment 1): built (`GPC_PLAN.md` §9.4); the author's launch with
   `runpod/launch_gpc2_thunder.sh` (RunPod fallback `runpod/launch_gpc2.sh`).
2. The author's ruling on a rebuilt W-self (targets written as fresh answers; GPC-A2), about 0.9
   A100-h with training, and whether it counts toward done-when.
3. Done (GPC-1): after the download: the Codex validity read of a seeded 30 kept pressure slots
   (`scripts/gpc1_read.py DIR --codex`, then `scripts/gpc1_codex_validity.py DIR`). **Codex count
   (2026-11-15): the goal option carries the assigned violation in 30 of 30, the other option
   avoids it in 30 of 30.**

## GPC-2 (drafted as `GPC_PLAN.md` Amendment 1, 2026-11-15; zero GPU until the author launches)

Priority order X1, X2, X3, F, C2 (C2-div, and C2-mach as the format-matched diagnostic), B (B0 to
B3, B4 if B3 passes), A; batched by loaded model, cut from the end. X3 and C2 run only if format
gates. Floor about 3.25 A100-h ($3.90 on Thunder); ceiling inside one pod about 9.65 ($11.30 on
Thunder, $21.50 on Secure). The scaffold templates: `CONDUCT_SCAFFOLDS.md`. The earlier queue
(the fifth line, items A and B with their riders) is carried into the amendment.

## Next pod

**GPC-2** (built 2026-11-15): floor about 3.25 A100-h ($3.90 on Thunder) when format does not gate
and B3 does not pass; ceiling inside one pod about 9.65 ($11.30 on Thunder, $21.50 on Secure). The
runner opts into the editable limits. Results in `outputs/gpc2_v1`; the local read is
`scripts/gpc2_read.py`.

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

After GPC-1: **$163.74 remain** under the shared cap ($173.00 − $9.26).

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
