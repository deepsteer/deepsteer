# RunPod Session — Paper 6 (Phase 3 / Ablation Resistance)

Programmatic spin-up / execute / download / tear-down for the GPU-bound parts of
Phase 3. Same lineage as the Paper 3 and Paper 5 launchers: the pod is created
right before the run and **always** terminated afterward via a shell `trap`,
including on error or Ctrl-C, so billed GPU time is minimized.

Currently wired for **Sprint 5** (the natural moral-dependency trajectory across
the 25-state OLMo-3 pipeline). Sprint 6 (ART-SFT) and Sprint 7 (post-ART eval)
toggles are present and **auto-skip** until their scripts exist.

## Files

| File | Runs where | Purpose |
|---|---|---|
| `local_test_phase3.sh` | Local | Pre-flight: unit checks (+ optional capped real run). Gate before any GPU spend. |
| `run_session_phase3.sh` | Local | Orchestrator: create pod, rsync up, exec, rsync down, terminate |
| `remote_phase3.sh` | Pod | The experiment plan (toggle-driven, fail-soft) |
| `../scripts/local_test.py` | Local | The actual unit-level checks `local_test_phase3.sh` invokes |

## Prerequisites

- Local tools: `curl`, `jq`, `ssh`, `rsync` (the orchestrator checks for these).
- `RUNPOD_API_KEY` exported (RunPod console → Settings → API Keys).
- An SSH keypair at `$SSH_KEY` (default `~/.ssh/id_ed25519`). The **public** key
  is injected via the `PUBLIC_KEY` env var so the RunPod PyTorch image authorizes
  you and starts `sshd`.
- **Phase 2 outputs present locally.** Paper 6 ablates the moral directions from
  `papers/5_moral_alignment/outputs/olmo3_base/exp1_probe_directions.npz` (and,
  with `PER_STATE=1`, the per-state `pipeline/<label>/probe_directions.npz`).
  By default the sync ships only `olmo3_base/` (~13M); the 189M of per-state
  `pipeline/` dirs ride along **only** when `PER_STATE=1`. Total default
  upload is ~31M.

## The three-gate workflow (do these in order)

Anything that takes more than ~5 minutes gets a local test and a cheap remote
dry-run first.

```bash
# Gate 1 — LOCAL pre-flight (seconds; no large download)
bash papers/5_moral_alignment/runpod/local_test_phase3.sh
# optional: also run the real 7B path from cache (~1 min load)
REAL=1 bash papers/5_moral_alignment/runpod/local_test_phase3.sh

# Gate 2 — RunPod DRY-RUN (~5-10 min, one model, cache purged after)
export RUNPOD_API_KEY=...
cd papers/5_moral_alignment/runpod
VALIDATE=1 ./run_session_phase3.sh
#   inspect outputs/_validate_dependency/dependency_summary.json — score sane?

# Gate 3 — FULL Sprint 5 sweep (~3 GPU-hours, 25 states)
./run_session_phase3.sh
```

## What runs (and minimizing compute)

Steps are toggled by env vars (`1` = run). Defaults:

| Toggle | Default | Step | Notes |
|---|---|---|---|
| `RUN_DEPENDENCY` | 1 | Sprint 5.2 moral dependency across the grid | 25 states; `--purge-hf-cache` keeps disk near one 7B. |
| `RUN_ART_SFT` | 0 | Sprint 6.4 control-SFT + ART-SFT | Preps the Tülu 3 general+moral mix, then trains both (LoRA). Saves **adapters only** (`--no-merge`). `ONLY=art`. |
| `RUN_EVAL` | 0 | Sprint 7 post-ART battery + Heretic | Reconstructs merged models from adapters, runs the 4-cell comparison into `outputs/eval/`. `ONLY=eval`. |

Sprint 6 knobs: `ART_LAMBDA` (0.01), `ART_MAX_STEPS` (400), `N_GENERAL`/`N_MORAL` (1500 each).
Run train+eval together in one pod: `RUN_ART_SFT=1 RUN_EVAL=1 ./run_session_phase3.sh`.

Knobs:
- `VALIDATE=1` — cheap single-state smoke (`olmo3_base`, 16 capped texts), then stop.
- `DEP_KIND=probe|meandiff` — which moral direction set to ablate (default `probe`).
- `PER_STATE=1` — ablate each state's own directions instead of the base set
  (measures self-dependency; **not** cross-state comparable; adds ~189M to the upload).
- `DATASET_TARGET=40` — probing pairs per foundation.
- `ONLY="dependency"` — run only the named step(s).
- `KEEP_POD=1` — leave the pod running for debugging (terminate it yourself).
- `REUSE_POD_ID=<id>` — attach to an existing pod (no create/terminate).

Cost levers:
- Trap-based teardown means a crashed run still terminates the pod.
- `remote_phase3.sh` is fail-soft (no `set -e`): one failed state doesn't
  abort the sweep, and partial results still download.
- `--purge-hf-cache` deletes each 7B revision after it's measured, so the 25
  distinct revisions never co-reside on disk.

Rough estimate for the full Sprint 5 sweep: ~3 GPU-hours, ~$3–5 on an A100 80 GB.

## After the run (local)

Outputs land under `papers/5_moral_alignment/outputs/`:
- base-transfer sweep -> `dependency/` (`dependency_summary.json` = trajectory;
  `<label>/moral_dependency.json` per state)
- `PER_STATE=1` sweep -> `dependency_perstate/` (separate, so it never clobbers
  the base-transfer trajectory)

Figures (Sprint 5.3):
```bash
# base-transfer trajectory
python papers/5_moral_alignment/scripts/dependency_figures.py \
  --summary papers/5_moral_alignment/outputs/dependency/dependency_summary.json

# per-state vs base-transfer overlay (the confirmation comparison)
python papers/5_moral_alignment/scripts/dependency_figures.py \
  --summary papers/5_moral_alignment/outputs/dependency_perstate/dependency_summary.json \
  --overlay-summary papers/5_moral_alignment/outputs/dependency/dependency_summary.json \
  --label "per-state directions" --overlay-label "base directions (transfer)" \
  --figures-dir papers/5_moral_alignment/outputs/figures_perstate
```

## Persisting large artifacts across pod shutdown

The 14 GB merged models are **never** the durable unit. Sprint 6 saves only the
small LoRA **adapters** (`outputs/{art,control}_sft/adapter`, ~135 MB each),
which sync back automatically; the launcher's download excludes every weight blob
(`*.safetensors`, `merged_model/`, `ablated_model/`, `_merged/`). Sprint 7
reconstructs each merged model from base + adapter on the pod (ephemeral), so a
pod shutdown loses nothing reproducible — re-sync the adapters next session and
eval rebuilds the merged models.

If you ever need the *merged* 7B models themselves persisted (to deploy or hand
off), push them to a private HF Hub repo from the pod (region-independent,
durable, uses your existing HF token) rather than relying on container disk; a
RunPod network volume (`VOLUME_GB>0`, mounted at `/workspace`) also works but is
region-locked, which constrains the GPU-capacity search.

## Notes / caveats

- **GPU selection / capacity.** OLMo-3 7B is ~14 GB fp16, so a 24–48 GB card is
  plenty. The orchestrator tries an ordered list of GPU types across both clouds
  until one deploys. Override with `GPU_TYPES="a,b,c"`, `GPU_TYPE="NVIDIA L40S"`,
  or `CLOUD_TYPES=SECURE`.
- **Drop-proof execution.** The plan launches detached (`setsid`, logging to
  `session.log` on the pod); the launcher polls the log + a `.session_done`
  sentinel, so a blipped SSH connection doesn't kill the run. For the multi-hour
  full sweep, start the local launcher under `tmux`/`nohup` too — if the *local*
  process dies, its EXIT trap terminates the pod (no cost leak, but the run is lost).
- Uses RunPod's GraphQL API directly (`curl` + `jq`); no `runpodctl` needed.
