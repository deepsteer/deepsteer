# Direction 1 — Phase 2 RunPod launch

GPU run for the single-source `V_moral` Phase 2: build `V_moral` on OLMo-3 Base + Instruct,
run the contamination gate (G2), fragility (Track-1), and the two-same-model-point
refusal-overlap test (G3). All no-GPU gates already pass locally.

## Prereqs

```bash
export RUNPOD_API_KEY=...          # required
export HF_TOKEN=...                # OLMo-3 is public; pass for rate limits / future gating
# SSH key at ~/.ssh/id_ed25519(.pub), or set SSH_KEY=...
```

## Flow (cost-minimizing)

```bash
cd papers/d1_moral_subspace/runpod

# 1. Local gates (already green; re-run anytime, no GPU):
python ../scripts/phase2_local_test.py
VALIDATE=1 bash ./phase2_session.sh         # local model-level smoke (OLMo-2-1B)

# 2. Cheap GPU plumbing smoke on the pod (tiny model both tags):
VALIDATE=1 ./run_session.sh

# 3. Real run (OLMo-3-7B Base + Instruct):
./run_session.sh
```

The launcher spins up a GPU pod, rsyncs the repo, runs `remote_phase2.sh` (which installs
deps + a recent transformers for OLMo-3, then drives `phase2_session.sh`), streams the log,
rsyncs results back, and **terminates the pod on exit** (`KEEP_POD=1` to keep it).

## Sequence (what runs on the pod)

```
[base]     extract --mft → G-AXIS(single-source) → assemble V_moral → FROZEN null → G2 → Track-1
[instruct] extract → G-AXIS → assemble V_moral → FROZEN null
G3 (Point A = base proto-refusal × Base-V_moral ; Point B = instruct gate × Instruct-V_moral)
```

Two structural constraints are enforced in code: the frozen null is written by `phase2_null.py`
and consumed read-only by G3 (predates-the-result), and each refusal point is measured within
its own model (no Base↔Instruct projection).

## GPU / disk

One 7B model is resident at a time (each `model.release()`d before the next), so a single
**48 GB** card suffices (80 GB also fine). `DISK_GB=150` holds both 7B checkpoints (~14 GB
each) + the tiny smoke model + HF cache headroom. Override `GPU_TYPES`, `DISK_GB`, `IMAGE` as needed.

## Results (rsynced back to `outputs/phase2/`)

| File | Meaning |
|---|---|
| `base/g2_result.json` | G2 contamination **PASS/STOP** on the narrative slice (acc_surf vs acc_para; PASS iff acc_para ≥ 0.60 and gap ≤ 0.10). A real-run STOP halts the pod run. |
| `base/track1_result.json` | σ* (RMS-normalized) V_moral vs MFT baseline + eff-dim contrast. |
| `g3_result.json` | Point A (base proto) + Point B (instruct gate, the 0.1044 comparison) projections vs each model's null+control; **POSITIVE iff both clear**, else NULL. |
| `{base,instruct}/{g_axis_decision,v_moral.npz,null_artifact}.json` | per-tag V_moral, eff-dim, frozen null. |

## Knobs

| Env | Default | Notes |
|---|---|---|
| `BASE_MODEL` / `INSTRUCT_MODEL` | `allenai/Olmo-3-1025-7B` / `-Instruct` | the two models |
| `VALIDATE` | `0` | `1` = tiny-model plumbing smoke (also config-checks OLMo-3) |
| `TRANSFORMERS_VERSION` | `5.12.1` | pinned (GPU-smoke-validated); override for a different build |
| `PIP_EXTRA` | — | extra pip override, e.g. `transformers==<ver>` |
| `GPU_TYPES` | 48 GB-class first | comma list searched in order |
| `KEEP_POD` / `REUSE_POD_ID` | — | debug: keep / attach to a pod |

## Pre-GPU checklist

- [x] No-GPU structural test passes (`phase2_local_test.py`).
- [x] Two-tag `VALIDATE` dry run passes end-to-end (local, OLMo-2-1B).
- [x] GPU plumbing smoke passed on A6000 (full sequence end-to-end, pod terminated;
      transformers now **pinned to 5.12.1**, the version that smoke validated).
- [x] OLMo-3 config gate passed on the corrected IDs (`Olmo-3-1025-7B` base +
      `Olmo-3-7B-Instruct`) on the pinned transformers 5.12.1 — both resolve, `model_type='olmo3'`.
- [ ] Real `./run_session.sh`. **(All pre-GPU gates pass; cleared to run.)** Only the OLMo-3
      *weight* load (vs config) is first exercised here; if it trips, set `TRANSFORMERS_VERSION`.

Note: all smoke numbers (G2/G3/Track-1) are artifacts of the 8-pair OLMo-2-1B run, not
signal. In particular Track-1 `σ*(V_moral)=0` is expected in the smoke (the 8-pair direction
doesn't separate the 8 eval pairs at baseline); the real run's 877-train/96-eval direction
will give meaningful σ*.
