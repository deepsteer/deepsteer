#!/usr/bin/env bash
# Runs ON the RunPod pod. Executes the Paper 5 (Phase 2) GPU session.
# Intentionally NOT `set -e`: one failed step must not abort the rest, so we
# salvage GPU time and download whatever completed.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"

P5="papers/5_moral_alignment"
SCRIPTS="$P5/scripts"
OUT="$P5/outputs"
BASE_DIR="${BASE_DIR:-$OUT/olmo3_base}"        # base directions (synced up from local)
GRID="${GRID:-$P5/checkpoint_grid.json}"

INSTRUCT_MODEL="${INSTRUCT_MODEL:-allenai/Olmo-3-7B-Instruct}"
DEVICE="${DEVICE:-cuda}"
INPUT_FORMAT="${INPUT_FORMAT:-raw}"            # pipeline probing format; Sprint 1 decided raw
                                               # (coupling + behavioral use chat internally)
STABLE_LAYER="${STABLE_LAYER:-16}"             # in the 15-31 stable band

# Step toggles (1 = run). VALIDATE=1 overrides to a cheap single-model smoke.
VALIDATE="${VALIDATE:-0}"
RUN_SETUP="${RUN_SETUP:-1}"
RUN_TRANSFER="${RUN_TRANSFER:-1}"      # Sprint 1.1/1.2: probe transfer raw + chat
RUN_BEHAVIORAL="${RUN_BEHAVIORAL:-1}"  # Sprint 1.5: behavioral baseline (instruct)
RUN_PERSONA="${RUN_PERSONA:-1}"        # Sprint 1.4: persona probe on instruct
RUN_PIPELINE="${RUN_PIPELINE:-1}"      # Sprint 2.2: full grid probing + geometry
RUN_COUPLING="${RUN_COUPLING:-1}"      # Sprint 2.3: coupling on post-training states
RUN_HERETIC="${RUN_HERETIC:-0}"        # Sprint 3: refusal ablation + post-ablation battery
RUN_MALLEABILITY="${RUN_MALLEABILITY:-0}"  # Tier 2: proto-refusal extraction + malleability
REFUSAL_PROMPTS="${REFUSAL_PROMPTS:-$P5/refusal_prompts.json}"

# Post-training states for coupling (instruct-capable only; pretraining ckpts
# can't comply, so coupling there is uninformative). "label repo revision".
COUPLING_SPECS="${COUPLING_SPECS:-
olmo3_sft_final allenai/Olmo-3-7B-Instruct-SFT main
olmo3_dpo_final allenai/Olmo-3-7B-Instruct-DPO main
olmo3_instruct_final allenai/Olmo-3-7B-Instruct main}"

# Cap CPU threads: the pod is a container on a huge shared host; torch otherwise
# spawns ~100 intra-op threads for tiny probe matmuls (50-100x slower). Export
# BEFORE python imports torch.
CPU_THREADS="${CPU_THREADS:-8}"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" \
       OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 \
       TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

log() { echo -e "\n=== [$(date +%H:%M:%S)] $* ==="; }
run_step() { local name="$1"; shift; log "START $name"
  if "$@"; then log "OK    $name"; else echo "WARN: $name FAILED (continuing)"; fi; }

if [ "$RUN_SETUP" = 1 ]; then
  log "Environment"
  python -c "import torch; print('CUDA', torch.cuda.is_available(), \
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
  log "Installing deepsteer (editable)"
  pip install -e ".[dev]" --break-system-packages -q || pip install -e ".[dev]" -q
fi

# Base directions must be present (computed locally, synced up). Warn if not.
[ -f "$BASE_DIR/exp1_probe_directions.npz" ] || \
  echo "WARN: $BASE_DIR/exp1_probe_directions.npz missing — transfer/pipeline will fail."

echo "Instruct=$INSTRUCT_MODEL Device=$DEVICE Format=$INPUT_FORMAT Layer=$STABLE_LAYER VALIDATE=$VALIDATE"

# --------------------------- VALIDATION RUN (cheap) --------------------------
# One instruct load + transfer (~5 min, ~14 GB download). Confirms the pod,
# deps, base directions, and the transfer code path all work BEFORE the full
# (multi-hour, ~350 GB) sweep. Always purge the model afterward.
if [ "$VALIDATE" = 1 ]; then
  run_step "VALIDATE: probe transfer (instruct, raw)" \
    python "$SCRIPTS/probe_transfer.py" --model "$INSTRUCT_MODEL" \
      --probe-dir "$BASE_DIR" --input-format raw --device "$DEVICE" \
      --output-dir "$OUT/_validate_instruct_raw"
  run_step "VALIDATE: pipeline 2-state smoke" \
    python "$SCRIPTS/pipeline_study.py" --grid "$GRID" --base-dir "$BASE_DIR" \
      --input-format "$INPUT_FORMAT" --device "$DEVICE" --purge-hf-cache \
      --only olmo3_base,olmo3_instruct_final --output-dir "$OUT/_validate_pipeline"
  if [ "$RUN_MALLEABILITY" = 1 ]; then
    run_step "VALIDATE: proto-refusal smoke (base, 4 prompts) + analysis" \
      bash -c "python '$SCRIPTS/extract_proto_refusal.py' --only olmo3_base \
        --prompts '$REFUSAL_PROMPTS' --max-prompts 4 --device '$DEVICE' --purge-hf-cache \
        --output-dir '$OUT/_validate_measurement/stage3' && \
        python '$SCRIPTS/malleability_analysis.py' \
        --stage3-dir '$OUT/_validate_measurement/stage3' --no-figure \
        --output-dir '$OUT/_validate_measurement'"
  fi
  log "VALIDATION complete. Inspect $OUT/_validate_* then run full (VALIDATE=0)."
  touch "$REPO_DIR/.session_done"; exit 0
fi

# ------------------------------- SPRINT 1 ------------------------------------
if [ "$RUN_TRANSFER" = 1 ]; then
  run_step "1.1 probe transfer (instruct, raw)" \
    python "$SCRIPTS/probe_transfer.py" --model "$INSTRUCT_MODEL" \
      --probe-dir "$BASE_DIR" --input-format raw --device "$DEVICE" \
      --output-dir "$OUT/olmo3_instruct_raw"
  run_step "1.2 probe transfer (instruct, chat)" \
    python "$SCRIPTS/probe_transfer.py" --model "$INSTRUCT_MODEL" \
      --probe-dir "$BASE_DIR" --input-format chat --device "$DEVICE" \
      --output-dir "$OUT/olmo3_instruct_chat"
fi

if [ "$RUN_PERSONA" = 1 ]; then
  run_step "1.4 persona probe (instruct, chat)" \
    python "$SCRIPTS/persona_probe_base.py" --model "$INSTRUCT_MODEL" \
      --input-format chat --device "$DEVICE" --output-dir "$OUT/olmo3_instruct"
  run_step "1.4 persona-morality angles (instruct)" \
    python "$SCRIPTS/persona_morality_angles.py" \
      --moral-npz "$OUT/olmo3_instruct_chat/fresh_probe_directions.npz" \
      --persona-npz "$OUT/olmo3_instruct/persona_directions.npz" \
      --output-dir "$OUT/olmo3_instruct" --label "OLMo-3 instruct"
fi

if [ "$RUN_BEHAVIORAL" = 1 ]; then
  run_step "1.5 behavioral baseline (instruct)" \
    python "$SCRIPTS/behavioral_baseline.py" --model "$INSTRUCT_MODEL" \
      --benchmark both --input-format chat --device "$DEVICE" \
      --output-dir "$OUT/olmo3_instruct"
fi

# ------------------------------- SPRINT 2 ------------------------------------
# Full grid (25 states). --purge-hf-cache keeps disk near one model at a time.
if [ "$RUN_PIPELINE" = 1 ]; then
  run_step "2.2 pipeline study (full grid)" \
    python "$SCRIPTS/pipeline_study.py" --grid "$GRID" --base-dir "$BASE_DIR" \
      --input-format "$INPUT_FORMAT" --device "$DEVICE" --purge-hf-cache \
      --output-dir "$OUT/pipeline"
fi

# Coupling only on instruct-capable states (loop; purge each repo after).
if [ "$RUN_COUPLING" = 1 ]; then
  while read -r label repo rev; do
    [ -z "${label:-}" ] && continue
    run_step "2.3 coupling: $label" \
      python "$SCRIPTS/coupling_measurement.py" --model "$repo" --revision "$rev" \
        --probe-dir "$BASE_DIR" --layer "$STABLE_LAYER" --input-format chat \
        --device "$DEVICE" --output-dir "$OUT/pipeline/$label"
    python - "$repo" <<'PY' || true
import sys, shutil
from pathlib import Path
try:
    from huggingface_hub.constants import HF_HUB_CACHE
except Exception:
    HF_HUB_CACHE = str(Path.home() / ".cache/huggingface/hub")
d = Path(HF_HUB_CACHE) / ("models--" + sys.argv[1].replace("/", "--"))
shutil.rmtree(d, ignore_errors=True)
PY
  done <<< "$COUPLING_SPECS"
fi

# ------------------------------- SPRINT 3 ------------------------------------
# Refusal ablation (Arditi single-direction) on instruct, then the post-ablation
# battery: the high-comprehension / low-compliance ("psychopath") cell.
if [ "$RUN_HERETIC" = 1 ]; then
  run_step "3.1 heretic ablation (instruct)" \
    python "$SCRIPTS/heretic_ablation.py" --model "$INSTRUCT_MODEL" \
      --prompts "$REFUSAL_PROMPTS" \
      --moral-npz "$BASE_DIR/exp1_probe_directions.npz" \
      --refusal-layer "$STABLE_LAYER" --input-format chat --device "$DEVICE" \
      --output-dir "$OUT/heretic"
  ABL="$OUT/heretic/ablated_model"
  if [ -d "$ABL" ]; then
    # Probe the ablated model like any other pipeline state (lands in pipeline/).
    echo "[{\"label\":\"olmo3_ablated\",\"repo\":\"$ABL\",\"revision\":null}]" \
      > "$OUT/heretic/ablated_grid.json"
    run_step "3.3 ablated: probing + geometry + persona" \
      python "$SCRIPTS/pipeline_study.py" --grid "$OUT/heretic/ablated_grid.json" \
        --base-dir "$BASE_DIR" --input-format "$INPUT_FORMAT" --device "$DEVICE" \
        --output-dir "$OUT/pipeline"
    run_step "3.3 ablated: coupling" \
      python "$SCRIPTS/coupling_measurement.py" --model "$ABL" \
        --probe-dir "$BASE_DIR" --layer "$STABLE_LAYER" --input-format chat \
        --device "$DEVICE" --output-dir "$OUT/pipeline/olmo3_ablated"
    run_step "3.3 ablated: behavioral (expect compliance collapse)" \
      python "$SCRIPTS/behavioral_baseline.py" --model "$ABL" --benchmark both \
        --input-format chat --device "$DEVICE" --output-dir "$OUT/heretic"
  else
    echo "WARN: ablated model not at $ABL; skipping post-ablation battery."
  fi
fi

# ----------------------- MEASUREMENT COUPLING (TIER 2) -----------------------
# Proto-refusal contrast per stage-3 checkpoint (+ base), then the malleability
# analysis (M2/M3/M4). The per-checkpoint MFT + persona directions are already
# cached under outputs/pipeline/ (synced up); only the proto-refusal contrast is
# new, so this loads each stage-3 checkpoint once. --purge-hf-cache bounds disk.
if [ "$RUN_MALLEABILITY" = 1 ]; then
  run_step "T2.1 proto-refusal extraction (stage-3 + base)" \
    python "$SCRIPTS/extract_proto_refusal.py" --grid "$GRID" \
      --prompts "$REFUSAL_PROMPTS" --input-format raw --device "$DEVICE" \
      --purge-hf-cache --output-dir "$OUT/measurement/stage3"
  run_step "T2.2 malleability analysis (M2/M3/M4 + figure)" \
    python "$SCRIPTS/malleability_analysis.py" \
      --stage3-dir "$OUT/measurement/stage3" --pipeline-dir "$OUT/pipeline" \
      --instruct-persona-npz "$OUT/olmo3_instruct/persona_directions.npz" \
      --output-dir "$OUT/measurement"
fi

log "Session complete. Output directories:"
ls -d "$OUT"/olmo3_instruct* "$OUT"/pipeline "$OUT"/heretic "$OUT"/measurement 2>/dev/null || true

# Completion sentinel the launcher polls for (survives SSH drops).
touch "$REPO_DIR/.session_done"
