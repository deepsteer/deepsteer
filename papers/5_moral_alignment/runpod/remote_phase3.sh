#!/usr/bin/env bash
# Runs ON the RunPod pod. Executes the Paper 6 (Phase 3 / ablation-resistance) session.
# Intentionally NOT `set -e`: one failed step must not abort the rest, so we
# salvage GPU time and download whatever completed.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"

P6="papers/5_moral_alignment"
SCRIPTS="$P6/scripts"
OUT="$P6/outputs"
P5="papers/5_moral_alignment"
# Moral directions Paper 6 ablates (computed in Phase 2, synced up from local).
BASE_DIRECTIONS="${BASE_DIRECTIONS:-$P5/outputs/olmo3_base/exp1_probe_directions.npz}"
GRID="${GRID:-$P5/checkpoint_grid.json}"
PIPELINE_DIR="${PIPELINE_DIR:-$P5/outputs/pipeline}"

DEVICE="${DEVICE:-cuda}"
DEP_KIND="${DEP_KIND:-probe}"            # probe | meandiff
PER_STATE="${PER_STATE:-0}"              # 1 -> ablate each state's own directions
DATASET_TARGET="${DATASET_TARGET:-40}"   # probing pairs per foundation

# Sprint 6 ART-SFT config.
BASE_MODEL="${BASE_MODEL:-allenai/Olmo-3-1025-7B}"
INSTRUCT_TEMPLATE_FROM="${INSTRUCT_TEMPLATE_FROM:-allenai/Olmo-3-7B-Instruct}"
SFT_DATA="${SFT_DATA:-$P6/data/sft_mix.jsonl}"
N_GENERAL="${N_GENERAL:-1500}"
N_MORAL="${N_MORAL:-1500}"
ART_LAMBDA="${ART_LAMBDA:-0.01}"
ART_MAX_STEPS="${ART_MAX_STEPS:-400}"

# Forced-coupling Stage 1 config (representational pre-check; no SFT, no Heretic).
STAGE1_REVISION="${STAGE1_REVISION:-stage3-step11921}"   # late stage-3 (Tier 2: no window)
STAGE1_MORAL_NPZ="${STAGE1_MORAL_NPZ:-$PIPELINE_DIR/olmo3_pretrain_stage3_step11921/probe_directions.npz}"
STAGE1_CAPACITY="${STAGE1_CAPACITY:-r16_qv}"   # rung 1; climb via STAGE1_CAPACITY=r64_qv_mlp|full
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-300}"
STAGE1_GENERAL="${STAGE1_GENERAL:-}"   # general LM corpus jsonl; empty -> probing-text fallback
STAGE1_EXTRA="${STAGE1_EXTRA:-}"       # extra flags, e.g. "--probe-monitor --stop-on-breach"

# Step toggles (1 = run). VALIDATE=1 overrides to a cheap single-model smoke.
VALIDATE="${VALIDATE:-0}"
RUN_SETUP="${RUN_SETUP:-1}"
RUN_DEPENDENCY="${RUN_DEPENDENCY:-1}"    # Sprint 5.2: dependency across the grid
RUN_ART_SFT="${RUN_ART_SFT:-0}"          # Sprint 6 (auto-skips until art_sft.py exists)
RUN_EVAL="${RUN_EVAL:-0}"                # Sprint 7 (auto-skips until eval_pipeline.py exists)
RUN_COUPLING_STAGE1="${RUN_COUPLING_STAGE1:-0}"  # Forced-coupling Stage 1 (off by default)

# Cap CPU threads: the pod is a container on a huge shared host; torch otherwise
# spawns ~100 intra-op threads for tiny matmuls (50-100x slower). Export BEFORE
# python imports torch.
CPU_THREADS="${CPU_THREADS:-8}"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" \
       OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 \
       TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# Reduce CUDA fragmentation (ART's paired forward is memory-heavy on 80 GB).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo -e "\n=== [$(date +%H:%M:%S)] $* ==="; }
run_step() { local name="$1"; shift; log "START $name"
  if "$@"; then log "OK    $name"; else echo "WARN: $name FAILED (continuing)"; fi; }

# Per-state ablation lands in a SEPARATE output dir so it never clobbers the
# base-transfer trajectory (the two are different measurements: self-dependency
# vs dependency on the fixed base subspace).
PER_STATE_FLAG=()
DEP_OUT="$OUT/dependency"
if [ "$PER_STATE" = 1 ]; then
  PER_STATE_FLAG=(--per-state-directions --pipeline-dir "$PIPELINE_DIR")
  DEP_OUT="$OUT/dependency_perstate"
fi

if [ "$RUN_SETUP" = 1 ]; then
  log "Environment"
  python -c "import torch; print('CUDA', torch.cuda.is_available(), \
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
  log "Installing deepsteer (editable)"
  pip install -e ".[dev]" --break-system-packages -q || pip install -e ".[dev]" -q
fi

# Moral directions must be present (computed in Phase 2, synced up). Warn if not.
[ -f "$BASE_DIRECTIONS" ] || \
  echo "WARN: $BASE_DIRECTIONS missing — the dependency sweep will fail. Did Paper 5 outputs sync?"

echo "Directions=$BASE_DIRECTIONS Kind=$DEP_KIND PerState=$PER_STATE Device=$DEVICE VALIDATE=$VALIDATE"

# --------------------------- VALIDATION RUN (cheap) --------------------------
# One base-model load + dependency measure on 16 capped texts (~5-10 min, ~14 GB
# download, purged after). Confirms the pod, deps, synced directions, the
# ablation hooks, and the CE/DiD path all work BEFORE the full 25-state sweep.
if [ "$VALIDATE" = 1 ]; then
  if [ "$RUN_DEPENDENCY" = 1 ]; then
    run_step "VALIDATE: dependency on olmo3_base (16 texts)" \
      python "$SCRIPTS/moral_dependency_pipeline.py" --grid "$GRID" \
        --base-directions "$BASE_DIRECTIONS" --direction-kind "$DEP_KIND" \
        --only olmo3_base --max-texts 16 --no-per-text \
        --device "$DEVICE" --purge-hf-cache \
        --output-dir "$OUT/_validate_dependency" "${PER_STATE_FLAG[@]}"
  fi
  if [ "$RUN_COUPLING_STAGE1" = 1 ]; then
    GEN_FLAG=(); [ -n "$STAGE1_GENERAL" ] && GEN_FLAG=(--general-jsonl "$STAGE1_GENERAL")
    run_step "VALIDATE: forced-coupling stage1 smoke (4 steps, purge after)" \
      python "$SCRIPTS/forced_coupling_stage1.py" --model "$BASE_MODEL" \
        --revision "$STAGE1_REVISION" --moral-npz "$STAGE1_MORAL_NPZ" \
        --capacity "$STAGE1_CAPACITY" --max-steps 4 --eval-every 2 --warmup-steps 1 \
        --calibrate --device "$DEVICE" "${GEN_FLAG[@]}" \
        --output-dir "$OUT/_validate_stage1"
  fi
  log "VALIDATION complete. Inspect $OUT/_validate_*,"
  log "then run the full sweep with VALIDATE=0."
  touch "$REPO_DIR/.session_done"; exit 0
fi

# ------------------------------- SPRINT 5 ------------------------------------
# Full grid (25 states). --purge-hf-cache keeps disk near one 7B at a time.
if [ "$RUN_DEPENDENCY" = 1 ]; then
  run_step "5.2 moral dependency (full grid -> $DEP_OUT)" \
    python "$SCRIPTS/moral_dependency_pipeline.py" --grid "$GRID" \
      --base-directions "$BASE_DIRECTIONS" --direction-kind "$DEP_KIND" \
      --dataset-target "$DATASET_TARGET" \
      --device "$DEVICE" --purge-hf-cache \
      --output-dir "$DEP_OUT" "${PER_STATE_FLAG[@]}"
fi

# ------------------------------- SPRINT 6 (ART-SFT) --------------------------
# Prepare the general+moral chat mix (once), then train control-SFT (λ=0) and
# ART-SFT (λ>0, calibrated) on the SAME data — only the ART term differs.
if [ "$RUN_ART_SFT" = 1 ]; then
  if [ ! -f "$SFT_DATA" ]; then
    run_step "6.3 prepare SFT mix (Tülu 3 general + moral)" \
      python "$SCRIPTS/prepare_sft_data.py" --output "$SFT_DATA" \
        --n-general "$N_GENERAL" --n-moral "$N_MORAL"
  else
    echo "SFT mix already present at $SFT_DATA; skipping prep."
  fi
  # --no-merge: save only the small LoRA adapter (the durable artifact that syncs
  # back). Sprint 7 reconstructs the 14 GB merged model from base+adapter on the
  # pod, so nothing 14 GB needs to survive a shutdown. Each condition is skipped
  # if its adapter WEIGHTS already exist (idempotent re-runs; the adapter syncs up
  # from a prior successful run).
  if [ -f "$OUT/control_sft/adapter/adapter_model.safetensors" ]; then
    echo "control_sft adapter already present; skipping control-SFT."
  else
    run_step "6.4 control-SFT (λ=0)" \
      python "$SCRIPTS/art_sft.py" --model "$BASE_MODEL" \
        --chat-template-from "$INSTRUCT_TEMPLATE_FROM" --data "$SFT_DATA" \
        --art-lambda 0.0 --max-steps "$ART_MAX_STEPS" --no-merge \
        --output-dir "$OUT/control_sft" --device "$DEVICE"
  fi
  if [ -f "$OUT/art_sft/adapter/adapter_model.safetensors" ]; then
    echo "art_sft adapter already present; skipping ART-SFT."
  else
    run_step "6.4 ART-SFT (λ=$ART_LAMBDA, calibrated, cap 1.0)" \
      python "$SCRIPTS/art_sft.py" --model "$BASE_MODEL" \
        --chat-template-from "$INSTRUCT_TEMPLATE_FROM" --data "$SFT_DATA" \
        --moral-directions "$BASE_DIRECTIONS" --direction-kind "$DEP_KIND" \
        --art-lambda "$ART_LAMBDA" --art-calibrate --max-steps "$ART_MAX_STEPS" --no-merge \
        --output-dir "$OUT/art_sft" --device "$DEVICE"
  fi
fi

# ------------------------------- SPRINT 7 (eval) -----------------------------
# Post-ART battery + Heretic on both conditions, reconstructing merged models
# from the durable adapters. The 4-cell comparison lands in $OUT/eval/.
if [ "$RUN_EVAL" = 1 ]; then
  run_step "7 post-ART evaluation battery + Heretic" \
    python "$SCRIPTS/eval_pipeline.py" \
      --art-adapter "$OUT/art_sft/adapter" \
      --control-adapter "$OUT/control_sft/adapter" \
      --base-model "$BASE_MODEL" \
      --base-dir "$P5/outputs/olmo3_base" --base-directions "$BASE_DIRECTIONS" \
      --refusal-prompts "$P5/refusal_prompts.json" \
      --output-dir "$OUT/eval" --device "$DEVICE"
fi

# ------------------- FORCED COUPLING (STAGE 1, limited) ----------------------
# Representational pre-check: can a bounded regularizer move proto-refusal->MFT
# projection without tripping the four specificity guards? No SFT, no Heretic.
# Hard stop after this for human review (do NOT chain Stage 2). Climb the
# capacity ladder via STAGE1_CAPACITY=r16_qv -> r64_qv_mlp -> full.
if [ "$RUN_COUPLING_STAGE1" = 1 ]; then
  [ -f "$STAGE1_MORAL_NPZ" ] || \
    echo "WARN: $STAGE1_MORAL_NPZ missing — Stage 1 needs the cached V directions (did pipeline/ sync?)."
  GEN_FLAG=(); [ -n "$STAGE1_GENERAL" ] && GEN_FLAG=(--general-jsonl "$STAGE1_GENERAL")
  run_step "Stage1 forced coupling ($STAGE1_CAPACITY, $STAGE1_MAX_STEPS steps)" \
    python "$SCRIPTS/forced_coupling_stage1.py" --model "$BASE_MODEL" \
      --revision "$STAGE1_REVISION" --moral-npz "$STAGE1_MORAL_NPZ" \
      --capacity "$STAGE1_CAPACITY" --max-steps "$STAGE1_MAX_STEPS" --calibrate \
      --label "coupling_$STAGE1_CAPACITY" --device "$DEVICE" "${GEN_FLAG[@]}" $STAGE1_EXTRA \
      --output-dir "$OUT/intervention_stage1"
fi

log "Session complete. Output directories:"
ls -d "$OUT"/dependency* "$OUT"/art_sft "$OUT"/control_sft "$OUT"/intervention_stage1 2>/dev/null || true

# Completion sentinel the launcher polls for (survives SSH drops).
touch "$REPO_DIR/.session_done"
