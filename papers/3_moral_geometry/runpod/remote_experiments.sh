#!/usr/bin/env bash
# Runs ON the RunPod pod. Executes the Paper 3 GPU session.
# Intentionally NOT `set -e`: one failed experiment must not abort the rest,
# so we salvage GPU time and download whatever completed.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"

MODEL="${MODEL:-allenai/OLMo-2-1124-7B}"
DEVICE="${DEVICE:-cuda}"
N_BOOTSTRAP="${N_BOOTSTRAP:-200}"
P3="papers/3_moral_geometry"
# exp1 probe directions matching $MODEL's scale (used by B/E/C/A.4 steps).
DIRECTIONS_NPZ="${DIRECTIONS_NPZ:-$P3/outputs/exp1_2_3_7B/exp1_probe_directions.npz}"

# Step toggles (1 = run). Steps whose script does not exist yet auto-skip.
RUN_SETUP="${RUN_SETUP:-1}"
RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-1}"   # A.2  Exp 3 bootstrap stability (uses --bootstrap-only)
RUN_FRAGILITY="${RUN_FRAGILITY:-1}"   # A.3  Exp 7 framework fragility
RUN_CAUSAL="${RUN_CAUSAL:-1}"         # C.2  direction ablation + steering injection
RUN_DILEMMA="${RUN_DILEMMA:-0}"       # A.4  dilemma probing + geometry (stretch)
RUN_TAXONOMY="${RUN_TAXONOMY:-1}"     # B    data-driven taxonomy (skips if script absent)
RUN_EXTERNAL="${RUN_EXTERNAL:-1}"     # E    MFV external robustness (skips if script absent)

log()  { echo -e "\n=== [$(date +%H:%M:%S)] $* ==="; }
run_step() { local name="$1"; shift; log "START $name"
  if "$@"; then log "OK    $name"; else echo "WARN: $name FAILED (continuing)"; fi; }

if [ "$RUN_SETUP" = 1 ]; then
  log "Environment"
  python -c "import torch; print('CUDA', torch.cuda.is_available(), \
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
  log "Installing deepsteer (editable)"
  pip install -e ".[dev]" --break-system-packages -q || \
    pip install -e ".[dev]" -q
fi

echo "Model=$MODEL  Device=$DEVICE  N_BOOTSTRAP=$N_BOOTSTRAP"

# --- A.2: Exp 3 bootstrap stability (reuses synced 7B Exp 1 directions) ---
if [ "$RUN_BOOTSTRAP" = 1 ]; then
  run_step "A.2 bootstrap stability (7B)" \
    python "$P3/scripts/exp1_2_3_framework_geometry.py" \
      --model "$MODEL" --output-dir "$P3/outputs/exp1_2_3_7B" \
      --bootstrap-only --n-bootstrap "$N_BOOTSTRAP" --device "$DEVICE"
fi

# --- A.3: Exp 7 framework fragility ---
if [ "$RUN_FRAGILITY" = 1 ]; then
  run_step "A.3 framework fragility (7B)" \
    python "$P3/scripts/exp7_framework_fragility.py" \
      --model "$MODEL" --output-dir "$P3/outputs/exp7_fragility_7B" \
      --olmo-only --device "$DEVICE"
fi

# --- C.2: causal validation (writes to outputs/probe_engineering; moved to _7B) ---
if [ "$RUN_CAUSAL" = 1 ]; then
  run_step "C.2 direction ablation (7B)" \
    python "$P3/scripts/probe_engineering/direction_ablation.py" \
      --model "$MODEL" \
      --probe-directions "$P3/outputs/exp1_2_3_7B/exp1_probe_directions.npz" \
      --device "$DEVICE"
  run_step "C.2 steering injection (7B)" \
    python "$P3/scripts/probe_engineering/steering_injection.py" \
      --model "$MODEL" \
      --probe-directions "$P3/outputs/exp1_2_3_7B/exp1_probe_directions.npz" \
      --device "$DEVICE"
  if [ -d "$P3/outputs/probe_engineering" ]; then
    rm -rf "$P3/outputs/probe_engineering_7B"
    mv "$P3/outputs/probe_engineering" "$P3/outputs/probe_engineering_7B"
    log "Moved causal outputs -> probe_engineering_7B (avoids clobbering 1B)"
  fi
fi

# --- A.4: dilemma probing + geometry (stretch) ---
if [ "$RUN_DILEMMA" = 1 ]; then
  run_step "A.4 dilemma probing (7B)" \
    python "$P3/scripts/dilemma_probing.py" \
      --model "$MODEL" \
      --directions "$P3/outputs/exp1_2_3_7B/exp1_probe_directions.npz" \
      --output-dir "$P3/outputs/dilemma_probing_7B" --device "$DEVICE"
  run_step "A.4 dilemma geometry (7B)" \
    python "$P3/scripts/dilemma_geometry.py" \
      --dilemma-directions "$P3/outputs/dilemma_probing_7B/dilemma_probe_directions.npz" \
      --foundation-directions "$P3/outputs/exp1_2_3_7B/exp1_probe_directions.npz" \
      --output-dir "$P3/outputs/dilemma_geometry_7B"
fi

# --- B: data-driven taxonomy (auto-skips until the script is written) ---
if [ "$RUN_TAXONOMY" = 1 ]; then
  if [ -f "$P3/scripts/data_driven_taxonomy.py" ]; then
    run_step "B taxonomy (7B)" \
      python "$P3/scripts/data_driven_taxonomy.py" \
        --model "$MODEL" --output-dir "$P3/outputs/taxonomy" \
        --directions "$DIRECTIONS_NPZ" --device "$DEVICE"
  else
    echo "SKIP B taxonomy: $P3/scripts/data_driven_taxonomy.py not present yet"
  fi
fi

# --- E: external dataset (MFV) robustness (auto-skips until written) ---
if [ "$RUN_EXTERNAL" = 1 ]; then
  if [ -f "$P3/scripts/external_dataset_robustness.py" ]; then
    run_step "E external robustness (7B)" \
      python "$P3/scripts/external_dataset_robustness.py" \
        --model "$MODEL" --output-dir "$P3/outputs/external_robustness" \
        --directions "$DIRECTIONS_NPZ" --device "$DEVICE"
  else
    echo "SKIP E external robustness: $P3/scripts/external_dataset_robustness.py not present yet"
  fi
fi

log "Session complete. Output directories:"
ls -d "$P3"/outputs/*_7B "$P3"/outputs/taxonomy "$P3"/outputs/external_robustness 2>/dev/null || true
