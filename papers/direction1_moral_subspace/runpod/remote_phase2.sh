#!/usr/bin/env bash
# Remote Phase 2 runner — executed ON the pod by run_session.sh.
#
# Installs deepsteer + a recent transformers (for OLMo-3), then runs the Phase 2 sequence
# (phase2_session.sh): base chain (extract --mft -> G-AXIS -> assemble -> FROZEN null -> G2
# -> Track-1) + instruct chain + two same-model-point G3. Always drops the .session_done
# sentinel on exit (even on failure) so the launcher never leaks the billed pod.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
BASE_MODEL="${BASE_MODEL:-allenai/OLMo-3-7B}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-allenai/OLMo-3-7B-Instruct}"
cd "$REPO_DIR"

# G2's transfer probes + Track-1's σ* run on CPU/numpy; cap threads so tiny matmuls don't
# thrash every core while the GPU does the activation extraction.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1

trap 'touch "$REPO_DIR/.session_done"' EXIT

echo ">> cpu: $(nproc 2>/dev/null || echo '?') vCPUs; OMP/MKL capped at $OMP_NUM_THREADS"
echo ">> python: $(python --version 2>&1)"
python -c 'import torch; print(">> cuda:", torch.cuda.is_available(),
  (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))' 2>&1 || true

echo ">> installing deepsteer (editable, core deps)..."
pip install -q --break-system-packages -e . 2>&1 | tail -2 \
  || pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -2

# OLMo-3 is a recent architecture; the base image's transformers predates it. Upgrade to a
# recent transformers (+ accelerate). Override with PIP_EXTRA="transformers==X" to pin.
echo ">> upgrading transformers/accelerate for OLMo-3 support..."
pip install -q --break-system-packages -U transformers accelerate 2>&1 | tail -2 || true
if [ -n "${PIP_EXTRA:-}" ]; then
  echo ">> PIP_EXTRA: installing $PIP_EXTRA"
  pip install -q --break-system-packages -U $PIP_EXTRA 2>&1 | tail -2 || true
fi

# Fast, resilient large-shard downloads (each OLMo-3-7B is ~14 GB).
if pip install -q --break-system-packages hf_xet >/dev/null 2>&1; then
  export HF_XET_HIGH_PERFORMANCE=1
  echo ">> hf_xet high-performance transfer: enabled"
fi
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

echo ">> transformers: $(python -c 'import transformers; print(transformers.__version__)' 2>&1)"
# Fail fast (with a clear message) if this transformers can't resolve the OLMo-3 config,
# so a real run aborts at install time rather than mid-extraction. Skipped in VALIDATE
# (the smoke uses the tiny OLMo-2-1B, which any recent transformers supports).
if [ "$VALIDATE" != "1" ]; then
  python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$BASE_MODEL'); \
    print('>> OLMo-3 config resolves OK')" 2>&1 | tail -3 \
    || echo ">> WARN: AutoConfig for $BASE_MODEL failed; set PIP_EXTRA='transformers==<ver>' with OLMo-3 support."
fi

if [ "$VALIDATE" = "1" ]; then
  echo ">> VALIDATE=1: GPU plumbing smoke (tiny model both tags, few pairs)"
else
  echo ">> REAL run: BASE_MODEL=$BASE_MODEL  INSTRUCT_MODEL=$INSTRUCT_MODEL"
fi

# phase2_session.sh enforces the stage sequence + both structural constraints. It honors
# VALIDATE (tiny model) and BASE_MODEL/INSTRUCT_MODEL.
VALIDATE="$VALIDATE" BASE_MODEL="$BASE_MODEL" INSTRUCT_MODEL="$INSTRUCT_MODEL" \
  bash "$REPO_DIR/papers/direction1_moral_subspace/runpod/phase2_session.sh"

echo ">> Phase 2 sequence finished. Key results:"
for f in base/g2_result.json base/track1_result.json g3_result.json; do
  p="$REPO_DIR/papers/direction1_moral_subspace/outputs/phase2/$f"
  [ -f "$p" ] && { echo "---- $f ----"; cat "$p"; echo; }
done
