#!/usr/bin/env bash
# Remote Phase 2 runner — executed ON the pod by run_session.sh
# (REMOTE_SCRIPT=papers/6_cross_model/runpod/remote_phase2.sh).
#
# Per family: Heretic-ablate the instruct model, then run the comprehension
# battery (probe acc + eff-dim + moral judgment + dependency) on instruct vs
# ablated, deleting the 14 GB ablated model afterwards. Always drops the
# .session_done sentinel on exit so the launcher never leaks the billed pod.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
MODELS="${MODELS:-all}"
VALIDATE="${VALIDATE:-0}"
cd "$REPO_DIR"

trap 'touch "$REPO_DIR/.session_done"' EXIT

# Cap CPU threads (same reason as Phase 1: per-layer probes train on CPU; on a
# many-vCPU pod Torch thread-thrashes tiny matmuls). No numerical change.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

echo ">> cpu: $(nproc 2>/dev/null || echo '?') vCPUs; OMP/MKL threads capped at $OMP_NUM_THREADS"
echo ">> python: $(python --version 2>&1)"
python -c 'import torch; print(">> cuda:", torch.cuda.is_available(),
  (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))' 2>&1 || true

echo ">> installing deepsteer (editable)..."
pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -2 \
  || pip install -q --break-system-packages -e . 2>&1 | tail -2

if [ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo ">> HF_TOKEN: set"
else
  echo ">> HF_TOKEN: UNSET -- gated Llama-3.1 steps will 401"
fi
echo ">> Phase 2 [VALIDATE=$VALIDATE] models=$MODELS"

VALIDATE="$VALIDATE" python papers/6_cross_model/scripts/run_phase2.py --models "$MODELS"
RC=$?
echo ">> run_phase2 exit code: $RC"
exit $RC
