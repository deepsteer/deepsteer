#!/usr/bin/env bash
# Remote Phase 1 runner — executed ON the pod by run_session.sh.
#
# Installs deepsteer, then runs the identical five-step decomposition chain for
# each requested model family (run_phase1.py reads per-model band/layer from the
# registry). Always drops the .session_done sentinel on exit (even on failure)
# so the launcher never leaks the billed pod.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
MODELS="${MODELS:-all}"
VALIDATE="${VALIDATE:-0}"
cd "$REPO_DIR"

trap 'touch "$REPO_DIR/.session_done"' EXIT

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
echo ">> Phase 1 [VALIDATE=$VALIDATE] models=$MODELS"

VALIDATE="$VALIDATE" python papers/6_cross_model/scripts/run_phase1.py --models "$MODELS"
RC=$?
echo ">> run_phase1 exit code: $RC"
exit $RC
