#!/usr/bin/env bash
# Remote runner for the rank-3 G3 re-run extraction. Launch via:
#   REMOTE_SCRIPT=papers/d1_moral_subspace/runpod/remote_g3_respec.sh ./run_session.sh
# Extracts instruct d_fables/d_ethics + base proto-refusal + instruct gate-refusal (vectors
# SAVED); the rank-3 V_moral assembly + recomputed null/control + projection + rank-sweep run
# locally (phase2_g3_respec.py). Drops .session_done on exit so the billed pod never leaks.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
BASE_MODEL="${BASE_MODEL:-allenai/Olmo-3-1025-7B}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-allenai/Olmo-3-7B-Instruct}"
cd "$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
trap 'touch "$REPO_DIR/.session_done"' EXIT

echo ">> cuda: $(python -c 'import torch;print(torch.cuda.is_available())' 2>&1)"
pip install -q --break-system-packages -e . 2>&1 | tail -1 || true
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"
echo ">> transformers==$TRANSFORMERS_VERSION + accelerate..."
pip install -q --break-system-packages "transformers==$TRANSFORMERS_VERSION" -U accelerate 2>&1 | tail -1 || true
[ -n "${PIP_EXTRA:-}" ] && pip install -q --break-system-packages -U $PIP_EXTRA 2>&1 | tail -1 || true
pip install -q --break-system-packages hf_xet >/dev/null 2>&1 && export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
echo ">> transformers: $(python -c 'import transformers;print(transformers.__version__)' 2>&1)"
if [ "$VALIDATE" != "1" ]; then
  python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$BASE_MODEL'); AutoConfig.from_pretrained('$INSTRUCT_MODEL'); print('>> OLMo-3 configs OK')" \
    || { echo "ERROR: transformers $TRANSFORMERS_VERSION cannot resolve OLMo-3 config; set TRANSFORMERS_VERSION."; exit 1; }
fi

echo ">> extracting instruct axes + base/instruct refusal vectors ..."
VALIDATE="$VALIDATE" python "$REPO_DIR/papers/d1_moral_subspace/scripts/phase2_g3_respec_extract.py" \
  --base-model "$BASE_MODEL" --instruct-model "$INSTRUCT_MODEL"
echo ">> done. Run locally after rsync-back:"
echo "   python papers/d1_moral_subspace/scripts/phase2_g3_respec.py"
