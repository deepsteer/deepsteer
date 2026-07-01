#!/usr/bin/env bash
# Remote runner for the rich-subspace axis extraction (rank>1 test). Launch via:
#   REMOTE_SCRIPT=papers/d1_moral_subspace/runpod/remote_axis.sh ./run_session.sh
# Extracts d_fables + d_ethics on OLMo-3 Base (phase2_axis_extract.py); the cosine / rank /
# spectrum analysis (phase2_axis_analysis.py) runs locally on the rsynced-back artifacts.
# Drops .session_done on exit so the billed pod never leaks.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
BASE_MODEL="${BASE_MODEL:-allenai/Olmo-3-1025-7B}"
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
  python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$BASE_MODEL'); print('>> OLMo-3 config OK')" \
    || { echo "ERROR: transformers $TRANSFORMERS_VERSION cannot resolve OLMo-3 config; set TRANSFORMERS_VERSION."; exit 1; }
fi

echo ">> extracting candidate moral-axis directions on $BASE_MODEL ..."
VALIDATE="$VALIDATE" python "$REPO_DIR/papers/d1_moral_subspace/scripts/phase2_axis_extract.py" \
  --model "$BASE_MODEL"
echo ">> axis extraction done. Run locally after rsync-back:"
echo "   python papers/d1_moral_subspace/scripts/phase2_axis_analysis.py"
cat "$REPO_DIR/papers/d1_moral_subspace/outputs/phase2/axis/axis_meta.json" 2>/dev/null
