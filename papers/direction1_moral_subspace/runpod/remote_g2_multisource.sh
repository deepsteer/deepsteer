#!/usr/bin/env bash
# Remote runner for multi-source GATE G2 (fables + ETHICS held-out + ETHICS extraction-gap).
# Launch via:
#   REMOTE_SCRIPT=papers/direction1_moral_subspace/runpod/remote_g2_multisource.sh ./run_session.sh
# One BASE model load; collects layer-16 activations on each source's eval surf + clean
# paraphrase, projects onto that source's frozen base mean-diff direction, reports acc_surf /
# acc_para / gap. No gate raises (G2 is reported, not a STOP here; the G2<->G3 distinction in
# RESULTS.md). Drops .session_done on exit so the billed pod never leaks.
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

echo ">> multi-source G2 (fables + ETHICS) on the base model ..."
VALIDATE="$VALIDATE" python "$REPO_DIR/papers/direction1_moral_subspace/scripts/phase2_g2_multisource.py" \
  --model "$BASE_MODEL"
echo ">> done. Result: papers/direction1_moral_subspace/outputs/phase2/g2_multisource_result.json"
