#!/usr/bin/env bash
# Re-run ONLY the 4-position refusal on OLMo-3-7B-Think (the </think> detection fix).
# The moral/persona/act_sample + fables/ethics axis are already extracted and rsynced back
# locally (think/ + think_axis/), and refusal lives in the SAME activation space (same model,
# layer 16), so only phase3_think_refusal needs re-running -- saving steps [1/3] and [2/3].
# Launch via:
#   REMOTE_SCRIPT=papers/direction1_moral_subspace/runpod/remote_think_refusal_only.sh ./run_session.sh
# Drops .session_done on exit so the billed pod never leaks. cap = MAX_NEW_TOKENS (default 2048).
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
THINK_MODEL="${THINK_MODEL:-allenai/Olmo-3-7B-Think}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
cd "$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
trap 'touch "$REPO_DIR/.session_done"' EXIT

echo ">> cuda: $(python -c 'import torch;print(torch.cuda.is_available())' 2>&1)"
pip install -q --break-system-packages -e . 2>&1 | tail -1 || true
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"
pip install -q --break-system-packages "transformers==$TRANSFORMERS_VERSION" -U accelerate 2>&1 | tail -1 || true
[ -n "${PIP_EXTRA:-}" ] && pip install -q --break-system-packages -U $PIP_EXTRA 2>&1 | tail -1 || true
# Disable xet (it wedges on the Think weights and ignores HF_HUB_DOWNLOAD_TIMEOUT); standard
# HTTP downloader is reliable and timeout-honoring.
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
echo ">> transformers: $(python -c 'import transformers;print(transformers.__version__)' 2>&1) | xet=disabled"
if [ "$VALIDATE" != "1" ]; then
  python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$THINK_MODEL'); print('>> OLMo-3-Think config OK')" \
    || { echo "ERROR: transformers $TRANSFORMERS_VERSION cannot resolve OLMo-3-Think config; set TRANSFORMERS_VERSION."; exit 1; }
fi
export THINK_MODEL
python - <<'PY' || { echo "ERROR: Think weight pre-pull failed; rerun (cache resumes)."; exit 1; }
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ.get("THINK_MODEL", "allenai/Olmo-3-7B-Think"))
print(">> Think weights cached")
PY

echo ">> 4-position refusal re-run on Think (</think> fix, cap=$MAX_NEW_TOKENS) ..."
VALIDATE="$VALIDATE" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  python "$REPO_DIR/papers/direction1_moral_subspace/scripts/phase3_think_refusal.py" \
  --model "$THINK_MODEL"
echo ">> done. closed-rate + gen_len in think/think_refusal_meta.json; samples in"
echo "   think/think_refusal_debug.json. Run locally: phase3_think_g3.py"
