#!/usr/bin/env bash
# Remote runner for the reasoning-model extension (OLMo-3-7B-Think) G3 + content check.
# Launch via:
#   REMOTE_SCRIPT=papers/direction1_moral_subspace/runpod/remote_think_g3.sh ./run_session.sh
# Three GPU steps on the Think model (V_moral recomputed fresh in Think's space, no transfer):
#   1) phase2_extract       -> think/  moral_directions, diffs_moral_stories, act_sample, persona
#   2) phase2_axis_extract  -> think_axis/  fables + ethics axis directions
#   3) phase3_think_refusal -> think/  4-position refusal (P0 t_inst, P1 gate, P2 in-trace, P3 ans)
# The rank-3 span assembly, content-dominated check, null/control recompute, and per-position
# projection run locally afterward (phase3_think_g3.py). Drops .session_done on exit.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
THINK_MODEL="${THINK_MODEL:-allenai/Olmo-3-7B-Think}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
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
# Resilient downloader: hf_transfer (multi-threaded HTTP, honors HF_HUB_DOWNLOAD_TIMEOUT and
# auto-retries) instead of the xet backend, which stalled on the Think weights and ignores the
# timeout (HF_XET_HIGH_PERFORMANCE never errors out -> the 14GB pull wedges forever).
export HF_HUB_DISABLE_XET=1
pip install -q --break-system-packages hf_transfer >/dev/null 2>&1 && export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
echo ">> downloader: xet=disabled hf_transfer=${HF_HUB_ENABLE_HF_TRANSFER:-0}"
# Pre-pull the full repo once, explicitly, so a stalled shard errors+retries here (not mid-load).
# Whole repo (no allow_patterns) so the chat-template file -- which lives OUTSIDE
# tokenizer_config.json on this model -- is guaranteed to come down.
export THINK_MODEL
python - <<'PY' || { echo "ERROR: Think weight pre-pull failed; rerun (cache resumes)."; exit 1; }
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ.get("THINK_MODEL", "allenai/Olmo-3-7B-Think"))
print(">> Think weights cached")
PY
echo ">> transformers: $(python -c 'import transformers;print(transformers.__version__)' 2>&1)"
if [ "$VALIDATE" != "1" ]; then
  python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$THINK_MODEL'); print('>> OLMo-3-Think config OK')" \
    || { echo "ERROR: transformers $TRANSFORMERS_VERSION cannot resolve OLMo-3-Think config; set TRANSFORMERS_VERSION."; exit 1; }
fi

D="$REPO_DIR/papers/direction1_moral_subspace/scripts"
echo ">> [1/3] moral/persona/act_sample on Think ..."
VALIDATE="$VALIDATE" python "$D/phase2_extract.py" --model "$THINK_MODEL" \
  --out "$REPO_DIR/papers/direction1_moral_subspace/outputs/phase2/think"
echo ">> [2/3] fables + ethics axis on Think ..."
VALIDATE="$VALIDATE" python "$D/phase2_axis_extract.py" --model "$THINK_MODEL" \
  --out "$REPO_DIR/papers/direction1_moral_subspace/outputs/phase2/think_axis"
echo ">> [3/3] 4-position refusal on Think (max_new_tokens=$MAX_NEW_TOKENS) ..."
VALIDATE="$VALIDATE" python "$D/phase3_think_refusal.py" --model "$THINK_MODEL" \
  --max-new-tokens "$MAX_NEW_TOKENS"
echo ">> done. Run locally after rsync-back:  python $D/phase3_think_g3.py"
