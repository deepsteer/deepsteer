#!/usr/bin/env bash
# GPT-OSS-20B = TERMINAL Direction-1 experiment. THREE payloads on one A100-80GB pod:
#   (1) instruct MFT extraction (OLMo-3-Instruct) -> instruct-MFT-null (tightest 0.1044 repro)
#   (2) GPT-OSS extraction (moral/persona/axis, layer 12, harmony) -> gpt_oss/ + gpt_oss_axis/
#   (3) GPT-OSS 4-position refusal (harmony in-trace anchor; P0/P1/P2 symmetric-window + P3)
# PILOT_N>0 runs ONLY the GPT-OSS refusal pilot (fast: measure GPT-OSS trace lengths to SIZE
# COT_WINDOW_N + MAX_NEW_TOKENS before the full run -- GPT-OSS traces are short, so N/cap must
# be re-derived, not copied from OLMo). Launch:
#   PILOT_N=8 GPU_TYPES='NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA H100 80GB HBM3' \
#     DISK_GB=220 REMOTE_SCRIPT=papers/direction1_moral_subspace/runpod/remote_gptoss.sh ./run_session.sh
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
GPTOSS_MODEL="${GPTOSS_MODEL:-openai/gpt-oss-20b}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-allenai/Olmo-3-7B-Instruct}"
PILOT_N="${PILOT_N:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"   # GPT-OSS traces are short (~280); 1024 is generous
P23_N="${P23_N:-0}"
COT_WINDOW_N="${COT_WINDOW_N:-}"           # set from the pilot (GPT-OSS: likely ~64, not 256)
cd "$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
# Timestamp every log line (throughput on the long run); redirect installed here so the trap fires.
exec > >(while IFS= read -r l; do printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$l"; done) 2>&1
trap 'touch "$REPO_DIR/.session_done"' EXIT

D="$REPO_DIR/papers/direction1_moral_subspace/scripts"
echo ">> cuda: $(python -c 'import torch;print(torch.cuda.is_available())' 2>&1)"
pip install -q --break-system-packages -e . 2>&1 | tail -1 || true
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"   # >=4.55 for gpt_oss; 5.12.1 also runs OLMo-3
pip install -q --break-system-packages "transformers==$TRANSFORMERS_VERSION" -U accelerate 2>&1 | tail -1 || true
[ -n "${PIP_EXTRA:-}" ] && pip install -q --break-system-packages -U $PIP_EXTRA 2>&1 | tail -1 || true
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

# mxfp4 dequant needs torch>=2.6 (torch.accelerator). Upgrade in place if the image is older.
if [ "$VALIDATE" != "1" ]; then
  python -c "import torch,sys; sys.exit(0 if hasattr(torch,'accelerator') else 1)" 2>/dev/null || {
    echo ">> upgrading torch->2.6.0+cu124 for mxfp4 (torch.accelerator) ..."
    pip install -q --break-system-packages torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
      --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1 || true
  }
  python -c "import torch;print('>> torch',torch.__version__,'accelerator',hasattr(torch,'accelerator'))"
  python -c "from transformers import Mxfp4Config; print('>> Mxfp4Config OK')" \
    || echo ">> WARN: Mxfp4Config unavailable; will rely on torch_dtype auto-dequant"
  python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$GPTOSS_MODEL'); print('>> gpt-oss config OK')" \
    || { echo "ERROR: transformers $TRANSFORMERS_VERSION cannot resolve gpt-oss config."; exit 1; }
fi
echo ">> transformers: $(python -c 'import transformers;print(transformers.__version__)' 2>&1) | xet=disabled"

# ---- PILOT: GPT-OSS refusal only (size N + cap from real trace lengths) ----
if [ "$PILOT_N" -gt 0 ] 2>/dev/null; then
  echo ">> PILOT: GPT-OSS refusal N=$PILOT_N cap=$MAX_NEW_TOKENS -> gpt_oss/pilot/ (harmony, layer 12)"
  VALIDATE="$VALIDATE" THINK_TAG=gpt_oss PILOT_N="$PILOT_N" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    COT_WINDOW_N="${COT_WINDOW_N:-64}" \
    python "$D/phase3_think_refusal.py" --model "$GPTOSS_MODEL"
  echo ">> pilot done. Inspect gpt_oss/pilot/think_refusal_debug.json (reason_len, closed, window-ok)."
  exit 0
fi

# ---- FULL: three payloads ----
echo ">> [1/3] instruct MFT extraction (OLMo-3-Instruct, layer 16) -> instruct/mft_directions.npz"
VALIDATE="$VALIDATE" MATCH_LAYER=16 python "$D/phase2_extract.py" --mft \
  --model "$INSTRUCT_MODEL" --out "$REPO_DIR/papers/direction1_moral_subspace/outputs/phase2/instruct"

echo ">> [2/3] GPT-OSS extraction (moral/persona/act_sample + fables/ethics axis, layer 12)"
VALIDATE="$VALIDATE" MATCH_LAYER=12 python "$D/phase2_extract.py" \
  --model "$GPTOSS_MODEL" --out "$REPO_DIR/papers/direction1_moral_subspace/outputs/phase2/gpt_oss"
VALIDATE="$VALIDATE" MATCH_LAYER=12 python "$D/phase2_axis_extract.py" \
  --model "$GPTOSS_MODEL" --out "$REPO_DIR/papers/direction1_moral_subspace/outputs/phase2/gpt_oss_axis"

echo ">> [3/3] GPT-OSS 4-position refusal (harmony, cap=$MAX_NEW_TOKENS, P23_N=$P23_N, N=${COT_WINDOW_N:-64})"
VALIDATE="$VALIDATE" THINK_TAG=gpt_oss MAX_NEW_TOKENS="$MAX_NEW_TOKENS" P23_N="$P23_N" \
  COT_WINDOW_N="${COT_WINDOW_N:-64}" \
  python "$D/phase3_think_refusal.py" --model "$GPTOSS_MODEL"
echo ">> done. Run locally after rsync-back:"
echo "   THINK_TAG=gpt_oss python $D/phase3_think_g3.py       # GPT-OSS P0/P1/P2(+P2_FULL)/P3"
echo "   python $D/compute_instruct_mft_null.py               # instruct-gate judged-vs-judged"
