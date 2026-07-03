#!/usr/bin/env bash
# Remote runner for the GPT-OSS-20B Tier-1 commitment-axis session (Amendment 5 finalization).
# One modest 20B session. Launch via the shared launcher:
#   VALIDATE=1 REMOTE_SCRIPT=papers/d3_decision_anatomy/scripts/remote_tier1.sh \
#     SELF_PAPER=papers/d3_decision_anatomy RESULTS_SUBPATH=outputs MODELS="gpt_oss_20b" \
#     ./papers/d1_moral_subspace/runpod/run_session.sh
#
# Flow (compute-ordering + test-gates-before-GPU):
#   1. local pure-math gate (always, fast, no model)
#   2. VALIDATE smoke: tiny model, full Tier-1 integration (gate -> psychometric -> prefill -> commitment)
#   3. real run: openai/gpt-oss-20b (bf16-dequant), one session
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
MODELS="${MODELS:-gpt_oss_20b}"
MAXTOK="${MAXTOK:-}"
cd "$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
trap 'touch "$REPO_DIR/.session_done"' EXIT

D3="$REPO_DIR/papers/d3_decision_anatomy"
OUT="$D3/outputs"; mkdir -p "$OUT"

echo ">> cuda: $(python -c 'import torch;print(torch.cuda.is_available())' 2>&1)"
pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -1 \
  || pip install -q --break-system-packages -e . 2>&1 | tail -1
[ -n "${PIP_EXTRA:-}" ] && pip install -q --break-system-packages -U $PIP_EXTRA 2>&1 | tail -2 || true
pip install -q --break-system-packages hf_xet >/dev/null 2>&1 && export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

# GPT-OSS mxfp4 quantizer calls torch.accelerator (torch>=2.6); upgrade the matched trio only if needed.
if ! python -c 'import torch; torch.accelerator' 2>/dev/null; then
  echo ">> upgrading torch trio to 2.6.0+cu124 (matched torchvision/torchaudio) for GPT-OSS..."
  pip install -q --break-system-packages \
    "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3 \
    || echo ">> WARN: torch trio upgrade failed; GPT-OSS load will fail-fast."
fi
echo ">> torch: $(python -c 'import torch;print(torch.__version__, "accel", hasattr(torch,"accelerator"))' 2>&1)"
echo ">> transformers: $(python -c 'import transformers;print(transformers.__version__)' 2>&1)"
[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ] && echo ">> HF_TOKEN: set" || echo ">> HF_TOKEN: unset (gpt-oss public)"

# ---- zero-GPU pure-math gate always runs first (fast, no model) ----
echo ">> Tier-1 local pure-math gate:"
python "$D3/scripts/local_test.py" || { echo "LOCAL GATE FAILED"; exit 1; }

# ---- cheap full-integration smoke (tiny model: gate -> psychometric -> prefill -> commitment) ----
if [ "$VALIDATE" = "1" ]; then
  echo ">> VALIDATE smoke: full Tier-1 integration on the tiny model"
  VALIDATE=1 python "$D3/scripts/gptoss_tier1.py" --key gpt_oss_20b --out "$OUT/_smoke" || exit 1
  echo ">> VALIDATE smoke OK (integration). Launch without VALIDATE for the real 20B session."
  exit 0
fi

# ---- real GPT-OSS-20B Tier-1 session ----
ARGS=""
[ -n "${LAYER_OVERRIDE:-}" ] && ARGS="$ARGS --layer $LAYER_OVERRIDE"     # depth override (default = registry primary=12)
[ -n "$MAXTOK" ] && ARGS="$ARGS --max-new-tokens $MAXTOK"
echo "==================== gpt_oss_20b Tier-1 (BOUNDARY=${BOUNDARY:-0}) ===================="
BOUNDARY="${BOUNDARY:-0}" python "$D3/scripts/gptoss_tier1.py" --key gpt_oss_20b --out "$OUT" $ARGS
RC=$?
echo ">> Tier-1 session done (exit $RC). rsync-back -> analyze tier1_session_gpt_oss_20b.json "
echo ">>   (position_gate.position_valid, psychometric bands, deliberation.asymmetry_A, commitment)."
exit $RC
