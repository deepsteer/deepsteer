#!/usr/bin/env bash
# Remote runner for Direction 3 / C1 (decision anatomy). One session per model (OLMo first, then
# Llama for the comparative prediction, then Qwen). Launch via the shared launcher:
#   VALIDATE=1 REMOTE_SCRIPT=papers/d3_decision_anatomy/scripts/remote_c1.sh \
#     SELF_PAPER=papers/d3_decision_anatomy RESULTS_SUBPATH=outputs \
#     SYNC_EXTRA=papers/d1_moral_subspace/outputs/full MODELS="olmo3" \
#     ./papers/d1_moral_subspace/runpod/run_session.sh
#
# Flow (compute-ordering + test-gates-before-GPU):
#   1. local gate (done):  python papers/d3_decision_anatomy/scripts/local_test.py
#   2. VALIDATE smoke:     tiny model, FULL integration (extract -> screen -> Stage1 -> Stage2)
#   3. real run:           one model per session (Stage 1+2; cells + riders ride the loaded model)
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
MODELS="${MODELS:-olmo3}"
cd "$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
trap 'touch "$REPO_DIR/.session_done"' EXIT

D3="$REPO_DIR/papers/d3_decision_anatomy"
OUT="$D3/outputs"; mkdir -p "$OUT"

echo ">> cuda: $(python -c 'import torch;print(torch.cuda.is_available())' 2>&1)"
pip install -q --break-system-packages -e . 2>&1 | tail -1 || true
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"
pip install -q --break-system-packages "transformers==$TRANSFORMERS_VERSION" -U accelerate 2>&1 | tail -1 || true
pip install -q --break-system-packages hf_xet >/dev/null 2>&1 && export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
echo ">> transformers: $(python -c 'import transformers;print(transformers.__version__)' 2>&1)"

# ---- zero-GPU pure-math gate always runs first (fast, no model) ----
echo ">> C1 local pure-math gate:"
python "$D3/scripts/local_test.py" || { echo "LOCAL GATE FAILED"; exit 1; }

# ---- cheap full-integration smoke (tiny model: extract -> screen -> Stage1 -> Stage2) ----
if [ "$VALIDATE" = "1" ]; then
  echo ">> VALIDATE smoke: full C1 integration on the tiny model"
  VALIDATE=1 python "$D3/scripts/c1_session.py" --key olmo3 --out "$OUT/_smoke" || exit 1
  echo ">> VALIDATE smoke OK (integration). Launch without VALIDATE for the real per-model run."
  exit 0
fi

# ---- real per-model run (registry layer; one model per session) ----
PANEL=("olmo3:allenai/Olmo-3-7B-Instruct" "qwen25:Qwen/Qwen2.5-7B-Instruct" "llama31:meta-llama/Llama-3.1-8B-Instruct")
repo_for () { local k="$1" e; for e in "${PANEL[@]}"; do [ "${e%%:*}" = "$k" ] && { echo "${e#*:}"; return; }; done; }

for key in $MODELS; do
  repo="$(repo_for "$key")"; [ -z "$repo" ] && { echo "WARN unknown key $key"; continue; }
  LAYER=$(python - "$key" <<'PY'
import sys; sys.path.insert(0,"papers/6_cross_model/scripts")
import model_registry as reg; print(reg.get(sys.argv[1]).primary_layer)
PY
)
  echo "==================== $key ($repo) layer=$LAYER ===================="
  python "$D3/scripts/c1_session.py" --model "$repo" --key "$key" --layer "$LAYER" --out "$OUT" || true
done
echo ">> C1 session done. rsync-back -> analyze c1_session_*.json (reconstruction, top heads, Stage-2 transport)."
