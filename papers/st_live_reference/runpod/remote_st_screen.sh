#!/usr/bin/env bash
# Remote runner for the ST T1.0 relaunch (papers/ST_PLAN.md v0.2.9 §2, §7; Amendments 2-7;
# T0.11). Orion launches (keys stay in Orion's terminal). Generation runs in a vLLM venv
# (Amendment 7); readouts run in the HF stack. One model at a time, in MODELS order; each runs
# phase V (vLLM: pilot + speculation probes, C0 answers, screen / parity) then phase H (HF: gates,
# padded check, fast-path rung, readouts / memory probe); the last output is the re-priced Tier 1
# table (outputs/t10/tier1_reprice.md) for G1.
#
#   GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe" MIN_FREE_GB=15 \
#   DISK_GB=250 REMOTE_SCRIPT=papers/st_live_reference/runpod/remote_st_screen.sh \
#     SELF_PAPER=papers/st_live_reference RESULTS_SUBPATH=outputs/t10 \
#     ./papers/d1_moral_subspace/runpod/run_session.sh
#   VALIDATE=1 <same> ......  the first real vLLM run: stacks, fast gates, the GPT-OSS-20B vLLM
#                             smoke (EAGLE3 on/off) and HF fast-path rung, dry runs, then the
#                             slow gates; its own envelope stop at VALIDATE_MAX_SECONDS (0.3 h).
#   MODELS="qwen38_27b" <same>  a subset, in the given order.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
MODELS="${MODELS:-gpt_oss_20b qwen38_27b muse_glimmer_30b}"
MODELS="${MODELS//,/ }"
CLEAN_HF_CACHE="${CLEAN_HF_CACHE:-1}"   # drop each model's (and drafter's) weights after its unit
PIP_INSTALL="${PIP_INSTALL:-1}"         # 0 = local VALIDATE on a machine with the stack already
VLLM_SMOKE="${VLLM_SMOKE:-1}"           # 0 = local VALIDATE (no CUDA): skip the real vLLM smoke
VLLM_VENV="${VLLM_VENV:-/workspace/venv_vllm}"
VLLM_VERSION=0.31.0                      # Amendment 7 implementation (default wheel CUDA 12.9)
VALIDATE_MAX_SECONDS="${VALIDATE_MAX_SECONDS:-1080}"     # 0.3 A100-h
SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-6.8}"            # 1.0 + 2.5 + 2.5 + setup
cd "$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
PY="${PY:-python}"
trap 'touch "$REPO_DIR/.session_done"' EXIT

# ---- envelope stop: a watchdog that ends the session, kills the workers, and touches the
# sentinel so the launcher still downloads what was banked and tears the pod down ----
if [ "$VALIDATE" = "1" ]; then MAX_S="$VALIDATE_MAX_SECONDS"; else
  MAX_S="$($PY -c "print(int(float('$SESSION_MAX_HOURS') * 3600))")"; fi
MAIN_PID=$$
( sleep "$MAX_S"
  echo ">> WATCHDOG: envelope stop after ${MAX_S}s (banked outputs are downloaded)"
  pkill -TERM -f "scripts/pod_st_screen.py" ; pkill -TERM -f "scripts/gen_vllm.py"
  pkill -TERM -f "scripts/smoke_st.py" ; pkill -TERM -f "pytest -q tests/scripts"
  sleep 20; touch "$REPO_DIR/.session_done"; kill -TERM "$MAIN_PID" 2>/dev/null ) &
WATCHDOG_PID=$!
trap 'pkill -P "$WATCHDOG_PID" 2>/dev/null; kill "$WATCHDOG_PID" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
echo ">> envelope stop armed: ${MAX_S}s"

OUT="$REPO_DIR/papers/st_live_reference/outputs/t10"; mkdir -p "$OUT"
S=papers/st_live_reference/scripts/pod_st_screen.py
DATE_PIN="$(date -u +%F)"   # one date for the whole session (Muse renders it; Amendment 2 item 6)
echo ">> date pin: $DATE_PIN   models: $MODELS"

if [ "$PIP_INSTALL" = "1" ]; then
  # HF stack: transformers 5.17.0 (muse_glimmer) on the W4/GPT-OSS torch 2.6.0+cu124 trio
  if ! $PY -c 'import torch,sys; v=tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]); sys.exit(0 if v>=(2,6) else 1)' 2>/dev/null; then
    echo ">> torch $($PY -c 'import torch;print(torch.__version__)') < 2.6; upgrading the trio to 2.6.0+cu124 (W4 stack)..."
    pip install -q --break-system-packages "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
      --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3
  fi
  pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -1 || pip install -q --break-system-packages -e . 2>&1 | tail -1
  pip install -q --break-system-packages "transformers==5.17.0" -U accelerate pytest pyyaml 2>&1 | tail -1 || true
  pip install -q --break-system-packages hf_xet >/dev/null 2>&1 && export HF_XET_HIGH_PERFORMANCE=1
fi
echo ">> HF stack: torch $($PY -c 'import torch;print(torch.__version__)' 2>&1)  transformers $($PY -c 'import transformers;print(transformers.__version__)' 2>&1)  cuda $($PY -c 'import torch;print(torch.cuda.is_available())' 2>&1)"
$PY -c 'import transformers,sys; sys.exit(0 if transformers.__version__=="5.17.0" else 1)' \
  || { echo "FATAL: transformers is not 5.17.0 (Amendment 4 item 7)"; exit 1; }
command -v nvidia-smi >/dev/null && echo ">> driver: $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"

# ---- inputs the screen needs (committed; never regenerated here) ----
for f in papers/st_live_reference/split_v1.json papers/st_live_reference/data/s40_reference.json \
         papers/kdg_panel/data/harm_divergence/hdiv_set.json \
         papers/kdg_panel/data/harm_divergence/h1_kept_ids.json papers/kdg_panel/models.yaml; do
  [ -f "$f" ] || { echo "FATAL: $f missing"; exit 1; }
done

# ---- fast gates first (no model) ----
GATE_TESTS="tests/scripts/test_st_split.py tests/scripts/test_st_formats.py tests/scripts/test_st_screen.py tests/scripts/test_st_analysis.py tests/scripts/test_st_probes.py tests/scripts/test_st_vllm_worker.py tests/core/test_model_interface_vlm.py tests/kdg"
echo ">> ST fast gates:"
$PY -m pytest -q $GATE_TESTS -m "not slow" || { echo "LOCAL GATE FAILED"; exit 1; }

# ---- the vLLM venv (Amendment 7): v0.31.0 on CUDA 12.9 torch, transformers 5.17.0 ----
if [ "$VLLM_SMOKE" = "1" ]; then
  if [ ! -x "$VLLM_VENV/bin/python" ]; then
    echo ">> creating $VLLM_VENV (vllm==$VLLM_VERSION)"
    $PY -m venv "$VLLM_VENV"
    "$VLLM_VENV/bin/pip" install -q --upgrade pip 2>&1 | tail -1
    "$VLLM_VENV/bin/pip" install -q "vllm==$VLLM_VERSION" \
      --extra-index-url https://download.pytorch.org/whl/cu129 2>&1 | tail -3
    "$VLLM_VENV/bin/pip" install -q "transformers==5.17.0" pyyaml 2>&1 | tail -1
  fi
  "$VLLM_VENV/bin/python" -c "import vllm, torch, transformers, sys; print('>> vLLM stack:', vllm.__version__, torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), 'transformers', transformers.__version__); sys.exit(0 if vllm.__version__=='$VLLM_VERSION' and torch.cuda.is_available() else 1)" \
    || { echo "FATAL: vLLM venv not usable (version or CUDA)"; exit 1; }
  export ST_VLLM_PYTHON="$VLLM_VENV/bin/python"
fi

if [ "$VALIDATE" = "1" ]; then
  if [ "$VLLM_SMOKE" = "1" ]; then
    echo "==================== VALIDATE: GPT-OSS-20B vLLM smoke + HF fast-path rung ===================="
    $PY papers/st_live_reference/scripts/smoke_st.py --out "$OUT/_validate" \
      || { echo "SMOKE FAILED"; exit 1; }
  fi
  for m in $MODELS; do
    $PY $S --model "$m" --dry-run --date-pin "$DATE_PIN" --out "$OUT/_dry" \
      || { echo "DRY RUN FAILED ($m)"; exit 1; }
  done
  echo ">> ST slow gates (tiny real checkpoints):"
  $PY -m pytest -q $GATE_TESTS -m slow || { echo "SLOW GATE FAILED"; exit 1; }
  echo ">> VALIDATE: smoke, dry runs and gates OK. Launch without VALIDATE for the real run."; exit 0
fi

for m in $MODELS; do
  $PY $S --model "$m" --dry-run --date-pin "$DATE_PIN" --out "$OUT/_dry" \
    || { echo "DRY RUN FAILED ($m)"; exit 1; }
done
VRAM_GB="$($PY -c 'import torch;print(int(torch.cuda.get_device_properties(0).total_memory/1e9)) if torch.cuda.is_available() else 0' 2>/dev/null || echo 0)"
echo ">> GPU VRAM: ${VRAM_GB} GB"
[ "${VRAM_GB:-0}" -lt 75 ] && { echo "FATAL: need an 80 GB GPU (Muse Glimmer 30B is ~60 GB in bf16)."; exit 1; }

hf_cache_dirs() {  # HF hub cache directories of a registry key's model and drafter repos
  $PY - "$1" <<'PY'
import sys, yaml
from huggingface_hub.constants import HF_HUB_CACHE
e = yaml.safe_load(open("papers/kdg_panel/models.yaml"))["st"][sys.argv[1]]
repos = [e["repo"]] + [r for r in [((e.get("vllm") or {}).get("draft") or {}).get("repo")] if r]
print(" ".join(f"{HF_HUB_CACHE}/models--{r.replace('/', '--')}" for r in repos))
PY
}

RC_ALL=0
for m in $MODELS; do
  echo "==================== t10/$m ===================="
  $PY $S --model "$m" --date-pin "$DATE_PIN" --out "$OUT"; rc=$?
  echo ">> $m exit $rc (0 ok, 2 gate failed, 3 envelope stop; banked either way)"
  $PY -c "import sys; sys.path.insert(0,'papers/kdg_panel/scripts'); from kdg_pod_lib import verify_manifest as v; bad=v('$OUT/$m/manifest_kdg.json'); print('manifest verify:', 'OK' if not bad else bad[:5])" \
    || echo "WARN: $m manifest verify did not run"
  [ "$rc" -ne 0 ] && RC_ALL=$rc
  if [ "$CLEAN_HF_CACHE" = "1" ]; then
    for d in $(hf_cache_dirs "$m"); do [ -d "$d" ] && rm -rf "$d" && echo ">> removed $d"; done
  fi
done

echo "==================== t10/reprice ===================="
$PY papers/st_live_reference/scripts/reprice_tier1.py --screen-dir "$OUT" || echo "WARN: re-price failed"
echo ">> ST T1.0 relaunch done (rc=$RC_ALL). rsync-back -> papers/st_live_reference/outputs/t10/"
exit $RC_ALL
