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
#   STEPS=rider RESULTS_SUBPATH=outputs/t10b SYNC_OUTPUTS="outputs/t10/qwen38_27b/,outputs/t10/muse_glimmer_30b/,outputs/t10/gpt_oss_20b/" <same>
#                             the G1 rider pod (author, 2026-10-19; Amendment 8): Qwen3.8 phase H
#                             on its banked relaunch phase V (--resume-from), Muse's dose-0
#                             re-check on the corrected rung (rider (a)), GPT-OSS's corrected rung
#                             with per-row arrays and the eager HF memory probe (rider (b)); writes
#                             outputs/t10b; envelope 3.0 h.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
STEPS="${STEPS:-}"
if [ "$STEPS" = "rider" ]; then
  MODELS="${MODELS:-qwen38_27b muse_glimmer_30b gpt_oss_20b}"; OUT_SUB=t10b
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-3.0}"
else
  MODELS="${MODELS:-gpt_oss_20b qwen38_27b muse_glimmer_30b}"; OUT_SUB=t10
fi
MODELS="${MODELS//,/ }"
CLEAN_HF_CACHE="${CLEAN_HF_CACHE:-1}"   # drop each model's (and drafter's) weights after its unit
PIP_INSTALL="${PIP_INSTALL:-1}"         # 0 = local VALIDATE on a machine with the stack already
VLLM_SMOKE="${VLLM_SMOKE:-1}"           # 0 = local VALIDATE (no CUDA): skip the real vLLM smoke
VLLM_VENV="${VLLM_VENV:-/workspace/venv_vllm}"
VLLM_VERSION=0.31.0                      # Amendment 7 implementation
# the PyPI wheel of 0.31.0 is a CUDA 13 build (requires_dist pins [cu13] kernels; its _C links
# libcudart.so.13: the first VALIDATE pod, 2026-10-18); the CUDA 12.9 build is the GitHub release
# asset, pinned by the sha256 GitHub publishes for it, over torch 2.13.0+cu129
VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_SHA256=5176ab746f4e736c4a354a11d0620abf0a941de287cad937ac34039293ca6f5f
CU129_INDEX=https://download.pytorch.org/whl/cu129
VLLM_CONSTRAINTS=papers/st_live_reference/runpod/vllm_constraints.txt
VALIDATE_MAX_SECONDS="${VALIDATE_MAX_SECONDS:-1080}"     # 0.3 A100-h
SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-6.8}"            # 1.0 + 2.5 + 2.5 + setup (rider: 3.0)
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
  sleep 20; cp "$REPO_DIR/session.log" "$REPO_DIR/papers/st_live_reference/outputs/$OUT_SUB/session_watchdog.log" 2>/dev/null
  touch "$REPO_DIR/.session_done"; kill -TERM "$MAIN_PID" 2>/dev/null ) &
WATCHDOG_PID=$!
# on exit: the session log goes into the downloaded outputs before the sentinel (the launcher
# syncs only RESULTS_SUBPATH; the second and third VALIDATE pods lost their logs)
trap 'pkill -P "$WATCHDOG_PID" 2>/dev/null; kill "$WATCHDOG_PID" 2>/dev/null; mkdir -p "$REPO_DIR/papers/st_live_reference/outputs/$OUT_SUB"; cp "$REPO_DIR/session.log" "$REPO_DIR/papers/st_live_reference/outputs/$OUT_SUB/session_$(date -u +%Y%m%dT%H%M%S).log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
echo ">> envelope stop armed: ${MAX_S}s"

OUT="$REPO_DIR/papers/st_live_reference/outputs/$OUT_SUB"; mkdir -p "$OUT"
BANKED="$REPO_DIR/papers/st_live_reference/outputs/t10"
# the rider pod's per-model mode (author, 2026-10-19): Qwen phase H on its banked phase V, Muse's
# dose-0 re-check (rider (a)), GPT-OSS's corrected rung + eager memory probe (rider (b))
mode_args() {
  [ "$STEPS" = "rider" ] || return 0
  case "$1" in
    qwen38_27b) echo "--resume-from $2/qwen38_27b" ;;
    muse_glimmer_30b) echo "--resume-from $2/muse_glimmer_30b --dose0-only" ;;
    gpt_oss_20b) echo "--rider-oss" ;;
  esac
}
# session wall time for the re-price's cumulative A100-h (author, 2026-10-19)
echo "{\"session_start_utc\": \"$(date -u +%FT%TZ)\", \"validate\": $VALIDATE}" > "$OUT/session_start.json"
S=papers/st_live_reference/scripts/pod_st_screen.py
DATE_PIN="$(date -u +%F)"   # one date for the whole session (Muse renders it; Amendment 2 item 6)
echo ">> date pin: $DATE_PIN   models: $MODELS   steps: ${STEPS:-screen}   out: outputs/$OUT_SUB"

# ---- 1. the vLLM venv first, with the deep import check (a missing CUDA library shows in
# minutes, before the HF stack and the gates; author, 2026-10-18) ----
if [ "$VLLM_SMOKE" = "1" ]; then
  command -v nvidia-smi >/dev/null && echo ">> driver: $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"
  if [ ! -x "$VLLM_VENV/bin/python" ]; then
    echo ">> creating $VLLM_VENV (vllm==$VLLM_VERSION, CUDA 12.9 line)"
    $PY -m venv "$VLLM_VENV"
    "$VLLM_VENV/bin/pip" install -q --upgrade pip 2>&1 | tail -1
    "$VLLM_VENV/bin/pip" install -q "torch==2.13.0+cu129" "torchvision==0.28.0+cu129" \
      "torchaudio==2.11.0+cu129" --index-url "$CU129_INDEX" 2>&1 | tail -3
    "$VLLM_VENV/bin/pip" install -q "vllm @ ${VLLM_WHEEL_URL}#sha256=${VLLM_WHEEL_SHA256}" \
      "transformers==5.17.0" pyyaml ninja -c "$VLLM_CONSTRAINTS" --extra-index-url "$CU129_INDEX" \
      2>&1 | tail -3
  fi
  PATH="$VLLM_VENV/bin:$PATH" VLLM_VERSION="$VLLM_VERSION" "$VLLM_VENV/bin/python" - <<'PY' || { echo "FATAL: vLLM venv not usable (see the check above)"; exit 1; }
import importlib.metadata as md, os, shutil, sys
import torch
from vllm import LLM, SamplingParams  # noqa: F401  (the worker's own import; pulls torchcodec)
import flashinfer, torchcodec  # noqa: F401,E401
from cuda.bindings import nvrtc
from vllm.model_executor.models import gpt_oss, muse_glimmer, qwen3_5  # noqa: F401
v = {p: md.version(p) for p in ("vllm", "torch", "torchcodec", "cuda-bindings", "cuda-python",
                                "flashinfer-python", "transformers")}
err, major, minor = nvrtc.nvrtcVersion()  # loads libnvrtc: the CUDA 13 failure mode
print(">> vLLM stack:", v, "| torch cuda", torch.version.cuda, torch.cuda.is_available(),
      "| nvrtc", major, minor, err)
checks = {
    "vllm": v["vllm"].split("+")[0] == os.environ["VLLM_VERSION"],
    "torch_cuda_12.9": (torch.version.cuda or "").startswith("12.9"),
    "cuda_available": torch.cuda.is_available(),
    "transformers": v["transformers"] == "5.17.0",
    "cuda_bindings_12": v["cuda-bindings"].startswith("12."),
    "torchcodec_cu129": v["torchcodec"].endswith("+cu129"),
    "nvrtc_12": int(major) == 12,
    "ninja_on_worker_path": shutil.which("ninja") is not None,
}
bad = [k for k, ok in checks.items() if not ok]
print(">> vLLM venv checks:", "OK" if not bad else f"FAILED {bad}")
sys.exit(1 if bad else 0)
PY
  export ST_VLLM_PYTHON="$VLLM_VENV/bin/python"
fi

# ---- 2. the HF stack ----
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

# ---- inputs the screen needs (committed; never regenerated here) ----
for f in papers/st_live_reference/split_v1.json papers/st_live_reference/data/s40_reference.json \
         papers/kdg_panel/data/harm_divergence/hdiv_set.json \
         papers/kdg_panel/data/harm_divergence/h1_kept_ids.json papers/kdg_panel/models.yaml; do
  [ -f "$f" ] || { echo "FATAL: $f missing"; exit 1; }
done
if [ "$STEPS" = "rider" ]; then  # the banked relaunch records the rider resumes from (SYNC_OUTPUTS)
  for m in qwen38_27b muse_glimmer_30b gpt_oss_20b; do
    [ -f "$BANKED/$m/screen_record.json" ] || { echo "FATAL: banked $BANKED/$m missing (SYNC_OUTPUTS)"; exit 1; }
  done
  for m in qwen38_27b muse_glimmer_30b; do  # provenance before any download (fatal on mismatch)
    $PY $S --model "$m" --out "$OUT/_provenance" --resume-from "$BANKED/$m" --provenance-only \
      || { echo "FATAL: provenance check failed ($m)"; exit 1; }
  done
fi
# dry runs: the screen; for the rider, each mode on a dry banked run of the same model
dry_runs() {
  for m in $MODELS; do
    $PY $S --model "$m" --dry-run --date-pin "$DATE_PIN" --out "$OUT/_dry" \
      || { echo "DRY RUN FAILED ($m)"; return 1; }
    if [ "$STEPS" = "rider" ]; then
      $PY $S --model "$m" --dry-run --date-pin "$DATE_PIN" --out "$OUT/_dry_rider" $(mode_args "$m" "$OUT/_dry") \
        || { echo "DRY RUN FAILED ($m, rider mode)"; return 1; }
    fi
  done
}
GATE_TESTS="tests/scripts/test_st_split.py tests/scripts/test_st_formats.py tests/scripts/test_st_screen.py tests/scripts/test_st_analysis.py tests/scripts/test_st_probes.py tests/scripts/test_st_vllm_worker.py tests/core/test_model_interface_vlm.py tests/kdg"

if [ "$VALIDATE" = "1" ]; then
  # ---- 3. VALIDATE: the real vLLM smoke right after the stacks, then the gates ----
  if [ "$VLLM_SMOKE" = "1" ]; then
    echo "==================== VALIDATE: GPT-OSS-20B vLLM smoke + HF fast-path rung ===================="
    $PY papers/st_live_reference/scripts/smoke_st.py --out "$OUT/_validate" \
      || { echo "SMOKE FAILED"; exit 1; }
  fi
  echo ">> ST fast gates:"
  $PY -m pytest -q $GATE_TESTS -m "not slow" || { echo "LOCAL GATE FAILED"; exit 1; }
  dry_runs || exit 1
  echo ">> ST slow gates (tiny real checkpoints):"
  $PY -m pytest -q $GATE_TESTS -m slow || { echo "SLOW GATE FAILED"; exit 1; }
  echo ">> VALIDATE: smoke, dry runs and gates OK. Launch without VALIDATE for the real run."; exit 0
fi

# ---- 3. real run: fast gates and dry runs before any model ----
echo ">> ST fast gates:"
$PY -m pytest -q $GATE_TESTS -m "not slow" || { echo "LOCAL GATE FAILED"; exit 1; }
dry_runs || exit 1
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
  echo "==================== $OUT_SUB/$m ===================="
  $PY $S --model "$m" --date-pin "$DATE_PIN" --out "$OUT" $(mode_args "$m" "$BANKED"); rc=$?
  echo ">> $m exit $rc (0 ok, 2 gate failed, 3 envelope stop; banked either way)"
  $PY -c "import sys; sys.path.insert(0,'papers/kdg_panel/scripts'); from kdg_pod_lib import verify_manifest as v; bad=v('$OUT/$m/manifest_kdg.json'); print('manifest verify:', 'OK' if not bad else bad[:5])" \
    || echo "WARN: $m manifest verify did not run"
  [ "$rc" -ne 0 ] && RC_ALL=$rc
  if [ "$CLEAN_HF_CACHE" = "1" ]; then
    for d in $(hf_cache_dirs "$m"); do [ -d "$d" ] && rm -rf "$d" && echo ">> removed $d"; done
  fi
done

echo "{\"session_end_utc\": \"$(date -u +%FT%TZ)\", \"rc\": $RC_ALL}" > "$OUT/session_end.json"
echo "==================== $OUT_SUB/reprice ===================="
if [ "$STEPS" = "rider" ]; then RP_DIRS="$BANKED $OUT"; else RP_DIRS="$OUT"; fi
$PY papers/st_live_reference/scripts/reprice_tier1.py --screen-dir $RP_DIRS || echo "WARN: re-price failed"
echo ">> ST T1.0 ${STEPS:-relaunch} done (rc=$RC_ALL). rsync-back -> papers/st_live_reference/outputs/$OUT_SUB/"
exit $RC_ALL
