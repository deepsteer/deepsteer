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
#   STEPS=rider2 RESULTS_SUBPATH=outputs/t10c SYNC_OUTPUTS=none <same>
#                             rider pod 2 (author, 2026-10-19; G1 items 3-4, Amendment 9): on
#                             GPT-OSS-20B and Muse, the padded check and rung at the dose-0
#                             position with the bf16 and fp32 heads, the readout-mode rule, the
#                             stack-parity read (torch 2.6 vs the vLLM venv's 2.13), and on
#                             GPT-OSS the fold validation and the MoE probes; envelope 1.5 h.
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
# STEPS may name the results subdirectory after a slash (STEPS=rider2/t10d): it must equal the
# launch's RESULTS_SUBPATH (outputs/<it>), which the launcher downloads and nothing else (the
# GPT-OSS rerun of rider pod 2 wrote t10c while the launch fetched t10d, and its results were lost)
OUT_SUB_OVERRIDE=""
if [[ "$STEPS" == */* ]]; then OUT_SUB_OVERRIDE="${STEPS#*/}"; STEPS="${STEPS%%/*}"; fi
if [ "$STEPS" = "rider" ]; then
  # the bounded riders first (~0.65 h), then Qwen's phase H, which has never run on a GPU and
  # cannot stop itself mid-panel: the 3.0 h watchdog then leaves it ~2.1 h (about twice its estimate)
  MODELS="${MODELS:-gpt_oss_20b muse_glimmer_30b qwen38_27b}"; OUT_SUB=t10b
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-3.0}"
elif [ "$STEPS" = "t11" ]; then
  # T1.1 on the development model (author, 2026-10-19): the refusal-retention selection, vLLM
  # 264 x 8, HF readouts + banking on the ST stack of record, the independent none re-run, Muse's
  # QLoRA rider, the re-price last
  MODELS="${MODELS:-gpt_oss_20b}"; OUT_SUB=t11
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-2.0}"
elif [ "$STEPS" = "t13" ]; then
  # T1.3 on the development model (author, 2026-10-19; Amendments 11-12): the strength sweep, the
  # target-selection gate, the transport control and GPT-OSS's none panel with the carrier call,
  # one HF process on the ST stack; inputs from outputs/t13_inputs (SYNC_OUTPUTS)
  MODELS="${MODELS:-gpt_oss_20b}"; OUT_SUB=t13
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-5.0}"
elif [ "$STEPS" = "t14" ]; then
  # T1.4 on the development model (author, 2026-10-21; Amendments 14-16): two gate arms, the three
  # adapters, the conditional fourth, one HF process; inputs from outputs/t14_inputs, the T1.1
  # rollouts and the T1.3 gate rows (SYNC_OUTPUTS)
  MODELS="${MODELS:-gpt_oss_20b}"; OUT_SUB=t14
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-6.5}"
elif [ "$STEPS" = "t15" ]; then
  # T1.5 on the development model (author, 2026-10-21; Amendments 17-20): L-beh's cap rerun, the
  # bf16 baseline and parity gates, each adapter's HF phase (parity rows, specificity, the panel
  # guard where it applies, the merge) and vLLM phase (H1_held x 8, capability), the readouts last;
  # inputs from outputs/t14_inputs and the T1.4 adapters (SYNC_OUTPUTS). The 7.0 h cap keeps the
  # envelope at 36 with the second family's price (22.84 spent + 7.1 + 5.97).
  MODELS="${MODELS:-gpt_oss_20b}"; OUT_SUB=t15
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-7.0}"
elif [ "$STEPS" = "st2a" ]; then
  # ST Phase 2 pod A (author, 2026-10-21; Amendment 22 and its riders): Cell 1, the full-n gate on
  # GPT-OSS (45 engaged H1_train x 16, five arms), one HF process; inputs from outputs/st2_inputs
  MODELS="${MODELS:-gpt_oss_20b}"; OUT_SUB=st2a
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-8.0}"
elif [ "$STEPS" = "st2b" ]; then
  # ST Phase 2 pod B (author, 2026-10-21; Amendment 22, its riders and Cell 2 decisions): the Qwen
  # top-up (vLLM, first, on an empty GPU), Cell 2 (V_rule) and Cell 3 (V_qwen) on GPT-OSS; inputs
  # from outputs/st2_inputs, T1.1's rows, Qwen's screen rows and pod A's Cell 1 none (SYNC_OUTPUTS)
  # G3 (author, 2026-10-22): T1.5's follow-up first (HF none and L-anchor on the 38 engaged, the
  # ST-A17 merge read, the readouts), on a copy of T1.5's rows under outputs/st2b/t15_followup
  # the -V_harm arm after Cell 2 (author, 2026-10-22; 45 x 16, ~1.35 A100-h): 8.0 h stop
  MODELS="${MODELS:-gpt_oss_20b}"; OUT_SUB=st2b
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-8.0}"
elif [ "$STEPS" = "st2c" ]; then
  # ST Phase 2 pod C (author, 2026-10-22; Amendment 25): -V_harm's refusal retention (60 plain x 4)
  # and the specificity set at inference, the rider's positive control, V_qwen at 133.6 on the
  # engaged 45 x 8 and, if coherent, its two reads; one HF process; inputs from outputs/st2_inputs,
  # outputs/st2c_inputs and pod A's Cell 1 none (SYNC_OUTPUTS from sync_st2c.txt); Amendment 26:
  # the two conditionals (rr_fill, rr_ablate_refusal) before V_qwen, whose reads run last; 4.5 h
  MODELS="${MODELS:-gpt_oss_20b}"; OUT_SUB=st2c
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-4.5}"
elif [ "$STEPS" = "st2m" ]; then
  # ST Phase 2 Cell 4, Muse's transfer test (author, 2026-10-22; Amendments 29-30): vLLM H1 264 x 8,
  # then one HF process (dose-0, bank, derive, pilot, directions, sweep, every spec read, the arms);
  # Qwen3.8's traces from pod B and its screen (SYNC_OUTPUTS from sync_st2m.txt); 6.5 h stop
  MODELS="${MODELS:-muse_glimmer_30b}"; OUT_SUB=st2m
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-6.5}"
elif [ "$STEPS" = "st2fx" ]; then
  # the format x harm matrix (author, 2026-10-23; Amendment 33): dose-0 reads of cells A, B and the
  # H1_held rider under none, L-beh-2 and L-beh, each arm its own process; the two adapters synced
  # (sync_st2fx.txt; sha256 checked against data/st_fxh_adapters_v1.json); then the D_rb rider
  # (Amendments 34-35: specificity-set traces on vLLM, prefills under five arms); 2.5 h stop
  MODELS="${MODELS:-gpt_oss_20b}"; OUT_SUB=st2fx
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-2.5}"
elif [ "$STEPS" = "st2mr" ]; then
  # the care_harm replicate at 0.3x on Muse (author, 2026-10-23; Amendment 32): none, fill and
  # care_harm on Cell 4's 45 engaged x 16, fresh seeds and fill draws; spec reads first; Cell 4's
  # files and the fresh draws synced (sync_st2mr.txt); 4.5 h stop
  MODELS="${MODELS:-muse_glimmer_30b}"; OUT_SUB=st2mr
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-4.5}"
elif [ "$STEPS" = "st2lb" ]; then
  # L-beh-2 on GPT-OSS (author, 2026-10-23; Amendment 31 item 3 and its addition): rule-right
  # rollouts on vLLM, the DPO adapter with H1_train and rule-right pairs, T1.5's gate on it; inputs
  # from outputs/t14_inputs and T1.1's rows (SYNC_OUTPUTS from sync_st2lb.txt); 5.5 h stop
  MODELS="${MODELS:-gpt_oss_20b}"; OUT_SUB=st2lb
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-5.5}"
elif [ "$STEPS" = "st2ml" ]; then
  # the care_harm ladder on Muse (author, 2026-10-23; Amendment 31 item 4, ST-A20): 0.3x and 0.4x on
  # Cell 4's engaged x 8 with its seeds, spec reads first; Cell 4's derived files and rows synced
  # (sync_st2ml.txt); one HF process; 3.0 h stop
  MODELS="${MODELS:-muse_glimmer_30b}"; OUT_SUB=st2ml
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-3.0}"
elif [ "$STEPS" = "st2dl" ]; then
  # ST Phase 2 D-lite (author, 2026-10-22; Amendment 29): rr_none, rr_fill and spec_fill at alpha
  # 267.3 and 133.6, spec_none parity; pod C's inputs (SYNC_OUTPUTS from sync_st2dl.txt); ~1.5 h
  MODELS="${MODELS:-gpt_oss_20b}"; OUT_SUB=st2dl
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-2.0}"
elif [ "$STEPS" = "rider2" ]; then
  # rider pod 2 (author, 2026-10-19; G1 items 3-4, Amendment 9): the models of record only
  MODELS="${MODELS:-gpt_oss_20b muse_glimmer_30b}"; OUT_SUB=t10c
  SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-1.5}"
else
  MODELS="${MODELS:-gpt_oss_20b qwen38_27b muse_glimmer_30b}"; OUT_SUB=t10
fi
[ -n "$OUT_SUB_OVERRIDE" ] && OUT_SUB="$OUT_SUB_OVERRIDE"
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
SESSION_T0="$(date +%s)"  # the phases' deadline is measured from here (T1.1)
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
T13_INPUTS="$REPO_DIR/papers/st_live_reference/outputs/t13_inputs/gpt_oss_20b"  # SYNC_OUTPUTS (T1.3)
T14_INPUTS="$REPO_DIR/papers/st_live_reference/outputs/t14_inputs/gpt_oss_20b"  # SYNC_OUTPUTS (T1.4)
T14_ADAPTERS="$REPO_DIR/papers/st_live_reference/outputs/t14/gpt_oss_20b/adapters"  # SYNC_OUTPUTS (T1.5)
ST2_INPUTS="$REPO_DIR/papers/st_live_reference/outputs/st2_inputs/gpt_oss_20b"  # SYNC_OUTPUTS (Phase 2)
ST2C_INPUTS="$REPO_DIR/papers/st_live_reference/outputs/st2c_inputs/gpt_oss_20b"  # SYNC_OUTPUTS (pod C)
T11_DIR="$REPO_DIR/papers/st_live_reference/outputs/t11/gpt_oss_20b"
T13_DIR="$REPO_DIR/papers/st_live_reference/outputs/t13/gpt_oss_20b"
# the rider pod's per-model mode (author, 2026-10-19): Qwen phase H on its banked phase V, Muse's
# dose-0 re-check (rider (a)), GPT-OSS's corrected rung + eager memory probe (rider (b))
mode_args() {
  if [ "$STEPS" = "rider2" ]; then
    case "$1" in
      gpt_oss_20b) echo "--rider2" ;;
      muse_glimmer_30b) echo "--rider2 --effort medium" ;;  # Muse's effort of record (screen)
      # ST-A5's discriminator (author, 2026-10-19), last in the bail order
      qwen38_27b) echo "--resume-from $2/qwen38_27b --dose0-candidates think_on_empty --dose0-brake" ;;
    esac
    return 0
  fi
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
    # the ST stack of record (Amendment 9): HF readouts and T0.6's training run here too, so the
    # venv carries deepsteer (+ peft) and bitsandbytes for the QLoRA path; torch stays 2.13 cu129
    "$VLLM_VENV/bin/pip" install -q -e ".[lora]" "peft==0.21.2" "bitsandbytes==0.50.2" \
      "transformers==5.17.0" -c "$VLLM_CONSTRAINTS" --extra-index-url "$CU129_INDEX" 2>&1 | tail -3
  fi
  PATH="$VLLM_VENV/bin:$PATH" VLLM_VERSION="$VLLM_VERSION" "$VLLM_VENV/bin/python" - <<'PY' || { echo "FATAL: vLLM venv not usable (see the check above)"; exit 1; }
import importlib.metadata as md, os, shutil, sys
import torch
from vllm import LLM, SamplingParams  # noqa: F401  (the worker's own import; pulls torchcodec)
import flashinfer, torchcodec  # noqa: F401,E401
from cuda.bindings import nvrtc
from vllm.model_executor.models import gpt_oss, muse_glimmer, qwen3_5  # noqa: F401
import bitsandbytes, peft  # noqa: F401,E401  (T0.6 on the ST stack of record)
from deepsteer.steering import objective_lora  # noqa: F401
sys.path[:0] = ["papers/st_live_reference/scripts", "papers/kdg_panel/scripts"]
import pod_st_screen, st_train, t11_driver, t13_driver, panel_call, t14_driver, general_slice, t15_driver, screen_stub, st2_driver  # noqa: F401,E401
v = {p: md.version(p) for p in ("vllm", "torch", "torchcodec", "cuda-bindings", "cuda-python",
                                "flashinfer-python", "transformers", "peft", "bitsandbytes")}
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
    "peft": v["peft"] == "0.21.2",
    "bitsandbytes": v["bitsandbytes"] == "0.50.2",
    "torch_still_2.13": torch.__version__.startswith("2.13.0"),
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
if [ "$STEPS" = "rider2" ] && [[ " $MODELS " == *" qwen38_27b "* ]]; then  # ST-A5 resumes Qwen
  [ -f "$BANKED/qwen38_27b/screen_record.json" ] || { echo "FATAL: banked $BANKED/qwen38_27b missing (SYNC_OUTPUTS)"; exit 1; }
  $PY $S --model qwen38_27b --out "$OUT/_provenance" --resume-from "$BANKED/qwen38_27b" --provenance-only \
    || { echo "FATAL: provenance check failed (qwen38_27b)"; exit 1; }
fi
if [ "$STEPS" = "t13" ]; then  # built locally after the constancy read (build_t13_inputs.py)
  for f in "$T13_INPUTS/t13_inputs.npz" "$T13_INPUTS/t13_inputs.json" \
           papers/st_live_reference/data/t12_constancy_v1.json papers/st_live_reference/data/t13_draw_v1.json; do
    [ -f "$f" ] || { echo "FATAL: $f missing (T1.3 inputs; SYNC_OUTPUTS=outputs/t13_inputs/)"; exit 1; }
  done
fi
if [ "$STEPS" = "t14" ]; then  # built locally (build_t14_inputs.py); T1.1 rollouts; T1.3 gate rows
  for f in "$T14_INPUTS/t14_inputs.npz" "$T14_INPUTS/t14_inputs.json" \
           "$T11_DIR/gen/h1_rollouts_read.jsonl" "$T13_DIR/gen/gate_none.jsonl" \
           "$T13_DIR/gen/gate_fill_rerun.jsonl" papers/st_live_reference/data/general_slice_v1.json; do
    [ -f "$f" ] || { echo "FATAL: $f missing (T1.4 inputs; SYNC_OUTPUTS)"; exit 1; }
  done
fi
if [ "$STEPS" = "t15" ]; then  # T1.4's inputs and adapters (SYNC_OUTPUTS); the T1.5 sets (committed)
  for f in "$T14_INPUTS/t14_inputs.npz" "$T14_INPUTS/t14_inputs.json" \
           papers/st_live_reference/data/t13_draw_v1.json papers/st_live_reference/data/t15_specificity_v1.json \
           papers/st_live_reference/data/t15_capability_v1.json papers/st_live_reference/data/t15_panel_half_v1.json \
           papers/st_live_reference/data/t15_engaged_held_v1.json papers/st_live_reference/data/t15_adapters_v1.json; do
    [ -f "$f" ] || { echo "FATAL: $f missing (T1.5 inputs)"; exit 1; }
  done
  for arm in beh beh_epoch3 anchor self; do
    for f in adapter_config.json adapter_model.safetensors; do  # sha256 checked by the driver
      [ -s "$T14_ADAPTERS/$arm/$f" ] || { echo "FATAL: $T14_ADAPTERS/$arm/$f missing (SYNC_OUTPUTS)"; exit 1; }
    done
  done
fi
if [ "$STEPS" = "st2b" ]; then  # G3: the T1.4 adapters for the follow-up (sha256 checked by the driver)
  for arm in beh beh_epoch3 anchor self; do
    for f in adapter_config.json adapter_model.safetensors; do
      [ -s "$T14_ADAPTERS/$arm/$f" ] || { echo "FATAL: $T14_ADAPTERS/$arm/$f missing (SYNC_OUTPUTS)"; exit 1; }
    done
  done
fi
ST2A_DIR="$REPO_DIR/papers/st_live_reference/outputs/st2a/gpt_oss_20b"  # SYNC_OUTPUTS (pod B)
QWEN_SCREEN="$REPO_DIR/papers/st_live_reference/outputs/t10/qwen38_27b/screen/h1_screen.jsonl"
T15_JOBS="$REPO_DIR/papers/st_live_reference/outputs/t15_early/gpt_oss_20b/vllm/mxfp4_parity.jobs.jsonl"  # ST-A15
T15_DIR="$REPO_DIR/papers/st_live_reference/outputs/t15/gpt_oss_20b"  # SYNC_OUTPUTS (G3 follow-up)
QWEN_TOPUP="$REPO_DIR/papers/st_live_reference/outputs/st2b/gpt_oss_20b/gen/c3_qwen_topup.jsonl"  # Cell 4
# pod C (Amendments 25-26): -V_harm's two reads, the rider's positive control, the two conditionals,
# then V_qwen at 133.6 and its reads, last
ST2C_PHASES="rr_none,rr_neg_V_harm,spec_none,spec_neg_V_harm,rr_neg_refusal,rr_fill,rr_ablate_refusal,qwen_half,rr_V_qwen,spec_V_qwen"
# the D-lite (Amendment 29): the random-vector references, retention then specificity
ST2DL_PHASES="rr_none,rr_fill,rr_fill_half,spec_none,spec_fill,spec_fill_half"
# Cell 4 (Amendment 30): the HF phases after v1, every spec read before any arm, arms in bail order reversed
ST2M_PHASES="dose0,bank,derive,pilot,directions,sweep,spec_none,spec_fill,spec_V_qwen,spec_neg_V_harm,spec_care_harm,arm_none,arm_fill,arm_V_qwen,arm_neg_V_harm,arm_care_harm"
ST2B_PHASES="c3_qwen,c2_extract,c2_direction,c2_anchor,c2_ablate,c2_random,c2_twin_none,c2_transport,c2_freegen,c1_neg_V_harm,c3_reencode,c3_direction,c3_qwen_arm"
if [ "$STEPS" = "st2b" ]; then  # built locally; synced (sync_st2b.txt); twins and ids committed
  for f in "$ST2_INPUTS/st2_inputs.npz" "$T11_DIR/gen/h1_rollouts_read.jsonl" "$QWEN_SCREEN" \
           "$ST2A_DIR/gen/c1_none.jsonl" "$T15_JOBS" "$T15_DIR/t15_record.json" \
           "$T15_DIR/gen/h1_none.jsonl" "$T15_DIR/gen/h1_anchor.jsonl" "$T14_INPUTS/t14_inputs.npz" papers/st_live_reference/data/st2_engaged_train_v1.json \
           papers/st_live_reference/data/st2_rule_twins_v1.json; do
    [ -f "$f" ] || { echo "FATAL: $f missing (pod B inputs; SYNC_OUTPUTS from sync_st2b.txt)"; exit 1; }
  done
fi
if [ "$STEPS" = "st2c" ] || [ "$STEPS" = "st2dl" ]; then  # built locally (build_st2_inputs.py cell1, podc); synced (sync_st2c.txt, sync_st2dl.txt)
  for f in "$ST2_INPUTS/st2_inputs.npz" "$ST2C_INPUTS/st2c_inputs.npz" "$ST2C_INPUTS/st2c_inputs.json" \
           "$ST2A_DIR/gen/c1_none.jsonl" papers/st_live_reference/data/st2_engaged_train_v1.json \
           papers/st_live_reference/data/refusal_retention_v1.json papers/st_live_reference/data/t15_specificity_v1.json; do
    [ -f "$f" ] || { echo "FATAL: $f missing (pod C inputs; SYNC_OUTPUTS from sync_st2c.txt)"; exit 1; }
  done
fi
CELL4_DIR="$REPO_DIR/papers/st_live_reference/outputs/st2m/muse_glimmer_30b"  # SYNC_OUTPUTS (ladder)
if [ "$STEPS" = "st2lb" ]; then  # T1.4's inputs and T1.1's rows (synced); the pool and T1.5's sets (committed)
  for f in "$T14_INPUTS/t14_inputs.npz" "$T14_INPUTS/t14_inputs.json" "$T11_DIR/gen/h1_rollouts_read.jsonl" \
           papers/st_live_reference/data/st_lbeh2_pool_v1.json papers/st_live_reference/data/t15_specificity_v1.json \
           papers/st_live_reference/data/t15_capability_v1.json papers/st_live_reference/data/t15_panel_half_v1.json \
           papers/st_live_reference/data/t15_engaged_held_v1.json; do
    [ -f "$f" ] || { echo "FATAL: $f missing (L-beh-2 inputs; SYNC_OUTPUTS from sync_st2lb.txt)"; exit 1; }
  done
fi
ST2MR_INPUTS="$REPO_DIR/papers/st_live_reference/outputs/st2mr_inputs/muse_glimmer_30b"  # SYNC_OUTPUTS (replicate)
FX_ADAPTERS="$OUT/_fx_adapters"  # the adapters, gathered from their synced dirs (Amendments 33-35)
DRB_POOL="$REPO_DIR/papers/st_live_reference/outputs/st2lb/gpt_oss_20b/gen/rr_rollouts.jsonl"  # SYNC_OUTPUTS (D_rb)
DRB_T11="$REPO_DIR/papers/st_live_reference/outputs/t11/gpt_oss_20b/gen/h1_rollouts.jsonl"  # SYNC_OUTPUTS (D_rb)
if [ "$STEPS" = "st2fx" ]; then  # the cells and the manifest (committed); the adapters (synced)
  for f in papers/st_live_reference/data/st_fxh_h1fmt_ruleright_v1.json papers/st_live_reference/data/st_fxh_panelfmt_ruleharm_v1.json \
           papers/st_live_reference/data/st_fxh_adapters_v1.json "$T14_ADAPTERS/beh/adapter_model.safetensors" \
           "$T14_ADAPTERS/beh/adapter_config.json" "$REPO_DIR/papers/st_live_reference/outputs/st2lb/gpt_oss_20b/adapters/beh2/adapter_model.safetensors" \
           "$REPO_DIR/papers/st_live_reference/outputs/st2lb/gpt_oss_20b/adapters/beh2/adapter_config.json" \
           "$T14_ADAPTERS/anchor/adapter_model.safetensors" "$T14_ADAPTERS/anchor/adapter_config.json" \
           "$T14_ADAPTERS/beh_epoch3/adapter_model.safetensors" "$T14_ADAPTERS/beh_epoch3/adapter_config.json" \
           "$DRB_POOL" "$DRB_T11"; do
    [ -f "$f" ] || { echo "FATAL: $f missing (matrix and D_rb inputs; SYNC_OUTPUTS from sync_st2fx.txt)"; exit 1; }
  done
  mkdir -p "$FX_ADAPTERS" && ln -sfn "$T14_ADAPTERS/beh" "$FX_ADAPTERS/beh" \
    && ln -sfn "$REPO_DIR/papers/st_live_reference/outputs/st2lb/gpt_oss_20b/adapters/beh2" "$FX_ADAPTERS/beh2" \
    && ln -sfn "$T14_ADAPTERS/anchor" "$FX_ADAPTERS/anchor" && ln -sfn "$T14_ADAPTERS/beh_epoch3" "$FX_ADAPTERS/beh_epoch3"
fi
if [ "$STEPS" = "st2mr" ]; then  # Cell 4's files and the fresh draws (synced)
  for f in "$CELL4_DIR/st2m_record.json" "$CELL4_DIR/derive.npz" "$CELL4_DIR/directions.npz" \
           "$CELL4_DIR/gen/arm_none.jsonl" "$CELL4_DIR/spec/none/dl_chat_neutral.jsonl" \
           "$ST2MR_INPUTS/fill_seed1.npz" "$QWEN_TOPUP" "$QWEN_SCREEN"; do
    [ -f "$f" ] || { echo "FATAL: $f missing (replicate inputs; SYNC_OUTPUTS from sync_st2mr.txt)"; exit 1; }
  done
fi
if [ "$STEPS" = "st2ml" ]; then  # Cell 4's derived files and rows (synced)
  for f in "$CELL4_DIR/st2m_record.json" "$CELL4_DIR/derive.npz" "$CELL4_DIR/directions.npz" \
           "$CELL4_DIR/gen/arm_none.jsonl" "$CELL4_DIR/gen/arm_fill.jsonl" \
           "$CELL4_DIR/spec/none/dl_chat_neutral.jsonl" "$QWEN_TOPUP" "$QWEN_SCREEN"; do
    [ -f "$f" ] || { echo "FATAL: $f missing (ladder inputs; SYNC_OUTPUTS from sync_st2ml.txt)"; exit 1; }
  done
fi
if [ "$STEPS" = "st2m" ]; then  # Qwen3.8's traces (synced); every other input is derived on the pod
  for f in "$QWEN_TOPUP" "$QWEN_SCREEN" papers/st_live_reference/data/t15_specificity_v1.json \
           papers/5_moral_alignment/refusal_prompts.json; do
    [ -f "$f" ] || { echo "FATAL: $f missing (Cell 4 inputs; SYNC_OUTPUTS from sync_st2m.txt)"; exit 1; }
  done
fi
if [ "$STEPS" = "st2a" ]; then  # built locally (build_st2_inputs.py); the engaged ids (committed)
  for f in "$ST2_INPUTS/st2_inputs.npz" "$ST2_INPUTS/st2_inputs.json" \
           papers/st_live_reference/data/st2_engaged_train_v1.json; do
    [ -f "$f" ] || { echo "FATAL: $f missing (Phase 2 inputs; SYNC_OUTPUTS=outputs/st2_inputs/)"; exit 1; }
  done
fi
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
    if [ "$STEPS" = "rider" ] || [ "$STEPS" = "rider2" ]; then
      $PY $S --model "$m" --dry-run --date-pin "$DATE_PIN" --out "$OUT/_dry_rider" $(mode_args "$m" "$OUT/_dry") \
        || { echo "DRY RUN FAILED ($m, rider mode)"; return 1; }
    fi
    if [ "$STEPS" = "t11" ] && [ "$m" = "gpt_oss_20b" ]; then  # every T1.1 phase, dry
      for ph in v1 h v2 qlora; do
        $PY papers/st_live_reference/scripts/t11_driver.py --phase "$ph" --out "$OUT/_dry_t11" \
          --dry-run --date-pin "$DATE_PIN" || { echo "DRY RUN FAILED (t11 $ph)"; return 1; }
      done
    fi
    if [ "$STEPS" = "t13" ] && [ "$m" = "gpt_oss_20b" ]; then  # every T1.3 phase, dry (tiny model)
      "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/t13_driver.py --out "$OUT/_dry_t13" --dry-run \
        --inputs "$T13_INPUTS" --date-pin "$DATE_PIN" || { echo "DRY RUN FAILED (t13)"; return 1; }
    fi
    if [ "$STEPS" = "t14" ] && [ "$m" = "gpt_oss_20b" ]; then  # every T1.4 phase, dry (tiny model)
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/t14_driver.py \
        --out "$OUT/_dry_t14" --dry-run --inputs "$T14_INPUTS" --t11 "$T11_DIR" --t13 "$T13_DIR" \
        --date-pin "$DATE_PIN" || { echo "DRY RUN FAILED (t14)"; return 1; }
    fi
    if [ "$STEPS" = "t15" ] && [ "$m" = "gpt_oss_20b" ]; then  # every T1.5 phase, dry, both parity branches
      for extra in "" "--dry-fail-base"; do
        rm -rf "$OUT/_dry_t15"
        PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/t15_driver.py \
          --out "$OUT/_dry_t15" --dry-run $extra --t14-inputs "$T14_INPUTS" \
          --adapters "$OUT/_dry_t15/adapters" --date-pin "$DATE_PIN" \
          || { echo "DRY RUN FAILED (t15 $extra)"; return 1; }
      done
      rm -rf "$OUT/_dry_t15"
    fi
    if [ "$STEPS" = "st2a" ] && [ "$m" = "gpt_oss_20b" ]; then  # every Cell 1 phase, dry (tiny model)
      rm -rf "$OUT/_dry_st2"
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/st2_driver.py \
        --out "$OUT/_dry_st2" --dry-run --inputs "$ST2_INPUTS" --date-pin "$DATE_PIN" \
        || { echo "DRY RUN FAILED (st2a)"; return 1; }
      rm -rf "$OUT/_dry_st2"
    fi
    if [ "$STEPS" = "st2b" ] && [ "$m" = "gpt_oss_20b" ]; then  # Cell 1 dry (its none), then pod B's order, dry
      rm -rf "$OUT/_dry_st2" "$OUT/_dry_t15fu"
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/t15_driver.py \
        --out "$OUT/_dry_t15fu" --dry-run --t14-inputs "$T14_INPUTS" --adapters "$OUT/_dry_t15fu/adapters" \
        --date-pin "$DATE_PIN" || { echo "DRY RUN FAILED (st2b, t15 base)"; return 1; }
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/t15_driver.py \
        --out "$OUT/_dry_t15fu" --dry-run --t14-inputs "$T14_INPUTS" --adapters "$OUT/_dry_t15fu/adapters" \
        --date-pin "$DATE_PIN" --phase followup --force || { echo "DRY RUN FAILED (st2b, t15 follow-up)"; return 1; }
      rm -rf "$OUT/_dry_t15fu"
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/router_disc.py \
        --phase hf --dry-run --jobs "$T15_JOBS" --out "$OUT/_dry_st2/rd" \
        || { echo "DRY RUN FAILED (st2b, router_disc hf)"; return 1; }
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/st2_driver.py \
        --out "$OUT/_dry_st2" --dry-run --inputs "$ST2_INPUTS" --date-pin "$DATE_PIN" \
        || { echo "DRY RUN FAILED (st2b, cell 1)"; return 1; }
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/st2_driver.py \
        --out "$OUT/_dry_st2" --dry-run --inputs "$ST2_INPUTS" --date-pin "$DATE_PIN" \
        --phases "$ST2B_PHASES" --t11 "$T11_DIR" --c1 "$OUT/_dry_st2/gpt_oss_20b" \
        --qwen-screen "$QWEN_SCREEN" || { echo "DRY RUN FAILED (st2b)"; return 1; }
      rm -rf "$OUT/_dry_st2"
    fi
    if [ "$STEPS" = "st2c" ] && [ "$m" = "gpt_oss_20b" ]; then  # Cell 1's none dry, then pod C's order, dry
      rm -rf "$OUT/_dry_st2" "$OUT/_dry_st2c"
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/st2_driver.py \
        --out "$OUT/_dry_st2" --dry-run --inputs "$ST2_INPUTS" --date-pin "$DATE_PIN" --phases c1_none \
        || { echo "DRY RUN FAILED (st2c, cell 1 none)"; return 1; }
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/st2c_driver.py \
        --out "$OUT/_dry_st2c" --dry-run --inputs "$ST2_INPUTS" --podc-inputs "$ST2C_INPUTS" \
        --c1 "$OUT/_dry_st2/gpt_oss_20b" --phases "$ST2C_PHASES" || { echo "DRY RUN FAILED (st2c)"; return 1; }
      rm -rf "$OUT/_dry_st2" "$OUT/_dry_st2c"
    fi
    if [ "$STEPS" = "st2fx" ] && [ "$m" = "gpt_oss_20b" ]; then  # every arm, dry (tiny model, zero adapters)
      rm -rf "$OUT/_dry_st2fx"
      for arm in none beh2 beh; do
        PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/st2fx_driver.py \
          --arm "$arm" --out "$OUT/_dry_st2fx" --adapters "$OUT/_dry_st2fx/adapters" --dry-run \
          || { echo "DRY RUN FAILED (st2fx $arm)"; return 1; }
      done
      rm -rf "$OUT/_dry_st2fx"
      $PY papers/st_live_reference/scripts/st2drb_driver.py --step gen --out "$OUT/_dry_st2drb" --dry-run \
        || { echo "DRY RUN FAILED (st2drb gen)"; return 1; }
      for arm in none beh beh2 anchor beh_epoch3; do  # every rider arm, dry (tiny model, zero adapters)
        PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/st2drb_driver.py \
          --step prefill --arm "$arm" --out "$OUT/_dry_st2drb" --adapters "$OUT/_dry_st2drb/adapters" \
          --pool "$DRB_POOL" --t11 "$DRB_T11" --dry-run || { echo "DRY RUN FAILED (st2drb $arm)"; return 1; }
      done
      rm -rf "$OUT/_dry_st2drb"
    fi
    if [ "$STEPS" = "st2mr" ] && [ "$m" = "muse_glimmer_30b" ]; then  # a tiny Cell 4, then the replicate on it
      rm -rf "$OUT/_dry_st2m" "$OUT/_dry_st2mr"
      $PY papers/st_live_reference/scripts/st2m_driver.py --out "$OUT/_dry_st2m" --dry-run \
        --phases v1 --date-pin "$DATE_PIN" || { echo "DRY RUN FAILED (st2mr, cell 4 v1)"; return 1; }
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/st2m_driver.py \
        --out "$OUT/_dry_st2m" --dry-run --qwen-topup "$QWEN_TOPUP" --qwen-screen "$QWEN_SCREEN" \
        || { echo "DRY RUN FAILED (st2mr, cell 4 hf)"; return 1; }
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/st2mr_driver.py \
        --out "$OUT/_dry_st2mr" --dry-run --cell4 "$OUT/_dry_st2m/muse_glimmer_30b" \
        --inputs "$ST2MR_INPUTS" || { echo "DRY RUN FAILED (st2mr)"; return 1; }
      rm -rf "$OUT/_dry_st2m" "$OUT/_dry_st2mr"
    fi
    if [ "$STEPS" = "st2lb" ] && [ "$m" = "gpt_oss_20b" ]; then  # every L-beh-2 step, dry (stub, tiny model)
      rm -rf "$OUT/_dry_st2lb"
      $PY papers/st_live_reference/scripts/st2lb_driver.py --step rr_gen --out "$OUT/_dry_st2lb" \
        --dry-run --date-pin "$DATE_PIN" || { echo "DRY RUN FAILED (st2lb rr_gen)"; return 1; }
      for st in train gate; do
        PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/st2lb_driver.py \
          --step "$st" --out "$OUT/_dry_st2lb" --t14-inputs "$T14_INPUTS" --t11 "$T11_DIR" --dry-run \
          --date-pin "$DATE_PIN" || { echo "DRY RUN FAILED (st2lb $st)"; return 1; }
      done
      rm -rf "$OUT/_dry_st2lb"
    fi
    if [ "$STEPS" = "st2ml" ] && [ "$m" = "muse_glimmer_30b" ]; then  # a tiny Cell 4, then the ladder on it
      rm -rf "$OUT/_dry_st2m" "$OUT/_dry_st2ml"
      $PY papers/st_live_reference/scripts/st2m_driver.py --out "$OUT/_dry_st2m" --dry-run \
        --phases v1 --date-pin "$DATE_PIN" || { echo "DRY RUN FAILED (st2ml, cell 4 v1)"; return 1; }
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/st2m_driver.py \
        --out "$OUT/_dry_st2m" --dry-run --qwen-topup "$QWEN_TOPUP" --qwen-screen "$QWEN_SCREEN" \
        || { echo "DRY RUN FAILED (st2ml, cell 4 hf)"; return 1; }
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/st2ml_driver.py \
        --out "$OUT/_dry_st2ml" --dry-run --cell4 "$OUT/_dry_st2m/muse_glimmer_30b" \
        || { echo "DRY RUN FAILED (st2ml)"; return 1; }
      rm -rf "$OUT/_dry_st2m" "$OUT/_dry_st2ml"
    fi
    if [ "$STEPS" = "st2m" ] && [ "$m" = "muse_glimmer_30b" ]; then  # v1 (stub), then every HF phase on a tiny Muse
      rm -rf "$OUT/_dry_st2m"
      $PY papers/st_live_reference/scripts/st2m_driver.py --out "$OUT/_dry_st2m" --dry-run \
        --phases v1 --date-pin "$DATE_PIN" || { echo "DRY RUN FAILED (st2m v1)"; return 1; }
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/st2m_driver.py \
        --out "$OUT/_dry_st2m" --dry-run --phases "$ST2M_PHASES" --qwen-topup "$QWEN_TOPUP" \
        --qwen-screen "$QWEN_SCREEN" || { echo "DRY RUN FAILED (st2m hf)"; return 1; }
      rm -rf "$OUT/_dry_st2m"
    fi
    if [ "$STEPS" = "st2dl" ] && [ "$m" = "gpt_oss_20b" ]; then  # Cell 1's none dry, then the D-lite's order, dry
      rm -rf "$OUT/_dry_st2" "$OUT/_dry_st2dl"
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/st2_driver.py \
        --out "$OUT/_dry_st2" --dry-run --inputs "$ST2_INPUTS" --date-pin "$DATE_PIN" --phases c1_none \
        || { echo "DRY RUN FAILED (st2dl, cell 1 none)"; return 1; }
      PYTHONPATH="$REPO_DIR" "$VLLM_VENV/bin/python" papers/st_live_reference/scripts/st2dl_driver.py \
        --out "$OUT/_dry_st2dl" --dry-run --inputs "$ST2_INPUTS" --podc-inputs "$ST2C_INPUTS" \
        --c1 "$OUT/_dry_st2/gpt_oss_20b" --phases "$ST2DL_PHASES" || { echo "DRY RUN FAILED (st2dl)"; return 1; }
      rm -rf "$OUT/_dry_st2" "$OUT/_dry_st2dl"
    fi
    if [ "$m" = "gpt_oss_20b" ] && [ "$STEPS" != "t13" ] && [ "$STEPS" != "t14" ] && [ "$STEPS" != "t15" ] && [ "$STEPS" != "st2a" ] && [ "$STEPS" != "st2b" ] && [ "$STEPS" != "st2c" ] && [ "$STEPS" != "st2dl" ] && [ "$STEPS" != "st2lb" ] && [ "$STEPS" != "st2fx" ]; then  # Amendment 10
      $PY $S --model "$m" --dry-run --date-pin "$DATE_PIN" --out "$OUT/_dry_rr" --rr-select \
        || { echo "DRY RUN FAILED ($m, refusal-retention selection)"; return 1; }
    fi
  done
}
GATE_TESTS="tests/scripts/test_st_split.py tests/scripts/test_st_formats.py tests/scripts/test_st_screen.py tests/scripts/test_st_analysis.py tests/scripts/test_st_probes.py tests/scripts/test_st_vllm_worker.py tests/scripts/test_st_rider2.py tests/scripts/test_st_refusal_retention.py tests/scripts/test_st_steering_gate.py tests/scripts/test_st_train.py tests/scripts/test_st_t11.py tests/scripts/test_st_t13.py tests/scripts/test_st_t14.py tests/scripts/test_st_t15.py tests/scripts/test_st2.py tests/scripts/test_st2c.py tests/scripts/test_st2dl.py tests/scripts/test_st2m.py tests/scripts/test_st2ml.py tests/scripts/test_st2lb.py tests/scripts/test_st2mr.py tests/scripts/test_st2fx.py tests/scripts/test_st2drb.py tests/scripts/test_st2_read.py tests/scripts/test_st_t15_read.py tests/scripts/test_router_disc.py tests/scripts/test_st_t13_draw.py tests/scripts/test_st_panel_call.py tests/scripts/test_st_constancy_read.py tests/causal/test_attribution.py tests/core/test_model_interface_vlm.py tests/kdg"

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

# rider pod 2, after each model's driver: GPT-OSS's fold check, vLLM half (own process); the
# stack-parity reads with the registry's HF pins (the same script on torch 2.6 and in the vLLM venv
# on torch 2.13) and their ladder; grouped_mm on 2.13 only if the ladder keeps 2.13 (rider 2c,
# author 2026-10-19; eager and batched_mm on 2.6 were measured on rider pod 2b, and MegaBlocks has
# no torch-2.6 build)
VPY="$VLLM_VENV/bin/python"
rider2_after() {
  local m="$1" repo rev hfargs
  [ "$m" = "qwen38_27b" ] && return 0  # ST-A5's discriminator: no stack read, no probe
  read -r repo rev < <($PY -c "import yaml;e=yaml.safe_load(open('papers/kdg_panel/models.yaml'))['st']['$m'];print(e['repo'],e['revision'])")
  # the registry's HF pins, the same on both stacks (GPT-OSS: eager experts, bf16 dequantization)
  hfargs="$($PY -c "import yaml;h=yaml.safe_load(open('papers/kdg_panel/models.yaml'))['st']['$m'].get('hf') or {};print(' '.join(([('--experts '+h['experts_implementation'])] if h.get('experts_implementation') else [])+(['--dequantize'] if h.get('mxfp4_dequantize') else [])))")"
  local d="$OUT/$m/stack"
  if [ "$m" = "gpt_oss_20b" ]; then
    # the fold check's vLLM half first, in its own process on an empty GPU; it removes the checkpoint
    $PY papers/st_live_reference/scripts/fold_vllm.py --dir "$OUT/$m" || echo "WARN: fold vLLM half failed"
    rm -rf "$OUT/$m/fold/ckpt"
  fi
  [ -f "$d/rows64.json" ] || { echo "WARN: $m wrote no stack rows; skipping its stack read"; return 0; }
  $PY papers/st_live_reference/scripts/stack_read.py --repo "$repo" --revision "$rev" --rows "$d/rows64.json" --out "$d/torch26.npz" $hfargs \
    || echo "WARN: $m stack read (torch 2.6) failed"
  $VPY papers/st_live_reference/scripts/stack_read.py --repo "$repo" --revision "$rev" --rows "$d/rows64.json" --out "$d/torch213.npz" $hfargs \
    || echo "WARN: $m stack read (torch 2.13) failed"
  [ -f "$d/torch26.npz" ] && [ -f "$d/torch213.npz" ] && $PY papers/st_live_reference/scripts/stack_compare.py \
    --a "$d/torch26.npz" --b "$d/torch213.npz" --out "$d/parity.json" --tie "$OUT/$m/screen_record.json"
  if [ "$m" = "gpt_oss_20b" ] && [ -f "$d/parity.json" ]; then
    # Amendment 9 (a), author 2026-10-19: grouped_mm on torch 2.13 only if 2.13 is kept, i.e. the
    # ladder of the readout head of record is inside the bar
    if $PY -c "import json,sys;r=json.load(open('$OUT/$m/screen_record.json'));v=json.load(open('$d/parity.json'));sys.exit(0 if v['ladders'][r['readout_mode']['head']]['inside'] else 1)"; then
      $VPY papers/st_live_reference/scripts/moe_probe.py --repo "$repo" --revision "$rev" --prompts "$d/probe_prompts.json" --out "$OUT/$m/moe_grouped_mm_t213.json" --experts grouped_mm --dequantize \
        || echo "WARN: moe probe grouped_mm failed"
    else
      echo ">> stack parity outside the bar for $m: torch 2.6 stays; no grouped_mm probe"
    fi
  fi
}

RC_ALL=0
for m in $MODELS; do
  if [ "$m" = "gpt_oss_20b" ] && [ "$STEPS" != "t13" ] && [ "$STEPS" != "t14" ] && [ "$STEPS" != "t15" ] && [ "$STEPS" != "st2a" ] && [ "$STEPS" != "st2b" ] && [ "$STEPS" != "st2c" ] && [ "$STEPS" != "st2dl" ] && [ "$STEPS" != "st2lb" ] && [ "$STEPS" != "st2fx" ]; then  # of record from T1.1 (t11/_rr)
    # Amendment 10 item 3 (author, 2026-10-19): the refusal-retention selection pass rides the
    # first pod that loads the development model after the amendment; vLLM only, its own record
    echo "==================== $OUT_SUB/_rr/gpt_oss_20b (refusal-retention selection) ===================="
    if grep -qs '"status": "ok_rr_select"' "$OUT/_rr/gpt_oss_20b/screen_record.json"; then
      echo ">> refusal-retention selection already of record (synced from an earlier pod); not re-run"
    else
      $PY $S --model gpt_oss_20b --date-pin "$DATE_PIN" --out "$OUT/_rr" --rr-select \
        || echo "WARN: refusal-retention selection failed (recorded; the session continues)"
    fi
  fi
  echo "==================== $OUT_SUB/$m ===================="
  if [ "$STEPS" = "st2b" ]; then
    DEADLINE="$($PY -c "import time;print(int($SESSION_T0 + $MAX_S - 900))")"  # 15 min to bank
    # G3 first (author, 2026-10-22): T1.5's follow-up on a copy of its rows; fail-soft
    FU="$OUT/t15_followup"; mkdir -p "$FU/gpt_oss_20b"
    cp -r "$T15_DIR/." "$FU/gpt_oss_20b/" && rm -rf "$FU/gpt_oss_20b/merged"
    PYTHONPATH="$REPO_DIR" $VPY papers/st_live_reference/scripts/t15_driver.py --phase followup \
      --out "$FU" --t14-inputs "$T14_INPUTS" --adapters "$T14_ADAPTERS" --deadline "$DEADLINE" \
      || echo "WARN: T1.5 follow-up stopped (recorded; Phase 2 goes on)"
    # ST-A15's discriminator (author, 2026-10-21): HF router capture + the bf16 checkpoint,
    # then each vLLM engine in its own process; fail-soft (recorded), the checkpoint deleted after
    RD="$OUT/router_disc"
    PYTHONPATH="$REPO_DIR" $VPY papers/st_live_reference/scripts/router_disc.py --phase hf \
      --jobs "$T15_JOBS" --out "$RD" || echo "WARN: router_disc hf failed (recorded; Phase 2 goes on)"
    [ -d "$RD/ckpt" ] && { PYTHONPATH="$REPO_DIR" $VPY papers/st_live_reference/scripts/router_disc.py \
      --phase vllm --jobs "$T15_JOBS" --out "$RD" || echo "WARN: router_disc vllm failed (recorded)"; }
    rm -rf "$RD/ckpt"
    DEADLINE="$($PY -c "import time;print(int($SESSION_T0 + $MAX_S - 900))")"  # 15 min to bank
    rc=0
    PYTHONPATH="$REPO_DIR" $VPY papers/st_live_reference/scripts/st2_driver.py --out "$OUT" \
      --inputs "$ST2_INPUTS" --date-pin "$DATE_PIN" --deadline "$DEADLINE" \
      --phases "$ST2B_PHASES" --t11 "$T11_DIR" --c1 "$ST2A_DIR" --qwen-screen "$QWEN_SCREEN" || rc=$?
    echo ">> $m exit $rc (0 ok, 3 envelope stop; banked either way)"
    [ "$rc" -ne 0 ] && RC_ALL=$rc
    continue
  fi
  if [ "$STEPS" = "st2fx" ]; then
    rc=0
    for arm in none beh2 beh; do   # each arm its own process (HF on torch 2.13)
      PYTHONPATH="$REPO_DIR" $VPY papers/st_live_reference/scripts/st2fx_driver.py --arm "$arm" \
        --out "$OUT" --adapters "$FX_ADAPTERS" || { rc=$?; break; }
    done
    echo ">> $m matrix exit $rc; the D_rb rider (Amendments 34-35)"
    DEADLINE="$($PY -c "import time;print(int($SESSION_T0 + $MAX_S - 900))")"  # 15 min to bank
    rd=0
    $PY papers/st_live_reference/scripts/st2drb_driver.py --step gen --out "$OUT" \
      --deadline "$DEADLINE" || rd=$?   # vLLM via its worker
    if [ "$rd" -eq 0 ]; then
      for arm in none beh beh2 anchor beh_epoch3; do   # each arm its own process (HF on torch 2.13)
        PYTHONPATH="$REPO_DIR" $VPY papers/st_live_reference/scripts/st2drb_driver.py --step prefill \
          --arm "$arm" --out "$OUT" --adapters "$FX_ADAPTERS" --pool "$DRB_POOL" --t11 "$DRB_T11" \
          --deadline "$DEADLINE" || { rd=$?; break; }
      done
    fi
    [ "$rc" -eq 0 ] && rc=$rd
    echo ">> $m exit $rc (0 ok; banked either way)"
    [ "$rc" -ne 0 ] && RC_ALL=$rc
    continue
  fi
  if [ "$STEPS" = "st2mr" ]; then
    DEADLINE="$($PY -c "import time;print(int($SESSION_T0 + $MAX_S - 900))")"  # 15 min to bank
    rc=0
    PYTHONPATH="$REPO_DIR" $VPY papers/st_live_reference/scripts/st2mr_driver.py --out "$OUT" \
      --cell4 "$CELL4_DIR" --inputs "$ST2MR_INPUTS" --deadline "$DEADLINE" || rc=$?   # HF on torch 2.13
    echo ">> $m exit $rc (0 ok, 3 envelope stop; banked either way)"
    [ "$rc" -ne 0 ] && RC_ALL=$rc
    continue
  fi
  if [ "$STEPS" = "st2lb" ]; then
    DEADLINE="$($PY -c "import time;print(int($SESSION_T0 + $MAX_S - 900))")"  # 15 min to bank
    LB="papers/st_live_reference/scripts/st2lb_driver.py --out $OUT --date-pin $DATE_PIN --deadline $DEADLINE"
    rc=0
    $PY $LB --step rr_gen || rc=$?   # vLLM via its worker
    [ "$rc" -eq 0 ] && { PYTHONPATH="$REPO_DIR" $VPY $LB --step train --t14-inputs "$T14_INPUTS" \
      --t11 "$T11_DIR" || rc=$?; }   # HF on torch 2.13
    [ "$rc" -eq 0 ] && { PYTHONPATH="$REPO_DIR" $VPY $LB --step gate --t14-inputs "$T14_INPUTS" \
      || rc=$?; }   # T1.5's phases, each its own process
    echo ">> $m exit $rc (0 ok, 3 envelope stop; banked either way)"
    [ "$rc" -ne 0 ] && RC_ALL=$rc
    continue
  fi
  if [ "$STEPS" = "st2ml" ]; then
    DEADLINE="$($PY -c "import time;print(int($SESSION_T0 + $MAX_S - 900))")"  # 15 min to bank
    rc=0
    PYTHONPATH="$REPO_DIR" $VPY papers/st_live_reference/scripts/st2ml_driver.py --out "$OUT" \
      --cell4 "$CELL4_DIR" --deadline "$DEADLINE" || rc=$?   # HF on torch 2.13
    echo ">> $m exit $rc (0 ok, 3 envelope stop; banked either way)"
    [ "$rc" -ne 0 ] && RC_ALL=$rc
    continue
  fi
  if [ "$STEPS" = "st2m" ]; then
    DEADLINE="$($PY -c "import time;print(int($SESSION_T0 + $MAX_S - 900))")"  # 15 min to bank
    rc=0
    $PY papers/st_live_reference/scripts/st2m_driver.py --out "$OUT" --phases v1 \
      --date-pin "$DATE_PIN" --deadline "$DEADLINE" || rc=$?   # vLLM via its worker
    [ "$rc" -eq 0 ] && { PYTHONPATH="$REPO_DIR" $VPY papers/st_live_reference/scripts/st2m_driver.py \
      --out "$OUT" --phases "$ST2M_PHASES" --qwen-topup "$QWEN_TOPUP" --qwen-screen "$QWEN_SCREEN" \
      --date-pin "$DATE_PIN" --deadline "$DEADLINE" || rc=$?; }   # HF on torch 2.13
    echo ">> $m exit $rc (0 ok, 3 envelope stop; banked either way)"
    [ "$rc" -ne 0 ] && RC_ALL=$rc
    continue
  fi
  if [ "$STEPS" = "st2dl" ]; then
    DEADLINE="$($PY -c "import time;print(int($SESSION_T0 + $MAX_S - 900))")"  # 15 min to bank
    rc=0
    PYTHONPATH="$REPO_DIR" $VPY papers/st_live_reference/scripts/st2dl_driver.py --out "$OUT" \
      --inputs "$ST2_INPUTS" --podc-inputs "$ST2C_INPUTS" --c1 "$ST2A_DIR" --deadline "$DEADLINE" \
      --phases "$ST2DL_PHASES" || rc=$?
    echo ">> $m exit $rc (0 ok, 3 envelope stop; banked either way)"
    [ "$rc" -ne 0 ] && RC_ALL=$rc
    continue
  fi
  if [ "$STEPS" = "st2c" ]; then
    DEADLINE="$($PY -c "import time;print(int($SESSION_T0 + $MAX_S - 900))")"  # 15 min to bank
    rc=0
    PYTHONPATH="$REPO_DIR" $VPY papers/st_live_reference/scripts/st2c_driver.py --out "$OUT" \
      --inputs "$ST2_INPUTS" --podc-inputs "$ST2C_INPUTS" --c1 "$ST2A_DIR" --deadline "$DEADLINE" \
      --phases "$ST2C_PHASES" || rc=$?
    echo ">> $m exit $rc (0 ok, 3 envelope stop; banked either way)"
    [ "$rc" -ne 0 ] && RC_ALL=$rc
    continue
  fi
  if [ "$STEPS" = "st2a" ]; then
    DEADLINE="$($PY -c "import time;print(int($SESSION_T0 + $MAX_S - 900))")"  # 15 min to bank
    rc=0
    PYTHONPATH="$REPO_DIR" $VPY papers/st_live_reference/scripts/st2_driver.py --out "$OUT" \
      --inputs "$ST2_INPUTS" --date-pin "$DATE_PIN" --deadline "$DEADLINE" || rc=$?
    echo ">> $m exit $rc (0 ok, 3 envelope stop; banked either way)"
    [ "$rc" -ne 0 ] && RC_ALL=$rc
    continue
  fi
  if [ "$STEPS" = "t15" ]; then
    DEADLINE="$($PY -c "import time;print(int($SESSION_T0 + $MAX_S - 900))")"  # 15 min to bank
    rc=0
    PYTHONPATH="$REPO_DIR" $VPY papers/st_live_reference/scripts/t15_driver.py --out "$OUT" \
      --t14-inputs "$T14_INPUTS" --adapters "$T14_ADAPTERS" --date-pin "$DATE_PIN" \
      --deadline "$DEADLINE" || rc=$?   # HF on torch 2.13 (Amendment 9); vLLM via its worker
    echo ">> $m exit $rc (0 ok, 3 envelope stop; banked either way)"
    [ "$rc" -ne 0 ] && RC_ALL=$rc
    continue
  fi
  if [ "$STEPS" = "t14" ]; then
    DEADLINE="$($PY -c "import time;print(int($SESSION_T0 + $MAX_S - 900))")"  # 15 min to bank
    rc=0
    PYTHONPATH="$REPO_DIR" $VPY papers/st_live_reference/scripts/t14_driver.py --out "$OUT" \
      --inputs "$T14_INPUTS" --t11 "$T11_DIR" --t13 "$T13_DIR" --date-pin "$DATE_PIN" \
      --deadline "$DEADLINE" || rc=$?   # HF on torch 2.13 (Amendment 9)
    echo ">> $m exit $rc (0 ok, 2 gate failed, 3 envelope stop; banked either way)"
    [ "$rc" -ne 0 ] && RC_ALL=$rc
    continue
  fi
  if [ "$STEPS" = "t13" ]; then
    DEADLINE="$($PY -c "import time;print(int($SESSION_T0 + $MAX_S - 900))")"  # 15 min to bank
    rc=0
    $VPY papers/st_live_reference/scripts/t13_driver.py --out "$OUT" --inputs "$T13_INPUTS" \
      --date-pin "$DATE_PIN" --deadline "$DEADLINE" || rc=$?   # HF on torch 2.13 (Amendment 9)
    echo ">> $m exit $rc (0 ok, 2 gate failed, 3 envelope stop; banked either way)"
    [ "$rc" -ne 0 ] && RC_ALL=$rc
    continue
  fi
  if [ "$STEPS" = "t11" ]; then
    DEADLINE="$($PY -c "import time;print(int($SESSION_T0 + $MAX_S - 900))")"  # 15 min to bank
    T11="papers/st_live_reference/scripts/t11_driver.py --out $OUT --date-pin $DATE_PIN --deadline $DEADLINE"
    rc=0
    $PY $T11 --phase v1 || rc=$?
    [ "$rc" -eq 0 ] && { $VPY $T11 --phase h || rc=$?; }       # HF on torch 2.13 (Amendment 9)
    [ "$rc" -eq 0 ] && { $PY $T11 --phase v2 || rc=$?; }
    [ "$rc" -eq 0 ] && { $VPY $T11 --phase qlora || echo "WARN: QLoRA rider failed (recorded)"; }
    echo ">> $m exit $rc (0 ok, 2 gate failed, 3 envelope stop; banked either way)"
    [ "$rc" -ne 0 ] && RC_ALL=$rc
    continue
  fi
  $PY $S --model "$m" --date-pin "$DATE_PIN" --out "$OUT" $(mode_args "$m" "$BANKED"); rc=$?
  echo ">> $m exit $rc (0 ok, 2 gate failed, 3 envelope stop; banked either way)"
  $PY -c "import sys; sys.path.insert(0,'papers/kdg_panel/scripts'); from kdg_pod_lib import verify_manifest as v; bad=v('$OUT/$m/manifest_kdg.json'); print('manifest verify:', 'OK' if not bad else bad[:5])" \
    || echo "WARN: $m manifest verify did not run"
  [ "$rc" -ne 0 ] && RC_ALL=$rc
  [ "$STEPS" = "rider2" ] && [ "$rc" -eq 0 ] && rider2_after "$m"
  if [ "$CLEAN_HF_CACHE" = "1" ] && ! { [ "$STEPS" = "rider2" ] && [ "$m" = "gpt_oss_20b" ]; }; then
    for d in $(hf_cache_dirs "$m"); do [ -d "$d" ] && rm -rf "$d" && echo ">> removed $d"; done
  fi
done
if [ "$STEPS" = "rider2" ] && [[ " $MODELS " == *" gpt_oss_20b "* ]] && [ "$CLEAN_HF_CACHE" = "1" ]; then
  for d in $(hf_cache_dirs gpt_oss_20b); do [ -d "$d" ] && rm -rf "$d" && echo ">> removed $d"; done
fi

echo "{\"session_end_utc\": \"$(date -u +%FT%TZ)\", \"rc\": $RC_ALL}" > "$OUT/session_end.json"
echo "==================== $OUT_SUB/reprice ===================="
if [ "$STEPS" = "rider" ]; then RP_DIRS="$BANKED $OUT"; else RP_DIRS="$OUT"; fi
if [ "$STEPS" != "rider2" ] && [ "$STEPS" != "t13" ] && [ "$STEPS" != "t14" ] && [ "$STEPS" != "t15" ] && [ "$STEPS" != "st2a" ] && [ "$STEPS" != "st2b" ] && [ "$STEPS" != "st2c" ] && [ "$STEPS" != "st2dl" ] && [ "$STEPS" != "st2m" ] && [ "$STEPS" != "st2ml" ] && [ "$STEPS" != "st2lb" ] && [ "$STEPS" != "st2mr" ] && [ "$STEPS" != "st2fx" ]; then  # re-priced locally
  $PY papers/st_live_reference/scripts/reprice_tier1.py --screen-dir $RP_DIRS || echo "WARN: re-price failed"
fi
echo ">> ST T1.0 ${STEPS:-relaunch} done (rc=$RC_ALL). rsync-back -> papers/st_live_reference/outputs/$OUT_SUB/"
exit $RC_ALL
