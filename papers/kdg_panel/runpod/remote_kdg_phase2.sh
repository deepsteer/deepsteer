#!/usr/bin/env bash
# Remote runner for KDG Phase 2 (papers/KDG_F6_F8_SPEC.md §10). Orion launches (keys stay in
# Orion's terminal); KDG_PROFILE selects the session:
#
#   KDG_PROFILE=p2a  pilot: OLMo-3-7B-Instruct and Llama-3.1-8B-Instruct (Meta), one load each:
#                    F6–F8 pilot cells (P2PILOT), turns-since-norm rider on the model's own Phase 1
#                    screen (TSN), forward-vs-generate check (VALIDATE)
#   KDG_PROFILE=p2b  turns-since-norm follow-ups (P2-A4): Llama-3.1 Meta counterbalanced filler order
#                    (TSN_ROT, KDG-A16) + token-distance ladder (TSN_LEN, 600 / 2,000 tokens) on both
#                    models, each on its own Phase 1 screen, with VALIDATE (incl. a 2,000-token prompt)
#   KDG_PROFILE=p2c  GPT-OSS-20B Tier 2 addition (papers/KDG_GPTOSS_SPEC.md v0.2), one load:
#                    gates (bail, G-A4), C0 readout check, five letter units with decision-token
#                    residuals. VALIDATE=1 loads the real model, runs the gates and times one full
#                    letter unit + one C0 batch; exits 3 if the projection passes 2.6 A100-h.
#                    A100 80GB only (SXM4 or PCIe), so VALIDATE's timing and the real run share a
#                    GPU class; the real run refuses (exit 4) without a VALIDATE timing.json from
#                    the same class (it is synced up from papers/kdg_panel/outputs/p2c/validate/):
#     GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe" DISK_GB=150 \
#     SYNC_OUTPUTS=outputs/p2c/validate/timing.json MAX_SYNC_GB=1 \
#     REMOTE_SCRIPT=papers/kdg_panel/runpod/remote_kdg_phase2.sh KDG_PROFILE=p2c \
#     SELF_PAPER=papers/kdg_panel RESULTS_SUBPATH=outputs/p2c VALIDATE=1 \
#     ./papers/d1_moral_subspace/runpod/run_session.sh      (then the same without VALIDATE=1)
#   SYNC_OUTPUTS ships that one file and no other output (the KDG outputs tree, ~108 GB locally,
#   is excluded in rsync_exclude.txt); MAX_SYNC_GB refuses the launch if the upload would exceed
#   1 GB (the repo without outputs measures 0.28 GB, 2026-10-07).
#   KDG_PROFILE=p2f  GPT-OSS-20B dose-matched C0 only (KDG_GPTOSS_SPEC G-A11): the p2c gates, then
#                    64 x 4 generations after the empty closed analysis turn; no other cell, no
#                    timing record needed (about 0.3 A100-h with setup):
#     GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe" DISK_GB=150 SYNC_OUTPUTS=none \
#     MAX_SYNC_GB=1 REMOTE_SCRIPT=papers/kdg_panel/runpod/remote_kdg_phase2.sh KDG_PROFILE=p2f \
#     SELF_PAPER=papers/kdg_panel RESULTS_SUBPATH=outputs/p2f \
#     ./papers/d1_moral_subspace/runpod/run_session.sh
#   KDG_PROFILE=p2g  p2e then p2f in one pod (author, 2026-10-08): the Qwen2.5 C3 re-read on the
#                    image's stack of record (torch 2.4.1) first, then the torch 2.6 upgrade and the
#                    GPT-OSS dose-matched C0. Independent: a failure in one does not skip the other;
#                    the exit code is non-zero if either failed. About 0.8 A100-h; ~6.5 GB download:
#     GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe" DISK_GB=150 SYNC_OUTPUTS=none \
#     MAX_SYNC_GB=1 MIN_FREE_GB=20 REMOTE_SCRIPT=papers/kdg_panel/runpod/remote_kdg_phase2.sh \
#     KDG_PROFILE=p2g SELF_PAPER=papers/kdg_panel RESULTS_SUBPATH=outputs/p2g \
#     ./papers/d1_moral_subspace/runpod/run_session.sh
#   KDG_PROFILE=p2e  KDG-A20 item 2: Qwen2.5-7B-Instruct C3 re-read (four letter cells, readout v2,
#                    same order and batch size) on the stack of record, VALIDATE unit first:
#     GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe" DISK_GB=150 SYNC_OUTPUTS=none \
#     MAX_SYNC_GB=1 REMOTE_SCRIPT=papers/kdg_panel/runpod/remote_kdg_phase2.sh KDG_PROFILE=p2e \
#     SELF_PAPER=papers/kdg_panel RESULTS_SUBPATH=outputs/p2e \
#     ./papers/d1_moral_subspace/runpod/run_session.sh
#   KDG_PROFILE=p2d  KDG-A20 VALIDATE-only record (no cells) on OLMo-3, Llama-3.1 Meta, Tulu 3 and
#                    Qwen2.5 on the stack of record (torch 2.4.1, transformers 5.12.1; no torch
#                    upgrade): per prompt, readout v1 (the cells of record) and v2 vs generation,
#                    the unpadded read and pad counts on the G6 batch and a 0-70-token pad ladder.
#     GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe" DISK_GB=150 SYNC_OUTPUTS=none \
#     MAX_SYNC_GB=1 REMOTE_SCRIPT=papers/kdg_panel/runpod/remote_kdg_phase2.sh KDG_PROFILE=p2d \
#     SELF_PAPER=papers/kdg_panel RESULTS_SUBPATH=outputs/p2d \
#     ./papers/d1_moral_subspace/runpod/run_session.sh      (HF_TOKEN with Llama-3.1 access)
#   KDG_PROFILE=p2a_e1 / p2a_e2  extras E1 (OLMo-3 SFT/DPO) and E2 (within-RL sweep): refused here
#                    until the author schedules them at the pod gate (E2 also needs P2-A1 pushed)
#
#   GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA H100 80GB HBM3,NVIDIA H100 PCIe" \
#   DISK_GB=150 REMOTE_SCRIPT=papers/kdg_panel/runpod/remote_kdg_phase2.sh KDG_PROFILE=p2a \
#     SELF_PAPER=papers/kdg_panel RESULTS_SUBPATH=outputs/p2a \
#     ./papers/d1_moral_subspace/runpod/run_session.sh
#   VALIDATE=1 <same> ......  gates + dry run on the pod, then exit. Run this FIRST.
#   (Needs HF_TOKEN exported with Llama-3.1 access.)
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
PROFILE="${KDG_PROFILE:-}"
cd "$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
trap 'touch "$REPO_DIR/.session_done"' EXIT

case "$PROFILE" in
  p2a|p2b|p2c|p2d|p2e|p2f|p2g) ;;
  p2a_e1|p2a_e2) echo "FATAL: $PROFILE is an optional extra (spec §10) and is not scheduled; the author decides at the pod gate"; exit 1;;
  *) echo "FATAL: KDG_PROFILE must be p2a..p2g (got '$PROFILE')"; exit 1;;
esac
OUT="$REPO_DIR/papers/kdg_panel/outputs/$PROFILE"; mkdir -p "$OUT"
S=papers/kdg_panel/scripts/pod_kdg_phase1.py
D=papers/kdg_panel/data

echo ">> cuda: $(python -c 'import torch;print(torch.cuda.is_available())' 2>&1)"
pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -1 || pip install -q --break-system-packages -e . 2>&1 | tail -1
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"
pip install -q --break-system-packages "transformers==$TRANSFORMERS_VERSION" -U accelerate pytest pyyaml 2>&1 | tail -1 || true
pip install -q --break-system-packages hf_xet >/dev/null 2>&1 && export HF_XET_HIGH_PERFORMANCE=1
echo ">> transformers: $(python -c 'import transformers;print(transformers.__version__)' 2>&1)"

# ---- p2c: GPT-OSS-20B (KDG_GPTOSS_SPEC v0.2); its own driver, gates inside it ----
if [ "$PROFILE" = "p2c" ] || [ "$PROFILE" = "p2f" ]; then
  G=papers/kdg_panel/scripts/pod_kdg_gptoss.py
  # GPT-OSS's mxfp4 quantizer calls torch.accelerator (torch >= 2.6). The image ships 2.4, so
  # upgrade the matched trio exactly as W4 did (manifest_w4: torch 2.6.0+cu124, transformers
  # 5.12.1) and stop before any model download if it did not take.
  if ! python -c 'import torch; torch.accelerator' 2>/dev/null; then
    echo ">> torch $(python -c 'import torch;print(torch.__version__)') lacks torch.accelerator; upgrading the trio to 2.6.0+cu124 (W4 stack)..."
    pip install -q --break-system-packages "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
      --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3
  fi
  python -c 'import torch; torch.accelerator; assert torch.cuda.is_available()' \
    || { echo "FATAL: torch.accelerator missing or CUDA unavailable after the upgrade; GPT-OSS cannot load."; exit 1; }
  echo ">> torch: $(python -c 'import torch;print(torch.__version__)')  transformers: $(python -c 'import transformers;print(transformers.__version__)')"
  echo ">> KDG local gates (p2c):"
  python -m pytest -q tests/kdg tests/scripts/test_pod_kdg_phase1.py tests/scripts/test_pod_kdg_gptoss.py \
    || { echo "LOCAL GATE FAILED"; exit 1; }
  N_ROWS="$(python -c 'import glob,json;print(sum(len(json.load(open(p))["scenarios"]) for p in glob.glob("papers/kdg_panel/data/*_scenarios_*.json")))')"
  echo ">> scenario rows on pod: $N_ROWS"
  [ "${N_ROWS:-0}" -ge 580 ] || { echo "FATAL: fewer than 580 union scenario rows"; exit 1; }
  VRAM_GB="$(python -c 'import torch;print(int(torch.cuda.get_device_properties(0).total_memory/1e9)) if torch.cuda.is_available() else 0' 2>/dev/null || echo 0)"
  echo ">> GPU VRAM: ${VRAM_GB} GB"
  [ "${VRAM_GB:-0}" -lt 75 ] && { echo "FATAL: need an 80 GB card for GPT-OSS-20B bf16 dequant."; exit 1; }
  GPU_NAME="$(python -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null || echo none)"
  echo ">> GPU: $GPU_NAME"
  case "$GPU_NAME" in
    *A100*80GB*) ;;
    *) echo "FATAL: p2c runs on A100 80GB only (timing parity with VALIDATE, G-A4); got '$GPU_NAME'"; exit 1;;
  esac
  python $G --dry-run --out "$OUT/_dry" || { echo "DRY RUN FAILED"; exit 1; }
  if [ "$PROFILE" = "p2f" ]; then  # G-A11: dose-matched C0 only
    python $G --dry-run --c0-dm --out "$OUT/_dry_c0dm" || { echo "DRY RUN (C0-dm) FAILED"; exit 1; }
    [ "$VALIDATE" = "1" ] && { echo ">> VALIDATE=1: gates + dry runs OK. Launch without VALIDATE=1."; exit 0; }
    echo "==================== p2f/c0dm ===================="
    python $G --c0-dm --out "$OUT/c0dm"; rc=$?
    python papers/kdg_panel/scripts/pod_kdg_phase1.py --verify-manifest --out "$OUT/c0dm" \
      || echo "WARN: manifest verify reported mismatches"
    echo ">> KDG p2f done (rc=$rc). rsync-back -> papers/kdg_panel/outputs/p2f/; then"
    echo "   python3 papers/kdg_panel/scripts/analyze_gptoss_c0dm.py"
    exit $rc
  fi
  if [ "$VALIDATE" = "1" ]; then
    python $G --validate --batch-invariance --out "$OUT/validate"; rc=$?
    case $rc in
      0) echo ">> VALIDATE OK (gates + timing). Launch without VALIDATE for the real run.";;
      2) echo ">> BAIL: a G-A4 gate failed (see $OUT/validate/manifest_kdg.json). No launch.";;
      3) echo ">> STOP: timing projection over 2.6 A100-h (see $OUT/validate/timing.json). Report before launch.";;
      *) echo ">> VALIDATE crashed (rc=$rc).";;
    esac
    exit $rc
  fi
  echo "==================== p2c/gpt_oss_20b ===================="
  python $G --out "$OUT/gpt_oss_20b" --require-timing "$OUT/validate/timing.json" --batch-invariance; rc=$?
  [ $rc -eq 4 ] && echo ">> REFUSED: no matching VALIDATE timing record (see log above)."
  python papers/kdg_panel/scripts/pod_kdg_phase1.py --verify-manifest --out "$OUT/gpt_oss_20b" \
    || echo "WARN: manifest verify reported mismatches"
  echo ">> KDG p2c done (rc=$rc). rsync-back -> papers/kdg_panel/outputs/p2c/"
  exit $rc
fi

# ---- p2d: KDG-A20 VALIDATE-only record on the four panel models (no cells) ----
if [ "$PROFILE" = "p2d" ]; then
  # the bound is for the cells of record, so it is measured on their stack (p1a/p1b/p2a manifests)
  STACK="$(python -c 'import torch,transformers;print(torch.__version__, transformers.__version__)')"
  echo ">> stack: $STACK"
  case "$STACK" in
    "2.4.1+cu124 5.12.1") ;;
    *) echo "FATAL: p2d must run on the stack of record (torch 2.4.1+cu124, transformers 5.12.1)"; exit 1;;
  esac
  [ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ] || { echo "FATAL: HF_TOKEN unset (Llama-3.1 is gated)"; exit 1; }
  echo ">> KDG local gates (p2d):"
  python -m pytest -q tests/kdg tests/scripts/test_pod_kdg_phase1.py tests/scripts/test_kdg_readout_v2.py \
    || { echo "LOCAL GATE FAILED"; exit 1; }
  GPU_NAME="$(python -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null || echo none)"
  echo ">> GPU: $GPU_NAME"
  case "$GPU_NAME" in *A100*80GB*) ;; *) echo "FATAL: p2d runs on A100 80GB only; got '$GPU_NAME'"; exit 1;; esac
  P2D_MODELS=olmo3_instruct,llama31_instruct_meta,tulu3_final,qwen25_instruct_p1
  python $S --dry-run --models $P2D_MODELS --units VALIDATE --out "$OUT/_dry" || { echo "DRY RUN FAILED"; exit 1; }
  if [ "$VALIDATE" = "1" ]; then
    echo ">> VALIDATE=1: gates + dry run OK. Launch without VALIDATE=1 for the record."; exit 0
  fi
  echo "==================== p2d/validate ===================="
  # one load per model; a v2 gate miss is recorded (the record is written before the unit raises)
  python $S --out "$OUT/validate" --models $P2D_MODELS --units VALIDATE; rc=$?
  python $S --verify-manifest --out "$OUT/validate" || echo "WARN: manifest verify reported mismatches"
  for k in olmo3_instruct llama31_instruct_meta tulu3_final qwen25_instruct_p1; do
    [ -f "$OUT/validate/$k/forward_matches_generate.json" ] || echo "WARN: no VALIDATE record for $k"
  done
  echo ">> KDG p2d done (rc=$rc). rsync-back -> papers/kdg_panel/outputs/p2d/; then"
  echo "   python3 papers/kdg_panel/scripts/analyze_kdg_a20_validate.py"
  exit $rc
fi

# ---- p2g: p2e (Qwen2.5 re-read, stack of record) then p2f (GPT-OSS C0-dm, torch 2.6) ----
if [ "$PROFILE" = "p2g" ]; then
  G=papers/kdg_panel/scripts/pod_kdg_gptoss.py
  P2E_UNITS=validate_forward_matches_generate,dl_chat_neutral,jl_chat_neutral,dl_chat_neutral_pressure_removed,jl_chat_neutral_pressure_removed
  # every check that can fail runs before either model is downloaded
  STACK="$(python -c 'import torch,transformers;print(torch.__version__, transformers.__version__)')"
  echo ">> stack: $STACK"
  case "$STACK" in
    "2.4.1+cu124 5.12.1") ;;
    *) echo "FATAL: the Qwen2.5 re-read needs the stack of record (torch 2.4.1+cu124, transformers 5.12.1)"; exit 1;;
  esac
  echo ">> KDG local gates (p2g):"
  python -m pytest -q tests/kdg tests/scripts/test_pod_kdg_phase1.py tests/scripts/test_kdg_readout_v2.py \
    tests/scripts/test_pod_kdg_gptoss.py || { echo "LOCAL GATE FAILED"; exit 1; }
  GPU_NAME="$(python -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null || echo none)"
  VRAM_GB="$(python -c 'import torch;print(int(torch.cuda.get_device_properties(0).total_memory/1e9))' 2>/dev/null || echo 0)"
  echo ">> GPU: $GPU_NAME ($VRAM_GB GB)"
  case "$GPU_NAME" in *A100*80GB*) ;; *) echo "FATAL: p2g runs on A100 80GB only; got '$GPU_NAME'"; exit 1;; esac
  N_ROWS="$(python -c 'import glob,json;print(sum(len(json.load(open(p))["scenarios"]) for p in glob.glob("papers/kdg_panel/data/*_scenarios_*.json")))')"
  [ "${N_ROWS:-0}" -ge 580 ] || { echo "FATAL: fewer than 580 union scenario rows"; exit 1; }
  python $S --dry-run --models qwen25_instruct_p1 --units $P2E_UNITS --out "$OUT/_dry_reread" \
    || { echo "DRY RUN (re-read) FAILED"; exit 1; }
  python $G --dry-run --c0-dm --out "$OUT/_dry_c0dm" || { echo "DRY RUN (C0-dm) FAILED"; exit 1; }
  [ "$VALIDATE" = "1" ] && { echo ">> VALIDATE=1: gates + dry runs OK. Launch without VALIDATE=1."; exit 0; }

  echo "==================== p2g 1/2: Qwen2.5 re-read (KDG-A20 item 2) ===================="
  python $S --out "$OUT/reread" --models qwen25_instruct_p1 --units $P2E_UNITS; rc_q=$?
  python $S --verify-manifest --out "$OUT/reread" || echo "WARN: re-read manifest verify reported mismatches"

  echo "==================== p2g 2/2: GPT-OSS dose-matched C0 (G-A11) ===================="
  rc_g=1
  echo ">> upgrading the torch trio to 2.6.0+cu124 for GPT-OSS (W4 stack)..."
  pip install -q --break-system-packages "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3
  if python -c 'import torch; torch.accelerator; assert torch.cuda.is_available()'; then
    echo ">> torch: $(python -c 'import torch;print(torch.__version__)')"
    python $G --c0-dm --out "$OUT/c0dm"; rc_g=$?
    python $S --verify-manifest --out "$OUT/c0dm" || echo "WARN: C0-dm manifest verify reported mismatches"
  else
    echo "FATAL (C0-dm skipped): torch.accelerator missing after the upgrade"
  fi
  echo ">> KDG p2g done (re-read rc=$rc_q, C0-dm rc=$rc_g). rsync-back -> papers/kdg_panel/outputs/p2g/; then"
  echo "   python3 papers/kdg_panel/scripts/analyze_kdg_a20_qwen_reread.py"
  echo "   python3 papers/kdg_panel/scripts/analyze_gptoss_c0dm.py"
  [ $rc_q -eq 0 ] && [ $rc_g -eq 0 ] && exit 0 || exit 1
fi

# ---- p2e: KDG-A20 item 2, Qwen2.5 C3 re-read on the stack of record ----
if [ "$PROFILE" = "p2e" ]; then
  STACK="$(python -c 'import torch,transformers;print(torch.__version__, transformers.__version__)')"
  echo ">> stack: $STACK"
  case "$STACK" in
    "2.4.1+cu124 5.12.1") ;;
    *) echo "FATAL: p2e must run on the stack of record (torch 2.4.1+cu124, transformers 5.12.1)"; exit 1;;
  esac
  echo ">> KDG local gates (p2e):"
  python -m pytest -q tests/kdg tests/scripts/test_pod_kdg_phase1.py tests/scripts/test_kdg_readout_v2.py \
    || { echo "LOCAL GATE FAILED"; exit 1; }
  GPU_NAME="$(python -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null || echo none)"
  echo ">> GPU: $GPU_NAME"
  case "$GPU_NAME" in *A100*80GB*) ;; *) echo "FATAL: p2e runs on A100 80GB only; got '$GPU_NAME'"; exit 1;; esac
  P2E_UNITS=validate_forward_matches_generate,dl_chat_neutral,jl_chat_neutral,dl_chat_neutral_pressure_removed,jl_chat_neutral_pressure_removed
  python $S --dry-run --models qwen25_instruct_p1 --units $P2E_UNITS --out "$OUT/_dry" || { echo "DRY RUN FAILED"; exit 1; }
  [ "$VALIDATE" = "1" ] && { echo ">> VALIDATE=1: gates + dry run OK. Launch without VALIDATE=1."; exit 0; }
  echo "==================== p2e/reread ===================="
  # default scenario set and order = the record's (checked 2026-10-07: rows in load order, seeds 0..7)
  python $S --out "$OUT/reread" --models qwen25_instruct_p1 --units $P2E_UNITS; rc=$?
  python $S --verify-manifest --out "$OUT/reread" || echo "WARN: manifest verify reported mismatches"
  echo ">> KDG p2e done (rc=$rc). rsync-back -> papers/kdg_panel/outputs/p2e/; then"
  echo "   python3 papers/kdg_panel/scripts/analyze_kdg_a20_qwen_reread.py"
  exit $rc
fi

# ---- no-model gates always run first ----
echo ">> KDG local gates:"
python -m pytest -q tests/kdg tests/scripts/test_pod_kdg_phase1.py tests/scripts/test_pod_kdg_phase2.py \
  || { echo "LOCAL GATE FAILED"; exit 1; }
# author rule (2026-10-01): the Phase 3 action-position pre-registration exists before this pod
[ -f papers/KDG_PHASE3_SPEC.md ] || { echo "FATAL: papers/KDG_PHASE3_SPEC.md missing (Phase 3 prereg is due before the Phase 2 pod)"; exit 1; }
ITEMS="$(ls $D/p2pilot_items_A_*.json $D/p2pilot_items_B_*.json 2>/dev/null | tr '\n' ' ')"
if [ "$PROFILE" = "p2a" ]; then
[ "$(echo $ITEMS | wc -w)" -eq 2 ] || { echo "FATAL: need exactly one merged item file per half (got: $ITEMS)"; exit 1; }
python - $ITEMS <<'PY' || { echo "FATAL: item files not pilot-complete"; exit 1; }
import sys
from deepsteer.kdg.phase2 import load_items
n = {}
for p in sys.argv[1:]:
    items, meta = load_items(p)
    assert meta.get("external_label_rater"), f"{p}: external labels missing (spec §2.1)"
    for it in items:
        n[it.family] = n.get(it.family, 0) + 1
assert n == {"F6": 24, "F7": 12, "F8": 12}, n
print(">> pilot items:", n)
PY
fi
for f in screened_ids_a17_union.json screened_ids_llama31_meta.json tsn_filler_turns.json \
         tsn_filler_turns_600.json tsn_filler_turns_2000.json; do
  [ -f "$D/$f" ] || { echo "FATAL: $D/$f missing"; exit 1; }
done

VRAM_GB="$(python -c 'import torch;print(int(torch.cuda.get_device_properties(0).total_memory/1e9)) if torch.cuda.is_available() else 0' 2>/dev/null || echo 0)"
echo ">> GPU VRAM: ${VRAM_GB} GB"
[ "${VRAM_GB:-0}" -lt 38 ] && { echo "FATAL: need >= 40 GB for 7-8B bf16."; exit 1; }

if [ "$PROFILE" = "p2a" ]; then
  python $S --dry-run --models olmo3_instruct,olmo3_sft --units P2PILOT,TSN,VALIDATE --items $ITEMS \
    --scenario-ids-file $D/screened_ids_a17_union.json --out "$OUT/_dry" || { echo "DRY RUN FAILED"; exit 1; }
else
  python $S --dry-run --models llama31_instruct_meta --units VALIDATE,TSN_ROT,TSN_LEN \
    --scenario-ids-file $D/screened_ids_llama31_meta.json --out "$OUT/_dry" || { echo "DRY RUN FAILED"; exit 1; }
fi
if [ "$VALIDATE" = "1" ]; then
  echo ">> VALIDATE: gates + dry run OK on the pod. Launch without VALIDATE for the real run."; exit 0
fi

step() {  # step <name> <driver args...>
  local name="$1"; shift
  echo "==================== $PROFILE/$name ===================="
  python $S --out "$OUT/$name" "$@"; local rc=$?
  python $S --verify-manifest --out "$OUT/$name" || echo "WARN: $name manifest verify reported mismatches"
  return $rc
}

# Bail (spec §12): VALIDATE runs first in each step's unit list; a forward-vs-generate mismatch fails
# that unit and is recorded; the pilot cells still run and the gate analysis reports G6 per model.
if [ "$PROFILE" = "p2a" ]; then
  step olmo3_instruct --models olmo3_instruct --units VALIDATE,P2PILOT,TSN --items $ITEMS \
    --scenario-ids-file $D/screened_ids_a17_union.json
  step llama31_instruct_meta --models llama31_instruct_meta --units VALIDATE,P2PILOT,TSN --items $ITEMS \
    --scenario-ids-file $D/screened_ids_llama31_meta.json
else  # p2b: the KDG-A16 discriminator first (author), then the ladder
  step llama31_instruct_meta --models llama31_instruct_meta --units VALIDATE,TSN_ROT,TSN_LEN \
    --scenario-ids-file $D/screened_ids_llama31_meta.json
  step olmo3_instruct --models olmo3_instruct --units VALIDATE,TSN_LEN \
    --scenario-ids-file $D/screened_ids_a17_union.json
fi
echo ">> KDG Phase 2 $PROFILE done. rsync-back -> papers/kdg_panel/outputs/$PROFILE/ (one manifest per step)."
