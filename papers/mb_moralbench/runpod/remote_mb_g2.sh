#!/usr/bin/env bash
# Remote runner for MB-G2 (papers/MB_PLAN.md v0.7 Amendment 2). Orion launches (keys stay in
# Orion's terminal); the launcher passes HF_TOKEN, VALIDATE and SESSION_MAX_HOURS.
#
#   VALIDATE=1 GPU_COUNT=1 GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe" DISK_GB=80 \
#     REMOTE_SCRIPT=papers/mb_moralbench/runpod/remote_mb_g2.sh SELF_PAPER=papers/mb_moralbench \
#     RESULTS_SUBPATH=outputs/g2_validate SYNC_OUTPUTS=none ./papers/runpod_common/run_session.sh
#       stack, access, MoralBench fetch, gate tests, the tiny dry run on CUDA, and a 3-item Gemma
#       timing probe; stop at 0.3 h.
#   GPU_COUNT=2 GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe" DISK_GB=260 \
#     REMOTE_SCRIPT=papers/mb_moralbench/runpod/remote_mb_g2.sh SELF_PAPER=papers/mb_moralbench \
#     RESULTS_SUBPATH=outputs/g2 SYNC_OUTPUTS=none ./papers/runpod_common/run_session.sh
#       the session: GPU0 Gemma-1.1 (t07, cb07, t10) and GPU1 D4 zephyr-7b-gemma (t07) at once,
#       the Llama-2-70B download meanwhile, then Llama-2-70B-chat (t07, cb07, t10) on both GPUs.
#
# MB_LOCAL_DRY=1 (tests/scripts/test_mb_g2_remote.py): the same sequence on CPU with the tiny dry
# model, no installs, no GPU or access checks, outputs under OUT_ROOT.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
LOCAL_DRY="${MB_LOCAL_DRY:-0}"
PY="${PY:-python}"  # the local test passes its interpreter (no `python` on the Mac)
cd "$REPO_DIR"
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/papers/mb_moralbench}"
SUB=g2
[ "$VALIDATE" = 1 ] && SUB=g2_validate
OUT="$OUT_ROOT/outputs/$SUB"
mkdir -p "$OUT"
SESSION_MAX_HOURS="${SESSION_MAX_HOURS:-1.95}"
[ "$VALIDATE" = 1 ] && SESSION_MAX_HOURS="${SESSION_MAX_HOURS_VALIDATE:-0.3}"
START=$(date +%s)
# cells stop starting 15 min before the envelope: analysis, the manifest and the download remain
DEADLINE=$("$PY" -c "print($START + $SESSION_MAX_HOURS * 3600 - 900)")
SENTINEL_DIR="${SENTINEL_DIR:-$REPO_DIR}"  # the launcher waits for $REPO_DIR/.session_done
trap 'cp "$REPO_DIR/session.log" "$OUT/session.log" 2>/dev/null; touch "$SENTINEL_DIR/.session_done"' EXIT

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
RUN=papers/mb_moralbench/scripts/mb_g1_run.py
ANALYZE=papers/mb_moralbench/scripts/mb_g1_analyze.py
echo ">> MB-G2 start $(date -u +%FT%TZ) VALIDATE=$VALIDATE LOCAL_DRY=$LOCAL_DRY out=$OUT max_h=$SESSION_MAX_HOURS"

# watchdog: nothing runs past the envelope
# (the sleep runs as the subshell's own child so a TERM at the end also ends it; an orphaned sleep
# would hold the log pipe open for the whole envelope)
( trap 'kill $SLEEPER 2>/dev/null; exit 0' TERM
  sleep "$("$PY" -c "print(int($SESSION_MAX_HOURS * 3600))")" & SLEEPER=$!
  wait $SLEEPER
  echo ">> WATCHDOG: envelope ${SESSION_MAX_HOURS} h reached; stopping the drivers"
  pkill -f mb_g1_run.py ) &
WATCHDOG=$!

if [ "$LOCAL_DRY" != 1 ]; then
  echo ">> installing (transformers ${TRANSFORMERS_VERSION:=5.17.0}, the G1 stack's version)"
  pip install -q --break-system-packages -e . 2>&1 | tail -2
  pip install -q --break-system-packages "transformers==$TRANSFORMERS_VERSION" -U accelerate \
    hf_xet 2>&1 | tail -2
  export HF_XET_HIGH_PERFORMANCE=1
  if ! python -c 'import torch, transformers; from transformers import AutoModelForCausalLM; assert torch.cuda.is_available()' 2>/dev/null; then
    echo ">> torch $(python -c 'import torch;print(torch.__version__)' 2>&1) fails with transformers $TRANSFORMERS_VERSION; upgrading the trio to 2.6.0+cu124 (the KDG/W4 stack)"
    pip install -q --break-system-packages "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
      --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
  fi
  python -c 'import torch, transformers; assert torch.cuda.is_available(); print(">> stack: torch", torch.__version__, "transformers", transformers.__version__, "gpus", torch.cuda.device_count())' \
    || { echo "BAIL: the stack does not import with CUDA"; exit 1; }
  NGPU=$(python -c 'import torch; print(torch.cuda.device_count())')
  if [ "$VALIDATE" != 1 ] && [ "$NGPU" -lt 2 ]; then echo "BAIL: Llama-2-70B needs 2 GPUs, found $NGPU"; exit 1; fi
  python - <<'EOF' || { echo "BAIL: model access"; exit 1; }
from huggingface_hub import auth_check
for r in ("google/gemma-1.1-7b-it", "HuggingFaceH4/zephyr-7b-gemma-v0.1",
          "meta-llama/Llama-2-70b-chat-hf", "HuggingFaceH4/zephyr-7b-beta"):
    auth_check(r)
    print(">> access ok:", r)
EOF
fi

# dependency: MoralBench at the pinned commit, verified against the committed manifest
"$PY" -c 'from deepsteer.datasets.moralbench import fetch_moralbench; print(">> moralbench:", fetch_moralbench())' \
  || { echo "BAIL: MoralBench fetch or manifest check"; exit 1; }

if [ "$LOCAL_DRY" != 1 ]; then
  # gate tests (each names its most probable failure)
  python -m pytest -q -p no:cacheprovider tests/datasets/test_moralbench.py \
    tests/scripts/test_mb_g0_audit.py tests/scripts/test_mb_g1_run.py \
    tests/scripts/test_mb_g1_analyze.py tests/scripts/test_mb_g2_remote.py 2>&1 | tail -3
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }
fi

drv() {  # drv <model> <cells> <gpus> [extra args]: one driver call, raw replies under $OUT/_raw
  local m="$1" cells="$2" gpus="$3"; shift 3
  local extra=()
  [ "$LOCAL_DRY" = 1 ] && extra=(--dry-run --device cpu)
  [ "$LOCAL_DRY" != 1 ] && extra=(--device cuda)
  echo ">> unit $m cells=$cells gpus=$gpus $(date -u +%T)"
  CUDA_VISIBLE_DEVICES="$gpus" "$PY" "$RUN" --model "$m" --cells "$cells" --out "$OUT_ROOT" \
    --out-sub "$SUB" --raw-dir "$OUT/_raw/$m" --deadline-epoch "$DEADLINE" "${extra[@]}" "$@" \
    > "$OUT/driver_$m.log" 2>&1
  local rc=$?
  tail -3 "$OUT/driver_$m.log"
  echo ">> unit $m rc=$rc $(date -u +%T)"
  return $rc
}

if [ "$VALIDATE" = 1 ]; then
  drv zephyr t07 0 --dry-run --device cuda || echo "VALIDATE: dry run failed"
  drv gemma t07 0 --max-items 3 || echo "VALIDATE: Gemma probe failed"
  "$PY" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
m = json.loads((Path(sys.argv[1]) / "gemma" / "run_meta.json").read_text())["runs"][-1]
rate = m["timings_s"]["t07"] / 3
print(f">> VALIDATE: Gemma {rate:.1f} s/item on one GPU; its three cells project to "
      f"{rate * 264 / 60:.0f} min")
EOF
else
  # Llama-2-70B's weights download while the two 8.5B models run
  DL=""
  if [ "$LOCAL_DRY" != 1 ]; then
    "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-2-70b-chat-hf', revision='e9149a12809580e8602995856f8098ce973d1080', allow_patterns=['*.json','*.safetensors','tokenizer*','*.model'])" \
      > "$OUT/llama_download.log" 2>&1 &
    DL=$!
  fi
  drv gemma t07,cb07,t10 0 & G=$!
  drv zephyr_gemma t07 1 & Z=$!
  wait $G; wait $Z
  [ -n "$DL" ] && { wait $DL || echo "WARN: Llama-2-70B download failed (see llama_download.log)"; }
  drv llama2_70b t07,cb07,t10 0,1
  for m in gemma zephyr_gemma llama2_70b; do
    [ -f "$OUT/$m/t07.jsonl" ] && "$PY" "$ANALYZE" --model "$m" --root "$OUT_ROOT" --sub "$SUB" \
      > "$OUT/analysis_$m.log" 2>&1
  done
fi

kill "$WATCHDOG" 2>/dev/null
# session manifest: every output file with its sha256 (the raw replies included)
"$PY" - "$OUT" <<'EOF'
import hashlib, json, sys
from pathlib import Path
out = Path(sys.argv[1])
files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest_g2.json"}
(out / "manifest_g2.json").write_text(json.dumps({"files": files}, indent=1) + "\n")
print(f">> manifest: {len(files)} files")
EOF
echo ">> MB-G2 end $(date -u +%FT%TZ)"
