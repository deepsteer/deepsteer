#!/usr/bin/env bash
# Remote runner for C2-JAB's room pilot (GPC_PLAN.md §19.13), on Thunder, with no host and no adapter: the
# subject's vLLM server under 16,384 (the base only), its check, a smoke loop (the bail), the 240 loops on the 10
# pilot families (conflict and twin, 12 each) with the cold verdict at each decision point, and the read. No judge,
# no reasoning text read by any model. Writes papers/gpc_goal_pressure_conduct/outputs/c2_room (stages stop at
# 4.0 h, the watchdog at 4.5 h (§19.15), both movable with set_limits.sh; ceiling 14.0 h).
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
G="$REPO_DIR/papers/gpc_goal_pressure_conduct"
SESSION="${CST_SESSION:-}"
PRICE_H=4.25; WATCH_H=4.5   # §19.15 item 3: stages stop at 4.0 h, the watchdog at 4.5 h
OUT="$G/outputs/c2_room"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
source "$REPO_DIR/papers/runpod_common/session_limits.sh"
trap 'set +u +e; { kill ${WATCHDOG:-}; } 2>/dev/null; limits_save "$OUT/logs" || true; cp "$REPO_DIR/session.log" "$OUT/logs/session_c2room${SESSION:-}_${STAMP:-}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=$WATCH_H;;
  *) echo "BAIL: CST_SESSION must be 1"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.hf_token" 2>/dev/null)}"
PYBIN="${CST_PYTHON:-python3}"
export CST_PRICE_H=$PRICE_H CST_START_EPOCH=$START
export CST_DEADLINE_EPOCH=$(python3 -c "print(int($START + ($PRICE_H - 0.25) * 3600))")
limits_init "$REPO_DIR/.session_limits.json" "$START" "$(python3 -c "print($PRICE_H - 0.25)")" "$WATCH_H" \
  || { echo "BAIL: the limits file"; exit 1; }
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> C2 room session $SESSION start $(date -u +%FT%TZ) out=$OUT"
[ "$($PYBIN -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = 3.11 ] || { echo "BAIL: $PYBIN is not Python 3.11"; exit 1; }
limits_watchdog 'c2_room_pod.py|gpc4_pod.py|vllm'

# ---- the vLLM venv (the pins every GPC-4 pod used) ----
VENV=/workspace/venv_vllm
VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_SHA256=5176ab746f4e736c4a354a11d0620abf0a941de287cad937ac34039293ca6f5f
CU129_INDEX=https://download.pytorch.org/whl/cu129
CONSTRAINTS=papers/st_live_reference/runpod/vllm_constraints.txt
if [ ! -x "$VENV/bin/python" ]; then
  echo ">> creating $VENV"
  $PYBIN -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip 2>&1 | tail -1
  "$VENV/bin/pip" install -q "torch==2.13.0+cu129" "torchvision==0.28.0+cu129" \
    "torchaudio==2.11.0+cu129" --index-url "$CU129_INDEX" 2>&1 | tail -2
  "$VENV/bin/pip" install -q "vllm @ ${VLLM_WHEEL_URL}#sha256=${VLLM_WHEEL_SHA256}" \
    "transformers==5.17.0" "peft==0.21.2" pyyaml ninja pytest tokenizers -c "$CONSTRAINTS" \
    --extra-index-url "$CU129_INDEX" 2>&1 | tail -2
  "$VENV/bin/pip" install -q -e . "transformers==5.17.0" -c "$CONSTRAINTS" --extra-index-url "$CU129_INDEX" 2>&1 | tail -2
fi
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0 GPC4_VLLM="$VENV/bin/vllm"
$PY - <<'PYEOF' || { echo "BAIL: vLLM venv not usable"; exit 1; }
import importlib.metadata as md, torch, vllm  # noqa: E401,F401
v = {p: md.version(p) for p in ("vllm", "torch", "transformers")}
ok = v["vllm"].startswith("0.31.0") and v["transformers"] == "5.17.0" and torch.cuda.is_available()
print(">> stack:", v, "cuda", torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      "OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
PYEOF
$PY - <<'PYEOF' > "$OUT/logs/models_c2room${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
print(json.dumps({"qwen38_27b": snapshot_download("Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                                                  ignore_patterns=["original/*", "metal/*"])}))
PYEOF
echo ">> models: $(cat "$OUT/logs/models_c2room${SESSION}.json")"
echo ">> setup done at $(hours) h"

TESTS="tests/scripts/test_c2.py tests/scripts/test_session_limits.py"
GATE_MARK="not slow and not host_side"
GATE_FLOOR=24   # the room pod's gate tests (§19.13, §19.14); fewer means a file failed to collect
NCOL=$($PY -m pytest --collect-only -q -p no:cacheprovider -m "$GATE_MARK" $TESTS 2>/dev/null | grep -c '::' || true)
echo ">> gate tests collected: $NCOL (floor $GATE_FLOOR)"
[ "$NCOL" -ge "$GATE_FLOOR" ] || { echo "BAIL: the gate collected $NCOL tests, under its floor of $GATE_FLOOR"; exit 1; }
$PY -m pytest -q -p no:cacheprovider -m "$GATE_MARK" $TESTS > "$OUT/logs/gate_tests${SESSION}.log" 2>&1; GT=$?
grep -E '[0-9]+ (passed|failed)' "$OUT/logs/gate_tests${SESSION}.log" | tail -1
[ "$GT" = 0 ] || { echo "BAIL: gate tests (see logs/gate_tests${SESSION}.log)"; exit 1; }

DRY=${GPC4_DRY:+--dry-run}
GP=papers/gpc_goal_pressure_conduct/scripts/gpc4_pod.py
RP=papers/gpc_goal_pressure_conduct/scripts/c2_room_pod.py
run() { $PY "$1" --out "$OUT" "${@:2}" $DRY || echo ">> stage failed: ${*:2}"; echo ">> stage done at $(hours) h: ${*:2}"; }
stop() { echo ">> $1"; echo ">> done at $(hours) h"; kill ${WATCHDOG:-} 2>/dev/null; exit 0; }
end_session() { echo ">> $1"; run $GP --stage serve --part down; stop "$1"; }

# a. the server on the base only (no adapter is registered in this session), under 16,384, and its check
run $GP --stage serve --part up
run $GP --stage serve_check
$PY $GP --out "$OUT" --stage server_ok $DRY || end_session "the subject's server is dead: the session ends"
# b. the smoke (the bail), the 240 loops with their verdicts, the read
$PY $RP --out "$OUT" --stage smoke $DRY || end_session "the smoke loop failed: the session ends (see room/loops.jsonl)"
run $RP --stage loops
run $RP --stage read
run $GP --stage serve --part down
stop "all stages run"
