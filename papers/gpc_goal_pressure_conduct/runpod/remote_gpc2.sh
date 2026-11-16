#!/usr/bin/env bash
# Remote runner for GPC-2 (GPC_PLAN.md Amendment 1), provider-neutral (Thunder or RunPod): batched by
# loaded model, each item cut from the end of the priority order X1, X2, X3, F, C2, B, A (stages stop
# at 10.25 h unless the author moves the limits with set_limits.sh; watchdog ceiling 14.0 h). Writes
# papers/gpc_goal_pressure_conduct/outputs/gpc2_v1; the local read is gpc2_read.py.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
G="$REPO_DIR/papers/gpc_goal_pressure_conduct"
C="$REPO_DIR/papers/cst_constitutional_steering/outputs"
SESSION="${CST_SESSION:-}"
PRICE_H=10.5; WATCH_H=11.0
OUT="$G/outputs/gpc2_v1"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
source "$REPO_DIR/papers/runpod_common/session_limits.sh"
trap 'kill $RW 2>/dev/null; limits_save "$OUT/logs"; cp "$REPO_DIR/session.log" "$OUT/logs/session_gpc2${SESSION}_${STAMP}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=$WATCH_H;;
  *) echo "BAIL: CST_SESSION must be 1"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$C/_sync"
# the token: RunPod's launcher passes it from /root/.hf_token; elsewhere it is ~/.hf_token (mode 600)
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.hf_token" 2>/dev/null)}"
PYBIN="${CST_PYTHON:-python3}"
export CST_PRICE_H=$PRICE_H CST_START_EPOCH=$START
export CST_DEADLINE_EPOCH=$(python3 -c "print(int($START + ($PRICE_H - 0.25) * 3600))")
# editable limits (papers/runpod_common/session_limits.py): stages stop at 10.25 h, the watchdog at
# 11.0 h, both movable during the run with set_limits.sh (watchdog at most 14.0 h); each change logged
limits_init "$REPO_DIR/.session_limits.json" "$START" "$(python3 -c "print($PRICE_H - 0.25)")" "$WATCH_H" \
  || { echo "BAIL: the limits file"; exit 1; }
GP=papers/gpc_goal_pressure_conduct/scripts/gpc2_pod.py
TPOD=papers/cst_constitutional_steering/scripts/cst_teacher_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> GPC-2 session $SESSION start $(date -u +%FT%TZ) out=$OUT (stages stop at $(python3 -c "print($PRICE_H - 0.25)") h, watchdog ${MAX_H} h)"
[ "$($PYBIN -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = 3.11 ] || { echo "BAIL: $PYBIN is not Python 3.11"; exit 1; }
$PYBIN -c "import os, sysconfig; raise SystemExit(0 if os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')) else 1)" \
  || { echo "BAIL: no Python.h for $PYBIN (Triton compiles against it)"; exit 1; }

limits_watchdog 'gpc2_pod.py|cst_teacher_pod.py'  # sets WATCHDOG; re-reads the limits file

# ---- the vLLM venv (CST's pins, unchanged) plus the MACHIAVELLI extra ----
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
  "$VENV/bin/pip" install -q -e ".[machiavelli]" "transformers==5.17.0" -c "$CONSTRAINTS" \
    --extra-index-url "$CU129_INDEX" 2>&1 | tail -2
fi
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
$PY - <<'EOF' || { echo "BAIL: vLLM venv not usable"; exit 1; }
import importlib.metadata as md, shutil, torch, vllm, peft  # noqa: E401,F401
import gym, asteval  # noqa: F401,E401  (the MACHIAVELLI extra)
v = {p: md.version(p) for p in ("vllm", "torch", "transformers", "peft", "gym")}
ok = (v["vllm"].startswith("0.31.0") and v["transformers"] == "5.17.0" and v["peft"] == "0.21.2"
      and v["gym"] == "0.25.1" and torch.cuda.is_available()
      and (torch.version.cuda or "").startswith("12.9") and shutil.which("ninja") is not None)
print(">> stack:", v, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
EOF
$PY - <<'EOF' || { echo "BAIL: vLLM smoke"; exit 1; }
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", max_model_len=512, gpu_memory_utilization=0.3)
out = llm.generate(["Hello"], SamplingParams(temperature=0.2, top_p=1.0, seed=0, max_tokens=8))
print(">> vLLM smoke OK:", len(out[0].outputs[0].token_ids), "tokens")
EOF

# ---- CST's and ST's inputs, synced by name under GPC's outputs, put back at their paths; GPC-1's ----
I="$G/outputs/gpc2_in"
[ -d "$I/cst/outputs" ] && [ -d "$I/st/outputs" ] || { echo "BAIL: the CST or ST inputs did not sync ($I)"; exit 1; }
cp -r "$I/cst/outputs/." "$C/"
mkdir -p "$REPO_DIR/papers/st_live_reference/outputs" && cp -r "$I/st/outputs/." "$REPO_DIR/papers/st_live_reference/outputs/"
[ -f "$REPO_DIR/papers/st_live_reference/outputs/st2q/qwen38_27b/derive.npz" ] || { echo "BAIL: ST's fill draws did not sync"; exit 1; }
for f in conduct/slots.json conduct/kept_c1.jsonl conduct/teacher_c1.jsonl conduct/base_gap.jsonl conduct/holdout_base.jsonl \
         train_c1.json adapters/c1/612732/adapter_model.safetensors teacher/screen_c1.jsonl teacher/_raw/gpc_b.jsonl \
         teacher/_raw/screen_c1.jsonl; do
  [ -f "$G/outputs/gpc1_v1/$f" ] || { echo "BAIL: GPC-1's $f did not sync"; exit 1; }
done
for f in teacher_v1/teacher/base_good_behavior.jsonl finalist_c2_rule_right_mach/mach_states_base.jsonl \
         finalist_c2_rule_right_mach/mach/decisions.jsonl finalist_c2_rule_right_mach/mach/episodes.jsonl \
         _sync/eval_8gram_hashes_v1.npy _sync/C_anth.txt pilot_sdf/adapters/d_doc/1e+06/adapter_model.safetensors \
         pilot_sdf/parity/hf_d_doc_1M.json pilot_sdf/parity/hf_base.json; do
  [ -f "$C/$f" ] || { echo "BAIL: $f did not sync"; exit 1; }
done
for f in data/gpc_screen_sets_v1.json data/c_conduct_v1.json data/conduct_grid_v1.json data/neutral_lines_v1.json; do
  [ -f "$G/$f" ] || { echo "BAIL: $f did not sync"; exit 1; }
done

PART=mach $PY - <<'EOF' > "$OUT/logs/models_gpc2${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
got = {"qwen38_27b": snapshot_download("Qwen/Qwen3.8-27B",
                                       revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                                       ignore_patterns=["original/*", "metal/*"])}
got["gpt_oss_120b_tokenizer"] = snapshot_download(  # the CST gate tests' dry judge
    "openai/gpt-oss-120b", revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    allow_patterns=["tokenizer*", "special_tokens_map.json", "config.json",
                    "generation_config.json", "chat_template*"])
print(json.dumps(got))
EOF
echo ">> models: $(cat "$OUT/logs/models_gpc2${SESSION}.json")"
# the game bundle: Google Drive first; on a failure or a hash mismatch, the locally verified zip
# copied here by the launcher (ON_DEMAND_SYNC; MCH Amendment 8). Never re-hosted.
BUNDLE_PUSH=/workspace/mch_bundle/game_data.zip
if ! $PY -c 'from deepsteer.datasets.machiavelli import fetch_machiavelli; print(">> engine:", fetch_machiavelli())'; then
  echo ">> the Drive download failed or its hash mismatched; asking for the locally verified bundle"
  echo "NEED_SYNC mch_bundle"
  for _ in $(seq 1 120); do [ -f "$BUNDLE_PUSH" ] && break; sleep 10; done
  [ -f "$BUNDLE_PUSH" ] || { echo "BAIL: no bundle after 20 min (launch with ON_DEMAND_SYNC)"; exit 1; }
  $PY -c "from pathlib import Path; from deepsteer.datasets.machiavelli import fetch_machiavelli; print('>> engine:', fetch_machiavelli(zip_path=Path('$BUNDLE_PUSH')))" \
    || { echo "BAIL: the synced bundle failed its hash check"; exit 1; }
fi
echo ">> setup done at $(hours) h"

TESTS="tests/scripts/test_gpc2.py tests/scripts/test_gpc1.py tests/scripts/test_session_limits.py tests/scripts/test_cst_teacher.py tests/scripts/test_cst_parity.py tests/scripts/test_mch_thinking_pod.py"
# the slow dry chains are Z9's local check (run in the sync emulation); the pod runs the rest
$PY -m pytest -q -p no:cacheprovider -m "not slow" $TESTS > "$OUT/logs/gate_tests${SESSION}.log" 2>&1; GT=$?
grep -E '[0-9]+ (passed|failed)' "$OUT/logs/gate_tests${SESSION}.log" | tail -1
grep -E '^FAILED' "$OUT/logs/gate_tests${SESSION}.log" | head -5
[ "$GT" = 0 ] || { echo "BAIL: gate tests (see logs/gate_tests${SESSION}.log)"; exit 1; }

echo setup > "$OUT/.stage"
python3 papers/cst_constitutional_steering/scripts/rate_watch.py --log "$REPO_DIR/session.log" \
  --stage-file "$OUT/.stage" --out "$OUT/logs/rate_windows.jsonl" \
  --forecast '{"phase1": 1800, "phase2": 1800}' &
RW=$!

run() { echo "$2" > "$OUT/.stage"; $PY "$1" --out "$OUT" "${@:3}" || echo ">> stage failed: ${*:3}"; echo ">> stage done at $(hours) h: ${*:3}"; }
fits() { $PY $GP --out "$OUT" --stage fits --next "$1"; }
ok() { python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/$1')).get('$2') else 1)" 2>/dev/null; }
reading() { python3 -c "import json; print(json.load(open('$OUT/x_read.json')).get('reading'))" 2>/dev/null; }
stop() { echo ">> $1"; echo ">> done at $(hours) h"; kill $WATCHDOG 2>/dev/null; exit 0; }

# a. the training smoke and A's scaled copies (HF), the fixed smoke check, letters and served parity
run $GP smoke_train --stage smoke_train
ok smoke_train.json passes || stop "STOP: the training smoke failed; nothing is read"
run $TPOD smoke --stage smoke_serve
ok smoke_serve.json parity_passes || stop "STOP: the fixed smoke check failed; nothing is read"
run $GP parity_hf --stage parity_hf
run $GP parity --stage parity
# b. phase 1 (one LoRA engine): X1, X2 and the reading; X3 and C2's teachers if format gates; F; B1 and
#    the neutral lines' grid rollouts; A's screens; each item behind the cut rule by priority
run $GP phase1 --stage phase1
[ -f "$OUT/x_read.json" ] || stop "STOP: X1 and X2 did not read"
echo ">> reading: $(reading)"
# c. C2's adapters (HF), only if format gates and the arm's gate passed; their letters and parity
if [ "$(reading)" = "format gates" ]; then
  run $GP train --stage train --arm c2_div
  run $GP train --stage train --arm c2_mach
  run $GP parity_hf --stage parity_hf
  run $GP parity --stage parity
fi
# d. B0, B2, B3 (one HF process)
if fits b; then run $GP b_hf --stage b_hf; else echo ">> cut: B (the forecast passes the cut)"; fi
# e. phase 2 (one LoRA engine): C2's screens, the held-out cell, X1 for C2-mach, the registered read
if [ "$(reading)" = "format gates" ]; then run $GP phase2 --stage phase2; fi
# f. B4 (HF): the ladder, then the steered arms only if B3 passes at the coherent dose
if fits b4; then run $GP b4 --stage b4; else echo ">> cut: B4 (its forecast passes the cut)"; fi
$PY papers/gpc_goal_pressure_conduct/scripts/gpc2_read.py "$OUT" || echo ">> the read failed (rerun locally)"
stop "all stages run"
