#!/usr/bin/env bash
# Remote runner for CST Amendment 52's pod, provider-neutral (RunPod or the ported provider): the
# validation stages (3a; a failed validation stops the pod before any registered read), A0n with
# GOOD_BEHAVIOR on the nested 888 (3b), P-ctx's single-decision read (3c, Amendment 49 item 2d) and
# the strength sweep (3d, descriptive), in that order, cut from the end past 10.5 h (stages stop at
# 10.25 h, watchdog 11.0 h). Every stage's rate in 5-minute windows (item 4; rate_watch.py). Writes
# outputs/a52_v1; the read is a52_read.py.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
C="$REPO_DIR/papers/cst_constitutional_steering/outputs"
SESSION="${CST_SESSION:-}"
PRICE_H=10.5; WATCH_H=11.0
OUT="$C/a52_v1"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
trap 'kill $RW 2>/dev/null; cp "$REPO_DIR/session.log" "$OUT/logs/session_a52${SESSION}_${STAMP}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=$WATCH_H;;
  *) echo "BAIL: CST_SESSION must be 1"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$C/_sync"
# the token: RunPod's launcher passes it from /root/.hf_token; elsewhere it is ~/.hf_token (mode 600)
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.hf_token" 2>/dev/null)}"
# the interpreter the venv is built from (the ported provider's bring-up installs python3.11)
PYBIN="${CST_PYTHON:-python3}"
export CST_PRICE_H=$PRICE_H CST_START_EPOCH=$START
export CST_DEADLINE_EPOCH=$(python3 -c "print(int($START + ($PRICE_H - 0.25) * 3600))")
POD=papers/cst_constitutional_steering/scripts/cst_a52_pod.py
TPOD=papers/cst_constitutional_steering/scripts/cst_teacher_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST Amendment 52 session $SESSION start $(date -u +%FT%TZ) out=$OUT (stages stop at $(python3 -c "print($PRICE_H - 0.25)") h, watchdog ${MAX_H} h)"
[ "$($PYBIN -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = 3.11 ] || { echo "BAIL: $PYBIN is not Python 3.11"; exit 1; }
$PYBIN -c "import os, sysconfig; raise SystemExit(0 if os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')) else 1)" \
  || { echo "BAIL: no Python.h for $PYBIN (Triton compiles against it)"; exit 1; }

( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f 'cst_a52_pod.py|cst_teacher_pod.py' ) &
WATCHDOG=$!

# ---- the vLLM venv (VALIDATE's block, unchanged pins) plus the MACHIAVELLI extra ----
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

# ---- synced inputs: C_anth and A0n's text, P-ctx's record and adapter, the finalist run's decisions,
# the smoke pass's fixed adapter and its letters ----
I="$C/a52_in"
[ -f "$CST_CONST_DIR/C_anth.txt" ] || { echo "BAIL: C_anth did not sync"; exit 1; }
[ -f "$CST_CONST_DIR/a0n_faraday_v1.txt" ] || { echo "BAIL: A0n's text did not sync"; exit 1; }
cp "$I/train_p_ctx.json" "$OUT/" || { echo "BAIL: P-ctx's record did not sync"; exit 1; }
d=$(python3 -c "import json; print(list(json.load(open('$OUT/train_p_ctx.json'))['adapters'].values())[-1])")
[ -f "$d/adapter_model.safetensors" ] || { echo "BAIL: P-ctx's adapter did not sync ($d)"; exit 1; }
for f in decisions.jsonl episodes.jsonl; do
  [ -f "$C/finalist_c2_rule_right_mach/mach/$f" ] || { echo "BAIL: the finalist run's $f did not sync"; exit 1; }
done
for f in data/mach_teacher_states_v1.json outputs/pilot_sdf/adapters/d_doc/1e+06/adapter_model.safetensors outputs/pilot_sdf/parity/hf_d_doc_1M.json outputs/pilot_sdf/parity/hf_base.json; do
  [ -f "papers/cst_constitutional_steering/$f" ] || { echo "BAIL: $f did not sync"; exit 1; }
done

PART=mach $PY - <<'EOF' > "$OUT/logs/models_a52${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
import json, os
from huggingface_hub import snapshot_download
pins = {"qwen38_27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")}
if os.environ["PART"] == "bars":  # MoReBench's judge
    pins["gpt_oss_120b"] = ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a")
got = {k: snapshot_download(r, revision=v, ignore_patterns=["original/*", "metal/*"])
       for k, (r, v) in pins.items()}
if os.environ["PART"] == "mach":  # the gate tests' dry judge reads gpt-oss's tokenizer only
    got["gpt_oss_120b_tokenizer"] = snapshot_download(
        "openai/gpt-oss-120b", revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
        allow_patterns=["tokenizer*", "special_tokens_map.json", "config.json",
                        "generation_config.json", "chat_template*"])
print(json.dumps(got))
EOF
echo ">> models: $(cat "$OUT/logs/models_a52${SESSION}.json")"
$PY -c 'from deepsteer.datasets.morebench import fetch_morebench; from deepsteer.datasets.moralbench import fetch_moralbench; print(">> data:", fetch_morebench(), fetch_moralbench())' \
  || { echo "BAIL: MoReBench or MoralBench download"; exit 1; }
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

TESTS="tests/scripts/test_cst_a52.py tests/scripts/test_cst_teacher.py tests/scripts/test_cst_finalist.py tests/scripts/test_cst_parity.py tests/scripts/test_mch_thinking_pod.py"
$PY -m pytest -q -p no:cacheprovider $TESTS > "$OUT/logs/gate_tests${SESSION}.log" 2>&1; GT=$?
grep -E '[0-9]+ (passed|failed)' "$OUT/logs/gate_tests${SESSION}.log" | tail -1
grep -E '^FAILED' "$OUT/logs/gate_tests${SESSION}.log" | head -5
[ "$GT" = 0 ] || { echo "BAIL: gate tests (see logs/gate_tests${SESSION}.log)"; exit 1; }

# item 4: every stage's rate in 5-minute windows (forecast tokens/s: RunPod's measured rates times
# the provider probe's relative throughput, the adapter at 0.8 of the plain engine)
echo setup > "$OUT/.stage"
python3 papers/cst_constitutional_steering/scripts/rate_watch.py --log "$REPO_DIR/session.log" \
  --stage-file "$OUT/.stage" --out "$OUT/logs/rate_windows.jsonl" \
  --forecast '{"validate": 475, "a52_a0ngood": 280, "a49_pctx": 475, "sweep": 475}' &
RW=$!

run() { echo "$2" > "$OUT/.stage"; $PY "$1" --out "$OUT" "${@:3}" || echo ">> stage failed: ${*:3}"; echo ">> stage done at $(hours) h: ${*:3}"; }
fits() { $PY $POD --out "$OUT" --stage fits --next "$1"; }
stop() { echo ">> $1"; echo ">> done at $(hours) h"; kill $WATCHDOG 2>/dev/null; exit 0; }

# a. validation (Amendment 51 item D5): the fixed smoke check, P-ctx's parity, the reproduction
run $TPOD smoke --stage smoke_serve
python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/smoke_serve.json'))['parity_passes'] else 1)" 2>/dev/null \
  || stop "STOP: validation failed (the fixed smoke check); no registered read runs"
run $TPOD parity_hf --stage parity_hf
run $POD validate --stage validate
python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/validation.json'))['passes'] is True else 1)" 2>/dev/null \
  || stop "STOP: validation failed (the P-ctx reproduction); no registered read runs"
# b. A0n with GOOD_BEHAVIOR (item 3b)
fits a52_a0ngood && run $TPOD a52_a0ngood --stage a52_a0ngood || stop "cut: a52_a0ngood and after (the forecast passes 10.25 h)"
# c. P-ctx's single-decision read (item 3c, Amendment 49 item 2d)
fits a49_pctx && run $TPOD a49_pctx --stage a49_pctx || stop "cut: a49_pctx and after (the forecast passes 10.25 h)"
# d. the strength sweep (item 3d, descriptive)
fits sweep || stop "cut: the sweep (the forecast passes 10.25 h)"
run $POD sweep --stage sweep_prep
run $POD sweep --stage sweep
stop "all stages run"
