#!/usr/bin/env bash
# Remote runner for the CST pressure line's session 2-dpo (P-dpo, P-long, the legitimate-order reads) (Amendments 25 C4, 27 item 4, 28
# item 3; PRESSURE_BUILD.md). One A100-80GB, priced at about 4.8 A100-h (P-long 2.34 h of training; P-dpo not trained, its gate failed at 41 of 60) (watchdog 10.5 h).
# Orion launches with CST_SESSION=1 (launch_cst_pressure_dpo.sh) after 2-roll's line gate passed.
# First the smoke pass (about 10 min; any failure stops the pod), then training, the parity HF side,
# and the reads with a same-session base. Writes under outputs/pressure; inputs from 2-roll synced.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
OUT="$REPO_DIR/papers/cst_constitutional_steering/outputs/pressure"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SESSION="${CST_SESSION:-}"
PRICE_H=9.5
trap 'cp "$REPO_DIR/session.log" "$OUT/logs/session_pressure_dpo${SESSION}_${STAMP}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=10.5;;
  *) echo "BAIL: CST_SESSION must be 1"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$REPO_DIR/papers/cst_constitutional_steering/outputs/_sync"
TRAIN=papers/cst_constitutional_steering/scripts/cst_pressure_train_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST pressure dpo session $SESSION start $(date -u +%FT%TZ) out=$OUT (watchdog ${MAX_H} h)"

( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f 'cst_(pressure_train|s0a)_pod.py' ) &
WATCHDOG=$!

# ---- the vLLM venv (VALIDATE's block, unchanged pins) ----
VENV=/workspace/venv_vllm
VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_SHA256=5176ab746f4e736c4a354a11d0620abf0a941de287cad937ac34039293ca6f5f
CU129_INDEX=https://download.pytorch.org/whl/cu129
CONSTRAINTS=papers/st_live_reference/runpod/vllm_constraints.txt
if [ ! -x "$VENV/bin/python" ]; then
  echo ">> creating $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip 2>&1 | tail -1
  "$VENV/bin/pip" install -q "torch==2.13.0+cu129" "torchvision==0.28.0+cu129" \
    "torchaudio==2.11.0+cu129" --index-url "$CU129_INDEX" 2>&1 | tail -2
  "$VENV/bin/pip" install -q "vllm @ ${VLLM_WHEEL_URL}#sha256=${VLLM_WHEEL_SHA256}" \
    "transformers==5.17.0" "peft==0.21.2" pyyaml ninja pytest tokenizers -c "$CONSTRAINTS" \
    --extra-index-url "$CU129_INDEX" 2>&1 | tail -2
  "$VENV/bin/pip" install -q -e "." "transformers==5.17.0" -c "$CONSTRAINTS" \
    --extra-index-url "$CU129_INDEX" 2>&1 | tail -2
fi
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
$PY - <<'EOF' || { echo "BAIL: vLLM venv not usable"; exit 1; }
import importlib.metadata as md, shutil, torch, vllm, peft  # noqa: E401,F401
v = {p: md.version(p) for p in ("vllm", "torch", "transformers", "peft")}
ok = (v["vllm"].startswith("0.31.0") and v["transformers"] == "5.17.0" and v["peft"] == "0.21.2"
      and torch.cuda.is_available() and (torch.version.cuda or "").startswith("12.9")
      and shutil.which("ninja") is not None)
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

# ---- synced inputs (gitignored; SYNC_OUTPUTS, named files) ----
for f in C_anth.txt a0n_faraday_v1.txt; do
  [ -f "$CST_CONST_DIR/$f" ] || { echo "BAIL: $f did not sync"; exit 1; }
done
for f in scenarios.json act.jsonl act.json contrast.json; do
  [ -f "$OUT/$f" ] || { echo "BAIL: 2-roll's $f did not sync"; exit 1; }
done
for a in a0 a0s a0n; do
  ls "$OUT"/roll/$a/chunk_*.jsonl >/dev/null 2>&1 || { echo "BAIL: 2-roll's $a rollouts did not sync"; exit 1; }
done

$PY - <<'EOF' > "$OUT/logs/models_pressure_dpo${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
print(json.dumps({"qwen38_27b": snapshot_download(
    "Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    ignore_patterns=["original/*", "metal/*"])}))
EOF
echo ">> models: $(cat "$OUT/logs/models_pressure_dpo${SESSION}.json")"
echo ">> setup done at $(hours) h"

TESTS="tests/scripts/test_cst_pressure_train.py tests/scripts/test_cst_pressure_pod.py tests/scripts/test_cst_parity.py tests/scripts/test_cst_pilot_train.py"
$PY -m pytest -q -p no:cacheprovider $TESTS 2>&1 | grep -E '[0-9]+ (passed|failed)' | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }

run() { $PY $TRAIN --out "$OUT" "$@" || echo ">> stage failed: $*"; echo ">> stage done at $(hours) h: $*"; }
under_price() { python3 -c "import sys,time; sys.exit(0 if (time.time()-$START)/3600 < $PRICE_H else 1)"; }

# the smoke pass (Amendment 28 item 3): a failure stops the pod in its first minutes
run --stage smoke_hf --rungs p_dpo,p_long
[ -f "$OUT/smoke_hf.json" ] || { echo "BAIL: smoke_hf failed (see the log)"; exit 1; }
run --stage smoke_serve
python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/smoke_serve.json'))['parity_passes'] else 1)" 2>/dev/null \
  || { echo "BAIL: smoke_serve failed (see smoke_serve.json and the log)"; exit 1; }
echo ">> smoke pass done at $(hours) h"

run --stage train --rung p_dpo
run --stage train --rung p_long
run --stage parity_hf --tags p_dpo,p_long
run --stage read --tags p_dpo,p_long
# C7 last (it gates nothing): A0s and A0n on the legitimate-order cell, one engine per process
if under_price; then run --stage legit --arm a0s; else echo ">> over price: A0s legit read skipped"; fi
if under_price; then run --stage legit --arm a0n; else echo ">> over price: A0n legit read skipped"; fi
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
