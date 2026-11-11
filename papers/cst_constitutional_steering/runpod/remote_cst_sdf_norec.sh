#!/usr/bin/env bash
# Remote runner for SDF pod 2 (Amendment 29 items 5-7, 10): reads of the saved SDF adapters without
# recovery, on one same-session base. One A100-80GB, priced at about 7.2 A100-h of reads and training, then the D-doc corpus continuation to 10.3 h (watchdog 10.5 h).
# Orion launches with CST_SESSION=1 (launch_cst_sdf_norec.sh). Order: the smoke pass; the parity HF
# side; D-neutral 1M, D-doc 1M at k 4, D-chat 1M, D-doc 0.3M, D-raw; capability at n 200 on D-doc 1M
# and the base; the held-out free-form cell on the base, D-doc 1M and D-neutral 1M; the injected-
# order guard cell on the base and D-doc 1M; then, under the price, a second D-doc 1M draw and its
# read; last, the D-doc corpus continuation until 10.3 h. Writes under outputs/pilot_sdf2.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
IN="$REPO_DIR/papers/cst_constitutional_steering/outputs/pilot_sdf"  # the synced adapters and records
OUT="$REPO_DIR/papers/cst_constitutional_steering/outputs/pilot_sdf2"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SESSION="${CST_SESSION:-}"
PRICE_H=8.5
trap 'cp "$REPO_DIR/session.log" "$OUT/logs/session_sdf_norec${SESSION}_${STAMP}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=10.5;;
  *) echo "BAIL: CST_SESSION must be 1"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$REPO_DIR/papers/cst_constitutional_steering/outputs/_sync"
NOREC=papers/cst_constitutional_steering/scripts/cst_sdf_norec_pod.py
PILOT=papers/cst_constitutional_steering/scripts/cst_pilot_sdf_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST SDF no-recovery pod session $SESSION start $(date -u +%FT%TZ) out=$OUT (watchdog ${MAX_H} h)"

( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f 'cst_(sdf_norec|s0a)_pod.py' ) &
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
for a in d_neutral/1e+06 d_doc/1e+06 d_doc/300000 d_chat/1e+06 d_raw/102400; do
  [ -f "$IN/adapters/$a/adapter_model.safetensors" ] || { echo "BAIL: adapter $a did not sync"; exit 1; }
done
for f in train_d_doc.json train_d_neutral.json train_d_chat.json train_d_raw.json; do
  [ -f "$IN/$f" ] || { echo "BAIL: $f did not sync"; exit 1; }
  cp "$IN/$f" "$OUT/"
done
ls "$IN"/corpus/d_doc/chunk_*.jsonl >/dev/null 2>&1 || { echo "BAIL: D-doc's corpus did not sync"; exit 1; }
mkdir -p "$OUT/corpus" && cp -r "$IN/corpus/d_doc" "$OUT/corpus/"
[ -f "$REPO_DIR/papers/cst_constitutional_steering/outputs/pressure/scenarios.json" ] \
  || { echo "BAIL: the pressure scenarios did not sync"; exit 1; }
for f in C_anth.txt eval_8gram_hashes_v1.npy; do  # the corpus continuation's document and its gate
  [ -f "$CST_CONST_DIR/$f" ] || { echo "BAIL: $f did not sync"; exit 1; }
done

$PY - <<'EOF' > "$OUT/logs/models_sdf_norec${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
print(json.dumps({"qwen38_27b": snapshot_download(
    "Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    ignore_patterns=["original/*", "metal/*"]),
    "gpt_oss_120b": snapshot_download(  # the held-out and guard cells' action read
    "openai/gpt-oss-120b", revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    ignore_patterns=["original/*", "metal/*"])}))
EOF
echo ">> models: $(cat "$OUT/logs/models_sdf_norec${SESSION}.json")"
echo ">> setup done at $(hours) h"

TESTS="tests/scripts/test_cst_sdf_norec.py tests/scripts/test_cst_pilot_sdf.py tests/scripts/test_cst_parity.py tests/scripts/test_cst_pilot_train.py tests/scripts/test_cst_pressure_pod.py"
$PY -m pytest -q -p no:cacheprovider $TESTS 2>&1 | grep -E '[0-9]+ (passed|failed)' | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }

run() { $PY $NOREC --out "$OUT" "$@" || echo ">> stage failed: $*"; echo ">> stage done at $(hours) h: $*"; }
under_price() { python3 -c "import sys,time; sys.exit(0 if (time.time()-$START)/3600 < $PRICE_H else 1)"; }

# the smoke pass (Amendment 28 item 3): a failure stops the pod in its first minutes
run --stage smoke_hf
[ -f "$OUT/smoke_hf.json" ] || { echo "BAIL: smoke_hf failed (see the log)"; exit 1; }
run --stage smoke_serve
python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/smoke_serve.json'))['parity_passes'] else 1)" 2>/dev/null \
  || { echo "BAIL: smoke_serve failed (see smoke_serve.json and the log)"; exit 1; }
echo ">> smoke pass done at $(hours) h"

run --stage parity_hf --tags d_neutral_1M,d_doc_1M,d_doc_1M_k4,d_chat_1M,d_doc_03M,d_raw
run --stage read --tags d_neutral_1M
run --stage read --tags d_doc_1M_k4 --k4
run --stage read --tags d_chat_1M,d_doc_03M,d_raw
run --stage read --tags d_doc_1M --parity-only
run --stage capability --tags base,d_doc_1M
run --stage heldout --tags base,d_doc_1M,d_neutral_1M
run --stage heldout_act
run --stage guard --tags base,d_doc_1M
run --stage guard_act
if under_price; then
  run --stage train_draw2
  run --stage parity_hf --tags d_doc_1M_draw2
  run --stage read --tags d_doc_1M_draw2
else echo ">> over price: the second D-doc draw skipped"; fi
# the last stage (the author, 2026-11-11): the next D-doc corpus tokens under the plan of record
# extended to 2,600 documents (v2, its first 900 v1's) and the same gate, in resumable chunks,
# stopped at 10.3 h whatever is banked; nothing after it depends on it
REM=$(python3 -c "import time; print(max(0, int(10.3 * 3600 - (time.time() - $START))))")
if [ "$REM" -gt 900 ]; then
  timeout --signal=TERM "$REM" $PY $PILOT --out "$OUT" --stage corpus --arm d_doc \
    --plan data/pilot_corpus_plan_v2.json --target 3000000 \
    || echo ">> corpus continuation stopped at $(hours) h (the banked chunks are kept)"
  echo ">> stage done at $(hours) h: corpus continuation"
else echo ">> no time left for the corpus continuation"; fi
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
