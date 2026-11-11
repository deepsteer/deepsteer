#!/usr/bin/env bash
# Remote runner for tomorrow's SDF pod, the content branch of Amendment 29 item 6 (the author's trimmed
# ruling: no recovery anywhere; single model). One A100-80GB, priced at about 7.2 A100-h with a 1 h corpus top-up, at most 10.3 (watchdog 10.5 h).
# Orion launches with CST_SESSION=1 (launch_cst_sdf_content.sh) the night item 6's control lands.
# Writes under outputs/pilot_sdf3 (its own folder).
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
C="$REPO_DIR/papers/cst_constitutional_steering/outputs"
OUT="$C/pilot_sdf3"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SESSION="${CST_SESSION:-}"
PRICE_H=9.5
trap 'cp "$REPO_DIR/session.log" "$OUT/logs/session_sdf_content${SESSION}_${STAMP}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=10.5;;
  *) echo "BAIL: CST_SESSION must be 1"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$C/_sync"
NOREC=papers/cst_constitutional_steering/scripts/cst_sdf_norec_pod.py
PILOT=papers/cst_constitutional_steering/scripts/cst_pilot_sdf_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST SDF content pod session $SESSION start $(date -u +%FT%TZ) out=$OUT (watchdog ${MAX_H} h)"

( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f 'cst_(sdf_norec|pilot_sdf|s0a)_pod.py' ) &
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

# ---- synced inputs: D-doc's corpus (the re-read pod's 1M and pod 2's continuation), merged ----
mkdir -p "$OUT/corpus/d_doc"
cp "$C"/pilot_sdf/corpus/d_doc/chunk_*.jsonl "$OUT/corpus/d_doc/" 2>/dev/null || { echo "BAIL: D-doc's 1M corpus did not sync"; exit 1; }
cp "$C"/pilot_sdf2/corpus/d_doc/chunk_*.jsonl "$OUT/corpus/d_doc/" 2>/dev/null || echo ">> no continuation chunks from pod 2"
for f in C_anth.txt eval_8gram_hashes_v1.npy; do
  [ -f "$CST_CONST_DIR/$f" ] || { echo "BAIL: $f did not sync"; exit 1; }
done
[ -f "$C/pressure/scenarios.json" ] || { echo "BAIL: the pressure scenarios did not sync"; exit 1; }
echo ">> D-doc corpus chunks: $(ls "$OUT"/corpus/d_doc/chunk_*.jsonl | wc -l)"

$PY - <<'EOF' > "$OUT/logs/models_sdf_content${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
print(json.dumps({"qwen38_27b": snapshot_download(
    "Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    ignore_patterns=["original/*", "metal/*"]),
    "gpt_oss_120b": snapshot_download(  # the held-out and guard cells' action read
    "openai/gpt-oss-120b", revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    ignore_patterns=["original/*", "metal/*"])}))
EOF
echo ">> models: $(cat "$OUT/logs/models_sdf_content${SESSION}.json")"
echo ">> setup done at $(hours) h"

TESTS="tests/scripts/test_cst_sdf_norec.py tests/scripts/test_cst_pilot_sdf.py tests/scripts/test_cst_parity.py tests/scripts/test_cst_pilot_train.py tests/scripts/test_cst_pressure_pod.py"
$PY -m pytest -q -p no:cacheprovider $TESTS 2>&1 | grep -E '[0-9]+ (passed|failed)' | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }

run() { $PY $NOREC --out "$OUT" "$@" || echo ">> stage failed: $*"; echo ">> stage done at $(hours) h: $*"; }
under_price() { python3 -c "import sys,time; sys.exit(0 if (time.time()-$START)/3600 < $PRICE_H else 1)"; }

# the smoke pass (Amendments 28 item 3, 31): a failure stops the pod in its first minutes
run --stage smoke_hf --smoke-arm d_doc
[ -f "$OUT/smoke_hf.json" ] || { echo "BAIL: smoke_hf failed (see the log)"; exit 1; }
run --stage smoke_serve
python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/smoke_serve.json'))['parity_passes'] else 1)" 2>/dev/null \
  || { echo "BAIL: smoke_serve failed (see smoke_serve.json and the log)"; exit 1; }
echo ">> smoke pass done at $(hours) h"

# the corpus top-up to 3M under the extended plan and the same gate, stopped by 4.9 h so the
# rest fits under 10.3 h; a short corpus trains to its kept tokens and the record says so
REM=$(python3 -c "import time; print(max(0, int(4.9 * 3600 - (time.time() - $START))))")
if [ "$REM" -gt 300 ]; then
  timeout --signal=TERM "$REM" $PY $PILOT --out "$OUT" --stage corpus --arm d_doc \
    --plan data/pilot_corpus_plan_v2.json --target 3000000 \
    || echo ">> corpus top-up stopped at $(hours) h (the banked chunks are kept)"
  echo ">> stage done at $(hours) h: corpus top-up"
fi
run --stage train_dose --rung d_doc_3M
run --stage parity_hf --tags d_doc_3M,d_doc_3M_at1M
run --stage read --tags d_doc_3M,d_doc_3M_at1M
run --stage capability --tags base,d_doc_3M,d_doc_3M_at1M
run --stage heldout --tags base,d_doc_3M,d_doc_3M_at1M
run --stage heldout_act
run --stage guard --tags base,d_doc_3M,d_doc_3M_at1M
run --stage guard_act
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
