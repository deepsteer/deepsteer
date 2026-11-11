#!/usr/bin/env bash
# Remote runner for the kind-split pod (CST Amendment 32 items 6-10): four SDF adapters from the
# existing corpora's subsets (stories / without stories, D-neutral and D-doc), no recovery; their
# screen reads on one session base; the content gate; then, only if it can finish by 6 h, D-neutral
# 1M's guard read and the adapters' plain cells. One A100-80GB, priced at 6.0 A100-h (watchdog
# 6.5 h). Orion launches with CST_SESSION=1 (launch_cst_sdf_kinds.sh). Writes outputs/pilot_sdf4.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
C="$REPO_DIR/papers/cst_constitutional_steering/outputs"
OUT="$C/pilot_sdf4"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SESSION="${CST_SESSION:-}"
trap 'cp "$REPO_DIR/session.log" "$OUT/logs/session_sdf_kinds${SESSION}_${STAMP}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=6.5;;
  *) echo "BAIL: CST_SESSION must be 1"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$C/_sync"
NOREC=papers/cst_constitutional_steering/scripts/cst_sdf_norec_pod.py
KINDS=papers/cst_constitutional_steering/scripts/cst_sdf_kinds_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST SDF kind-split pod session $SESSION start $(date -u +%FT%TZ) out=$OUT (watchdog ${MAX_H} h)"

( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f 'cst_(sdf_norec|sdf_kinds|pilot_sdf|s0a)_pod.py' ) &
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

# ---- synced inputs: D-doc 0.3M's adapter and record, D-neutral's corpus ----
[ -f "$C/pilot_sdf/adapters/d_doc/300000/adapter_model.safetensors" ] || { echo "BAIL: D-doc 0.3M did not sync"; exit 1; }
cp "$C/pilot_sdf/train_d_doc.json" "$OUT/" || { echo "BAIL: train_d_doc.json did not sync"; exit 1; }
# the saved D-neutral 1M, for Amendment 30 item 1's reproduction check against the retrained endpoint
[ -f "$C/pilot_sdf/adapters/d_neutral/1e+06/adapter_model.safetensors" ] || { echo "BAIL: D-neutral 1M did not sync"; exit 1; }
cp "$C/pilot_sdf/train_d_neutral.json" "$OUT/" || { echo "BAIL: train_d_neutral.json did not sync"; exit 1; }
mkdir -p "$OUT/corpus/d_neutral"
cp "$C"/pilot_sdf/corpus/d_neutral/chunk_*.jsonl "$OUT/corpus/d_neutral/" 2>/dev/null || { echo "BAIL: D-neutral's corpus did not sync"; exit 1; }
# ---- synced inputs: both corpora, D-neutral 1M and its record (the guard read), C_anth, the
# content gate's scenarios (committed), the pressure scenarios ----
for arm in d_doc d_neutral; do
  mkdir -p "$OUT/corpus/$arm"
  cp "$C"/pilot_sdf/corpus/$arm/chunk_*.jsonl "$OUT/corpus/$arm/" 2>/dev/null || { echo "BAIL: $arm's corpus did not sync"; exit 1; }
done
[ -f "$C/pilot_sdf/adapters/d_neutral/1e+06/adapter_model.safetensors" ] || { echo "BAIL: D-neutral 1M did not sync"; exit 1; }
cp "$C/pilot_sdf/train_d_neutral.json" "$OUT/" || { echo "BAIL: train_d_neutral.json did not sync"; exit 1; }
[ -f "$C/_sync/C_anth.txt" ] || { echo "BAIL: C_anth did not sync"; exit 1; }
[ -f "papers/cst_constitutional_steering/data/content_scenarios_v1.json" ] || { echo "BAIL: the content scenarios are missing"; exit 1; }
[ -f "$C/pressure/scenarios.json" ] || { echo "BAIL: the pressure scenarios did not sync"; exit 1; }

$PY - <<'EOF' > "$OUT/logs/models_sdf_kinds${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
print(json.dumps({"qwen38_27b": snapshot_download(
    "Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    ignore_patterns=["original/*", "metal/*"]),
    "gpt_oss_120b": snapshot_download(  # the guard cell's action read (D-neutral 1M, if under the price)
    "openai/gpt-oss-120b", revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    ignore_patterns=["original/*", "metal/*"])}))
EOF
echo ">> models: $(cat "$OUT/logs/models_sdf_kinds${SESSION}.json")"
echo ">> setup done at $(hours) h"


TESTS="tests/scripts/test_cst_sdf_kinds.py tests/scripts/test_cst_sdf_norec.py tests/scripts/test_cst_pilot_sdf.py tests/scripts/test_cst_parity.py tests/scripts/test_cst_pilot_train.py"
$PY -m pytest -q -p no:cacheprovider $TESTS 2>&1 | grep -E '[0-9]+ (passed|failed)' | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }

nrun() { $PY $NOREC --out "$OUT" "$@" || echo ">> stage failed: $*"; echo ">> stage done at $(hours) h: $*"; }
krun() { $PY $KINDS --out "$OUT" "$@" || echo ">> stage failed: $*"; echo ">> stage done at $(hours) h: $*"; }
# a stage starts only if it can finish by the 6.0 h price: fits <hours the stage takes>
fits() { python3 -c "import sys,time; sys.exit(0 if (time.time()-$START)/3600 + $1 <= 6.0 else 1)"; }

# the smoke pass (Amendments 28 item 3, 31): a failure stops the pod in its first minutes
nrun --stage smoke_hf --smoke-arm d_neutral
[ -f "$OUT/smoke_hf.json" ] || { echo "BAIL: smoke_hf failed (see the log)"; exit 1; }
nrun --stage smoke_serve
python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/smoke_serve.json'))['parity_passes'] else 1)" 2>/dev/null \
  || { echo "BAIL: smoke_serve failed (see smoke_serve.json and the log)"; exit 1; }
echo ">> smoke pass done at $(hours) h"

# items 6-7: D-neutral's pair trained and read, then D-doc's (a stop leaves complete pairs)
krun --stage train_kind --kind dn_stories
krun --stage train_kind --kind dn_nostories
krun --stage parity_hf --tags dn_stories,dn_nostories
krun --stage read_base
krun --stage read_kinds --tags dn_stories,dn_nostories
krun --stage train_kind --kind dd_stories
krun --stage train_kind --kind dd_nostories
krun --stage parity_hf --tags dd_stories,dd_nostories
krun --stage read_kinds --tags dd_stories,dd_nostories
echo ">> kind-split reads done at $(hours) h"

# item 9: the content gate, A0 and A0s at k 4
if fits 0.5; then krun --stage content --arm a0; krun --stage content --arm a0s
else echo ">> content gate skipped at $(hours) h: it would not finish by 6.0 h"; fi
# item 10: D-neutral 1M's guard read on this pod's base (Amendment 32 item 2), then plain cells
if fits 0.6; then
  nrun --stage parity_hf --tags d_neutral_1M
  nrun --stage read --tags d_neutral_1M --parity-only
  nrun --stage guard --tags base,d_neutral_1M
  nrun --stage guard_act
else echo ">> D-neutral 1M guard read skipped at $(hours) h"; fi
if fits 0.7; then krun --stage read_plain --tags dn_stories,dn_nostories,dd_stories,dd_nostories
else echo ">> the adapters' plain cells skipped at $(hours) h"; fi
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
