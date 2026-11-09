#!/usr/bin/env bash
# Remote runner for the CST pilot SDF pod (PILOT_BUILD.md §5 card 1; Amendments 20-22), one
# A100-80GB, priced at about 10.3 A100-h. Orion launches with CST_SESSION=1 (launch_cst_pilot_sdf.sh).
#
# Stage order (the most decision-relevant first): D-doc corpus and its pilot gate; the recovery
# set; D-doc training (0.3M, 1M, each with and without recovery) and R0; reads of D-doc 1M with
# and without recovery and R0; D-neutral corpus, training, read; D-chat render, training, read;
# A0 on the legitimate-order cell (20-item check first); then, only under the price, the D-doc 0.3M
# reads and D-raw (Amendment 21 item 1's bail order). Everything writes under outputs/pilot_sdf; a
# rerun with the same CST_SESSION skips what is on disk. A failed stage does not stop the next,
# except the corpus gate. The .session_done trap is armed before any check that can bail.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
OUT="$REPO_DIR/papers/cst_constitutional_steering/outputs/pilot_sdf"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SESSION="${CST_SESSION:-}"
PRICE_H=10.3
trap 'cp "$REPO_DIR/session.log" "$OUT/logs/session_pilot_sdf${SESSION}_${STAMP}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=11.5;;
  *) echo "BAIL: CST_SESSION must be 1"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$REPO_DIR/papers/cst_constitutional_steering/outputs/_sync"
POD=papers/cst_constitutional_steering/scripts/cst_s0a_pod.py
PILOT=papers/cst_constitutional_steering/scripts/cst_pilot_sdf_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST pilot SDF pod session $SESSION start $(date -u +%FT%TZ) out=$OUT (watchdog ${MAX_H} h)"

( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f 'cst_(s0a|pilot_sdf)_pod.py' ) &
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

# ---- synced inputs (gitignored; SYNC_OUTPUTS) ----
for f in C_anth.txt a0n_faraday_v1.txt eval_8gram_hashes_v1.npy; do
  [ -f "$CST_CONST_DIR/$f" ] || { echo "BAIL: $f did not sync"; exit 1; }
done

$PY - <<'EOF' > "$OUT/logs/models_pilot_sdf${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
print(json.dumps({"qwen38_27b": snapshot_download(
    "Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    ignore_patterns=["original/*", "metal/*"])}))
EOF
echo ">> models: $(cat "$OUT/logs/models_pilot_sdf${SESSION}.json")"
echo ">> setup done at $(hours) h"

TESTS="tests/scripts/test_cst_pilot_sdf.py tests/scripts/test_cst_pilot_train.py tests/scripts/test_cst_pressure.py"
$PY -m pytest -q -p no:cacheprovider $TESTS 2>&1 | grep -E '[0-9]+ (passed|failed)' | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }

run() { $PY $PILOT --out "$OUT" "$@" || echo ">> stage failed: $*"; echo ">> stage done at $(hours) h: $*"; }
under_price() { python3 -c "import sys,time; sys.exit(0 if (time.time()-$START)/3600 < $PRICE_H else 1)"; }
gate_ok() { python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/corpus_gate_$1.json'))['proceed'] else 1)" 2>/dev/null; }

run --stage corpus --arm d_doc
gate_ok d_doc || { echo "BAIL: D-doc corpus gate (documents banked; see corpus_gate_d_doc.json)"; exit 1; }
run --stage recovery
run --stage train --rung d_doc
run --stage train --rung r0
run --stage read --tags d_doc_1M_rec,d_doc_1M,r0
run --stage corpus --arm d_neutral
if gate_ok d_neutral; then
  run --stage train --rung d_neutral
  run --stage read --tags d_neutral_1M_rec
else echo ">> D-neutral corpus gate failed: D-neutral not trained"; fi
run --stage chat
run --stage train --rung d_chat
run --stage read --tags d_chat_1M_rec
run --stage legit
# bail order (Amendment 21 item 1): past the price, the 0.3M without-recovery read goes first, then D-raw
if under_price; then run --stage read --tags d_doc_03M_rec; else echo ">> over price: D-doc 0.3M reads skipped"; fi
if under_price; then run --stage read --tags d_doc_03M; else echo ">> over price: D-doc 0.3M without recovery skipped"; fi
if under_price; then run --stage train --rung d_raw && run --stage read --tags d_raw_rec; else echo ">> over price: D-raw skipped"; fi

$PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
count = lambda g: sum(sum(1 for _ in open(f)) for f in o.glob(g))  # noqa: E731
print(">> on disk:", json.dumps({
    "corpus": {a: count(f"corpus/{a}/*.jsonl") for a in ("d_doc", "d_neutral", "d_chat")},
    "adapters": sorted(str(p.relative_to(o)) for p in o.glob("adapters/*/*") if p.is_dir()),
    "reads": sorted(p.name for p in o.glob("gen/*") if p.is_dir())}))
EOF
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
