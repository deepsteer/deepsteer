#!/usr/bin/env bash
# Remote runner for the single-decision teacher check (CST Amendments 46 item C, 47 item 2), with
# P-ctx's riders (Amendment 47 item 1). One A100-80GB, about 5.95 A100-h for the check (ceiling 6) and
# about 0.5 for the riders (ceiling 1); stages stop at 6.75 h, watchdog 7.5 h. Orion launches with
# CST_SESSION=1 (launch_cst_teacher.sh). The finalist MACHIAVELLI run's saved decisions in by name, the
# game bundle (Drive, else ON_DEMAND_SYNC), C_anth; P-ctx's record and adapter and the smoke pass's
# fixed adapter for the riders. Writes outputs/teacher_v1.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
C="$REPO_DIR/papers/cst_constitutional_steering/outputs"
SESSION="${CST_SESSION:-}"
PRICE_H=7.0; WATCH_H=7.5
OUT="$C/teacher_v1"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
trap 'cp "$REPO_DIR/session.log" "$OUT/logs/session_teacher${SESSION}_${STAMP}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=$WATCH_H;;  # mach: Amendment 43 item 2
  *) echo "BAIL: CST_SESSION must be 1"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$C/_sync"
# the stages stop 0.25 h before the price (MoReBench's judge and MACHIAVELLI read it); the rest is
# the download
export CST_PRICE_H=$PRICE_H
export CST_DEADLINE_EPOCH=$(python3 -c "print(int($START + ($PRICE_H - 0.25) * 3600))")
POD=papers/cst_constitutional_steering/scripts/cst_teacher_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST teacher check session $SESSION start $(date -u +%FT%TZ) out=$OUT (stages stop at $(python3 -c "print($PRICE_H - 0.25)") h, watchdog ${MAX_H} h)"

( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f 'cst_teacher_pod.py' ) &
WATCHDOG=$!

# ---- the vLLM venv (VALIDATE's block, unchanged pins) plus the MACHIAVELLI extra ----
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

# ---- synced inputs: the candidate's record and adapter, C_anth, the smoke pass's fixed adapter
# and its letters (Amendment 31) ----
[ -f "$CST_CONST_DIR/C_anth.txt" ] || { echo "BAIL: C_anth did not sync"; exit 1; }
cp "$C/teacher_in/train_p_ctx.json" "$OUT/" || { echo "BAIL: P-ctx's record did not sync"; exit 1; }
d=$(python3 -c "import json; print(list(json.load(open('$OUT/train_p_ctx.json'))['adapters'].values())[-1])")
[ -f "$d/adapter_model.safetensors" ] || { echo "BAIL: P-ctx's adapter did not sync ($d)"; exit 1; }
for f in decisions.jsonl episodes.jsonl; do
  [ -f "$C/finalist_c2_rule_right_mach/mach/$f" ] || { echo "BAIL: the finalist run's $f did not sync"; exit 1; }
done
[ -f "papers/cst_constitutional_steering/data/mach_teacher_states_v1.json" ] || { echo "BAIL: the states of record are missing"; exit 1; }
for f in outputs/pilot_sdf/adapters/d_doc/1e+06/adapter_model.safetensors outputs/pilot_sdf/parity/hf_d_doc_1M.json outputs/pilot_sdf/parity/hf_base.json; do
  [ -f "papers/cst_constitutional_steering/$f" ] || { echo "BAIL: $f did not sync"; exit 1; }
done

PART=mach $PY - <<'EOF' > "$OUT/logs/models_teacher${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
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
echo ">> models: $(cat "$OUT/logs/models_teacher${SESSION}.json")"
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
echo ">> setup done at $(hours) h"

TESTS="tests/scripts/test_cst_teacher.py tests/scripts/test_cst_finalist.py tests/scripts/test_cst_parity.py tests/scripts/test_mch_thinking_pod.py"
# the full pytest output stays in the downloaded logs
$PY -m pytest -q -p no:cacheprovider $TESTS > "$OUT/logs/gate_tests${SESSION}.log" 2>&1; GT=$?
grep -E '[0-9]+ (passed|failed)' "$OUT/logs/gate_tests${SESSION}.log" | tail -1
grep -E '^FAILED' "$OUT/logs/gate_tests${SESSION}.log" | head -5
[ "$GT" = 0 ] || { echo "BAIL: gate tests (see logs/gate_tests${SESSION}.log)"; exit 1; }

run() { $PY $POD --out "$OUT" "$@" || echo ">> stage failed: $*"; echo ">> stage done at $(hours) h: $*"; }

# the teacher check first (Amendment 47 item 2): base and GOOD_BEHAVIOR, then the constitution
run --stage plain
run --stage constitution
# P-ctx's riders (Amendment 47 item 1): the smoke pass and parity, then refusal retention and MoralBench
run --stage smoke_hf
[ -f "$OUT/smoke_hf.json" ] || { echo ">> riders skipped: smoke_hf failed (see the log)"; echo ">> done at $(hours) h"; exit 0; }
run --stage smoke_serve
python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/smoke_serve.json'))['parity_passes'] else 1)" 2>/dev/null \
  || { echo ">> riders skipped: smoke_serve failed"; echo ">> done at $(hours) h"; exit 0; }
run --stage parity_hf
run --stage parity
python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/parity/p_ctx.json')).get('read') else 1)" 2>/dev/null \
  || { echo ">> riders skipped: P-ctx's parity did not pass or did not finish"; echo ">> done at $(hours) h"; exit 0; }
run --stage rr
run --stage mb
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
