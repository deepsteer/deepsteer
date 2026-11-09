#!/usr/bin/env bash
# Remote runner for CST S0b (S0B_BUILD.md; Amendments 14, 16-18), one A100-80GB.
# Orion launches with CST_SESSION=1 (run_session.sh passes it to the pod); the launcher passes
# HF_TOKEN. VALIDATE's pinned vLLM 0.31.0. Session 2 (MoReBench, MACHIAVELLI) is not built yet.
#
#   SESSION=1  A0n: a pilot of two 256-rollout chunks and its gate (parse >= 0.95, capped <= 3%,
#              >= 600 rollouts/h, else bank and stop), then the rest of panel plain and operator at
#              k 4 and the dose-0 letter reads; gpt-oss-120b: A0n's bulk scoring and the CST-A15
#              cut stage on A0, A0s (S0a's rollouts, synced) and A0n.
# Everything writes under outputs/s0b, which the launcher downloads (RESULTS_SUBPATH) and a rerun
# syncs back up (SYNC_EXTRA), so a chunk or file on disk is skipped and a session cut short is rerun
# with the same CST_SESSION. A failed stage does not stop the next, except the pilot gate. The
# .session_done trap is armed before any check that can bail.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
OUT="$REPO_DIR/papers/cst_constitutional_steering/outputs/s0b"
S0A="$REPO_DIR/papers/cst_constitutional_steering/outputs/s0a"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SESSION="${CST_SESSION:-}"
trap 'cp "$REPO_DIR/session.log" "$OUT/logs/session_s0b${SESSION}_${STAMP}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=8.0;;
  *) echo "BAIL: CST_SESSION must be 1 (session 2 is not built)"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$REPO_DIR/papers/cst_constitutional_steering/outputs/_sync"
POD=papers/cst_constitutional_steering/scripts/cst_s0a_pod.py
PODB=papers/cst_constitutional_steering/scripts/cst_s0b_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST S0b session $SESSION start $(date -u +%FT%TZ) out=$OUT (watchdog ${MAX_H} h)"

( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f 'cst_s0[ab]_pod.py' ) &
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

# ---- synced inputs (gitignored; SYNC_EXTRA) ----
[ -f "$CST_CONST_DIR/C_anth.txt" ] || { echo "BAIL: the pinned constitution text did not sync"; exit 1; }
[ -f "$CST_CONST_DIR/a0n_faraday_v1.txt" ] || { echo "BAIL: the pinned A0n text did not sync"; exit 1; }
for a in a0 a0s; do
  n=$(cat "$S0A"/gen/$a/panel_operator_[0-9]*.jsonl 2>/dev/null | wc -l)
  [ "$n" = 2344 ] || { echo "BAIL: S0a's $a operator rollouts did not sync ($n of 2344)"; exit 1; }
done

# ---- models at their pins ----
$PY - <<'EOF' > "$OUT/logs/models_s0b${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
pins = {"qwen38_27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
        "gpt_oss_120b": ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a")}
print(json.dumps({k: snapshot_download(r, revision=v, ignore_patterns=["original/*", "metal/*"])
                  for k, (r, v) in pins.items()}))
EOF
echo ">> models: $(cat "$OUT/logs/models_s0b${SESSION}.json")"
echo ">> setup done at $(hours) h"

TESTS="tests/scripts/test_cst_s0a.py tests/scripts/test_cst_s0b.py tests/scripts/test_cst_pressure.py"
# S0b runs no long-task stage, and SYNC_OUTPUTS ships no committed outputs/*.json (the long-task
# tests read outputs/t01_tcore_votes_v2.json); CI keeps running both (sync emulation, 2026-11-09)
TESTS="$TESTS --deselect tests/scripts/test_cst_s0a.py::test_long_tasks_vote_on_the_record_splits"
TESTS="$TESTS --deselect tests/scripts/test_cst_s0a.py::test_long_stage_runs_olmo_twice"
$PY -m pytest -q -p no:cacheprovider $TESTS 2>&1 | grep -E '[0-9]+ (passed|failed)' | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }

run() { $PY $POD --out "$OUT" "$@" || echo ">> stage failed: $*"; echo ">> stage done at $(hours) h: $*"; }
case "$SESSION" in
  1)
    PARTS=panel_plain,panel_operator
    run --stage gen --arm a0n --parts $PARTS --max-chunks 2
    $PY $PODB --stage gate --arm a0n --out "$OUT" \
      || { echo "BAIL: A0n pilot gate (chunks banked; see gate_a0n.json)"; exit 1; }
    run --stage gen --arm a0n --parts $PARTS
    run --stage dose0 --arm a0n
    run --stage score --arm a0n --judge gpt_oss_120b
    $PY $PODB --stage cut --arms a0,a0s,a0n --out "$OUT" --src "$S0A" || echo ">> stage failed: cut"
    echo ">> stage done at $(hours) h: cut"
    ;;
esac
$PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
count = lambda g: sum(sum(1 for _ in open(f)) for f in o.glob(g))  # noqa: E731
print(">> on disk:", json.dumps({
    "gen_a0n": count("gen/a0n/*.jsonl"), "score_a0n_gpt": count("score/a0n/gpt_oss_120b/*.jsonl"),
    "dose0": sorted(f.name for f in o.glob("dose0/*.jsonl")),
    "cut": {a: count(f"score_cut/{a}/gpt_oss_120b/*.jsonl") for a in ("a0", "a0s", "a0n")}}))
EOF
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
