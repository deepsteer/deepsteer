#!/usr/bin/env bash
# Remote runner for CST S0a (S0A_BUILD.md; Amendments 7-12), one A100-80GB, three sessions.
# Orion launches with CST_SESSION=1|2|3 (run_session.sh passes it to the pod); the launcher passes
# HF_TOKEN. VALIDATE's pinned vLLM 0.31.0.
#
#   SESSION=1  A0: generation (12,304), dose-0 and E, recall, capability; gpt-oss-120b scoring
#              (blind, effort low), the retest, the relabel, Layer A; Olmo's 10% and the votes.
#   SESSION=2  A0s on the panel (9,376).
#   SESSION=3  A0s on H1_held and cell A', dose-0 and E, capability; gpt-oss and Olmo on A0s.
# Everything writes under outputs/s0a, which the launcher downloads (RESULTS_SUBPATH) and the next
# session syncs back up (SYNC_OUTPUTS outputs/s0a/), so a chunk or file on disk is skipped and a
# session cut short is rerun with the same CST_SESSION. A failed stage does not stop the next. The
# .session_done trap is armed before any check that can bail, so a bail never leaves the launcher
# waiting on a billing pod (the first S0a-1 launch, 2026-11-08).
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
OUT="$REPO_DIR/papers/cst_constitutional_steering/outputs/s0a"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SESSION="${CST_SESSION:-}"
trap 'cp "$REPO_DIR/session.log" "$OUT/logs/session_s0a${SESSION}_${STAMP}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=10.0;; 2) MAX_H=12.0;; 3) MAX_H=10.0;;
  *) echo "BAIL: CST_SESSION must be 1, 2 or 3"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$REPO_DIR/papers/cst_constitutional_steering/outputs/_sync"
POD=papers/cst_constitutional_steering/scripts/cst_s0a_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST S0a session $SESSION start $(date -u +%FT%TZ) out=$OUT (watchdog ${MAX_H} h)"

( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f cst_s0a_pod.py ) &
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
[ -f "$CST_CONST_DIR/C_anth.txt" ] || { echo "BAIL: the pinned constitution text did not sync"; exit 1; }

# ---- models at their pins (the judges only in the sessions that score) ----
JUDGES_NEEDED=$([ "$SESSION" = 2 ] && echo 0 || echo 1)
$PY - "$JUDGES_NEEDED" <<'EOF' > "$OUT/logs/models_s0a${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
import json, sys
from huggingface_hub import snapshot_download
pins = {"qwen38_27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")}
if sys.argv[1] == "1":
    pins |= {"gpt_oss_120b": ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"),
             "olmo31_32b_instruct": ("allenai/Olmo-3.1-32B-Instruct",
                                     "ac0587e4a7744a551c059d8cd17ba220bc940dae")}
print(json.dumps({k: snapshot_download(r, revision=v, ignore_patterns=["original/*", "metal/*"])
                  for k, (r, v) in pins.items()}))
EOF
echo ">> models: $(cat "$OUT/logs/models_s0a${SESSION}.json")"

# ---- Scruples Dilemmas (Layer A, session 1): evaluation only, fetched at runtime, never saved ----
SCR="$HOME/.cache/deepsteer/scruples"
if [ "$SESSION" = 1 ] && [ ! -f "$SCR/dilemmas/dev.scruples-dilemmas.jsonl" ]; then
  mkdir -p "$SCR"
  curl -sSfL -o "$SCR/dilemmas.tar.gz" \
    https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/dilemmas.tar.gz \
    || { echo "BAIL: Scruples fetch"; exit 1; }
  echo "71c32cea28a7f455463dea0f458771854378748f1f5c164bb90e9ada500fa182  $SCR/dilemmas.tar.gz" \
    | sha256sum -c - >/dev/null || { echo "BAIL: Scruples tarball sha256 differs from the pin"; exit 1; }
  tar -xzf "$SCR/dilemmas.tar.gz" -C "$SCR"
fi
echo ">> setup done at $(hours) h"

TESTS="tests/scripts/test_cst_s0a.py tests/scripts/test_cst_pressure.py"
[ "$SESSION" = 1 ] || TESTS="$TESTS --deselect tests/scripts/test_cst_s0a.py::test_long_tasks_vote_on_the_record_splits_and_score_only_the_overlap"
$PY -m pytest -q -p no:cacheprovider $TESTS 2>&1 | grep -E '[0-9]+ (passed|failed)' | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }

run() { $PY $POD --out "$OUT" "$@" || echo ">> stage failed: $*"; echo ">> stage done at $(hours) h: $*"; }
case "$SESSION" in
  1)
    run --stage gen --arm a0
    run --stage dose0 --arm a0
    run --stage recall
    run --stage capability --arm a0
    run --stage score --arm a0 --judge gpt_oss_120b
    run --stage long --judge gpt_oss_120b
    run --stage score --arm a0 --judge olmo31_32b_instruct
    run --stage long --judge olmo31_32b_instruct
    ;;
  2)
    run --stage gen --arm a0s --parts panel_plain,panel_user,panel_operator,panel_operator_pr
    ;;
  3)
    run --stage gen --arm a0s
    run --stage dose0 --arm a0s
    run --stage capability --arm a0s
    run --stage score --arm a0s --judge gpt_oss_120b
    run --stage score --arm a0s --judge olmo31_32b_instruct
    ;;
esac
$PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
count = lambda g: sum(sum(1 for _ in open(f)) for f in o.glob(g))  # noqa: E731
print(">> on disk:", json.dumps({
    "gen_a0": count("gen/a0/*.jsonl"), "gen_a0s": count("gen/a0s/*.jsonl"),
    "score_a0_gpt": count("score/a0/gpt_oss_120b/*.jsonl"),
    "score_a0s_gpt": count("score/a0s/gpt_oss_120b/*.jsonl"),
    "dose0": sorted(f.name for f in o.glob("dose0/*.jsonl")),
    "cap": sorted(f.name for f in o.glob("cap/*.jsonl")),
    "long": sorted(str(f.relative_to(o)) for f in o.glob("long/*/*.jsonl"))}))
EOF
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
