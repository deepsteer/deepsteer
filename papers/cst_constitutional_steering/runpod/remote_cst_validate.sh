#!/usr/bin/env bash
# Remote runner for CST VALIDATE (CST_PLAN T0.8; V1-V10; T06_T08_SPECS.md), one A100-80GB.
# Second run (2026-11-07): V3-V7 again (V7 on its own load, checkpointing active), rates at a
# saturating batch, the A0s cap probe, the judges at an 8,192-token budget with raw replies, and
# the judge candidates measured for the author.
# Orion launches; the launcher passes HF_TOKEN. ST's, MRB's and MCH's pinned vLLM 0.31.0 stack.
#
#   setup (venv, stack check, vLLM smoke, models, gate tests) -> render (V8) -> gen (V1, V2 and the
#   subject's rate) -> lora_hf (V3, V5) -> vllm_lora (V4, V6) -> gradnec (V7) -> judges, one at a
#   time (V9, V10) -> price (S0a). Each stage writes outputs/validate/<stage>.json; a failed stage stops the
#   stages that depend on it, and each judge is independent. Adapters stay in outputs_raw (not
#   synced back). Wall cap 4.0 h.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
OUT="$REPO_DIR/papers/cst_constitutional_steering/outputs/validate"
RAWD="$REPO_DIR/papers/cst_constitutional_steering/outputs_raw/validate"
mkdir -p "$OUT" "$RAWD"
START=$(date +%s)
trap 'cp "$REPO_DIR/session.log" "$OUT/session_validate.log" 2>/dev/null; cp "$RAWD"/*.json "$RAWD"/judges_*.jsonl "$RAWD"/gen_*.jsonl "$OUT"/ 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$REPO_DIR/papers/cst_constitutional_steering/outputs/_sync"
POD=papers/cst_constitutional_steering/scripts/cst_validate_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST VALIDATE start $(date -u +%FT%TZ) out=$OUT"

MAX_H=4.0
( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f cst_validate_pod.py ) &
WATCHDOG=$!

# ---- the vLLM venv (MRB-G1's and MCH's block, unchanged pins) plus peft ----
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

# ---- models: the subject and the three judges at their pins ----
$PY - <<'EOF' > "$OUT/models.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
pins = {"qwen38_27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
        "gpt_oss_120b": ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"),
        "olmo31_32b_think": ("allenai/Olmo-3.1-32B-Think", "97ade57a9cc55ed48bcd76be4b7a3b6fee61b80d"),
        "glm47_flash": ("zai-org/GLM-4.7-Flash", "7dd20894a642a0aa287e9827cb1a1f7f91386b67"),
        "olmo31_32b_instruct": ("allenai/Olmo-3.1-32B-Instruct", "ac0587e4a7744a551c059d8cd17ba220bc940dae")}
print(json.dumps({k: snapshot_download(r, revision=v, ignore_patterns=["original/*", "metal/*"])
                  for k, (r, v) in pins.items()}))
EOF
echo ">> models: $(cat "$OUT/models.json")"
echo ">> setup done at $(hours) h"
$PY -m pytest -q -p no:cacheprovider tests/scripts/test_cst_validate.py tests/scripts/test_cst_pressure.py \
  2>&1 | grep -E '[0-9]+ (passed|failed)' | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }

run() { $PY $POD --stage "$1" --out "$RAWD" ${2:+--judges "$2"}; }
run render || echo ">> render (V8) failed; continuing (renders are re-checked at S0)"
run gen || { echo "BAIL: gen (V1/V2) failed; the later stages need its rollouts"; exit 1; }
echo ">> gen done at $(hours) h"
if run lora_hf; then
  run vllm_lora || echo ">> vllm_lora (V4/V6) failed"
else
  echo ">> lora_hf (V3/V5) failed; vllm_lora skipped"
fi
run gradnec || echo ">> gradnec (V7) failed"
echo ">> subject stages done at $(hours) h"
# the registered configs first, then the candidates; the slowest (Olmo Think) last
for J in gpt_oss_120b glm47_flash glm47_flash_nothink olmo31_32b_instruct olmo31_32b_think_fp8kv; do
  run judges "$J" && cp "$RAWD/judges.json" "$RAWD/judges_$J.json" || echo ">> judge $J failed"
done
$PY - "$RAWD" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
merged = {}
for f in sorted(d.glob("judges_*.json")):
    merged.update(json.loads(f.read_text()))
(d / "judges.json").write_text(json.dumps(merged, indent=1))
print(">> judges:", {k: (v.get("parse_rate"), round(v.get("calls_per_hour", 0))) for k, v in merged.items()})
EOF
run price || echo ">> price failed"
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
