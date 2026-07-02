#!/usr/bin/env bash
# Remote runner for Direction-2 Phase B SESSION 1 (loaded-model batch: B1 + B3 + B5).
# Launch via the shared launcher:
#   REMOTE_SCRIPT=papers/d2_decision_coupling/scripts/remote_session1.sh \
#     SELF_PAPER=papers/d2_decision_coupling ./papers/d1_moral_subspace/runpod/run_session.sh
#
# Cost-minimizing flow (per feedback_test_gates_before_gpu):
#   1. local gate (done):  python papers/d2_decision_coupling/scripts/b_session1_local_test.py
#   2. cheap GPU smoke:     VALIDATE=1 <launch>     (tiny model; plumbing only, no external npz)
#   3. real run:            <launch>                (OLMo-3 / Qwen2.5 / Llama-3.1 instruct)
#
# Per model the sequence is: extract V_moral source dirs + refusal + persona + act_sample in the
# model's OWN space (directions never transfer), then B1 (judgment-decision keystone), B3
# (non-moral control dirs), B5 (moral fragility of refusal). B3 rotate (R6) runs on OLMo base+SFT.
# Drops .session_done on exit so the billed pod never leaks.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
cd "$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
trap 'touch "$REPO_DIR/.session_done"' EXIT

D2="$REPO_DIR/papers/d2_decision_coupling"
D1S="$REPO_DIR/papers/d1_moral_subspace/scripts"
P6S="$REPO_DIR/papers/6_cross_model/scripts"
OUT="$D2/outputs"
mkdir -p "$OUT"

# ---- env setup (mirror the d1 remote runners) ----
echo ">> cuda: $(python -c 'import torch;print(torch.cuda.is_available())' 2>&1)"
pip install -q --break-system-packages -e . 2>&1 | tail -1 || true
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"
pip install -q --break-system-packages "transformers==$TRANSFORMERS_VERSION" -U accelerate 2>&1 | tail -1 || true
[ -n "${PIP_EXTRA:-}" ] && pip install -q --break-system-packages -U $PIP_EXTRA 2>&1 | tail -1 || true
pip install -q --break-system-packages hf_xet >/dev/null 2>&1 && export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
echo ">> transformers: $(python -c 'import transformers;print(transformers.__version__)' 2>&1)"

# ---- cheap self-contained plumbing smoke (no external npz; B-scripts skip measurements) ----
if [ "$VALIDATE" = "1" ]; then
  echo ">> VALIDATE smoke: B1 / B3 / B5 on the tiny model (plumbing only)"
  VALIDATE=1 python "$D2/scripts/b1_judgment_direction.py" --key olmo3 --no-ablation --out "$OUT/_smoke" || exit 1
  VALIDATE=1 python "$D2/scripts/b3_batched_extractions.py" --key olmo3 --out "$OUT/_smoke" || exit 1
  VALIDATE=1 python "$D2/scripts/b5_moral_fragility.py" --key olmo3 --out "$OUT/_smoke" || exit 1
  echo ">> VALIDATE smoke OK (plumbing). Stop here; launch without VALIDATE for the real run."
  exit 0
fi

# ---- per-model real run ----
# Panel: key -> instruct repo (base repo used only for the R6 rotation on OLMo).
PANEL=("olmo3:allenai/Olmo-3-7B-Instruct" "qwen25:Qwen/Qwen2.5-7B-Instruct" "llama31:meta-llama/Llama-3.1-8B-Instruct")
REFUSAL_PROMPTS="$REPO_DIR/papers/5_moral_alignment/refusal_prompts.json"

extract_inputs () {  # $1=key $2=instruct_repo -> writes $OUT/$key/{vmoral_sources.npz,refusal.npz,persona_direction.npz,act_sample.npz}
  local key="$1" repo="$2" mdir="$OUT/$1"
  mkdir -p "$mdir"
  # layer/band from the paper-6 registry (single source of truth).
  read -r LAYER B0 B1 < <(python - "$key" <<'PY'
import sys; sys.path.insert(0,"papers/6_cross_model/scripts")
import model_registry as reg; s=reg.get(sys.argv[1]); print(s.primary_layer, s.band[0], s.band[1])
PY
)
  echo ">> [$key] layer=$LAYER band=$B0-$B1 : extracting V_moral sources + refusal + persona"
  # moral_stories dir + persona + act_sample (+ mft) in the instruct model's space.
  MATCH_LAYER="$LAYER" python "$D1S/phase2_extract.py" --model "$repo" --out "$mdir" --mft || return 1
  # fables / ethics axis directions.
  MATCH_LAYER="$LAYER" python "$D1S/phase2_axis_extract.py" --model "$repo" --out "$mdir" || return 1
  # refusal (Arditi/Heretic last-input-token, chat) via the paper-6 extractor.
  python "$P6S/extract_refusal.py" --model "$repo" --prompts "$REFUSAL_PROMPTS" \
    --layer "$LAYER" --band "$B0" "$B1" --output-dir "$mdir" || return 1
  # assemble the rank-3 V_moral source-dir npz that B1/B3/B5 consume as --vmoral-npz.
  python - "$mdir" "$LAYER" <<'PY'
import sys, numpy as np
from pathlib import Path
mdir, L = Path(sys.argv[1]), int(sys.argv[2])
md = dict(np.load(mdir/"moral_directions.npz"))
ad = dict(np.load(mdir/"axis_directions.npz"))
out = {f"moral_stories_layer{L}": md[f"moral_stories_layer{L}"],
       f"fables_layer{L}": ad[f"fables_layer{L}"], f"ethics_layer{L}": ad[f"ethics_layer{L}"]}
np.savez(mdir/"vmoral_sources.npz", **out)
# refusal file from extract_refusal is refusal_directions.npz keyed per layer -> normalize name
import glob
rf = mdir/"refusal_directions.npz"
if rf.exists():
    rz = np.load(rf); key = f"refusal_layer{L}" if f"refusal_layer{L}" in rz.files else rz.files[0]
    np.savez(mdir/"refusal.npz", refusal=rz[key], layer=L)
print(">> assembled", mdir/"vmoral_sources.npz")
PY
}

for entry in "${PANEL[@]}"; do
  key="${entry%%:*}"; repo="${entry#*:}"; mdir="$OUT/$key"
  echo "==================== $key ($repo) ===================="
  extract_inputs "$key" "$repo" || { echo "ERROR extract $key"; continue; }
  VM="$mdir/vmoral_sources.npz"; RF="$mdir/refusal.npz"
  PS="$mdir/persona_direction.npz"; AS="$mdir/act_sample.npz"

  echo ">> [$key] B1 judgment-decision direction (R2/R3 + cross-ablation)"
  python "$D2/scripts/b1_judgment_direction.py" --model "$repo" --key "$key" \
    --vmoral-npz "$VM" --refusal-npz "$RF" --persona-npz "$PS" --act-sample-npz "$AS" --out "$mdir" || true

  echo ">> [$key] B3 non-moral control directions (R5 + fable-schema)"
  python "$D2/scripts/b3_batched_extractions.py" --model "$repo" --key "$key" \
    --vmoral-npz "$VM" --refusal-npz "$RF" --out "$mdir" || true

  echo ">> [$key] B5 moral fragility of refusal (R8)"
  python "$D2/scripts/b5_moral_fragility.py" --model "$repo" --key "$key" \
    --vmoral-npz "$VM" --persona-npz "$PS" --act-sample-npz "$AS" --out "$mdir" || true
done

# ---- R6 rotation-specificity: OLMo-3 base + SFT control directions, then compare ----
echo "==================== R6 rotation (OLMo-3 base + SFT) ===================="
python "$D2/scripts/b3_batched_extractions.py" --model "allenai/Olmo-3-1025-7B" --key "olmo3_base" --out "$OUT" || true
SFT_MODEL="${SFT_MODEL:-allenai/Olmo-3-7B-Instruct-SFT}"
python "$D2/scripts/b3_batched_extractions.py" --model "$SFT_MODEL" --key "olmo3_sft" --out "$OUT" || true
python "$D2/scripts/b3_batched_extractions.py" --mode rotate --base-tag olmo3_base --sft-tag olmo3_sft \
  --moral-rotation-deg "${MORAL_ROTATION_DEG:-40}" --out "$OUT" || true

echo ">> session 1 done. rsync-back then analyze b1_result_*.json / b3_result_*.json / b5_fragility_*.json"
