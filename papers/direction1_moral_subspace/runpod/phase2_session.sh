#!/usr/bin/env bash
# Direction 1, Phase 2 driver. Two SAME-MODEL refusal points (no cross-model projection):
#   Point A = base proto-refusal x Base-V_moral ; Point B = instruct gate x Instruct-V_moral.
# Per tag, enforces the stage sequence as a hard order:
#   extract -> G-AXIS -> assemble V_moral -> FROZEN null ; then G3 reads BOTH tags' frozen nulls.
# Structural constraints (PREREGISTRATION §3A, §3.3, cross-model amendment):
#   1. phase2_null.py writes null_artifact.json; phase2_g3.py HARD-requires BOTH tags' nulls and
#      does not recompute them. Predates-the-result by structure.
#   2. assemble_vmoral branches on g_axis_decision.json (single-source path runs directly since
#      MORABLES was dropped).
#   3. Each refusal point is measured within its own model -> no Base<->Instruct projection.
# Gate before any real GPU run: VALIDATE=1 ./phase2_session.sh  (tiny model both tags, plumbing).
set -euo pipefail

S="$(cd "$(dirname "$0")/../scripts" && pwd)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)/outputs/phase2"
PY="${PYTHON:-python3}"
BASE_MODEL="${BASE_MODEL:-allenai/OLMo-3-7B}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-allenai/OLMo-3-7B-Instruct}"

run_tag () {  # $1=tag  $2=model
  local tag="$1" model="$2" art="$ROOT/$1"
  echo "== [$tag] stage 0: extract ($model) =="    ; "$PY" "$S/phase2_extract.py" --model "$model" --out "$art"
  echo "== [$tag] stage 1: G-AXIS =="              ; "$PY" "$S/phase2_gaxis.py" --artifacts "$art"
  echo "== [$tag] stage 2: assemble V_moral =="    ; "$PY" "$S/phase2_assemble_vmoral.py" --artifacts "$art"
  echo "== [$tag] stage 3: FROZEN null (no refusal) ==" ; "$PY" "$S/phase2_null.py" --artifacts "$art"
}

run_tag base "$BASE_MODEL"
run_tag instruct "$INSTRUCT_MODEL"

echo "== stage 4: G3 (two same-model points; consumes both frozen nulls) =="
"$PY" "$S/phase2_g3.py" --base-artifacts "$ROOT/base" --instruct-artifacts "$ROOT/instruct" \
  --base-model "$BASE_MODEL" --instruct-model "$INSTRUCT_MODEL"
# stage 5 Track-1 sigma* -> follow-on (G5 input), not in the critical G2/G3 path.
echo "== phase 2 sequence complete; artifacts in $ROOT =="
