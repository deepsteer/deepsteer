#!/usr/bin/env bash
# Direction 1, Phase 2 driver. Enforces the stage sequence as a hard order:
#   extract -> G-AXIS -> assemble V_moral -> FROZEN null -> G3 (consumes null) -> Track-1
# Two structural constraints (see PREREGISTRATION §3A, §3.3):
#   1. phase2_null.py writes null_artifact.json; phase2_g3.py HARD-requires it and does not
#      recompute it. Running g3 before null aborts. Predates-the-result by structure.
#   2. phase2_assemble_vmoral.py branches on g_axis_decision.json (two-source / single-source);
#      a G-AXIS fail runs the single-source fallback (NOT re-adding ETHICS).
# Gate before any real GPU run: VALIDATE=1 ./phase2_session.sh  (tiny model, few pairs).
set -euo pipefail

S="$(cd "$(dirname "$0")/../scripts" && pwd)"
ART="${ART:-$(cd "$(dirname "$0")/.." && pwd)/outputs/phase2}"
PY="${PYTHON:-python3}"

echo "== stage 0: extract (GPU) =="          ; "$PY" "$S/phase2_extract.py" --out "$ART"
echo "== stage 1: G-AXIS =="                 ; "$PY" "$S/phase2_gaxis.py" --artifacts "$ART"
echo "== stage 2: assemble V_moral (branch) ="; "$PY" "$S/phase2_assemble_vmoral.py" --artifacts "$ART"
echo "== stage 3: FROZEN null (no refusal) ==" ; "$PY" "$S/phase2_null.py" --artifacts "$ART"
echo "== stage 4: G3 (consumes frozen null) ="; "$PY" "$S/phase2_g3.py" --artifacts "$ART"
# stage 5 Track-1 sigma* -> follow-on (G5 input), not in the critical G2/G3 path.
echo "== phase 2 sequence complete; artifacts in $ART =="
