#!/usr/bin/env bash
# The standing step before any GPC-4 launch (the lead, after launches 1 and 2): run the pod's gate tests
# locally under the environment the pod runner has exported when it runs them, once quietly and then
# several times under heavy CPU load, all passing. Launch 1 bailed on a test that read an environment
# name the runner sets; launch 2 on a test whose order depended on load. Both were host-side tests,
# which the pod's gate now deselects (marker host_side; CI still runs them).
#
#   bash papers/gpc_goal_pressure_conduct/runpod/gpc4_pod_gates_local.sh        # LOAD_RUNS=3 by default
#
# Reads the runner's own marker expression, floor, TESTS list and `export` lines (those before its gate
# tests), so they never drift. Leaves out PATH (the local interpreter stays) and HF_TOKEN (never
# copied). Prints the variables' names only, never their values, the collected count against the floor,
# and each run's summary; exits nonzero on any failure.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=papers/gpc_goal_pressure_conduct/runpod/remote_gpc4.sh
LOAD_RUNS=${LOAD_RUNS:-3}
gate_at=$(grep -n 'pytest -q -p no:cacheprovider -m "\$GATE_MARK" \$TESTS' "$R" | head -1 | cut -d: -f1)
[ -n "$gate_at" ] || { echo "BAIL: the runner's gate-test line not found"; exit 1; }
tests=$(head -n "$gate_at" "$R" | grep -E '^TESTS=' | tail -1)
mark=$(head -n "$gate_at" "$R" | grep -E '^GATE_MARK=' | tail -1)
floor=$(head -n "$gate_at" "$R" | grep -E '^GATE_FLOOR=' | tail -1 | sed 's/ *#.*//')
[ -n "$tests" ] && [ -n "$mark" ] && [ -n "$floor" ] || { echo "BAIL: the runner's TESTS, GATE_MARK or GATE_FLOOR not found"; exit 1; }
exports=$(head -n "$gate_at" "$R" | grep -E '^export ' | grep -vE '^export (PATH|HF_TOKEN)=')
PY=${PY:-/opt/homebrew/bin/python3.13}
run_gate() {  # <label>
  bash -c "
set -euo pipefail
REPO_DIR=/workspace/deepsteer G=\$REPO_DIR/papers/gpc_goal_pressure_conduct
C=\$REPO_DIR/papers/cst_constitutional_steering/outputs I=\$G/outputs/gpc4_in VENV=/workspace/venv_vllm
START=\$(date +%s) PRICE_H=10.5 SESSION=1
$exports
$tests
$mark
$floor
cd '$PWD'
if [ '$1' = quiet ]; then
  echo \">> the pod runner's exports at its gate tests: \$(printf '%s\n' \"$exports\" | grep -oE '[A-Z_0-9]+=' | tr -d = | sort -u | tr '\n' ' ')\"
  n=\$('$PY' -m pytest --collect-only -q -p no:cacheprovider -m \"\$GATE_MARK\" \$TESTS 2>/dev/null | grep -c '::' || true)
  echo \">> the pod's gate (-m \\\"\$GATE_MARK\\\"): \$n tests collected, floor \$GATE_FLOOR\"
  [ \"\$n\" -ge \"\$GATE_FLOOR\" ] || { echo 'BAIL: under the floor'; exit 1; }
fi
'$PY' -m pytest -q -p no:cacheprovider -m \"\$GATE_MARK\" \$TESTS 2>&1 | tail -1
exit \${PIPESTATUS[0]}
"
}
# every path the runner exports into the repo exists (launch 3 stopped at prep on a doubled outputs/ in
# GPC4_PCTX): the runner's paths are evaluated with the repo here, where the synced inputs come from
all_exports=$(grep -E '^export ' "$R" | grep -vE '^export (PATH|HF_TOKEN)=')
missing=$(bash -c "
REPO_DIR='$PWD' G=\$REPO_DIR/papers/gpc_goal_pressure_conduct
C=\$REPO_DIR/papers/cst_constitutional_steering/outputs I=\$G/outputs/gpc4_in VENV=/workspace/venv_vllm
START=\$(date +%s) PRICE_H=10.5 SESSION=1
$all_exports
for v in \$(printf '%s\n' \"$all_exports\" | grep -oE '[A-Z_0-9]+=' | tr -d =); do
  val=\${!v}; case \"\$val\" in '$PWD'/*) [ -e \"\$val\" ] || echo \"\$v=\${val#'$PWD'/}\";; esac
done" 2>/dev/null)
n_paths=$(bash -c "REPO_DIR='$PWD' G=\$REPO_DIR/papers/gpc_goal_pressure_conduct C=\$REPO_DIR/papers/cst_constitutional_steering/outputs I=\$G/outputs/gpc4_in VENV=/workspace/venv_vllm START=0 PRICE_H=10.5 SESSION=1
$all_exports
for v in \$(printf '%s\n' \"$all_exports\" | grep -oE '[A-Z_0-9]+=' | tr -d =); do case \"\${!v}\" in '$PWD'/*) echo x;; esac; done" 2>/dev/null | wc -l | tr -d ' ')
[ -z "$missing" ] || { echo "BAIL: the runner exports paths into the repo that do not exist: $missing"; exit 1; }
echo ">> the runner's $n_paths exported repo paths all exist"
out=$(run_gate quiet) || { echo "$out"; echo "BAIL: the quiet run failed"; exit 1; }
echo "$out" | sed '$d'; echo ">> run 1, quiet: $(echo "$out" | tail -1)"
burners=()
stop_load() { [ ${#burners[@]} -gt 0 ] && kill "${burners[@]}" 2>/dev/null; burners=(); }
trap stop_load EXIT
for _ in $(seq 1 $(( $(sysctl -n hw.ncpu 2>/dev/null || nproc) * 2 ))); do ( while :; do :; done ) & burners+=($!); done
fails=0
for k in $(seq 1 "$LOAD_RUNS"); do
  out=$(run_gate loaded) || fails=$((fails + 1))
  echo ">> run $((k + 1)), under load (two busy loops a core): $out"
done
stop_load
[ "$fails" = 0 ] || { echo "BAIL: $fails of $LOAD_RUNS loaded runs failed"; exit 1; }
echo ">> the standing step passed: 1 quiet run and $LOAD_RUNS under load"
