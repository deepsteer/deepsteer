#!/usr/bin/env bash
# The standing step before any GPC-4 launch (the lead, after the first launch): run the pod's gate tests
# locally under the environment the pod runner has exported when it runs them. The first launch bailed at
# a test that read GPC4_IB_DATA, which the runner sets for the pod's own use; locally it was unset, so the
# test passed here and failed there.
#
#   bash papers/gpc_goal_pressure_conduct/runpod/gpc4_pod_gates_local.sh
#
# Takes the runner's own `export` lines that come before its gate tests, and its TESTS list, so the two
# never drift. Leaves out PATH (the local interpreter stays) and HF_TOKEN (never copied). Prints the
# variables' names only, never their values, and the test summary; exits nonzero on any failure.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=papers/gpc_goal_pressure_conduct/runpod/remote_gpc4.sh
gate_at=$(grep -n 'pytest -q -p no:cacheprovider -m "not slow" \$TESTS' "$R" | head -1 | cut -d: -f1)
[ -n "$gate_at" ] || { echo "BAIL: the runner's gate-test line not found"; exit 1; }
tests=$(head -n "$gate_at" "$R" | grep -E '^TESTS=' | tail -1)
[ -n "$tests" ] || { echo "BAIL: the runner's TESTS list not found"; exit 1; }
exports=$(head -n "$gate_at" "$R" | grep -E '^export ' | grep -vE '^export (PATH|HF_TOKEN)=')
PY=${PY:-/opt/homebrew/bin/python3.13}
bash -c "
set -euo pipefail
REPO_DIR=/workspace/deepsteer G=\$REPO_DIR/papers/gpc_goal_pressure_conduct
C=\$REPO_DIR/papers/cst_constitutional_steering/outputs I=\$G/outputs/gpc4_in VENV=/workspace/venv_vllm
START=\$(date +%s) PRICE_H=10.5 SESSION=1
$exports
$tests
echo \">> the pod runner's exports at its gate tests: \$(printf '%s\n' \"$exports\" | grep -oE '[A-Z_0-9]+=' | tr -d = | sort -u | tr '\n' ' ')\"
cd '$PWD'
'$PY' -m pytest -q -p no:cacheprovider -m 'not slow' \$TESTS 2>&1 | tail -3
exit \${PIPESTATUS[0]}
"
