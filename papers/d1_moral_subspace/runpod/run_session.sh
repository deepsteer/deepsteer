#!/usr/bin/env bash
# Forwarding script: the shared launcher moved to papers/runpod_common/run_session.sh (2026-10-30).
# Committed launch lines that name this path keep working; new launch lines use the new path.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../runpod_common" && pwd)/run_session.sh" "$@"
