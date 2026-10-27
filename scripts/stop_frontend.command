#!/bin/bash
# Double-click in Finder, or run from any directory in Terminal.
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/frontend_env.sh"
echo "Stopping this project's MRI frontend. Unrelated services will be left alone."
frontend_python "$MRI_PROJECT_ROOT/scripts/stop_frontend.py" "$@"
