#!/bin/bash
# Double-click in Finder, or run from any directory in Terminal.
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/frontend_env.sh"
echo "Starting MRI reading room using Conda environment: $MRI_SELECTED_ENV"
frontend_python "$MRI_PROJECT_ROOT/scripts/run_frontend.py" "$@"
