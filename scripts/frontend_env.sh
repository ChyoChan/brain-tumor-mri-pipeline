# Shared environment setup for the macOS start/stop launchers.
# Source this file; do not execute it directly.
MRI_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MRI_PROJECT_ROOT"

# CONDA_EXE is provided by active Conda shells; type -P ignores shell functions.
MRI_CONDA_BIN="${MRI_CONDA_EXE:-${CONDA_EXE:-$(type -P conda || true)}}"
if [[ -z "$MRI_CONDA_BIN" ]]; then
    # Finder does not always inherit the Terminal's Conda PATH.
    for candidate in "$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda" \
                     "$HOME/miniforge3/bin/conda" /opt/anaconda3/bin/conda \
                     /opt/miniconda3/bin/conda /opt/homebrew/bin/conda; do
        if [[ -x "$candidate" ]]; then
            MRI_CONDA_BIN="$candidate"
            break
        fi
    done
fi

if [[ -z "$MRI_CONDA_BIN" || ! -x "$MRI_CONDA_BIN" ]]; then
    echo "Conda not found. Set MRI_CONDA_EXE to its executable path."
    echo "Alternatively, activate your Python environment and run:"
    echo "  python scripts/run_frontend.py      # start"
    echo "  python scripts/stop_frontend.py     # stop"
    exit 1
fi

MRI_SELECTED_ENV="${MRI_CONDA_ENV:-rag}"

frontend_python() {
    exec "$MRI_CONDA_BIN" run --no-capture-output -n "$MRI_SELECTED_ENV" \
        env PYTHONUNBUFFERED=1 python "$@"
}
