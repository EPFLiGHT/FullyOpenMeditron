#!/bin/bash
# Create the local (non-cluster) python environment at .venv, using uv.
#
#   bash setup_local_env.sh
#   source .venv/bin/activate
#
# Works the same on a laptop and on a the cluster login node. It is deliberately NOT
# the cluster job environment: compute-node jobs run inside the container EDF
# and use requirements-inference.txt. Nothing installed here pulls torch or
# vLLM.
#
# Why uv and not `python -m venv`: the login node ships only python 3.6 and
# 3.11, and inspect-ai needs 3.12. uv downloads a standalone interpreter rather
# than depending on whatever the host happens to have. That mismatch is what
# broke `inspect view` on the login node -- the launcher installed from inside
# the container had a shebang pointing at a python that only exists there.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# 1. uv
if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
else
    echo "==> uv not found; installing to ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$HOME/.local/bin/uv"
fi
echo "==> uv: $("$UV" --version)"

# 2. venv on a pinned interpreter
echo "==> creating $VENV_DIR (python $PYTHON_VERSION)"
"$UV" venv --python "$PYTHON_VERSION" "$VENV_DIR"

# 3. dependencies
echo "==> installing requirements-local.txt"
"$UV" pip install --python "$VENV_DIR/bin/python" \
    -r "$PROJECT_ROOT/requirements-local.txt"

# 4. verify the two things that actually break
echo "==> verifying"
"$VENV_DIR/bin/python" - <<'PY'
import importlib.util, sys
missing = [m for m in ("inspect_ai", "numpy", "pandas", "scipy",
                       "statsmodels", "sklearn", "matplotlib", "krippendorff")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"MISSING: {', '.join(missing)}")
import inspect_ai
print(f"  python      {sys.version.split()[0]}")
print(f"  inspect-ai  {inspect_ai.__version__}")
PY

# the `inspect` launcher's shebang must point inside this venv, not at a python
# that only exists in the container
head -1 "$VENV_DIR/bin/inspect" | grep -q "$VENV_DIR" \
    && echo "  inspect launcher shebang OK" \
    || { echo "  WARNING: unexpected shebang in $VENV_DIR/bin/inspect"; exit 1; }

cat <<EOF

Done. Activate with:

    source ${VENV_DIR#$PROJECT_ROOT/}/bin/activate

Then, for example:

    inspect view --log-dir logs/inspect_auto_moove
    python auto_moove/results_to_inspect.py <results>.jsonl \\
        --log-dir logs/inspect_auto_moove

On a laptop, pull the logs down first:

    rsync -av <login-node>:meditronFO/logs/inspect_auto_moove/ ~/moove-logs/
    inspect view --log-dir ~/moove-logs
EOF
