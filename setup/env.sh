# Source in every terminal:  source setup/env.sh   (from the repo root, or with a full path)
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_ROOT/.venv/bin/activate" ]; then
  source "$_ROOT/.venv/bin/activate"
else
  echo "no .venv yet -- run: bash $_ROOT/setup/install_sim.sh" >&2
fi
export ACADOS_SOURCE_DIR="$_ROOT/external/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
# Terminals started outside the desktop session (ssh, agents) have no DISPLAY for the viewer/renderer.
if [ -z "${DISPLAY:-}" ] && [ -S /tmp/.X11-unix/X1 ]; then export DISPLAY=:1; fi
unset _ROOT
