#!/usr/bin/env bash
# One-time setup of the simulation environment (no sudo needed). Ubuntu 22.04/24.04, Python 3.11 via uv.
#   bash setup/install_sim.sh          then, in every terminal:   source setup/env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACADOS_DIR="$ROOT/external/acados"
ACADOS_COMMIT="a525ff47a392c7a1a1e2a0a134ab8bbe36886a1e"   # tested commit (newer master changed behaviour)

command -v uv >/dev/null || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
cd "$ROOT"
[ -d .venv ] || uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e "src/uam_quad[mpc,dev]"

if [ ! -f "$ACADOS_DIR/lib/libacados.so" ]; then
  mkdir -p "$ROOT/external"
  [ -d "$ACADOS_DIR" ] || git clone https://github.com/acados/acados.git "$ACADOS_DIR"
  git -C "$ACADOS_DIR" checkout -q "$ACADOS_COMMIT"
  git -C "$ACADOS_DIR" submodule update --recursive --init -q
  mkdir -p "$ACADOS_DIR/build" && cd "$ACADOS_DIR/build"
  cmake -DACADOS_INSTALL_DIR=.. .. >/dev/null && make -j"$(nproc)" install >/dev/null
  cd "$ROOT"
fi
uv pip install -e "$ACADOS_DIR/interfaces/acados_template"
# ACADOS renders its C templates with the "tera" binary; fetch the version this commit expects (no prompt).
[ -x "$ACADOS_DIR/bin/t_renderer" ] || ACADOS_SOURCE_DIR="$ACADOS_DIR" python -c "from acados_template.utils import get_tera; get_tera(force_download=True)"
echo "Done. Next: source setup/env.sh && cd src/uam_quad && python -m pytest -q tests"
