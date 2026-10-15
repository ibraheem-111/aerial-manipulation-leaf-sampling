#!/usr/bin/env bash
# Builds PX4 SITL and the PX4 <-> ROS 2 link in user space (no sudo). Run after setup/install_ros2_px4.sh:
#   bash setup/build_px4.sh
# Produces: external/PX4-Autopilot (v1.16.2, SITL build), external/Micro-XRCE-DDS-Agent (DDS agent),
# and, when ROS 2 Kilted is installed, ros2_px4_ws/ with px4_msgs built.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$ROOT/external"
PX4_TAG="v1.16.2"
AGENT_TAG="v2.4.3"          # agent version PX4's ROS 2 guide uses
mkdir -p "$EXT"

echo "== PX4 $PX4_TAG =="
if [ ! -d "$EXT/PX4-Autopilot" ]; then
  git clone --branch "$PX4_TAG" --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/PX4/PX4-Autopilot.git "$EXT/PX4-Autopilot"
fi
git -C "$EXT/PX4-Autopilot" submodule update --init --recursive --depth 1
# A shallow clone has no NuttX release tags; PX4's version header wants one (NuttX isn't used in SITL).
git -C "$EXT/PX4-Autopilot/platforms/nuttx/NuttX/nuttx" tag -f nuttx-11.0.0 >/dev/null
[ -d "$EXT/.venv-px4" ] || uv venv -q "$EXT/.venv-px4" --python 3.12
source "$EXT/.venv-px4/bin/activate"
uv pip install -q -r "$EXT/PX4-Autopilot/Tools/setup/requirements.txt" ninja
make -C "$EXT/PX4-Autopilot" px4_sitl_default
deactivate

echo "== Micro XRCE-DDS Agent $AGENT_TAG =="
if [ ! -x "$EXT/Micro-XRCE-DDS-Agent/install/bin/MicroXRCEAgent" ]; then
  [ -d "$EXT/Micro-XRCE-DDS-Agent" ] || git clone --branch "$AGENT_TAG" --depth 1 \
    https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "$EXT/Micro-XRCE-DDS-Agent"
  cmake -S "$EXT/Micro-XRCE-DDS-Agent" -B "$EXT/Micro-XRCE-DDS-Agent/build" \
    -DCMAKE_INSTALL_PREFIX="$EXT/Micro-XRCE-DDS-Agent/install" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$EXT/Micro-XRCE-DDS-Agent/build" -j"$(nproc)"
  cmake --install "$EXT/Micro-XRCE-DDS-Agent/build"
fi

if [ -f /opt/ros/kilted/setup.bash ]; then
  echo "== px4_msgs for ROS 2 Kilted =="
  mkdir -p "$ROOT/ros2_px4_ws/src"
  MSGS_BRANCH="release/$(echo "${PX4_TAG#v}" | cut -d. -f1,2)"      # e.g. release/1.16 (must match PX4)
  [ -d "$ROOT/ros2_px4_ws/src/px4_msgs" ] || git clone --branch "$MSGS_BRANCH" --depth 1 \
    https://github.com/PX4/px4_msgs.git "$ROOT/ros2_px4_ws/src/px4_msgs"
  set +u; source /opt/ros/kilted/setup.bash; set -u
  (cd "$ROOT/ros2_px4_ws" && colcon build --symlink-install)
  echo "== Python 3.12 venv for ROS 2 nodes (.venv-ros) =="
  [ -d "$ROOT/.venv-ros" ] || uv venv -q "$ROOT/.venv-ros" --python /usr/bin/python3.12 --system-site-packages
  VIRTUAL_ENV="$ROOT/.venv-ros" uv pip install -q -e "$ROOT/src/uam_quad[mpc]" pymavlink \
    -e "$EXT/acados/interfaces/acados_template"
else
  echo "ROS 2 Kilted not found in /opt/ros/kilted: run setup/install_ros2_px4.sh, then this script again."
fi
echo "Done."
