# Source in every terminal for the PX4 + ROS 2 setup:   source setup/env_ros.sh
# ROS 2 Kilted + px4_msgs + the Python 3.12 venv (.venv-ros: MuJoCo, CasADi, ACADOS, rclpy) + the DDS agent.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/kilted/setup.bash
[ -f "$_ROOT/ros2_px4_ws/install/setup.bash" ] && source "$_ROOT/ros2_px4_ws/install/setup.bash"
if [ -f "$_ROOT/.venv-ros/bin/activate" ]; then
  source "$_ROOT/.venv-ros/bin/activate"
else
  echo "no .venv-ros yet -- run: bash $_ROOT/setup/build_px4.sh" >&2
fi
export ACADOS_SOURCE_DIR="$_ROOT/external/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
export PATH="$_ROOT/external/Micro-XRCE-DDS-Agent/install/bin:$PATH"
export LD_LIBRARY_PATH="$_ROOT/external/Micro-XRCE-DDS-Agent/install/lib:$LD_LIBRARY_PATH"
if [ -z "${DISPLAY:-}" ] && [ -S /tmp/.X11-unix/X1 ]; then export DISPLAY=:1; fi
unset _ROOT
