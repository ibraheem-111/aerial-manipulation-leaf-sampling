#!/usr/bin/env bash
# System packages for the PX4 + ROS 2 setup (needs sudo, run once):   bash setup/install_ros2_px4.sh
# Installs ROS 2 Kilted Kaiju (ros-base, no Gazebo), colcon and the PX4 SITL build tools.
# Everything else (PX4 itself, the DDS agent, px4_msgs) is built in user space by setup/build_px4.sh.
set -euo pipefail

if [ "$(. /etc/os-release && echo "$VERSION_CODENAME")" != "noble" ]; then
  echo "Expected Ubuntu 24.04 (noble). Aborting." >&2
  exit 1
fi

echo "== ROS 2 apt source =="
sudo apt-get update
sudo apt-get install -y software-properties-common curl locales
sudo locale-gen en_US en_US.UTF-8
sudo add-apt-repository -y universe
if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
  V=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F '"tag_name"' | awk -F'"' '{print $4}')
  curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${V}/ros2-apt-source_${V}.noble_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb
fi

echo "== ROS 2 Kilted (ros-base) + build tools =="
sudo apt-get update
sudo apt-get install -y \
  ros-kilted-ros-base ros-dev-tools python3-colcon-common-extensions \
  ninja-build ccache build-essential cmake git \
  python3-pip python3-venv python3-dev

echo "Done. Next (no sudo):  bash setup/build_px4.sh"
