#!/usr/bin/env bash
# One-time machine setup for the aerial-contact applied project.
# Ubuntu 24.04 (noble): ROS 2 Jazzy desktop, Gazebo Harmonic via ros_gz, colcon/rosdep, build tools.
# Run as your user; it calls sudo where needed:  bash setup/install_ros2_jazzy.sh
set -euo pipefail

if [ "$(. /etc/os-release && echo "$VERSION_CODENAME")" != "noble" ]; then
  echo "Expected Ubuntu 24.04 (noble). Aborting." >&2
  exit 1
fi

echo "== Locale (UTF-8) =="
sudo apt-get update
sudo apt-get install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

echo "== ROS 2 apt source =="
sudo apt-get install -y software-properties-common curl
sudo add-apt-repository -y universe
ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F '"tag_name"' | awk -F'"' '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.noble_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

echo "== ROS 2 Jazzy + Gazebo Harmonic + dev tools =="
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y \
  ros-jazzy-desktop \
  ros-dev-tools \
  ros-jazzy-ros-gz \
  ros-jazzy-xacro \
  ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-gz-ros2-control \
  python3-colcon-common-extensions python3-rosdep python3-vcstool \
  python3-venv python3-pip build-essential cmake git

echo "== rosdep =="
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
rosdep update

echo
echo "Done. Next:"
echo "  cd ~/ros2_ws && source ros_env.sh && ros2 doctor --report | head -20"
echo "  gz sim shapes.sdf     # Gazebo GUI smoke test"
