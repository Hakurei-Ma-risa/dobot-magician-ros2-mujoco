# Source this file in EVERY new terminal before using the robot or the sim:
#
#     source ~/dobot_ws/dobot-magician-ros2-mujoco/setup_env.bash
#
# It loads ROS 2 Humble, this project's Python venv and build, and a DDS
# setting that lets 640x480 camera images through at full rate.

_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
if [ -f "$_REPO/.venv_ros_mujoco/bin/activate" ]; then
  source "$_REPO/.venv_ros_mujoco/bin/activate"
else
  echo "setup_env: no venv yet ($_REPO/.venv_ros_mujoco); see README 'ROS 2 + MuJoCo venv'"
fi
if [ -f "$_REPO/install/setup.bash" ]; then
  source "$_REPO/install/setup.bash"
else
  echo "setup_env: not built yet; run: cd $_REPO && python -m colcon build --symlink-install"
fi
export FASTRTPS_DEFAULT_PROFILES_FILE="$_REPO/config/fastdds_large_images.xml"
unset _REPO
