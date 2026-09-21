# mani_dobot

`mani_dobot` is the first embodiment adapter for the Portable Mani runtime. It
has two paths with identical task-space semantics:

- `DobotMujocoBackend` for local simulation and tests.
- `dobot_bridge` for the existing `magician_ros2` hardware stack.

The generic pose action uses metres, radians, ROS `PoseStamped`, and an explicit
six-axis capability mask. Dobot accepts only `x/y/z/yaw`; requests for roll or
pitch are rejected instead of being projected silently.

## Build the ROS packages

```bash
cd /home/hongjin/Documents/Codex/2026-09-17/new-chat/outputs/dobot_magician_ros2
conda deactivate  # repeat if necessary; ROS must report /usr/bin/python3
source /opt/ros/humble/setup.bash
which python3
colcon build --symlink-install \
  --packages-select mani_interfaces mani_core mani_dobot
source install/setup.bash
```

`which python3` must print `/usr/bin/python3`. Do not build ROS packages from
the Conda base environment (its Python 3.13 is incompatible with Humble).

## Simulation smoke test

The existing `dobot-mujoco` Conda environment already provides MuJoCo and the
validated model. Install the two pure Python packages once in editable mode:

```bash
conda activate dobot-mujoco
python -m pip install -e src/portable_mani_ros2/mani_core
python -m pip install -e src/portable_mani_ros2/mani_dobot
python -m mani_dobot.mujoco_demo
```

The default command is a fast headless numerical smoke test. To watch the same
trajectory in the interactive MuJoCo viewer, run:

```bash
python -m mani_dobot.mujoco_demo --viewer
```

Close the viewer window to exit. For an automatically closing window, add for
example `--hold-seconds 5`.

For physical-contact pick evaluation rather than reaching only:

```bash
python -m mani_dobot.pick_evaluator --trials 100
python -m mani_dobot.pick_evaluator --trials 1 --viewer
```

The complete ROS graph and `/mani/pick_object` command are documented in
`../mani_mujoco/README.md`.

## Hardware path

Start the existing Magician stack first, then start only the portable bridge:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

# Terminal 1: existing driver stack (only after the read-only probe passes)
ros2 launch dobot_bringup dobot_magician_control_system.launch.py

# Terminal 2: generic façade
ros2 launch mani_dobot dobot_portable_bridge.launch.py

# Inspect the latched capability description
ros2 topic echo --once /mani/capabilities
```

After the real-robot read-only checks pass, send a conservative first goal:

```bash
ros2 action send_goal /mani/move_to_pose mani_interfaces/action/MoveToPose \
"{group_name: arm, target: {header: {frame_id: magician_base_link}, pose: \
{position: {x: 0.20, y: 0.0, z: 0.15}, orientation: {w: 1.0}}}, \
controlled_axes: [true, true, true, false, false, true], \
position_tolerance_m: 0.005, orientation_tolerance_rad: 0.01745, \
velocity_scale: 0.2, acceleration_scale: 0.2}" --feedback

ros2 service call /mani/command_end_effector \
mani_interfaces/srv/CommandEndEffector \
"{end_effector_name: gripper, command: 0, hold: false}"
```

The bridge refuses commands when TCP feedback is missing or older than 250 ms,
when alarms are active, when a target is outside the configured workspace, or
when roll/pitch is requested. A zero tolerance field selects the adapter
defaults of 5 mm and 1 degree.

Do not send a `/mani/move_to_pose` goal until the USB, joint-map and TCP checks
in `mujoco_sim/README.md` have passed on the real robot.
