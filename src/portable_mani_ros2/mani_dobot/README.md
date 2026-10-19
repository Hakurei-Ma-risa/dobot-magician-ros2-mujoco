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

## Fast taught suction pick-and-place

`fast_pick_place` runs the taught A -> B/C/D suction task from
`reports/calibration/2026-10-17/task_waypoints.json`. Each cycle is queued on
the Dobot controller in one go (MOVL up, across at `--safe-z`, down, suction,
dwell), so moves run back to back. R is held at the taught pick_A value, so the
block does not rotate. It talks to the serial port directly, shares the driver's
UART lock, and can run while `dobot_state_publisher` is up. No other ROS node is
needed.

```bash
cd /home/hongjin/Documents/Codex/2026-09-17/new-chat/outputs/dobot_magician_ros2
source /opt/ros/humble/setup.bash && source .venv_ros_mujoco/bin/activate && source install/setup.bash

ros2 run mani_dobot fast_pick_place --dry-run          # print queued steps only
ros2 run mani_dobot fast_pick_place --check            # read pose/alarms/tool bias
ros2 run mani_dobot fast_pick_place --targets B        # Challenge 1
ros2 run mani_dobot fast_pick_place                    # Challenge 3: A->B, A->C, A->D
```

Between cycles it waits for Enter while you put the next block at A (`q` quits),
or for `--refill-delay SECONDS`. Defaults: `--speed 150 --z-speed 80 --accel 150`
(mm/s, mm/s^2), `--grip-ms 400 --release-ms 400`. Before moving it checks the
controller tool bias is the taught 59.7 mm and clears latched alarms. While
moving it stops on a new alarm or a stalled queue. Ctrl+C force-stops and
clears the queue and leaves the suction as it is.

## Lab tools (2026-10)

These talk to the Dobot serial port directly (same UART lock as the driver) and
prompt in Chinese; teaching and runs advance on Enter. Usage, the two-robot
conveyor flow and the hardware facts measured on 2026-10-17 are documented in
the repository README, section "实验室真机工具（2026-10）".

| Command | Purpose |
|---|---|
| `dobot_home 1\|2` | manual homing; live J2/J3, clears alarms once the pose is legal |
| `sensor_check` | manual check of GP2 colour, GP4 input, EIO1-20 levels and the Stepper1 belt |
| `chip_tray teach\|park\|view\|click\|status\|move` | taught 15x6 chip tray, moves chips by cell |
| `conveyor_sort teach\|vision\|run\|release` | two-robot belt sorting; Gemini on a D435i frame gives the colour |

Modules: `fast_pick_place.Dobot` (queued serial driver with MOVL/MOVJ, homing,
thread-safe), `chip_tray` (grid fit, firmware IK incl. the parallelogram limit,
path checks), `belt_vision` (D435i frame via ROS, Gemini request, logs),
`conveyor_sort` (plans, teach flows, runner).
