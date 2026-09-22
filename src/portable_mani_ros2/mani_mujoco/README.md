# mani_mujoco

This package exposes the Dobot MuJoCo model through exactly the same portable
ROS 2 endpoints used by the hardware adapter:

- `/mani/move_to_pose`
- `/mani/command_end_effector`
- `/mani/capabilities`
- `/mani/scene_objects` (RGB-D reconstructed target pose)
- `/mani/sim/ground_truth` (evaluation only)
- `/mani/perception/detections_2d` (perfect simulator detector)
- `/mani/pick_object` (provided by `mani_tasks`)
- `/mani/sim/reset_scene` (simulation-only evaluation service)

With `publish_camera:=true` (the default), the same process also publishes the
robot-independent RGB-D contract used by the real D435i:

- `/camera/camera/color/image_raw`
- `/camera/camera/color/camera_info`
- `/camera/camera/depth/image_rect_raw`
- `/camera/camera/aligned_depth_to_color/image_raw`
- `/camera/camera/aligned_depth_to_color/camera_info`

MuJoCo RGB, metric depth and instance segmentation are rendered by a fixed
eye-to-hand camera. The target object's perfect segmentation bounding box is
published on `/mani/perception/detections_2d`. The generic RGB-D localizer
combines that box with aligned depth and publishes `/mani/scene_objects` for
the pick server. MuJoCo's true pose is isolated on `/mani/sim/ground_truth`
for scoring and is never consumed by the task layer. A static TF connects
`magician_base_link` to `camera_color_optical_frame`, so downstream perception
code can be shared with the real camera.

The simulation defaults to `320x240@10 Hz` to keep RGB, depth and instance
rendering responsive on a single Python process. Consumers must use
`CameraInfo` rather than hard-coded intrinsics, so the real D435i's 640x480
stream is a drop-in replacement. Override with `camera_width:=640
camera_height:=480` when image fidelity matters more than throughput.
The RGB-D path uses EGL; the interactive viewer uses GLFW. The clutter launch
runs the viewer in a separate process, allowing both a live window and the
camera stream. Use the default `viewer:=false publish_camera:=true` for legacy
perception and closed-loop evaluation, or `viewer:=true publish_camera:=false`
to inspect single-object motion visually.
Visual-only mode feeds simulator ground truth directly to the pick server;
only the default headless mode is a perception-driven benchmark.

## Base-camera clutter scene

`ros2 launch mani_mujoco dobot_clutter.launch.py` starts five randomized
tabletop objects, a simulated base-mounted D435i RGB-D feed at 640x480,
RGB-D localizer, top-down pick server, and separate MuJoCo viewer. The perfect
detector and `/mani/sim/ground_truth` are disabled. The viewer alone receives
`/mani/sim/qpos`; perception clients must not subscribe to it. The localizer
projects the depth ROI into the robot base and estimates tabletop footprint
midpoints. It falls back to the legacy single-point method when needed.

`/mani/sim/reset_scene` requires `randomize: true` in clutter mode. The seed
controls object arrangement; the response omits object poses. This scene is
still idealized geometry and RGB-D, not a calibrated RealSense or a suction
gripper digital twin.

The project-local `.venv_ros_mujoco` uses system Python 3.10 and inherits ROS
Humble packages. It exists because the original `dobot-mujoco` Conda env uses
Python 3.12 and cannot load Humble's Python 3.10 extensions.

## Run the visual ROS simulation

Terminal 1:

```bash
cd /home/hongjin/Documents/Codex/2026-09-17/new-chat/outputs/dobot_magician_ros2
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch mani_mujoco dobot_mujoco.launch.py viewer:=true publish_camera:=false
```

Terminal 2:

```bash
cd /home/hongjin/Documents/Codex/2026-09-17/new-chat/outputs/dobot_magician_ros2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 service call /mani/sim/reset_scene mani_interfaces/srv/ResetScene \
"{randomize: true, seed: 42}"

ros2 action send_goal /mani/pick_object mani_interfaces/action/PickObject \
"{object_uuid: pick-object-0, group_name: arm, end_effector_name: gripper, \
approach_distance_m: 0.08, lift_distance_m: 0.09, \
velocity_scale: 0.5, acceleration_scale: 0.5}" --feedback
```

Use the default launch arguments for headless RGB-D runs.

## Randomized physical-pick evaluation

The faster in-process evaluator runs without ROS transport overhead:

```bash
conda activate dobot-mujoco
python -m mani_dobot.pick_evaluator --trials 100
```

To watch one randomized physical pick:

```bash
python -m mani_dobot.pick_evaluator --trials 1 --viewer
```

The currently validated tabletop region is `x=[0.20, 0.28] m` and
`y=[-0.12, 0.12] m` in `magician_base_link`.

## RGB-D closed-loop evaluation

With `dobot_mujoco.launch.py` running in another terminal, evaluate randomized
perception-driven physical picks (MuJoCo ground truth is used only for scoring):

```bash
ros2 run mani_mujoco closed_loop_evaluator --ros-args \
  -p trials:=20 -p max_localization_error_m:=0.006 \
  -p min_success_rate:=0.95
```

The JSON report contains each RGB-D localization error, visual lift estimate
and physical-pick result.

## Rebuild the Python ROS packages

Build the packages that launch MuJoCo with the project-local Python 3.10 venv:

```bash
source /opt/ros/humble/setup.bash
source .venv_ros_mujoco/bin/activate
python -m colcon build --symlink-install \
  --packages-select mani_interfaces mani_core mani_tasks mani_dobot mani_mujoco
```
