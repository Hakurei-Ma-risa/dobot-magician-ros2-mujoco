# Gemini Robotics ER simulation adapter

This package has a single-frame diagnostic probe and a MuJoCo-only interactive
terminal. Gemini Robotics ER proposes target points and boxes from the
base-mounted RGB camera. RGB-D localization and portable actions execute
simulated picks and placements. No physical Dobot is controlled.

Install the optional client only in the MuJoCo environment:

```bash
python -m pip install google-genai pillow
export GEMINI_API_KEY='...'
```

Inspect the request without using the API:

```bash
python -m mani_gemini.probe \
  --image mujoco_sim/artifacts/magician_mujoco.png \
  --task 'find the card and propose a safe center point' \
  --dry-run
```

For a live MuJoCo camera frame, use the project-local Python 3.10 ROS venv.
Build `mani_gemini`, `mani_perception`, and `mani_mujoco` with that venv active. Then launch the
simulation with its perfect 2-D detector disabled:

```bash
source /opt/ros/humble/setup.bash
source .venv_ros_mujoco/bin/activate
source install/setup.bash
ros2 launch mani_mujoco dobot_mujoco.launch.py \
  viewer:=false publish_camera:=true publish_perfect_detections:=false \
  localizer_support_anchor:=bbox_bottom camera_width:=640 camera_height:=480
```

In another terminal with the same ROS environment and `GEMINI_API_KEY` set:

```bash
ros2 run mani_gemini gemini_er_sim_probe
```

The live probe requires the image publisher to be `mani_mujoco_server`. It
publishes one ER detection, waits for `/mani/scene_objects`, compares the
estimated 3-D pose with simulation ground truth for scoring, then clears its
detection. Ground truth is not used to create or localize the detection.

The default task targets the yellow cylinder, which is the physical pick object
in this MuJoCo scene. The red sphere in the overview snapshot is a target
marker. The API can take several seconds, so this probe assumes a stationary
scene during the request. The bottom-edge anchor corrects for the cylinder's
upper half being hidden by the tool. One 640x480 run localized the cylinder
within 0.71 mm of MuJoCo ground truth; this is a single-scene result, not an
accuracy guarantee. The diagnostic probe does not execute a pick.

## Interactive clutter simulation

Launch `ros2 launch mani_mujoco dobot_clutter.launch.py` in one terminal. In a
second terminal, source ROS Humble, `.venv_ros_mujoco`, and `install/setup.bash`,
then run `ros2 run mani_gemini gemini_sim_chat` with `GEMINI_API_KEY` set. Run
`/help` for examples; `/find 紫色方块`, `/pick 紫色方块`, `/place 右侧空旷位置`, and
`/reset 42` are supported. Plain text asks a scene question.

The terminal checks that its RGB publisher is the MuJoCo server and that the
perfect detector is off. It subscribes to RGB, aligned depth, CameraInfo, TF,
and the RGB-D localizer output, but not simulator ground truth or full state.
Known object dimensions are a geometry prior. A CSRT tracker updates the 2-D
box between Gemini calls; when post-pick tracking fails, Gemini can reacquire
the held object from a fresh frame for lift verification. Placement validates
an empty depth patch and robot kinematic reachability. This is a research
prototype: one successful seed=0 block pick/place is not a reliability claim,
and the simulated gripper differs from the real suction tool.
