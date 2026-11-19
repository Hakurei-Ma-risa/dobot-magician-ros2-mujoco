# mani_vision — find objects, choose one, pick it

Course package for the vision module chain. Each module is one ROS node and
talks to the next one only through a topic (its **contract**). Keep the
contract and you can replace the inside of a module without touching the
others.

```text
① camera ──► ② find_objects ──► ③ choose_target ──► ④ rgbd_localizer ──► ⑤ pick task ──► ⑥ robot
  (MuJoCo D435i or real D435i)        ▲ instruction                                   (MuJoCo or Dobot)
```

| # | Module | Node | In | Out |
|---|---|---|---|---|
| ① | camera | `mujoco_server` (sim) / `realsense2_camera` | – | `/camera/camera/color/image_raw`, `/camera/camera/aligned_depth_to_color/image_raw`, `/camera/camera/color/camera_info`, TF camera → `magician_base_link` |
| ② | find objects | `mani_vision find_objects` | depth, camera info, TF | `/mani/vision/candidates` (`SceneObjectArray`: `cand-0`, `cand-1`, … with box, position, size, yaw) |
| ③ | choose target | `mani_vision choose_target` | candidates, colour image, `/mani/vision/instruction` (`std_msgs/String`) | `/mani/perception/detections_2d` (one object, uuid `target`), `/mani/vision/choice` (JSON), `/mani/vision/marked_image` |
| ④ | locate in 3-D | `mani_perception rgbd_localizer` | detections, depth, camera info, TF | `/mani/scene_objects` |
| ⑤ | pick | `mani_tasks pick_server` | `/mani/scene_objects`, action goal | action `/mani/pick_object` |
| ⑥ | robot | `mujoco_server` / `dobot_bridge` | `/mani/move_to_pose`, `/mani/command_end_effector` | arm motion |

## Run it in simulation

Every terminal first: `source ~/dobot_ws/dobot-magician-ros2-mujoco/setup_env.bash`

```bash
# Terminal 1 — whole chain, MuJoCo window, rule-based chooser
ros2 launch mani_vision course_sim.launch.py

# Terminal 2 — one command does: park the arm → choose → back to ready → pick
ros2 run mani_vision pick "the purple block"
ros2 service call /mani/sim/reset_scene mani_interfaces/srv/ResetScene "{randomize: true, seed: 3}"
ros2 run mani_vision pick "the green block"

# See what the chooser saw (numbered boxes, chosen one in red)
ros2 run rqt_image_view rqt_image_view /mani/vision/marked_image
```

The same steps by hand, one contract at a time:

```bash
ros2 topic echo --once /mani/vision/candidates --field objects     # ② output
ros2 topic pub --once /mani/vision/instruction std_msgs/msg/String "data: 'the green block'"
ros2 topic echo --once /mani/vision/choice                         # ③ output
ros2 topic echo --once /mani/scene_objects                         # ④ output, uuid "target"
ros2 action send_goal /mani/pick_object mani_interfaces/action/PickObject \
  "{object_uuid: target, group_name: arm, end_effector_name: gripper, approach_distance_m: 0.05, lift_distance_m: 0.05, velocity_scale: 0.5, acceleration_scale: 0.5}"
```

(By hand, park the arm first or the arm itself shows up as a candidate; `pick`
does that for you.)

## Swap module ③: rule → DeepSeek

```bash
export DEEPSEEK_API_KEY=sk-...        # the course key, never commit it
ros2 launch mani_vision course_sim.launch.py selector:=deepseek
ros2 run mani_vision pick "the cube that looks like a grape"
```

- **rule** (default): free and offline. Understands colours (red, orange,
  yellow, green, blue, purple …) and left / right / nearest / farthest /
  biggest / smallest / tallest / shortest. Anything else → the nearest object.
- **deepseek**: draws the numbered boxes on the image and asks
  `deepseek-flash` which number matches the sentence. The model answers with
  a number only; position, size and yaw still come from depth, so the AI never
  invents coordinates. About 2 s and well under RM0.01 per choice.

Nothing else changes: same topics in, same topic out.

## How each module works (short)

- **find_objects** turns every depth pixel into a 3-D point in the robot base
  frame, takes the most common height as the table, keeps points 8 mm–20 cm
  above it inside the reachable ring (12–34 cm from the base) and groups
  touching pixels into blobs. It does not know what objects are. Each blob's
  top-view rectangle gives the yaw, so the gripper closes across the short side.
- **choose_target** picks one blob, then keeps following it from frame to
  frame (nearby in x/y, lowest one, still the same colour) so the pick task can
  check that the object really went up. A new instruction drops the old target.
- **rgbd_localizer** turns the chosen box + depth into the object's 3-D centre
  and reports a lifted object where it really is.

## Limits you will meet

- The arm must be out of the camera view while choosing (`pick --park …`).
  The defaults fit the MuJoCo scene (first straight up, so the gripper does not
  knock the yellow cylinder next to its rest pose); on a real setup pass your own park and
  ready paths: `--park "x,y,z;x,y,z" --ready "x,y,z;…"` (metres, base frame).
- Dobot reach: close to the base (r < 0.21 m) the arm cannot go much above
  z = 0.12 m, and at r ≈ 0.31 m not above ≈ 0.13 m. That is why `pick` uses a
  5 cm approach and lift, and why the 9 cm cans at the edge of the MuJoCo
  table are found and chosen but usually cannot be picked from above.
- The lift check can say `FAILED at verify` although the object is in the
  gripper: with the arm in view, the depth image may join the lifted object,
  the arm and a neighbour into one blob, and the follower loses the target.
  Look at the simulator window. (Measured: purple block 8/8 picked and
  confirmed; yellow cylinder 8/8 picked, confirmed 1 of 4 after the fix.)
- Fast DDS must have the large shared-memory profile (`setup_env.bash` sets
  it), or 640×480 images arrive at about 0.4 Hz instead of 10 Hz.
- Real camera: the camera → robot TF must come from a calibration. Without a
  good calibration, use the simulation.
