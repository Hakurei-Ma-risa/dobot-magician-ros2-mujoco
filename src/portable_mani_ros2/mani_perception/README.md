# Mani Perception

This package owns the robot-independent RGB-D input contract for Portable Mani.
It deliberately has no dependency on Dobot, MuJoCo, YOLO or a policy runtime.

## D435i launch

Keep ROS 2 outside Conda, disable USB autosuspend for the current D435i device,
then launch the tested profile:

```bash
conda deactivate
D435_USB_DEVICE=$(
  for device in /sys/bus/usb/devices/*; do
    if [ "$(cat "$device/idVendor" 2>/dev/null)" = "8086" ] &&
       [ "$(cat "$device/idProduct" 2>/dev/null)" = "0b3a" ]; then
      echo "$device"
      break
    fi
  done
)
test -n "$D435_USB_DEVICE" || echo "D435i not found"
test -z "$D435_USB_DEVICE" || echo on | sudo tee "$D435_USB_DEVICE/power/control"
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch mani_perception d435i.launch.py
```

The default contract is:

- RGB: `/camera/camera/color/image_raw`, `640x480`, nominal 15 Hz.
- raw depth: `/camera/camera/depth/image_rect_raw`, `640x480`, nominal 30 Hz.
- color-aligned depth: `/camera/camera/aligned_depth_to_color/image_raw`.
- color intrinsics: `/camera/camera/color/camera_info`.
- image QoS: ROS `SENSOR_DATA` (`BEST_EFFORT`, `VOLATILE`).

The lower RGB rate is intentional for the first reliable grasping baseline. It
leaves compute budget for object detection while depth continues at 30 Hz.
The launch explicitly disables all infrared image publishers; the depth engine
remains enabled, but unused IR streams do not consume ROS or USB bandwidth.
An optional `initial_reset:=true` launch argument exists for recovery tests, but
is off by default because a marginal USB link can renegotiate as USB 2 after a
reset.

## Health check

With the camera launch running, use a second ROS 2 terminal:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run mani_perception camera_healthcheck --ros-args -p duration_sec:=15.0
```

The command exits successfully only when RGB, raw depth, aligned depth,
intrinsics, USB 3 speed and USB power policy all satisfy the runtime contract.
It prints one JSON report suitable for CI logs or preflight scripts.

For the MuJoCo RGB-D provider, skip the physical USB requirement:

```bash
ros2 run mani_perception camera_healthcheck --ros-args \
  -p duration_sec:=10.0 -p require_d435_usb:=false -p min_color_hz:=5.0
```

## RGB-D localization

`rgbd_localizer` consumes detector output on
`/mani/perception/detections_2d`, color-aligned depth, `CameraInfo`, and TF. It
publishes robot-frame objects on `/mani/scene_objects`. For tabletop targets it
intersects the bbox-center camera ray with the configured support plane; depth
validates the ROI and takes over once the object has been lifted. The same node
works with MuJoCo segmentation today and a YOLO detector later.
