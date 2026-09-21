import mujoco
import numpy as np

from dobot_mujoco.model import load_model, reset_pick_object
from mani_mujoco.rgbd_camera import MujocoRgbdCamera


def test_rgbd_camera_observes_pick_object() -> None:
    model = load_model()
    data = mujoco.MjData(model)
    reset_pick_object(model, data, [0.22, 0.0])
    camera = MujocoRgbdCamera(model, width=320, height=240)
    try:
        frame = camera.render(data)
    finally:
        camera.close()

    assert frame.rgb.shape == (240, 320, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.depth_m.shape == (240, 320)
    assert frame.depth_m.dtype == np.float32
    assert frame.object_bbox_xywh is not None

    x, y, width, height = frame.object_bbox_xywh
    assert width > 2 and height > 2
    center_x = x + width // 2
    center_y = y + height // 2
    assert 0.48 < float(frame.depth_m[center_y, center_x]) < 0.56

    depth_mm = camera.depth_millimeters(frame.depth_m)
    assert depth_mm.dtype == np.uint16
    assert 480 < int(depth_mm[center_y, center_x]) < 560
