"""Display a ROS MuJoCo server in a separate GLFW process."""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from dobot_mujoco.model import load_model
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class StateViewer(Node):
    def __init__(self) -> None:
        super().__init__("mani_mujoco_state_viewer")
        self.declare_parameter("scene_mode", "single")
        self._latest_qpos: np.ndarray | None = None
        self.create_subscription(
            Float64MultiArray, "/mani/sim/qpos", self._on_state, 10
        )

    def _on_state(self, message: Float64MultiArray) -> None:
        self._latest_qpos = np.asarray(message.data, dtype=np.float64)

    def run(self) -> None:
        model = load_model()
        data = mujoco.MjData(model)
        if str(self.get_parameter("scene_mode").value) == "clutter":
            marker_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "target_marker"
            )
            data.mocap_pos[model.body_mocapid[marker_id]] = [2.0, 2.0, 2.0]
        viewer = mujoco.viewer.launch_passive(model, data)
        try:
            viewer.cam.lookat[:] = [0.23, 0.0, 0.08]
            viewer.cam.distance = 0.68
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -35
            while rclpy.ok() and viewer.is_running():
                rclpy.spin_once(self, timeout_sec=0.02)
                if self._latest_qpos is not None:
                    if self._latest_qpos.shape != data.qpos.shape:
                        raise RuntimeError(
                            "simulator and viewer have different model dimensions"
                        )
                    data.qpos[:] = self._latest_qpos
                    mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(1.0 / 30.0)
        finally:
            viewer.close()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StateViewer()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
