# Portable Mani ROS 2

This source tree is the first executable slice of a robot-agnostic manipulation
runtime.  Dobot Magician is the first embodiment; task, perception and policy
interfaces deliberately avoid Dobot-specific joint names.

Packages:

- `mani_interfaces`: stable ROS messages/actions/services shared by all robots.
- `mani_core`: transport-independent capability, state, trajectory and safety types.
- `mani_tasks`: robot-independent top-down pick planning and execution state machine.
- `mani_perception`: robot-independent RGB-D launch contract and camera preflight.
- `mani_dobot`: Dobot kinematics, MuJoCo backend and ROS 2 bridge to `magician_ros2`.
- `mani_mujoco`: ROS 2 simulation server exposing the same endpoints as hardware.

The portability boundary is intentional:

```text
task / perception / policy
          |
    mani_interfaces
          |
 embodiment adapter (mani_dobot today, mani_tron2 later)
          |
 simulation or hardware driver
```

See `mani_dobot/README.md` for hardware/reach commands and
`mani_mujoco/README.md` for the visual ROS pick workflow. See
`mani_perception/README.md` for the verified D435i launch and health check.
