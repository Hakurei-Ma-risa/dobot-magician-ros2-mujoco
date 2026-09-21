# Dobot Magician ROS 2 + MuJoCo + Portable Manipulation

面向 Dobot Magician 的一套可迁移操作栈：ROS 2 Humble 负责真机通信，MuJoCo
负责无实机验证，Portable Mani 提供与机器人型号无关的任务、感知和策略接口。
当前版本已经在真机上完成吸盘抓取、90° 搬运和放置，并验证了 Intel RealSense
D435i 的 RGB-D 数据通路。

> **安全声明**：本仓库会控制真实机械臂。首次运行必须低速、清空工作区并准备好
> 断电。README 中的真机坐标来自一台具体设备和桌面标定，不能直接复制到另一套
> 机械臂或安装环境。

## 当前状态

| 模块 | 状态 | 已验证内容 |
|---|---|---|
| ROS 2 Humble | 可用 | Ubuntu 22.04、系统 Python 3.10 |
| Dobot Magician | 可用 | Homing、PTP、状态反馈、报警、吸盘 |
| MuJoCo | 可用 | 4-DOF 模型、FK 对比、Gymnasium、抓取评估 |
| RealSense D435i | 可用 | RGB、深度、对齐深度、CameraInfo、USB 3 |
| Portable Mani | 原型可用 | 通用接口、Dobot adapter、MuJoCo backend、RGB-D localizer |
| LeRobot / ACT | 未集成 | 建议后续通过通用 observation/action contract 接入 |

真机测试结果：

- `/dobot_TCP` 与 `/dobot_joint_states` 稳定约 `20 Hz`；
- `/dobot_alarms` 稳定约 `10 Hz`，抓放过程中为空；
- 已验证吸盘抓取点 `(210, 0, -65) mm`；
- 已验证右侧放置点 `(0, -210, -65) mm`；
- 已验证复位位姿 `(150, 0, 100, 0)`；
- 自由空间搬运已测试至 `70%` 速度，接触阶段仍限制为低速。

上述坐标只对测试当天的安装位置、吸盘和桌面高度有效。

## 系统结构

```text
                           +-------------------+
D435i RGB-D -------------->| mani_perception   |----> /mani/scene_objects
                           +-------------------+
                                      |
Policy / YOLO / ACT / RL ---> mani_interfaces <--- mani_tasks
                                      |
                         +------------+------------+
                         |                         |
                   mani_dobot                mani_mujoco
                         |                         |
                  magician_ros2                 MuJoCo
                         |
                   USB serial port
                         |
                  Dobot Magician
```

Portable Mani 的上层接口使用米、弧度、`PoseStamped` 和 capability mask；Dobot
专用的毫米、角度、4-DOF 约束都留在 embodiment adapter 内。以后迁移到 Tron 2
或其他机械臂时，应新增 adapter，而不是重写感知、任务和策略层。

## 目录

```text
.
├── mujoco_sim/                    # MuJoCo 模型、Gymnasium 环境和验证脚本
├── src/
│   ├── magician_ros2/             # Dobot ROS 2 Humble 驱动及本项目补丁
│   └── portable_mani_ros2/
│       ├── mani_interfaces/       # 通用 msg/action/service
│       ├── mani_core/             # 与 ROS 无关的数据结构、安全和轨迹逻辑
│       ├── mani_tasks/            # 通用抓取任务状态机
│       ├── mani_perception/       # D435i 与 RGB-D 定位接口
│       ├── mani_dobot/            # Dobot adapter 与硬件 bridge
│       └── mani_mujoco/           # 与真机端点一致的 ROS 2 仿真后端
└── README.md
```

详细文档：

- [MuJoCo 环境](mujoco_sim/README.md)
- [URDF 审计](mujoco_sim/URDF_AUDIT.md)
- [Portable Mani 总览](src/portable_mani_ros2/README.md)
- [Dobot adapter](src/portable_mani_ros2/mani_dobot/README.md)
- [MuJoCo ROS 后端](src/portable_mani_ros2/mani_mujoco/README.md)
- [D435i 感知](src/portable_mani_ros2/mani_perception/README.md)
- [上游 Magician ROS 2 文档](src/magician_ros2/README.md)

## 环境边界

不要让 Conda 的 Python 3.12/3.13 覆盖 ROS 2 Humble 的 Python 3.10。

| 用途 | 推荐环境 |
|---|---|
| ROS 2、Dobot 真机、D435i | 退出 Conda，`/usr/bin/python3` 3.10 |
| 独立 MuJoCo、Gymnasium | Conda `dobot-mujoco`，Python 3.12 |
| ROS 2 + MuJoCo 同进程 | Python 3.10 `--system-site-packages` venv |

检查 ROS 环境：

```bash
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
which python3
python3 --version
```

应看到 `/usr/bin/python3` 和 Python 3.10。

## 安装与构建

### 1. 系统依赖

```bash
sudo apt update
sudo apt install -y \
  python3-colcon-common-extensions python3-pip python3-rosdep \
  ros-humble-xacro \
  ros-humble-realsense2-camera ros-humble-realsense2-description
```

克隆后在仓库根目录安装 ROS 依赖并构建：

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install --user -r src/magician_ros2/requirements.txt
colcon build --symlink-install
source install/setup.bash
```

### 2. MuJoCo 环境

```bash
conda create -n dobot-mujoco python=3.12 -y
conda activate dobot-mujoco
python -m pip install -r mujoco_sim/requirements.txt
python -m pip install -e mujoco_sim
python -m pip install -e src/portable_mani_ros2/mani_core
python -m pip install -e src/portable_mani_ros2/mani_dobot
```

## MuJoCo 验证

先运行无窗口检查：

```bash
conda activate dobot-mujoco
python -m pytest mujoco_sim/tests
python mujoco_sim/scripts/verify_model.py --samples 1000
python -m mani_dobot.mujoco_demo
```

打开可视化预演：

```bash
python mujoco_sim/scripts/preview_deploy_sequence.py \
  --duration 1.2 --dwell 0.5 --hold-seconds 3
```

运行随机物理抓取评估：

```bash
python -m mani_dobot.pick_evaluator --trials 100
python -m mani_dobot.pick_evaluator --trials 1 --viewer
```

模型适合验证接口、运动学、控制周期和任务逻辑。质量、惯量、摩擦和执行器动态仍是
近似值，不能当作高保真数字孪生直接做 sim-to-real 承诺。

## RealSense D435i

确认相机运行在 USB 3：

```bash
lsusb | grep -i realsense
lsusb -t
```

启动经过测试的 ROS 2 profile：

```bash
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch mani_perception d435i.launch.py
```

检查数据：

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
ros2 topic echo --once /camera/camera/color/camera_info
ros2 run mani_perception camera_healthcheck --ros-args -p duration_sec:=15.0
```

默认图像 QoS 为 `SENSOR_DATA`，即 `BEST_EFFORT + VOLATILE`。Humble 的
`ros2 topic hz` 不一定支持命令行 `--qos-reliability`，直接运行上述命令即可。

## Dobot 真机

### 1. 串口权限

```bash
sudo usermod -aG dialout <username>
```

注销并重新登录后检查：

```bash
groups
ls -l /dev/ttyACM* /dev/ttyUSB* /dev/serial/by-id/* 2>/dev/null
```

优先使用稳定的 `/dev/serial/by-id/...`，不要依赖可能变化的 `ttyACM0`。

### 2. 启动吸盘配置

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

export DOBOT_PORT=/dev/serial/by-id/<your-dobot-device>
export MAGICIAN_TOOL=suction_cup

ros2 launch dobot_bringup dobot_magician_control_system.launch.py
```

本项目对上游串口层做了两项关键修改：

1. `DOBOT_PORT` 环境变量支持稳定设备路径；
2. 使用进程间文件锁串行化多个 ROS 节点对同一 UART 的访问，并清理残留 ACK，
   避免状态、动作和报警节点互相读走响应。

### 3. 动作前只读检查

```bash
ros2 node list
ros2 topic echo --once /dobot_TCP
ros2 topic echo --once /dobot_alarms
ros2 topic hz /dobot_TCP
ros2 topic hz /dobot_alarms
/usr/bin/python3 mujoco_sim/scripts/validate_ros_live.py --seconds 5
```

只有在报警为空、反馈稳定且工作区无人时才允许动作。

### 4. 基本命令

复位到测试位姿，速度和加速度 20%：

```bash
ros2 action send_goal /PTP_action dobot_msgs/action/PointToPoint \
"{motion_type: 1, target_pose: [150.0, 0.0, 100.0, 0.0], \
velocity_ratio: 0.2, acceleration_ratio: 0.2}"
```

吸盘开关：

```bash
ros2 service call /dobot_suction_cup_service \
  dobot_msgs/srv/SuctionCupControl "{enable_suction: true}"

ros2 service call /dobot_suction_cup_service \
  dobot_msgs/srv/SuctionCupControl "{enable_suction: false}"
```

服务返回成功只表示控制命令已发送，不代表形成真空。必须连接吸盘的 GP1 线，并在
首次使用时确认能听到气泵启动声。

## 已验证吸盘抓放流程

测试桌面上的标定流程如下：

| 阶段 | 目标 `[x, y, z, yaw]` mm/deg | 建议速度 |
|---|---:|---:|
| 正前方高位 | `[210, 0, 50, 0]` | 70% |
| 抓取预接近 | `[210, 0, -55, 0]` | 自由空间速度 |
| 抓取接触 | `[210, 0, -65, 0]` | 5% 以下 |
| 右侧高位 | `[0, -210, 50, 0]` | 70% |
| 放置预接近 | `[0, -210, -55, 0]` | 自由空间速度 |
| 放置接触 | `[0, -210, -65, 0]` | 5% 以下 |
| 垂直抬离 | `[0, -210, -15, 0]` | 自由空间速度 |

实践规则：

1. 自由空间动作可以较快，但接触阶段必须慢；
2. 吸盘接触后等待约 2 秒再试提；
3. 第一次只抬高 10 mm，目视确认吸住后再升到搬运高度；
4. 放置时先接触桌面，再关闭吸盘，最后垂直抬离；
5. Dobot 没有末端力传感器，也没有可靠的真空闭环反馈，不能盲目下压；
6. 更换桌面、工具或安装位置后必须重新标定接触高度。

## Portable Mani 快速入口

启动现有 Dobot 驱动后，在第二个终端运行通用 bridge：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch mani_dobot dobot_portable_bridge.launch.py
ros2 topic echo --once /mani/capabilities
```

通用接口包括：

- `/mani/move_to_pose`
- `/mani/command_end_effector`
- `/mani/capabilities`
- `/mani/scene_objects`
- `/mani/pick_object`

仿真侧提供相同接口，参见
[mani_mujoco/README.md](src/portable_mani_ros2/mani_mujoco/README.md)。

## 常见问题

### `Permission denied: /dev/ttyACM0`

确认当前登录会话已经拥有 `dialout` 组。执行 `usermod` 后，仅打开新终端通常不够，
需要注销登录或重启。

### ROS 节点存在但状态话题停止

确认使用的是本仓库修改后的 `dobot_driver`，并且重新执行了 `colcon build` 和
`source install/setup.bash`。多个进程访问串口时必须启用本项目的进程间锁补丁。

### 吸盘服务成功但抓不起来

依次检查 GP1 线、气泵声音、软管、吸盘是否压实，以及目标表面是否光滑不透气。
普通纸张、粗糙纸板和弯曲物体可能无法形成密封。

### MuJoCo 窗口关闭时出现 `exit 139`

如果日志已经打印 `PREVIEW_PASS`，崩溃发生在 GLFW/viewer 释放阶段，不代表轨迹
验证失败。无窗口测试和 ROS 仿真不受影响。

### D435i 帧率波动

确认相机位于 `5000M` USB 3 链路，避免 USB 2 Hub，并检查图像 QoS 是否为
`BEST_EFFORT + VOLATILE`。

## 来源与许可证

`src/magician_ros2` 基于
[jkaniuka/magician_ros2](https://github.com/jkaniuka/magician_ros2)，按其
[MIT License](src/magician_ros2/LICENSE) 保留许可和归属。本项目新增的 Portable
  Mani packages 在各自 `package.xml` 中声明 Apache-2.0。模型网格和第三方代码仍受
  各自原始许可证约束；本仓库不以根目录许可证覆盖这些组件。
