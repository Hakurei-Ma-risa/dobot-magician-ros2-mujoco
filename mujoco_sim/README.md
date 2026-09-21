# Dobot Magician：MuJoCo 无实机验证环境

这个目录用于在连接真机前验证 Dobot Magician 的关节语义、运动学、控制周期和 RL 接口。它使用独立 Conda 环境，不向 ROS 2 Humble 的系统 Python 3.10 安装 MuJoCo。

## 已覆盖的验证

- MuJoCo 3.13.0 可编译 MJCF；模型为 4 个主动旋转关节、1 个主动夹爪关节和 3 个 mimic/equality 约束。
- 外观使用上游 URDF 的 DAE 网格转换出的 19 个分材质 OBJ parts，与官网 Magician 轮廓一致。
- 固化 Dobot 固件角到 URDF/MuJoCo 角的转换：`q3_relative = theta3 - theta2`。
- 与仓库 `dobot_forward_kin.py` 使用同一组尺寸：基座高度 131.305 mm、后臂 135 mm、前臂 147 mm。
- 1000 个随机合法姿态对比解析 FK 和 MuJoCo site。
- 位置执行器、平行四边形约束、20 Hz 上层控制周期与 Gymnasium API。
- EGL 无窗口渲染，便于之后在训练脚本或 CI 中运行。

当前 visual 是上游 CAD 外观；碰撞几何、质量和惯量仍是保守近似值，适合接口、运动学和控制软件验证。在辨识真机质量、摩擦、减速器和电机响应前，不应把它当作高保真动力学数字孪生。URDF 的详细检查见 [URDF_AUDIT.md](URDF_AUDIT.md)。

## 运行

```bash
conda activate dobot-mujoco
cd /home/hongjin/Documents/Codex/2026-09-17/new-chat/outputs/dobot_magician_ros2/mujoco_sim

# 完整自动测试
python -m pytest
python scripts/verify_model.py --samples 1000

# 打开窗口，自动扫动 4 个关节和夹爪（默认 30 秒）
python scripts/view_motion.py --seconds 30

# Gymnasium 随机动作烟雾测试
python scripts/run_random_env.py --steps 100

# 无显示器渲染
MUJOCO_GL=egl python scripts/render_snapshot.py
```

若只想打开 MuJoCo 自带界面并手动操作 actuator control：

```bash
python -m mujoco.viewer --mjcf model/dobot_magician.xml
```

## 关节映射（最重要）

| 语义 | Dobot `/dobot_joint_states` | ROS `/joint_states` | MuJoCo |
|---|---:|---:|---:|
| 底座偏航 | `theta1` | `theta1` | `joint_1` |
| 后臂俯仰 | `theta2` | `theta2` | `joint_2` |
| 前臂绝对俯仰 | `theta3` | `theta3 - theta2` | `joint_3_relative` |
| 末端偏航 | `theta4` | `theta4` | `joint_4` |
| 左夹爪 | 0 或 13.5 mm | 同左 | `gripper_left` |
| 右夹爪 | 未发布 | `-left`（mimic） | `gripper_right = -left` |

不要把 ROS `/joint_states` 的第三项再减一次 `theta2`。模型 API `set_raw_pose()` 和 `control_raw_pose()` 接受的始终是 Dobot 原始格式 `[theta1, theta2, theta3, theta4]`，单位为弧度。

## 坐标参考点

- `base_pivot_site`：肩关节原点，世界坐标 `z = 0.131305 m`。
- `wrist_site`：147 mm 前臂末端，严格对应现有 ROS `dobot_forward_kin.py` 的输出。
- `flange_site`：从 wrist 水平向外 60 mm。
- `tcp_site`：普通夹爪 DAE 的最低指尖，从 flange 向下 104.843 mm。

真机 `/dobot_pose_raw` 和 `/dobot_TCP` 会受 Dobot 内部 tool offset 影响。明天应先在静止姿态记录它们，与 `wrist_site` 和 `tcp_site` 分别比较，再决定策略使用哪个 TCP；不要通过“看起来差不多”猜偏移。

## 已发现的上游模型风险

1. `dobot_state_updater` 正确地向 `/joint_states` 发布 `theta3 - theta2`，但 Xacro 给第三关节的固定限位仍是原始 `theta3` 的 `[-15°, 70°]`。相对角理论包络应为 `[-105°, 75°]`；本 MJCF 使用该包络，同时仍对原始 theta2/theta3 独立限位。
2. Xacro 中右夹爪关节写成 `lower="0" upper="-0.0135"`，上下限倒置。本 MJCF 使用合法范围 `[-0.0135, 0]`。
3. 上游 URDF 没有 inertial 数据，且网格为 DAE；因此不能直接作为可靠的 MuJoCo 动力学模型。本环境使用简化碰撞体和近似质量。
4. 驱动把串口硬编码为 `/dev/ttyUSB0`。若明天枚举成 `/dev/ttyUSB1`，必须修改 `dobot_driver/dobot_driver/dobot_handle.py` 或建立稳定 udev 名称。

## 明天接真机：先读后动

1. 退出 Conda，确认 ROS 使用 `/usr/bin/python3`：

   ```bash
   conda deactivate
   which python3
   python3 --version
   ```

2. 插入 USB、打开 Dobot 电源，只检查设备和权限，不发运动命令：

   ```bash
   groups
   ls -l /dev/ttyUSB* /dev/serial/by-id/* 2>/dev/null
   ```

3. 启动控制栈后先运行只读探针。它不发布消息、不发送动作，会检查 20 Hz 频率、第三关节映射以及真机 TCP/FK：

   ```bash
   cd /home/hongjin/Documents/Codex/2026-09-17/new-chat/outputs/dobot_magician_ros2
   source /opt/ros/humble/setup.bash
   source install/setup.bash
   /usr/bin/python3 mujoco_sim/scripts/validate_ros_live.py --seconds 5
   ```

   预期结果为 `status: PASS`。若只有 FK 失败，先确认控制栈确实执行了 `set_end_effector_params(0, 0, 0)`，再检查 TCP/tool offset，而不是立即改关节符号。
4. 清空机械臂周围空间、随时准备断电，再执行 homing；不要让其它程序同时控制机械臂。
5. 第一次动作只用低速、小幅度、单关节目标。比较真机原始角、ROS 角和 MuJoCo 预测；方向或零位不一致就停止，不进入采集/训练。
6. 运动学与夹爪通过后再接 D435i，并固定相机序列号、分辨率、帧率和时间戳配置。

## 与 LeRobot / ACT 的边界

这个 Gymnasium 环境验证的是低维控制通路，不是 ACT 训练器。推荐保持三个边界：ROS 2 Python 3.10 负责硬件；LeRobot Python 3.12 负责数据和策略；两者通过 ROS topic、共享内存或明确的数据记录格式交换。之后接 ACT 时，动作字段继续保存 Dobot 原始四角或笛卡尔 TCP 二选一，并在数据集 metadata 中写清单位和关节映射。
