# Dobot Magician URDF 检查记录

检查对象：`magician_ros2/dobot_description` 中的 4-DOF、普通夹爪配置，并以用户提供的 Dobot 官网产品图作为外形参考。

## 结论

仓库没有拿错机器人模型。DAE 网格呈现的底座外壳、肩部电机、双侧平行四边形连杆、水平腕部和竖直夹爪，与官网图中的 Dobot Magician 一致。之前 MuJoCo 截图不相似，是因为第一版 MJCF 只使用胶囊体表达运动学主链，并不是 URDF 网格错误。

上游 URDF 可以被 ROS `check_urdf` 成功解析，关节树为：

```text
root
└── base_link                 固定，肩关节高度 0.131305 m
    └── joint_1 / link_1      Z 轴偏航
        └── joint_2 / link_2  Y 轴后臂俯仰
            └── joint_3 / link_3
                └── mimic_1 / virtual
                    └── mimic_2 / link_4
                        └── joint_4 / gripper_core
                            ├── prismatic_l / left jaw
                            └── prismatic_r / right jaw
```

## 几何核对

| 项目 | URDF/Xacro 数值 | 检查结果 |
|---|---:|---|
| 肩关节离安装平面 | 131.305 mm | 与网格基座对齐 |
| 后臂运动学长度 | 135 mm | 与 ROS FK 一致 |
| 前臂运动学长度 | 147 mm | 与 ROS FK 一致 |
| wrist 到 joint_4 | 60 mm | 与 Xacro 一致 |
| 普通夹爪 flange 到最低指尖 | 104.843 mm | 从官方 DAE 顶点测得 |
| 关节 1 | ±125° | 一致 |
| 原始 theta2 | -5°～90° | 一致 |
| 原始 theta3 | -15°～70° | 一致，但不能直接作为 URDF joint_3 的限位 |
| 关节 4 | ±150° | 一致 |

MuJoCo 现已加载由上游 DAE 按原始材质颜色转换得到的 19 个 OBJ visual parts；简化胶囊体和 box 仅作为不可见碰撞/惯量近似。因此外观使用原模型，碰撞和动力学仍保持可控、快速。

## 发现的问题

### 1. 第三关节语义与限位不一致

`dobot_state_updater` 向 `/joint_states` 发布：

```text
joint_3 = theta3 - theta2
```

但 Xacro 给 `joint_3` 的限位仍写成原始 theta3 的 `[-15°, 70°]`。如果 theta2 和 theta3 分别取合法范围，relative joint 的固定包络应为：

```text
[-15° - 90°, 70° - (-5°)] = [-105°, 75°]
```

固定包络只能防止 URDF/RViz 错误拒绝合法姿态；真正的机械限制仍必须分别检查原始 theta2、theta3，不能只检查 relative joint。

### 2. 右夹爪上下限倒置

上游写成：

```xml
<limit lower="0" upper="-0.0135"/>
```

合法写法应为 `lower="-0.0135" upper="0"`。MuJoCo 模型已经采用合法范围，并保持 `right = -left`。

### 3. 两个 mimic 关节的限位与实际符号不符

`joint_mimic_1` 和 `joint_mimic_2` 都写成 `[0, 2π]`，但 multiplier 为 `-1`，正常姿态会产生负值。部分可视化程序忽略 mimic joint limits，因此不一定立刻报错；规划器或严格 URDF 转换器可能出问题。

### 4. 没有任何 inertial 定义

所有运动 link 均缺少质量、质心和惯量。`check_urdf` 仍可解析，但 PyBullet 会为每个 link 临时假定 `mass=1`、单位惯量；直接从该 URDF 做动力学训练会得到错误结果。

### 5. collision 重用了高面数 visual DAE

主 Xacro 的 visual 和 collision 默认引用同一个网格。仓库虽然另有简化 collision DAE，但当前主模型没有使用它们。这样会增加碰撞检测成本，也可能产生不稳定接触。

### 6. virtual link 使用零尺寸 box

虚拟连杆的 visual/collision 是 `box size="0 0 0"`。ROS parser 接受，但部分转换器不接受零尺寸几何；更稳妥的做法是完全省略该 link 的 visual/collision，仅保留 link 与 joint。

### 7. 当前系统缺少 Xacro 命令

本机已安装 `ros-humble-urdf`，`check_urdf` 成功；但尚未安装 `ros-humble-xacro`，因此不能从 `.urdf.xacro` 重新生成 URDF。需要补装：

```bash
sudo apt install ros-humble-xacro
```

## 本次采取的处理

- 没有擅自修改上游 ROS 源码，避免改变真机栈行为。
- MuJoCo 中修正了 relative joint、mimic 和夹爪范围。
- 使用上游 DAE 生成真实外观，保留简化、不可见的碰撞体。
- 将普通夹爪 `tcp_site` 放到 DAE 实际最低指尖：flange 下方 104.843 mm。
- 重新运行 5 个单元测试、1000 个随机姿态 FK 和动态控制测试，全部通过。

