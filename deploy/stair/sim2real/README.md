# G1 sim2real 部署步骤

适用于 G1 29 自由度、腰部解锁、PR 控制、pelvis IMU（wxyz）和 RealSense 深度相机。
以下命令均在项目根目录执行。

## 1. 准备文件与环境

将 `deploy/` 和 `outputs/sim2sim/v2_model_46000/` 复制到部署主机，保持目录结构。
模型目录内必须保留 `policy.onnx`、`manifest.json`、`robot.zip` 三个文件。

使用 Python 3.10+，安装运行依赖：

```bash
python -m pip install -r deploy/stair/sim2real/requirements.txt
```

另按部署主机的系统和架构安装：

- RealSense SDK 和 `pyrealsense2`。
- Unitree SDK2 Python 及其 CycloneDDS 依赖。
- 如需 `--show-depth`，安装 `opencv-python` 并使用可用的桌面环境。

相机采集和控制进程需运行在同一台主机上。

## 2. 核对配置

编辑 [g1_real.json](g1_real.json)：

- `camera.serial`：多台相机时指定设备序列号。
- `camera.width/height/fps`：默认 `480×270@60 Hz`，需与设备支持的深度流一致。
- `camera.mount_pos/mount_quat_wxyz`：实际安装位姿必须与模型配置一致；采用 X前/Y左/Z上坐标系，相对于 torso，四元数顺序为 wxyz。仅修改数值不能校正实际安装误差。
- `joint_limits` 和 `control`：核对关节范围、超时、站姿过渡及停止参数。

使用其他配置文件时，在启动命令中添加 `--config 配置文件路径`。
以下命令中的 `eth0` 替换为与 G1 通信的实际网卡名。

## 3. 检查深度相机

仅采集相机，不连接 G1：

```bash
python deploy/stair/sim2real/g1_stair_sim2real.py \
  --bundle outputs/sim2sim/v2_model_46000 \
  --mode depth --duration 20 --record outputs/sim2real/depth_001
```

有桌面环境时可加 `--show-depth` 查看深度图。
`--record` 必须指定新目录；重复运行时更换目录名。

## 4. 检查状态读取与策略推理

此模式读取机器人状态并推理，不发送电机控制指令：

```bash
python deploy/stair/sim2real/g1_stair_sim2real.py \
  --bundle outputs/sim2sim/v2_model_46000 --interface eth0 \
  --mode monitor --duration 30 --record outputs/sim2real/monitor_001
```

确认相机、状态读取和推理正常后，再进入控制模式。

## 5. 启动实机控制

先让机器人进入低层/debug 模式，程序不会自动释放运动服务。
当前仅完成离线验证；首次上机需有支撑和独立急停，先验证站姿与停止行为。

```bash
python deploy/stair/sim2real/g1_stair_sim2real.py \
  --bundle outputs/sim2sim/v2_model_46000 --interface eth0 \
  --mode control --record outputs/sim2real/control_001
```

使用 G1 自带无线遥控器：

| 操作 | 作用 |
| --- | --- |
| 按 A | 默认用 3 秒过渡到站姿，然后保持 |
| 摇杆回中，再按 R1 | 站姿过渡完成且深度历史满 700 ms 后启动策略 |
| 左摇杆向前 | 前进 |
| 右摇杆左右 | 转向 |
| 按住 B | 速度指令归零；松开后恢复摇杆控制 |
| 按 R2 或 Ctrl+C | 停止并退出；恢复需重新启动程序 |

启动时已按住的 A/R1 需先松开再按。
退出时，若已发送控制指令且机器人状态有效，程序短暂发送阻尼命令后停发；
阻尼不保证站立，需保持外部支撑。B 仅将速度指令归零，不退出策略控制。
