# G1 Stair 部署

本项目仅包含 sim2sim（MuJoCo 仿真）和 sim2real（G1 实机）部署代码。
策略通过 ONNX Runtime 推理，无需 Torch、MJLab、RSL-RL 或训练数据。

- `deploy/stair/sim2sim/`：仿真、深度处理、策略推理和手柄控制。
- `deploy/stair/sim2real/`：实机通信、相机、控制、回放和离线测试。
- `outputs/sim2sim/`：仅保留 `v2_model_46000` 部署 bundle。

在项目根目录安装并运行 sim2sim：

```bash
python -m pip install -e '.[sim2sim,gamepad]'
python deploy/stair/sim2sim/g1_stair_sim2sim.py \
  --bundle outputs/sim2sim/v2_model_46000 \
  --joystick --joystick-type xbox \
  --duration 0 \
  --follow-camera
```

运行需要完整 bundle：`policy.onnx`、`robot.zip` 和 `manifest.json`。
机器人模型与网格已包含在 `robot.zip` 中。训练源码、AMP 动作数据和依赖训练环境的
checkpoint/ONNX 导出工具已移除；新模型需在独立训练项目中导出完整 bundle 后复制过来。

实机部署步骤与硬件依赖见 [sim2real](deploy/stair/sim2real/README.md)。

离线验证（不连接硬件）：

```bash
STAIR_TEST_BUNDLE=outputs/sim2sim/v2_model_46000 \
  python -m unittest discover -s deploy/stair/sim2real/tests -v
```
