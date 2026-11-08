# TD

参数在 `config/td.yaml`，NumPy 模型在 `models/TD/`。默认模型为 30 万局乐观 5-tuple TD（查表，不是 CNN）；原 CNN、原 n-tuple 也保留。所有评测都使用 NumPy，不需要 PyTorch。

```bash
uv run python -m game2048.wrappers.TD
uv run python -m game2048.wrappers.TD --config config/td-cnn.yaml
uv run python -m game2048.wrappers.TD --model models/TD/td-3x3-ntuple.npz
uv run --extra td python -m game2048.wrappers.TD.train --model models/TD/my-td.npz
uv run --extra td python -m game2048.wrappers.TD.train --resume --model models/TD/my-td.npz
uv run python -m game2048.wrappers.TD.train --network ntuple --model models/TD/my-ntuple.npz
```

`__main__.py` 默认评测 20 局；`train.py` 负责训练及续训，`training.network` 选择 CNN 或 n-tuple。四窗口演示入口是 `uv run game2048/play.py`，每个方法只运行一局并停留在最终棋盘。

CNN 的参数及优化器状态都以 NumPy 数组保存，元数据为 JSON；加载评测模型不导入 torch。CNN 续训恢复优化器和动作 RNG，但不保存进行中的棋局和 FIFO 队列。目标、达标奖励和价值缩放不能在续训时改变。

默认 5-tuple 在当前种子 10000–10019 的 20 局为平均 sum 612.1、通过 20/20；原 CNN 为 418.8、16/20。另有一组独立协议的 500 局复测为 639.408、463/500。模型来源与校验值在 `models/TD/optimistic-5tuple-300k.source.json`。

当前训练入口支持原 CNN 和原 n-tuple；导入的乐观 5-tuple 检查点用于评测，当前入口不复现该检查点的训练流程。CNN/PPO 仍保留可选训练依赖，基础运行环境不安装它们。
