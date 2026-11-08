# 2048 算法实验

## 运行

所有命令均通过 `uv run` 执行。四种算法的默认模型已经转换为 NumPy `.npz`，评测和四窗口只需要基础依赖，无需安装 PyTorch 或 SB3。

```bash
# 同时打开 MCTS、Expectimax、TD（默认 5-tuple）、PPO 四个窗口
uv run game2048/play.py

# 各方法独立做无界面评测，默认 20 局
uv run python -m game2048.wrappers.MCTS
uv run python -m game2048.wrappers.Expectimax
uv run python -m game2048.wrappers.TD
uv run python -m game2048.wrappers.ppo

# 修改次数、种子或保存结果
uv run python -m game2048.wrappers.TD --episodes 100 --seed 10000 --output results/td.json
uv run python -m game2048.wrappers.TD --model models/TD/td-3x3-ntuple.npz
```

## 目录

```text
config/
├── mcts.yaml               # MCTS 参数、评测次数和种子
├── expectimax.yaml         # Expectimax 参数及状态表目录
├── td.yaml                 # 默认 30 万局 5-tuple TD；保留原训练参数
├── td-cnn.yaml             # 原 CNN-TD 对照配置
├── ppo.yaml                # PPO 模型、训练及教师参数
└── player.yaml             # 四窗口布局、速度及结束行为
models/
├── MCTS/                   # 无需训练权重
├── Expectimax/             # 原生求解器自动生成的状态表
├── TD/
│   ├── optimistic-5tuple-300k.npz  # 30 万局默认乐观 5-tuple 模型
│   ├── optimistic-5tuple-300k.source.json # 校验值及复测摘要
│   ├── td-3x3-cnn.npz      # 原 CNN NumPy 权重及训练状态
│   └── td-3x3-ntuple.npz   # n-tuple NumPy 特征表
└── ppo/
    └── ppo-3x3.npz         # PPO NumPy 权重及训练状态
game2048/
├── play.py                 # 四窗口单局演示入口
├── core.py                 # 确定性移动规则
├── runner.py               # 配置加载、模型加载及统一评测
├── utils/                  # 跨模块复用的工具
│   ├── common.py           # 搜索启发式及随机发块
│   └── weights.py          # NPZ 存取及 NumPy CNN 运算
├── envs/                   # 原版环境
└── wrappers/
    ├── env.py              # 3×3、256 目标、独立 RNG 适配
    ├── MCTS/               # agent.py、__main__.py
    ├── Expectimax/         # agent.py、__main__.py、原生求解器
    ├── TD/                 # agent.py、ntuple.py、optimistic.py、train.py、__main__.py
    └── ppo/                # agent.py、network.py、env.py、train.py、__main__.py
tests/
```

## 配置、测评与训练结果

参数集中在 `config/*.yaml`。自定义 YAML 只需填写覆盖字段，其余沿用默认配置，未知字段会报错。模型、状态表和输出路径均按项目根目录解析；命令行 `--config` 文件按执行目录解析。

- `evaluation.episodes`：runner 默认评测 20 局。
- `evaluation.seed`：默认首局种子 10000；第 i 局使用 seed + i。
- `evaluation.output`：可选 JSON 路径；null 表示不写文件。
- `agent`：搜索参数。Expectimax 的 `depth` 为 null 时完整求解，正整数时有限深度搜索。
- `model`：对应方法的 `.npz` 文件；TD 根据文件结构识别 CNN、n-tuple 或乐观 5-tuple。
- `training`：训练预算、优化器、探索及续训参数。

下表使用相同的 3×3 实验环境和种子 10000–10019，固定模型各评测 20 局。`sum` 是自然终局棋盘数字总和的平均值，`≥256` 是最大块达到 256 或以上的局数。训练阶段结果只描述训练过程中的最后窗口或采样表现，不能替代固定模型评测。

| 方法 | 训练预算 / 训练阶段结果 | 20 局平均 sum | ≥256 | 平均步数 |
| --- | --- | ---: | ---: | ---: |
| MCTS | 无训练；每步 128 次模拟、深度 12 | 266.00 | 6/20 | 119.45 |
| 完整 Expectimax | 无训练；完整终局求解 | 761.00 | 20/20 | 344.55 |
| TD 乐观 5-tuple | 300,000 局固定检查点；无当前训练窗口记录 | 612.10 | 20/20 | 277.60 |
| TD CNN | 50,000 局自我对弈；最后 500 局平均 sum 406.59，≥256 为 369/500 | 418.80 | 16/20 | 188.50 |
| TD n-tuple | 5,000 局自我对弈；最后 500 局平均 sum 330.41，≥256 为 259/500 | 384.00 | 14/20 | 172.15 |
| PPO | 行为克隆 500 局、12 轮，再训练 50,176 个环境步；训练采样 626 局平均 sum 180.43，≥256 为 63/626 | 211.40 | 2/20 | 94.55 |

CNN、n-tuple 和 PPO 的训练阶段统计来自训练过程中的探索或采样策略；评测阶段使用固定模型、贪心动作和自然终局，不更新参数、不设置步数上限。不同方法的训练预算和在线计算量不同，表格适合查看当前检查点表现，不代表等成本比较。

```bash
uv run python -m game2048.wrappers.MCTS --config config/my-mcts.yaml
uv run python -m game2048.wrappers.TD --episodes 20 --output results/td.json
```

## 四窗口

`game2048/play.py` 是四窗口演示入口：每个方法只运行一局，结束后停留在最终棋盘，手动关闭窗口即可退出；`config/player.yaml` 指定方法配置、窗口尺寸、动作速度 `fps` 以及结束前暂停时间。批量评测请使用各方法的 runner 入口，例如 `uv run python -m game2048.wrappers.TD --episodes 20`。

窗口显示棋盘、sum、最大块、步数、已完成棋局的平均 sum 和通过次数。模型加载及搜索在后台执行。关闭一个窗口后，其余窗口继续；终端按 Ctrl+C 关闭全部窗口。默认完成后保留最终棋盘。

Expectimax 的完整状态表仍使用原生求解器的二进制缓存，不是神经网络权重。首次完整求解需要 C++17 编译器和约 3 GiB 内存，状态表约 1 GiB，保存在 `models/Expectimax/`；编译产物在 `.cache/expectimax/`。

## NPZ 模型与训练

默认运行环境没有安装 PyTorch/SB3，运行、评测及四窗口均不需要它们。`pyproject.toml` 中仍保留 `td`、`ppo` **可选训练依赖**，因为当前 CNN/PPO 的梯度训练和续训仍调用 PyTorch。NPZ 只是保存格式，不能替代训练框架。旧 `.pt`、`.pkl`、`.zip` 权重及本地备份已经删除。

CNN TD、n-tuple TD、PPO 的权重均保存为 NumPy 数组，元数据使用 JSON，以 `allow_pickle=False` 加载。CNN/PPO 的优化器状态和训练计数也保存在同一个 `.npz` 中，以支持续训。评测使用 NumPy 前向计算；PyTorch 和 SB3 仅在训练时加载。

```bash
# CNN TD
uv run --extra td python -m game2048.wrappers.TD.train --model models/TD/my-td.npz
uv run --extra td python -m game2048.wrappers.TD.train --resume --model models/TD/my-td.npz

# n-tuple TD，无需训练依赖
uv run python -m game2048.wrappers.TD.train --network ntuple --model models/TD/my-ntuple.npz

# PPO
uv run --extra ppo python -m game2048.wrappers.ppo.train --model models/ppo/my-ppo.npz
uv run --extra ppo python -m game2048.wrappers.ppo.train --resume --model models/ppo/my-ppo.npz
```

默认乐观 5-tuple 检查点作为已训练评测模型使用，当前 `train.py` 不复现它的训练过程。TD 的 `training.network` 仍选择 cnn 或原 ntuple；参数分别在 `training.cnn`、`training.ntuple`。新训练使用不存在的路径，续训使用 `--resume` 或 `training.resume`。CNN 续训不能更改目标、达标奖励或价值缩放。PPO 可用 `training.teacher_model` 和 `bc_episodes` 配置 TD 教师。

## 验证

```bash
# 验证基础运行环境、NPZ 推理和四窗口
uv run python -m unittest discover -s tests -v

# 额外验证训练、续训及 NumPy/PyTorch 数值一致性
uv run --extra ppo python -m unittest discover -s tests -v
```

原版环境 `game2048/envs/game2048.py` 保留原文件，实验适配在 wrappers 中完成。
