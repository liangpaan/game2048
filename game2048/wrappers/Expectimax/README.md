# Expectimax 方案

参考 Brian Galebach 的 [Solving the 3x3 Variant of 2048（2014）](https://www.probabilitysports.com/2048.html)，默认完整求解至自然终局，无需训练或预训练模型。只在显式设置 `depth` 时使用有限深度启发式近似。

## 完整终局求解

玩家节点取合法动作的最大价值，随机节点按真实生成概率求期望。设移动后的棋盘有 `k` 个空位：

```text
Q(s, a) = Σ_{i 属于空位} [0.9 × V(spawn(move(s,a), i, 2))
                       + 0.1 × V(spawn(move(s,a), i, 4))] / k

V(s) = sum(s) + target_bonus × I(max(s) ≥ target)    无合法移动时
       max_{a 属于合法动作} Q(s, a)                  其他情况
```

默认 `target=256`、`target_bonus=256`，最大化期望终局总和与达标概率的加权目标，和当前 TD 的终局效用一致。`target_bonus=0` 时只优化期望终局总和。达到目标后继续，不停止搜索或游戏。方块不能变小，因此当前最大块即可判断是否曾达标。生成方块已包含在终局总和中，不再累加单步奖励或传统合并分数。

每次合法移动后的随机生成使棋盘总和严格增加，因此状态转移无环，可用记忆化递归求解。八种旋转/镜像中取同一规范表示，玩家节点和随机节点分别缓存。C++17 内核将九格方块编码为四位指数，按行查表完成移动，使用双精度价值。

从全部 144 种两块初始棋盘展开后，得到 **48,713,519 个对称归一化的玩家状态、31,431,374 个随机节点**。玩家状态数与原文一致；默认搜索不采样、不裁剪概率分支、不使用启发式叶节点，数值精度受浮点运算限制。

## 编译与缓存

Linux 环境需要 C++17 编译器 `c++` 或 `g++`，也可通过 `CXX` 指定。首次调用自动编译 `native/solver.cpp`，并计算对应 `target/target_bonus` 的完整状态表。本机首次求解约 20 秒；预留约 3 GiB 内存。求解完成后释放随机节点表，只保留玩家价值表用于查询。

缓存默认放在仓库根目录的 `.cache/expectimax/`，每个目标配置约 1 GiB。可设置 `GAME2048_EXPECTIMAX_CACHE=/path/to/cache` 更换目录。文件名区分目标和奖励，写入后原子替换；文件锁防止多个进程同时生成同一张表。缓存是本机生成文件，不提交到 Git，删除后会重新计算。

后续进程直接加载玩家价值表，每个动作只需要对后续生成位置的价值查表求期望。本机缓存后 100 局含启动、加载和绘图约 3 秒，峰值内存约 1.1 GiB。时间和内存随硬件及运行条件变化。没有编译器或希望使用较小内存时，可显式设置 `--depth 3` 使用 Python 有限深度版本。

## 接口与命令

```python
from game2048.wrappers.Expectimax import ExpectimaxAgent, ExpectimaxConfig

agent = ExpectimaxAgent()  # depth=None，完整终局求解
board = (2, 2, 0, 4, 8, 0, 16, 32, 64)
values = agent.action_values(board)  # 合法动作的期望终局效用
action = agent.choose(board)        # 0 左，1 右，2 上，3 下

sum_only = ExpectimaxAgent(ExpectimaxConfig(target_bonus=0))
limited = ExpectimaxAgent(ExpectimaxConfig(depth=3))
```

完整求解支持 3×3，输入空格为 0、方块为 2 至 1024 的二次幂。动作始终对应输入棋盘方向；平局按左、右、上、下选取。同一棋盘和配置的结果确定。无合法动作时 `action_values()` 返回空字典，`choose()` 抛出 `ValueError`。不同局复用同一目标配置的状态表。

```bash
# 默认完整求解及默认一次性达标奖励
uv run game2048-play --algorithm expectimax --mode eval --episodes 20 --seed 10000 --plot docs/expectimax-evaluation-curve.png
uv run game2048-play --algorithm expectimax --mode watch --seed 5001

# 只最大化期望终局数字总和，单独计算并缓存状态表
uv run game2048-play --algorithm expectimax --mode eval --target-bonus 0 --episodes 100 --seed 10000

# 原来的有限深度近似，仅用于对比或低内存环境
uv run game2048-play --algorithm expectimax --mode eval --depth 3 --episodes 20 --seed 10000
```

支持现有日志、曲线、`--max-steps`、演示截图和 `plot` 模式。训练模式不适用。默认没有步数截断，达到 256 后继续。

## 有限深度近似

显式指定正整数 `depth` 时，一个深度单位包括一次移动及随机生成。到达深度边界且仍有合法动作时，用共用的总和、空位、蛇形排列及达标奖励启发式估计价值；默认 `empty_weight=18`、`order_weight=1`。这些启发式权重仅影响有限深度模式。自然终局仍使用实际终局效用，缓存包含节点类型与剩余深度，并在每次决策后清空。

旧版深度 3 在种子 10000–10019 上平均总和仅 325.1、10/20 达标，原因是搜索时域短且依赖蛇形启发式。原文强调 3×3 最优策略更偏向围绕角落组织大块；默认完整求解直接计算后续终局，无需手工指定角落布局。

## 评测与原文口径

默认完整求解、`target_bonus=256`，各局均自然结束：

| 种子 | 局数 | 平均终局总和 | 中位总和 | 最高总和 | 达到 256 | 达到 512 | 平均步数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10000–10019 | 20 | 761.0 | 833 | 1016 | 20/20 | 11/20 | 344.55 |
| 10000–10099 | 100 | 851.44 | 961 | 1022 | 100/100 | 80/100 | 385.33 |

100 局包含前 20 局。逐局数据与配置在 [评测数据](../../../docs/expectimax-evaluation.json)，20 局曲线在 [评测曲线](../../../docs/expectimax-evaluation-curve.png)。样本全部达标不表示理论上必然达标。

原文图 2 的 **6872 是累计合并分数**，九格数字和为 **1018**。原文分别求解了最大化 256/512/1024 达标概率和最大化期望合并分数四个目标；其 99.571% 达到 256、73.677% 达到 512、期望合并分数 5468.5 分别来自对应目标的策略。本项目采用相同的完整求解思路，但优化终局总和与达标奖励，不将这些不同目标的理论数值作为本项目成绩。

单元测试用独立 Python 终局穷举核对小型残局的原生价值，验证有/无达标奖励、达标后继续、2/4 概率和移动内核；另保留有限深度搜索的穷举对照、八种对称动作映射与 CLI 回归。单元测试不要求生成完整状态表。
