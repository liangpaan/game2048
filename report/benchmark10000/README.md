# 万局评测复现

该入口只评测完整Expectimax、MCTS与乐观5-tuple TD。生产环境和原有算法文件不变；所有真实对局仍调用`game2048.runner.play_episode`与原环境包装类。

实际运行环境：WSL Ubuntu、Python 3.12.14、g++、依赖见requirements.txt。从仓库根目录运行：

```sh
python -m pip install -r report/benchmark10000/requirements.txt
python report/benchmark10000/run.py --validate
python report/benchmark10000/run.py --method mcts --episodes 10000 --seed 3000000 --workers 8
python report/benchmark10000/run.py --method td --episodes 10000 --seed 3000000 --workers 6
GAME2048_EXPECTIMAX_CACHE="$HOME/.cache/game2048-10000" python report/benchmark10000/run.py --method expectimax --episodes 10000 --seed 3000000 --workers 1
```

需保留仓库原`models/TD/optimistic-5tuple-300k.npz`权重。完整搜索需要先构建或加载求解缓存，首次运行的时间、内存及磁盘开销明显较大，入口固定为单进程以免复制价值表。不要将生成的求解缓存或编译产物提交到Git。

输出写入本目录被忽略的`results/`，每局一行JSONL并实时落盘，另生成汇总JSON。本次实际30,000局记录以压缩CSV存放在[结果目录](../data/evaluation10000/)，避免产生数万行PR差异。种子、自然终局、目标与统计规则见结果说明。

## MCTS加速实现的一致性

Python原MCTS计算较慢，因此另提供仅用于本次评测的C++内核`native_mcts.cpp`。保留128次模拟、深度12、探索系数70、模拟策略、启发式、死局罚分、动作顺序与CPython随机流；不是另一个参评方法。当前入口固定使用原默认参数，不支持任意自定义MCTS配置。

`--validate`比较：10,000组随机原语；194个有合法动作的随机棋盘的移动、启发式、最终动作及完整随机状态；3局原环境完整对局的322次动作与随机状态。结果全部一致，记录见[equivalence.json](equivalence.json)，并用源文件SHA256绑定。评测前必须先运行验证，验证通过才接受对应加速源码。

实现细节：Python赋值语句先计算右侧，因此原模拟落块先抽取2/4再抽取位置；C++保持该顺序。真实环境落块仍由原环境负责。随机验证与少量完整对局是对照证据，不宣称形式化证明。
