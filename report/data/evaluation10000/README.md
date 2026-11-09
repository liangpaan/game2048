# 3×3 2048：每方法10,000局冻结策略评测

本目录是这次汇报的结果来源。`expectimax`为完整Expectimax，`td`为乐观5-tuple TD，`mcts`为128次模拟、深度12的MCTS。

每个`.csv.gz`包含10,000行数据，列为`seed,sum,max_tile,reached_256,steps`。布尔值为`True/False`，总和是终局所有数字之和。种子均为3000000至3009999，达到256后继续至自然终局，无步数截断。未筛除失败局。

各`_summary.json`记录均值、样本标准差、总和均值95%正态近似区间、成功率95% Wilson区间、最大块分布、实际配置、版本及源文件/权重哈希。`raw_csv_sha256`对应解压后的原始CSV字节。

```python
import csv, gzip, statistics
with gzip.open('report/data/evaluation10000/td.csv.gz', 'rt', newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 10000
print(statistics.fmean(int(r['sum']) for r in rows))
print(sum(r['reached_256'] == 'True' for r in rows) / len(rows))
```

复现程序与MCTS一致性记录见[评测说明](../../benchmark10000/README.md)。环境随机源与搜索随机源分开；相同种子不意味着不同策略每步会在同一格落块，因为空格集合随动作变化。

TD复用原300,000局检查点；MCTS无需训练；完整搜索复用求解缓存。本次只增加测试样本，没有统一三种方法的训练与推理计算量。并行运行、进程数不同，且MCTS使用加速内核，`wall_seconds`不能用来比较方法速度；完整搜索首次建表不计入该字段。区间只反映冻结策略的评测抽样误差，不涵盖重新训练的波动。
