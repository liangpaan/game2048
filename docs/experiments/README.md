# 3×3 2048 实验材料索引

## 当前保留的三个版本

- [选择依据、训练配置、模型和原始数据 SHA256](selected/selection.json)
- [500局共同测试原始数据](selected/raw/)、[恢复后的逐局一致性验证](selected/verification/summary.json)
- [TD 30万局训练曲线数据](selected/td-training.json)
- [完整规划求解记录](selected/solve.json)
- [完整规划独立10000局数据](history/round3/final/champion_holdout.csv)、[汇总与配对区间](history/round3/final/holdout_summary.json)
- [本次提交的检查记录](validation.json)

当前默认只提供 `mc`、`td`、`best`。历史方法与原仓库 CNN/PPO 接口保留用于复核旧结果；它们不属于本次选出的三个版本。

## 完整历史探索

[完整58项榜单](history/leaderboard/LEADERBOARD.md)包含45个去重配置、5个加长训练对照、2个训练种子对照与6个仓库基线条目，不代表58种独立算法。

- [榜单 CSV](history/leaderboard/leaderboard.csv)、[详细统计 JSON](history/leaderboard/summary.json)
- [逐策略500局原始数据](history/leaderboard/raw/)、[逐策略统计](history/leaderboard/results/)
- [评测协议](history/leaderboard/manifest.json)、[原配置与检查点来源](history/leaderboard/inventory.json)
- [配对比较](history/leaderboard/paired-comparisons.json)、[完整性检查](history/leaderboard/integrity.json)
- [第一轮](history/exploration/)、[加长训练](history/refinement/)、[阶段切换](history/phase/)、[第二轮](history/round2/)、[最后一轮](history/round3/)、[训练种子对照](history/stability/)

归档报告保留当时原文。JSON 中 `artifacts/...`、旧模型路径、旧运行命令及绝对路径是来源记录，不能直接用作本仓库当前命令。报告所需的统计与原始数据已放到对应 `history/<批次>/` 下；除所选模型外，不重复发布其余历史权重和大型求解缓存。

## 预算可比性与取消的实验

历史58项榜单统一的是测试规则、测试种子及局数，训练局数、交互步数、CPU耗时和推理预算不相同，不能用于证明同等资源下某算法更优。

后来进行了按环境交互步数统一预算的补充对照。用户要求停止时，原网络 PPO 仍未全部完成；因此没有完整新榜单，也没有用该批模型替换所选版本。

- [取消说明](cancelled-resource-comparison/CANCELLED.md)
- [停止前进度](cancelled-resource-comparison/PROGRESS.md)
- [已完成的长预算 TD 子实验](cancelled-resource-comparison/extended_td/REPORT.md)

这里只保留补充实验的状态和报告，不发布未完成 PPO 模型、临时环境或运行目录。补充报告中的链接可能指向原本地研究目录，不是当前模型入口。
