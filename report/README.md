# 实验报告与课程汇报

| 材料 | 文件 | 内容 |
| --- | --- | --- |
| 演示文稿 | [game2048_presentation.pdf](game2048_presentation.pdf) | 南开 Beamer 模板，正文10页，问答备用2页 |
| 汇报讲稿 | [game2048_speaker_notes.pdf](game2048_speaker_notes.pdf) | 逐页讲稿、5分钟时间安排和问答准备，共4页 |
| 实验报告 | [game2048_experiment_report.pdf](game2048_experiment_report.pdf) | 宋体小四、单倍行距，共5页，含方法、结果、超参数讨论和成员分工 |

演示稿依次介绍完整 Expectimax、MCTS、TD 和 PPO。TD 部分比较乐观5-tuple、CNN 和普通 n-tuple 三种价值表示。

测试使用相同的20个种子，达到256后继续运行至自然终局。训练和搜索预算尚未统一，榜单反映当前实现的结果。

原简版报告见 [report.md](report.md)，实验数据位于 [data](data/)。

## 超参数讨论

新增[详细讨论模块](hyperparameter_discussion.md)，整理相同训练局数下的对称特征、模式结构、初始化与奖励对照，以及训练预算、探索率和固定权重的搜索深度实验。历史500局/300局对照与最终20局榜单分别统计；未重新训练模型。

[精简数据索引](data/hyperparameter_evidence.json)包含15个配置、4组配对区间和原始归档摘要。正式报告第3.1节保留关键表格与分析，可直接修改[对应LaTeX模块](latex/report/sections/hyperparameters.tex)。

## LaTeX 源码

三份材料的可编辑源码见 [latex/README.md](latex/README.md)，其中包含编译方法、字体要求和图表、公式的修改位置。

- [实验报告](latex/report/main.tex)
- [演示文稿](latex/slides/main.tex)
- [汇报讲稿](latex/speech/main.tex)
