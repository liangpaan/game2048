# 实验报告与课程汇报

| 材料 | 文件 | 内容 |
| --- | --- | --- |
| 演示文稿 | [game2048_presentation.pdf](game2048_presentation.pdf) | 南开 Beamer 模板，正文10页，问答备用2页 |
| 汇报讲稿 | [game2048_speaker_notes.pdf](game2048_speaker_notes.pdf) | 逐页讲稿、5分钟时间安排和问答准备，共4页 |
| 实验报告 | [game2048_experiment_report.pdf](game2048_experiment_report.pdf) | 宋体小四、单倍行距，共4页，含方法、结果、超参数讨论和成员分工 |

演示稿依次介绍完整 Expectimax、MCTS、TD 和 PPO。TD 部分比较乐观5-tuple、CNN 和普通 n-tuple 三种价值表示。

测试使用相同的20个种子，达到256后继续运行至自然终局。训练和搜索预算尚未统一，榜单反映当前实现的结果。

原简版报告见 [report.md](report.md)，实验数据位于 [data](data/)。
