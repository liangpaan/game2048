> 最新的五分钟汇报大纲、素材与问答准备见 [presentation_materials](presentation_materials/README.md)。目录使用英文命名，不包含PPT模板；补充调参与更新后的MCTS结果见其中说明。本目录已有PDF、PPTX与LaTeX成品保留为此前版本，本次未同步修改。

# 三方法实验报告与课程汇报

汇报只介绍完整 Expectimax、MCTS、乐观5-tuple TD，顺序为完整搜索、采样搜索、学习估值。正文9页，按5分钟编排；另有2页问答备用。

| 材料 | 文件 | 内容 |
| --- | --- | --- |
| 演示文稿PDF | [game2048_presentation.pdf](game2048_presentation.pdf) | 南开 Beamer 模板，9页正文＋2页备用 |
| 可编辑PPTX | [game2048_presentation.pptx](game2048_presentation.pptx) | 结果图和表格可编辑，附逐页讲稿备注 |
| 汇报讲稿 | [game2048_speaker_notes.pdf](game2048_speaker_notes.pdf) | 5分钟逐页讲稿、1—2分钟问答准备，共3页 |
| 实验报告 | [game2048_experiment_report.pdf](game2048_experiment_report.pdf) | 宋体小四、单倍行距，共3页，含分工 |
| LaTeX源码 | [latex/README.md](latex/README.md) | 报告、Beamer演示稿、讲稿及所需模板资源 |

## 新增万局评测

三个方法各评测10,000局，共30,000局；种子3,000,000至3,009,999，达到256后继续，运行至自然终局。使用冻结TD权重，没有重新训练。

| 方法 | 平均终局总和 | 达到256 | 平均步数 |
| --- | ---: | ---: | ---: |
| 完整Expectimax | 813.2666 | 9954/10000（99.54%） | 367.6799 |
| 乐观5-tuple TD | 640.7430 | 9119/10000（91.19%） | 289.2224 |
| MCTS | 282.0126 | 3496/10000（34.96%） | 126.1532 |

[逐局压缩CSV与汇总](data/evaluation10000/README.md)包含置信区间、最大块分布、种子和来源校验值。[评测入口及说明](benchmark10000/README.md)保留原环境，并提供经动作与随机状态一致性核对的MCTS评测加速内核。训练、建表和每步计算资源未统一，因此这不是等资源算法排名。

## 简洁超参数讨论

正式材料只保留三组同为10万局训练、500局测试的TD历史对照：对称共享、初值0/300。它们与新万局评测分别统计；学习率、探索率与MCTS搜索预算没有独立多取值实验，不声称参数已最优。

可编辑[LaTeX讨论模块](latex/report/sections/hyperparameters.tex)。此前较完整的[历史讨论](hyperparameter_discussion.md)及[数据索引](data/hyperparameter_evidence.json)继续作为归档，不代表本次万局成绩。原有简版[report.md](report.md)也作为历史材料保留。
