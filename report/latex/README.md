# LaTeX 源码与编译

| 材料 | 主文件 | 对应 PDF |
| --- | --- | --- |
| 实验报告 | [report/main.tex](report/main.tex) | [game2048_experiment_report.pdf](../game2048_experiment_report.pdf) |
| 演示文稿 | [slides/main.tex](slides/main.tex) | [game2048_presentation.pdf](../game2048_presentation.pdf) |
| 汇报讲稿 | [speech/main.tex](speech/main.tex) | [game2048_speaker_notes.pdf](../game2048_speaker_notes.pdf) |

## 编译

使用包含中文支持的 TeX Live 或 MiKTeX，编译器选择 **XeLaTeX**。三个目录各自独立，均以 main.tex 为主文件。

进入要编译的目录，例如 report/latex/slides，执行：

~~~sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
~~~

如果没有 latexmk，连续执行两次：

~~~sh
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
~~~

结果为当前目录下的 main.pdf。修改后请同时更新上表中对应的成品 PDF。

字体配置位于各主文件开头：

- 报告、讲稿：宋体 SimSun、黑体 SimHei、Times New Roman。
- 演示稿：微软雅黑 Microsoft YaHei、Microsoft YaHei Bold、Calibri。

当前配置使用 Windows 字体。在其他系统或在线编译环境中，需要安装相应字体，或调整主文件中的字体名称。课程提交的实验报告需保留正文宋体小四、单倍行距、不超过5页的要求。

## 修改位置

- **报告正文、表格、分工**：[report/main.tex](report/main.tex)。
- **超参数对照与讨论**：[report/sections/hyperparameters.tex](report/sections/hyperparameters.tex)，由主文件引入；详细数据解释见[讨论模块](../hyperparameter_discussion.md)。
- **报告结果图**：[report/figures/results.tex](report/figures/results.tex)，使用 PGFPlots，可直接修改数据和坐标轴。
- **演示稿内容、表格、逐页备注**：[slides/main.tex](slides/main.tex)。
- **演示稿结果图**：[slides/figures/means.tex](slides/figures/means.tex)。
- **演示稿公式**：[slides/formulas](slides/formulas/)，每条公式单独保存。
- **讲稿、逐页时间安排、问答准备**：[speech/main.tex](speech/main.tex)。

南开 Beamer 主题、颜色文件及实际使用的校徽、背景位于 slides/ 和 slides/assets/，无需另行下载模板。主题原有授权说明保留在文件头和 [slides/LICENSE](slides/LICENSE) 中。

目前报告5页，演示稿正文10页、备用2页，讲稿4页。修改算法顺序或结果时，请同步修改演示稿、讲稿和报告中的对应内容。
