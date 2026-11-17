# ChemGPT RL课程Proposal LaTeX项目

英文正文采用NeurIPS 2026样式的课程proposal适配：非匿名展示，关闭出版/投稿声明和审稿行号。成稿5页：4页正文、1页参考文献。原始`neurips_2026.sty`保持不变，课程适配仅写在`main.tex`导言区。

三位作者按用户提供的姓名同字号并列：Zhang Zhichao、Liao Yang、Han Gaofei。姓名下方注明“Name order does not imply contribution ranking.”。无第一作者、通讯作者、单位、邮箱、学号或贡献等级标记。个人职责已明确：Zhang负责领域调研、整体方案、模型接入及proposal背景；Liao负责分子任务、评分、预算、实验与评价；Han侧重REINFORCE、baseline、KL和训练正确性。三人共同承担交叉复核、报告整合和展示。

## 文件

- `proposal.pdf`：可直接查看的当前英文稿。
- `main.tex`：正文、作者和排版入口。
- `references.bib`：11篇参考文献的BibTeX记录。
- `main.bbl`：已生成的参考文献，方便归档和快速复编译。
- `neurips_2026.sty`：官方模板样式。
- `build.sh`：本地编译入口。

## Overleaf

将源码压缩包整体上传到Overleaf，选择`main.tex`为主文件，使用pdfLaTeX编译即可；无需模型权重或Python环境。系统会根据`references.bib`生成参考文献。本次本地实测使用Tectonic 0.17.0（XeTeX引擎），已完成BibTeX和交叉引用检查。

## 本地编译

安装有TeX Live/MiKTeX的环境可运行：

```bash
latexmk -pdf main.tex
```

也可以运行`bash build.sh`，它依次检查latexmk、Tectonic或pdfLaTeX/BibTeX。成功后输出`main.pdf`并复制为`proposal.pdf`。编译脚本不自动安装软件。

## 内容与状态

当前版本在原方案上适度扩充摘要、背景动机、生成/训练流程、任务比较、评价解释和工作交付；实验规模、成员职责侧重、署名和硬件配置不变。

文章包含领域背景、相关工作、研究问题、REINFORCE与KL方法、GuacaMol v2任务、18-run比较、评价预算与指标、可行性、计算资源及共同工作安排。已有32样本CPU检查、小环境梯度验证与H100资源测量准确写为可行性证据；正式18-run性能比较仍为计划。

此文档为ME2502课程proposal，未发送邮件、未提交课程系统或arXiv。资源段仅列硬件配置，不写GPU小时预算；个人分工已按用户确认的方案写入。

## 模板来源

官方页面：https://neurips.cc/Conferences/2026/CallForPapers

官方模板包：https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip
