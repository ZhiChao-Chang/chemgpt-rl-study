# 本轮复现与运行边界

本目录包含2026-11-04选题阶段的本地CPU预试。原先领域综述与其completion记录是前一阶段的历史交付，保持原样；本轮入口为DECISION.md和STAGE.json。

## 文件

- `pilot/preflight.py`：公开prior适配、512次采样、轨迹概率一致性和一次真实DAP更新。
- `pilot/pilot.py`：4设置×3seed的开发预试；运行前写protocol，存在旧protocol时拒绝覆盖。
- `pilot/audit.py`：从保存分子重算identity、数量、主要指标，并用官方评分原语对照本地分数。
- `pilot/source_manifest.json`：公开源码/模型的commit、大小和SHA256。
- `pilot/results/summary.json`：可行性结果。
- `pilot/pilot_results/`：protocol、逐run完整生成记录、曲线、最终policy记录、聚合表与独立审计。
- `methods_recommendation.md`与`evaluation_recommendation.md`：独立方法/评价侧选题意见，本文最终判断见DECISION.md。

## 环境

本轮只读使用现有`/home/zhangzhichao/miniconda3/envs/ddc_model/bin/python`：Python3.10、torch2.7.1+cu128、RDKit2025.09.6、NumPy。没有修改环境、安装包、调用CUDA、使用服务器或访问既有DDC模型/数据。程序固定CPU、torch两个线程。模型和任务源码来自本轮单独下载，许可证保存在assets中；代码许可不自动替代原语料和全部第三方资产许可。

## 复现命令

下列`PATH_TO_SELECTION`替换为本目录绝对路径。运行会产生新计算，不能在已经存在的pilot_results上静默重跑。

```bash
TASK_PYTHON=/home/zhangzhichao/miniconda3/envs/ddc_model/bin/python
$TASK_PYTHON PATH_TO_SELECTION/pilot/audit.py
```

如要独立重跑，将`pilot/`复制到一个新的输出目录，保留原始结果，只移除新副本中的`results`和`pilot_results`后依次运行`preflight.py`、`pilot.py`、`audit.py`。脚本按自身目录解析资产，不依赖原`/tmp`位置。

## 解释限制

训练组为DAP/augmented-likelihood的简化适配，不是完整原始PMO训练脚本。checkpoint的原始GRUCell与融合GRU已数值验收。预试未使用原脚本额外likelihood penalty、SMILES增强、动态diversity reward或其early stopping，不得以官方REINVENT/PMO复现成绩命名。

每run主预算为512个独特有效分子；记录所有生成、处理和实际评分入口次数。重复性质本轮重新计算，没有cache。固定额外128次尝试用于最终policy评价，各run一致；这些评分单列，不冒充免费结果。没有matched-token的正式公平性结论，没有领域SOTA结论。

本轮没有生成结构插图、对接或生物实验；没有使用PyMOL。原任务中PyMOL偏好针对真实结构展示，不需要为选题阶段的CPU性质试验虚构结合位姿。
