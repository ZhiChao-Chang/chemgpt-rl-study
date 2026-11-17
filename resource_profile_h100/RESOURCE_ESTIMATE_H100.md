# ChemGPT课程项目的H100资源实测与预算建议

2026年11月4日。根据用户要求，先用H100测量并讨论资源口径；本轮不修改proposal、不运行正式18-run矩阵。成功测量作业为`acp-explore-chemgpt-profile-20261104-05`，UID `2ae6600d22da4606a1685c6e4dcd5de6`。

## 建议采用的具体资源口径

| 项目 | 建议写法 | 依据 |
|---|---|---|
| GPU | 1张NVIDIA H100 80GB，顺序执行各run | 本轮已在完整H100上运行，无MIG替代 |
| CPU | 4 vCPU | 本轮实际申请和使用的配套规格，torch线程数4 |
| 主机内存 | 64GiB | 本轮平台分配规格；不是独立测得的算法最低内存 |
| 项目磁盘 | 预留10GB，结果与选中权重预计控制在5GB内 | 单份发布checkpoint约70.8MB，不从零预训练、不下载PubChem10M原始训练集；输出空间是规划值 |
| 主要计算配置 | FP32，batch32，最大256个生成动作 | 与当前方案一致，TF32关闭 |
| 正常核心运行预计 | 约1–2 H100 GPU小时 | 以短测时间、每次3000个新有效identity、生成中50–95%成为新有效评价对象外推；包含18次运行和最终policy采样 |
| 低效率情景 | 约4 H100 GPU小时 | 假设仅20%的生成尝试成为本run首次评价的有效分子；不是已观察到的稳态比例 |
| 完整项目预算建议 | 8 H100 GPU小时 | 核心预算4h、开发1h、敏感性1h、重试/变化余量2h；为待讨论规划，不是承诺必须用满 |

这里“新”仅指本次运行中此前未评价过的canonical identity，不表示训练语料之外的新化合物。硬件是选定的可用执行环境，不将80GB显存或64GiB主存解释为模型的最低硬件需求。

## 直接测得的批次时间和显存

每个方法先做一次warmup，然后重新载入相同原始权重和新的Adam状态，再测4批。三组每批均为32条真实自回归轨迹，使用Perindopril MPO开发目标；没有读取两项正式任务的优化结果。以下为4批均值及各方法观测峰值：

| 方法 | 每批完整步骤均值 | 主要生成时间 | 峰值PyTorch保留显存 |
|---|---:|---:|---:|
| Frozen，生成和评价 | 1.287 s | 1.271 s | 0.502 GiB |
| REINFORCE | 1.695 s | 1.625 s | 3.252 GiB |
| REINFORCE + KL | 1.719 s | 1.612 s | 3.414 GiB |

RF/KL真实批次总时间大约0.97–2.33秒，随批次中最长序列变化。KL组平均reference forward为0.018秒；actor forward约0.038秒、backward约0.027秒、Adam约0.005秒，解码/性质评价约0.017秒。该小模型主要耗时来自逐token生成，不能只按GPU理论算力外推。

显存指标来自`torch.cuda.max_memory_reserved`，不含CUDA驱动/上下文的全部占用。另做了B32的合成teacher-forcing、backward与Adam形状压力测试：L128保留显存约2.43GiB，L256约6.30GiB。压力测试不是完整分子生成步骤，也不包含完整KL reference路径；它用于检查长序列训练形状，不能写成正式任务全程的严格峰值上界。选择完整H100留有充分余量。

实际环境：Python3.10.21、torch2.7.1+cu128、CUDA12.8、transformers4.51.3、RDKit2025.09.6、SELFIES distribution1.0.4。权重SHA256、actor初始参数及冻结reference均已核验，加载无missing/mismatched keys。

## 18次运行的计算量

正式矩阵仍为2任务×3方法×3seed=18run，其中12个训练run和6个冻结基线。每run的新有效identity预算为3000，故最多54000个首次逻辑评价；总生成attempt上限为270000。最终policy评价另有18×1024=18432个attempt，不回流训练或checkpoint选择。12个训练run合计最多5640次参数更新。

以每run需要约`ceil(3000/(32*q))`批估算发现过程，q为每次生成带来一个新有效identity的比例；再加入每run32批的最终policy评价：

| q情景 | 18run合计，含最终policy采样 | 再乘2倍时间余量 |
|---|---:|---:|
| 95% | 约61分钟 | 约122分钟 |
| 50% | 约103分钟 | 约206分钟 |
| 20% | 约235分钟 | 约470分钟 |

这是基于短测的算术外推，**没有完整跑完18run**。正式优化可能改变序列长度、重复和有效性；尾批也会变小。资源上限触发、停止原因和实际达到的unique数仍须如实报告。开发时冻结配置，不根据两正式任务的结果来调整预算以制造优势。

8小时的分项建议以“低效率情景核心约4小时”为预留依据，另外给开发1小时、预先声明的敏感性1小时、运行变化与重试2小时。原先12小时只能作为较保守的计划上限，不能称为实测预计耗时。是否采纳8小时、是否保留12小时外层上限，留待与用户讨论；当前PDF和协议尚未改。

## 可放入proposal的资源段候选

> We will use one NVIDIA H100 80GB GPU with 4 vCPUs, 64 GiB of host RAM, and 10 GB of project storage. Preliminary H100 profiling at batch size 32 projects approximately 1–2 GPU-hours for the 18-run core study and final-policy evaluation, assuming that 50–95% of generation attempts yield a previously unevaluated valid molecule. A low-efficiency scenario gives approximately 4 GPU-hours. We propose a total budget of 8 GPU-hours, including development, sensitivity analysis, and contingency.

这段是待讨论的候选，不自动替换论文中现有资源段。个人分工和Preprint页脚也没有在本轮修改。

## 本轮执行与收尾

CCI debug H100池最初48/48占用，未对该入口START或改动主网络。另经ACP独立前置核验后，以单H100/4CPU/64GiB运行专用短测。准备阶段四个attempt依次在环境创建、依赖安装、外网连接和CUDA设备索引接口处失败；这些失败不计作模型性能结果，终态与释放证据均保留。第五轮通过原生终端传输19.25MB压缩输入，逐文件核验后成功测量，不安装依赖、不依赖服务器外网。

五轮从创建到Worker终态的时长保守相加约20.48分钟，小于本任务1 GPU小时预算；这不是精确计费或纯GPU计算时长。最终预检显示活动ACP GPU为0，CCI仍为原SUSPENDED状态，未停止或占用其他项目训练。成功轮次的Worker为Succeeded。

原始数据见`actual_h100_profile.json`，计算外推见`campaign_estimate.json`，失败与资源释放记录见`all_attempt_terminal_evidence.json`。正式18-run训练未启动。
