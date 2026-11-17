# 预训练化学语言模型的强化学习微调

**课程：** ME2502 Reinforcement Learning。**状态：** 研究内容草稿，团队姓名、资源与实际分工待填写，尚未提交。

## 研究问题

本项目研究在有限分子评价预算下，强化学习微调能否比预训练分子语言模型直接采样并筛选，更有效地获得高分候选；同时考察相对初始模型的KL约束如何影响目标分数、生成有效性、重复和结构多样性。

预训练分子语言模型学习已有化学结构分布，而目标导向设计需要满足特定性质要求。直接扩大采样和筛选本身可能很强，因此必须与同预算的参数更新方法公平比较。项目将以一个固定公开模型、两个公开多性质任务和多个随机种子，检验RL接入及先验正则的实际价值。

## 背景和相关工作

ChemGPT由Frey等在Nature Machine Intelligence（2023）的《Neural scaling of deep chemical models》中提出，是基于GPT-Neo的自回归化学语言模型。REINVENT等工作建立了基于反馈的分子生成优化路线；GuacaMol提供标准目标导向生成任务，PMO强调有限oracle预算下的比较。我们的工作定位为在公开ChemGPT上独立实现和评价REINFORCE微调及固定reference KL正则，不声称首次将RL用于化学语言模型。

预期贡献是正确可审查的策略梯度实现、强筛选基线、预算与重复样本的规范处理，以及对收益和失败的充分分析。研究成功不以新药发现、发表级算法创新或必须超过已有SOTA为条件。

## 模型与方法

采用公开`ncfrey/ChemGPT-4.7M`及同版本词表，固定revision `7438a282460b3038e17a27e25b85b1376e9a23e2`。模型在PubChem10M上预训练，使用SELFIES/WordPiece表示，无需重新预训练。模型官方名称为4.7M，实测含embedding总参数约5.10M。

将生成建模为有限时域MDP：初始状态为BOS，状态为前缀与步数，动作为下一个token或终止符，完整分子获得终端MPO reward。模型/reference使用相同温度、特殊token mask和最大长度，非法、空、多片段和未自然终止的截断输出给reward0；重复样本使用缓存的确定性reward，但仍保留其真实轨迹参与当前策略更新。

比较三组：冻结模型的同预算采样筛选、REINFORCE、REINFORCE加固定初始reference KL。第三组优化：

\[
J_\beta=\mathbb E_{p_\theta}[R]-\beta D_{KL}(p_\theta\Vert p_0).
\]

用采样轨迹的`log pθ−log p0`构造正则化return，采用leave-one-out baseline和停止梯度的score-function surrogate；每批全新当前策略轨迹只更新一次，不使用replay或多epoch复用。训练起点为batch32、Adam lr1e-5、KL β0.01、全序列累计KL、最大256动作、FP32；最终配置在独立开发任务上检查并在正式比较前冻结。

## 任务与实验设计

采用GuacaMol goal-directed suite v2：Perindopril MPO用于开发，Fexofenadine MPO和Osimertinib MPO作为两项主要benchmark。后两者均使用参考相似度、TPSA、Crippen logP等连续组分的几何平均；Osimertinib还包含不同指纹下不过度相似的约束。我们直接复用公开定义，评分为可计算性质代理，不将其视为实测活性。

Fexofenadine此前在另一GRU模型上做过开发预试，报告中明确其探索历史；不将换模型后的结果冒称全新盲测。正式任务运行期间查询reward是目标导向优化的必要环节，配置选择则只依据独立开发阶段。

主矩阵为2任务×3方法×3随机种子，共18run。每run最多评价3,000个新合法独特分子，另设15,000生成attempt、470次更新和30分钟搜索/训练时间上限；三个方法采用同一协议并报告实际计算成本。以固定末状态作为最终模型，不按正式任务分数选择checkpoint。

主要指标为Top10 MPO随新独特分子评价数变化的归一化AUC，辅助报告预算末top-k、阈值候选数量、严格多条件同时达标、有效率、重复和结构多样性。每run结束后再固定采样1,024次，独立评价最终policy的常规输出；这些额外评价单列且不回流训练。报告每任务、每seed结果和成对差异，不用最佳一次运行替代总体表现。

## 假设和预期分析

H1：目标反馈更新可在固定评价预算下改善候选发现或最终生成分布。H2：KL能够减轻偏移或输出退化，但可能降低峰值分数；需要以质量、多样性和成本共同评价。

若RL不优于同预算筛选，将分析先验质量、反馈信号和实际训练成本；若搜索分数提高但最终policy重复或严格多条件成功率不升，则区分“找到高分点”与“整体分布改善”。所有失败和无提升结果保留，不事后更换主任务、阈值或随机种子。

## 可行性与资源

已完成固定公开权重的本地CPU加载、32次小样本生成、真实轨迹与teacher-forcing概率一致性以及非零可微梯度检查；小样本32条均自然结束并解码成功。另在可枚举两步MDP中验证了KL score-function公式和LOO baseline的期望梯度。以上只证明接口与公式可行，尚无ChemGPT强化学习效能结果。

拟使用本地单GPU顺序训练、CPU进行性质评价，不从零训练大模型。主阶段计划预算上限约12小时单GPU占用，开发计时后在正式实验前确认；GPU型号、可用时段和峰值显存待填写/实测。没有服务器、付费服务、湿实验或自动提交授权。

## 进度与分工

11月8日前完成开发验收并提交proposal；11月上中旬完成18run核心比较；11月下旬完成独立汇总和一项必要敏感性；12月6日前提交最终报告，按第13/14周课程安排展示。

团队成员：待填写，最多5人。工作包可分为模型接口、RL实现、任务/计账、评价/分析、报告/展示，按实际人数合并；每位成员均需能解释MDP、损失、对照与结果。

## 参考资料

1. Frey et al. Neural scaling of deep chemical models. Nature Machine Intelligence, 2023. https://doi.org/10.1038/s42256-023-00740-3
2. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 1992. https://doi.org/10.1007/BF00992696
3. Olivecrona et al. Molecular de-novo design through deep reinforcement learning. Journal of Cheminformatics, 2017. https://doi.org/10.1186/s13321-017-0235-x
4. Brown et al. GuacaMol: Benchmarking Models for De Novo Molecular Design. JCIM, 2019. https://doi.org/10.1021/acs.jcim.8b00839
5. Gao et al. Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization. NeurIPS, 2022. https://arxiv.org/abs/2206.12411
6. ChemGPT作者权重与模型卡. https://huggingface.co/ncfrey/ChemGPT-4.7M
7. GuacaMol官方任务实现. https://github.com/BenevolentAI/guacamol
