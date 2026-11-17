# 预训练化学语言模型的强化学习微调

**课程：** ME2502 Reinforcement Learning  
**团队成员：** [姓名与学号，最多5人]  
**项目联系人：** [姓名与邮箱]

## 1 研究背景与问题

分子生成旨在从广阔的化学空间中提出满足用途的候选，支持从头设计、先导结构改良与多性质优化。领域既关注能否学习并重现合理的分子分布，也关注能否在有限评价预算下找到符合目标要求的结构。后者除生成合法性外，还涉及目标分数、多个条件的共同满足、结构多样性和计算成本[3,4]。

已有方法使用SMILES/SELFIES序列、原子键图、片段及三维坐标等表示，并采用自回归网络、变分自编码器、生成对抗网络、可逆流和扩散模型学习生成分布。例如，JT-VAE通过化学子结构及其连接生成分子图，MoFlow学习图与潜空间之间的可逆映射，EDM通过等变去噪联合生成原子类型和三维坐标；ChemGPT则采用自回归化学语言建模[1,7–9]。这些路线提供了不同的结构先验和生成方式。

生成合理候选之后，还需要使输出满足具体设计目标。现有策略包括条件生成、采样后筛选、潜空间优化、遗传搜索和强化学习。强化学习可将外部性质评价用于更新生成策略，但收益取决于先验、反馈和优化规则，不能仅凭接入RL就认定优于同预算筛选；持续优化也可能改变有效性、重复率与结构覆盖[2,4,10–12]。

据此，本项目拟回答两个问题：

1. 在有限的新分子评价预算下，REINFORCE微调是否比同一预训练模型直接采样并筛选，更有效地获得高分候选？
2. 相对初始模型的KL约束，如何影响目标分数、有效输出、重复和结构多样性之间的权衡？

## 2 相关工作与项目定位

**分子生成模型。** 序列模型将生成视为逐token预测，可利用大规模分子语料预训练，并直接计算采样轨迹概率；ChemGPT是这一路线的公开基础[1]。图和片段模型直接建模连接关系，JT-VAE结合子结构树与图消息传递；MolGAN将对抗生成和性质优化用于小分子图[7,10]。可逆流与三维扩散分别提供可逆潜表示和几何生成机制，例如MoFlow与EDM[8,9]。这些方法的表示与学习机制可以交叉，RL是利用目标反馈的优化机制，也可以与不同生成架构结合。

**目标导向优化与RL。** 遗传算法等搜索方法可以作为有竞争力的优化基线[11]。REINVENT建立了生成先验与RL反馈更新相结合的路线，ChemRLformer进一步研究了Transformer、文本表示及策略优化中的设计选择[2,12]。因此，本项目沿用已有反馈优化思路，研究选定ChemGPT先验上的具体接入、预算收益及KL权衡，不以“首次对分子生成使用RL”为贡献。

**评价与本项目切入点。** GuacaMol提供公开生成与目标优化任务，PMO强调评价调用预算及强基线[3,4]。本项目选择公开自回归模型，使终端奖励和轨迹概率能够清楚对应策略梯度；固定生成架构与任务定义，比较冻结先验、REINFORCE及REINFORCE加KL。预期贡献是正确的适配实现、统一评价预算下的受控证据，以及对质量、分布保持和成本关系的分析。

## 3 模型与强化学习方法

采用作者公开的ChemGPT-4.7M预训练权重[5]。该模型使用SELFIES的WordPiece表示，在PubChem10M上预训练；本项目直接微调公开模型，无需从零预训练。

将分子生成建模为有限时域决策过程：状态为当前token前缀和步数，动作为下一个token或终止符，完整分子的多性质评分作为终端奖励。解码失败、空分子、多片段或未在长度上限内终止的输出奖励为0。相同分子的重复生成可复用缓存分数，但保留其原始采样轨迹参与当前策略更新。

基本算法为REINFORCE[6]。对KL组，固定初始模型作为参考策略，优化

\[
J_\beta(\theta)=\mathbb E_{\tau\sim p_\theta}[R(\tau)]
-\beta D_{KL}(p_\theta\Vert p_0).
\]

其中，\(\tau\)为完整token轨迹，\(R\)为任务奖励，\(p_0\)为冻结参考模型。采用轨迹log概率比构造正则化回报，并使用leave-one-out baseline降低方差。每批从当前策略重新采样，仅更新一次；三组使用相同的采样、终止和解码规则。

开发初值为batch size 32、Adam学习率\(10^{-5}\)、KL系数0.01、最大生成长度256个动作。上述设置将在独立开发任务上检查稳定性，并在正式实验前固定。

## 4 公开任务与实验设计

使用GuacaMol goal-directed suite v2[3]，直接复用其公开评分定义：

| 用途 | 任务 | 评价内容 |
|---|---|---|
| 开发 | Perindopril MPO | 参考结构相似度及环数要求 |
| 正式比较 | Fexofenadine MPO | 参考相似度、TPSA和logP |
| 正式比较 | Osimertinib MPO | 两种指纹相似度、TPSA和logP |

两项正式任务使用连续组分的几何平均作为奖励。这些分数是计算性质和结构相似度代理，实验结论限定于对应的目标导向生成能力。

核心比较为：

| 方法 | 设置 | 目的 |
|---|---|---|
| Frozen + best-of-B | 原模型不更新，按相同预算生成并筛选 | 建立强采样筛选基线 |
| REINFORCE | 仅使用任务奖励进行策略更新 | 检验RL的增益 |
| REINFORCE + KL | 使用任务奖励及固定参考模型约束 | 检验先验保持的作用 |

计划在两个任务上各运行三种方法、三个随机种子，共18次运行。每次最多评价3,000个新合法独特分子，同时限制总生成尝试、更新次数和运行时间，记录实际消耗。采用固定预算末状态，不根据正式结果选择checkpoint。

超参数与实现调试使用独立开发任务，正式比较前固定。Fexofenadine此前在另一个GRU模型上进行过探索性预试，该历史将在最终报告中说明；不会将其描述为从未观察过的盲测任务。

## 5 评价指标与结果分析

主要指标为top-10任务分数随新独特分子评价次数变化的归一化曲线面积，用于衡量候选发现效率。辅助指标包括预算末top-k分数、阈值候选数、各组分及严格联合条件通过情况、有效率、重复率、结构多样性和真实计算成本。

每次运行结束后，固定重新采样1,024次，独立评价最终策略的平均奖励、有效独特产出和多样性，区分搜索过程中找到的最好候选与模型常规输出能力。这部分样本不参与训练或模型选择。

报告全部随机种子的结果、均值和波动，分别分析两个任务。若RL未优于直接筛选，或KL以部分分数为代价保留了输出质量，将结合奖励分布、轨迹偏移、重复和计算成本分析原因，完整保留负结果。

## 6 可行性与计算资源

已完成ChemGPT公开权重的本地CPU加载、32次小样本生成、采样与训练重算概率一致性及可微梯度检查；另在可枚举的小环境中核验了所用KL策略梯度和baseline公式。上述检查支持实现可行性，正式强化学习性能实验尚未开展。

拟使用现有单GPU顺序训练、CPU进行性质评价，开发计时后确认运行预算。GPU型号与可用时段：[待填写]。核心实验计划预留不超过约12小时的单GPU占用预算，实际训练时间和显存消耗将记录报告。

## 7 进度与成员分工

| 阶段 | 计划交付 |
|---|---|
| Proposal提交前 | 开发验收、确认配置与团队分工 |
| 11月上中旬 | 完成18次核心比较 |
| 11月下旬 | 独立汇总、图表及一项必要敏感性分析 |
| 12月6日前 | 最终报告、复现材料及课程展示准备 |

| 成员 | 主要职责 |
|---|---|
| [成员姓名] | [模型与采样接口] |
| [成员姓名] | [RL与KL实现] |
| [成员姓名] | [任务评分、预算记录及实验运行] |
| [成员姓名] | [评价分析、图表与报告] |

按实际人数合并或拆分职责；所有成员共同参与结果讨论与展示，并理解研究问题、算法和结论。

## 参考文献

[1] Frey et al. Neural scaling of deep chemical models. Nature Machine Intelligence, 2023. https://doi.org/10.1038/s42256-023-00740-3

[2] Olivecrona et al. Molecular de-novo design through deep reinforcement learning. Journal of Cheminformatics, 2017. https://doi.org/10.1186/s13321-017-0235-x

[3] Brown et al. GuacaMol: Benchmarking Models for De Novo Molecular Design. Journal of Chemical Information and Modeling, 2019. https://doi.org/10.1021/acs.jcim.8b00839

[4] Gao et al. Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization. NeurIPS, 2022. https://arxiv.org/abs/2206.12411

[5] ChemGPT作者公开模型. https://huggingface.co/ncfrey/ChemGPT-4.7M

[6] Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 1992. https://doi.org/10.1007/BF00992696

[7] Jin et al. Junction Tree Variational Autoencoder for Molecular Graph Generation. ICML, 2018. https://proceedings.mlr.press/v80/jin18a.html

[8] Zang and Wang. MoFlow: An Invertible Flow Model for Generating Molecular Graphs. KDD, 2020. https://doi.org/10.1145/3394486.3403104

[9] Hoogeboom et al. Equivariant Diffusion for Molecule Generation in 3D. ICML, 2022. https://proceedings.mlr.press/v162/hoogeboom22a.html

[10] De Cao and Kipf. MolGAN: An implicit generative model for small molecular graphs. 2018 preprint. https://arxiv.org/abs/1805.11973

[11] Jensen. A graph-based genetic algorithm and generative model/Monte Carlo tree search for the exploration of chemical space. Chemical Science, 2019. https://doi.org/10.1039/C8SC05372C

[12] Ghugare et al. Searching for High-Value Molecules Using Reinforcement Learning and Transformers. ICLR, 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/792a4eb351b014e1d61bcfaab0ddd555-Abstract-Conference.html

