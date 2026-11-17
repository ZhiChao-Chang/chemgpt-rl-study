# 分子生成中的强化学习：算法基础与技术附录

**范围更新：用户当前要调研整个“分子生成与优化中的RL”，TamGen只是例子，不是预选模型。先读 [field_extensions.md](field_extensions.md) 的领域问题、跨表示方法及近期进展；本文保留算法关系与实现正确性细节作附录，原先的TamGen迁移段落仅供实例参考。**

核验日期：2026-11-04。范围：原论文、出版者页面、作者公开代码和文档的只读调研。未训练、未生成或筛选分子、未下载权重、未安装环境、未使用服务器或付费服务。团队与具体资源尚未确定，**本文件不锁定题目、主算法或实验规模**；先解释已有方法及其适用条件，再提供课程选题候选。仅考虑常规公开人源靶点的教学研究。

## 1. 这个领域应沿四个独立维度理解

“Transformer / PPO / 多目标 / 靶点感知”不是四个互斥类别，它们分别回答不同问题：

| 维度 | 主要选择 | 实际决定什么 |
|---|---|---|
| 生成对象与动作 | SMILES token；原子/键图编辑；片段/反应；连续latent或3D去噪轨迹 | 合法性约束、轨迹长度、能否直接复用生成器 |
| 条件信息 | 无条件；固定一个靶点的reward；显式蛋白序列/口袋输入；scaffold/起始分子；多靶点集合 | 模型是否在推理时接收新靶点，还是每个目标需要重新优化 |
| 更新范式 | REINFORCE / REINVENT-DAP；actor–critic / PPO；value-based Q-learning；偏好优化；奖励分布采样 | 如何把评价反馈转成生成策略参数更新 |
| 目标与约束 | 单标量；加权多目标；阈值约束；Pareto排序；先验/KL；多样性；replay | 优化压力、探索范围、候选集合质量与预算 |

历史上大量“targeted generation”是：无条件语言模型 π(y)，用某一个目标的QSAR/docking分数微调它。靶点存在于reward定义里，但未必作为生成器输入。显式条件模型则是 π(y|c)，c可以是口袋、蛋白序列、目标集合；不等于获得可靠的跨靶点泛化，也不等于必须用RL。

TamGen的论文模型包含口袋编码、GPT式分子解码和上下文细化，属于显式条件生成。**给已有TamGen候选打分并重排是搜索/筛选；真正的RL适配需要采样轨迹、计算reward/return、取得被采样动作的log-probability并更新policy参数。** [TamGen原文](https://www.nature.com/articles/s41467-024-53632-4)

## 2. 十二项核心原始证据：从它们能迁移什么

下面只归纳与方法选择有关的结论，不把各文献的最佳分数横向当排行榜。出版时间、证据位置、代码和范围记录在 `rl_method_sources.json`。

| ID / 来源 | 方法与贡献 | 何时适用；不能据此宣称什么 |
|---|---|---|
| R01 [Olivecrona et al., 2017](https://link.springer.com/article/10.1186/s13321-017-0235-x) | 预训练RNN作Prior；Agent生成SMILES，用先验log-likelihood与分数构造augmented likelihood后更新。 | 已有自回归生成器、黑盒终点reward；DRD2是预测活性，不是新分子的湿实验命中率。 |
| R02 [REINVENT4, 2024](https://link.springer.com/article/10.1186/s13321-024-00812-5) | 生产化DAP、可组合评分、scaffold diversity filter、staged learning；提供不同生成任务。 | 借鉴reward组件和训练控制；框架有多种生成器，不能把软件名当成单一架构或单一算法。 |
| R03 [DrugEx v2, 2021](https://link.springer.com/article/10.1186/s13321-021-00561-9) | SMILES policy gradient，比较动态weighted sum与Pareto排序；探索网络/交叉/变异帮助维持候选差异。 | 多靶点与off-target要求的取舍；案例依据QSAR预测，非实验选择性结论。 |
| R04 [DrugEx v3, 2023](https://link.springer.com/article/10.1186/s13321-023-00694-z) | 接收多片段scaffold；Graph Transformer逐项生成原子、键和连接位置，价态mask；Pareto多目标RL。 | 片段生长/连接与结构约束较强时；其图动作不是SMILES decoder上加一个PPO即可复现。 |
| R05 [MolDQN, 2019](https://www.nature.com/articles/s41598-019-47148-x) | 分子+剩余步数作state，合法原子/键增删作action；double Q-learning与随机value函数探索。 | 起始分子的有限步局部优化；不需预训练生成器。化学价态合法不等于可合成或有药效。 |
| R06 [GCPN, NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/d60678e8f2ba9c540798ebbde31177e8-Abstract.html) | GNN policy生成图上的连接与终止动作；化学规则、属性reward与判别器反馈，PPO训练。 | 希望把图动作合法性写入环境；没有直接证明靶点识别，早期属性优化提升不能解释为药物发现提升。 |
| R07 [PPO, 2017](https://arxiv.org/pdf/1707.06347) | 用旧采样policy的概率比构造clipped surrogate，支持同一rollout的多轮更新。 | 想研究更新幅度/actor–critic时；论文原实验是控制/游戏，不是分子有效性证明。 |
| R08 [DPO, NeurIPS 2023；arXiv v3 2024](https://arxiv.org/html/2305.18290v3) | 在偏好模型等假设下，从KL正则reward最大化得到reference-relative pairwise loss。 | 已有同一条件下的winner/loser对；可直接更新生成器但标准离线DPO没有在线环境采样循环。 |
| R09 [PolyTamGen, Bioinformatics 2026](https://academic.oup.com/bioinformatics/article/42/10/btag766/8911832) | ESM3+NatureLM接收蛋白目标集合，SFT后用docking/QED构造偏好对做DPO。 | 直接说明“靶点条件分子LM+偏好对齐”已有工作。没有antitarget角色；目标组合测试与未见蛋白测试不可混称。未建立湿实验多靶点药理。 |
| R10 [ACEGEN, JCIM 2024](https://arxiv.org/html/2405.04657v2) | TorchRL模块化实现REINFORCE、REINVENT、AHC、A2C、PPO及replay变体；同生成器比较有助分离算法因素。 | 适合读训练loop和做受控比较；作者消融显示replay、prior约束和高似然惩罚的影响不同，不能“组件越多越好”。 |
| R11 [PMO, NeurIPS 2022](https://proceedings.nips.cc/paper_files/paper/2022/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html) | 25方法、23标量任务、10K oracle预算；强调top-10随查询数的AUC。 | 支持等预算、多个随机种子的评估原则；这些oracle基准不是新分子的临床/实验疗效。 |
| R12 [Saturn正式版, Nature Machine Intelligence 2026](https://doi.org/10.1038/s42256-026-01200-4) | Mamba配合SMILES augmentation与高分replay；分析局部利用、样本效率和多样性取舍，含docking与DFT计算oracle。 | 适合研究昂贵oracle下的经验复用；DFT仍是计算验证。2024预印本与2026正式版作者/实验/比较数量不同，不拼接其数字。 |

## 3. SMILES生成究竟怎样成为RL问题

以下是针对自回归条件生成的**本任务形式化**，不是声称TamGen公开实现已经具备这些训练接口：

- 条件 c：固定的靶点口袋/蛋白表示，或起始分子、指定scaffold等。优化时必须固定其构建方法。
- 状态 s_t=(c,y_<t,t)：完整已生成token前缀、条件与步数/剩余预算。策略 πθ(a_t|s_t)来自decoder对下一个token的categorical分布。
- 动作 a_t：词表中的下一token（包含EOS）。SMILES token可能是原子、键或语法符号，不能说“每一步都加一个原子”。
- 转移：把a_t附到前缀；EOS终止，或达到预先规定的最大长度截断。
- reward：通常只有完成分子后才能算R(y,c)。中间r_t=0，末步r_T=R；γ=1时各步return均为R。若加入每步KL或别的shaping，应重新写清r_t和return。
- episode长度可变。分子是否可解析、非法结构/未正常EOS如何处理，必须预先写进环境；不能只把有效样本留到报告里而丢掉失败分母。

**MDP / POMDP区别。** 给定完整prefix和固定c，上述计算环境可定义成MDP；“我们不知道真正药效”本身不使这个SMILES环境成为POMDP。2017原文在仅以RNN cell state决策的表述下用了POMDP措辞，引用时保留其原语境。若reward依赖跨批次scaffold记忆，把记忆M纳入环境状态才能保留该Markov说明；否则应承认reward随训练历史变化。随机docking需要固定版本/参数/随机种子或明确其随机转移与评价分布。

### 必须具备而常被忽略的迁移接口

1. **随机采样与可导logprob分开核验。** 生成器能够beam search输出SMILES，不等于训练loop已支持on-policy sampling。可先无梯度采样，再对同一实际token序列teacher forcing重算可导logprob；不对离散采样操作本身求导。
2. **EOS算一次，padding不算。** log πθ(y|c)=Σ_{t≤T} log πθ(y_t|c,y_<t)，包含实际采样EOS，排除BOS、prompt、padding和EOS之后的位置。达到长度上限的未终止样本按预定失败/截断规则计分，不伪造一个“已采样EOS”。
3. **采样分布与打分分布一致。** 温度、top-k/top-p、词表mask改变了行为policy。若从过滤分布采样却拿原softmax logprob作on-policy梯度，就是换了目标/带来偏差。首版可保持完整词表与固定温度；若使用约束mask，采样、agent/prior似然和PPO概率比都须明确同一合法支持集。
4. **分子身份与轨迹不能混淆。** canonical SMILES用来去重和oracle cache；policy loss仍用实际采样的token轨迹。不能canonicalize后拿另一条SMILES的likelihood冒充采样概率。randomized SMILES可用于replay augmentation，但这是显式的额外训练分布。
5. **记录真正的参数变化。** prior frozen、agent有非零梯度且optimizer.step确实改变参数，更新后生成分布/诊断likelihood可测。仅排序、再采样、改temperature或改prompt不等于RL训练。
6. **冻结范围不等于必须全模型更新。** 可固定蛋白encoder、更新decoder/adapter；只有在计算可导动作概率并由reward驱动参数更新时才成立。是否需LoRA、哪层可更新需对代码与checkpoint再审计，当前不预设。

## 4. 几种容易被混称的更新方法

### 4.1 REINFORCE：最直接的reward驱动更新

以固定条件c、终点reward、γ=1为例：

\[
J(θ)=E_{y∼π_θ(·|c)}[R(y,c)],\qquad
∇J=E[(R-b(c))∇\log π_θ(y|c)].
\]

b可以是上一批/移动均值等不依赖本次动作的baseline；也可用训练value baseline。实现surrogate时对R和b停止梯度。标准REINFORCE是一批新样本上的on-policy更新；同一数据反复多轮套用相同loss不再自动具有原始on-policy无偏解释。batch均值baseline可能引入有限batch缩放，若追求严格表述可用leave-one-out或独立/历史baseline。

适合问题：已有生成器能正确给出sample和logprob，想清楚观察reward强度、方差、遗忘和diversity之间的关系。缺点：稀疏终点reward时方差高，直接最大化proxy容易离开训练分布。

### 4.2 固定reference的KL正则与REINVENT augmented likelihood

固定同条件参考π₀，一个清晰目标为

\[
J_β=E_{π_θ}[R]-β\,D_{KL}(π_θ(y|c)\|π₀(y|c)).
\]

令ℓθ=logπθ(y|c)、ℓ₀=logπ₀(y|c)。在共同支持集、固定R、完整序列分布上，梯度可写为

\[
∇J_β=E[(R-β(ℓθ-ℓ₀)-b)\,∇ℓθ].
\]

这里省掉的常数项期望为零。用该score-function形式时，括号整体detach；不能只对采样到的log-ratio直接反传而忘掉样本分布本身的梯度。另一选择是明确实现token分布的KL与对应训练目标，不能把两种估计式混搭却声称完全等价。

REINVENT/DAP的原式是 A=ℓ₀+σR、L=(A−ℓθ)²，其中augmented likelihood是**未归一化目标**，不应解释为一个已经合法归一化的新策略。

**本文件代数说明（不冒充论文定理）：** 对固定采样y求DAP loss梯度，梯度下降方向为2(ℓ₀+σR−ℓθ)∇ℓθ。若y严格来自当前πθ、prior/reward固定、没有replay/top-k选择/额外penalty/动态diversity，取期望后与2∇[σE(R)−KL(πθ||π₀)]一致。它说明与KL正则policy gradient的联系；**不表示平方loss数值等于KL目标，也不表示任意REINVENT实现都等于PPO或精确无偏的标准REINFORCE。** 增加经验回放、精选样本或历史惩罚后需单独解释其偏差/目标。2017正文亦给出其REINFORCE重写，并在补充材料中说明具体构造。 [2017 Methods](https://link.springer.com/article/10.1186/s13321-017-0235-x)

固定prior是分布锚点，不是拟合state value的critic；不能因为有两个网络就称actor–critic。REINVENT4可借鉴的还有scaffold桶、重复结构惩罚和Inception。当前官方文档说Inception只支持其Reinvent生成器，**不能直接假设支持外接TamGen**。 [官方RL文档](https://github.com/MolecularAI/REINVENT4/blob/main/contrib/reinvent-doc/tutorials/rl.md)

### 4.3 PPO / actor–critic：旧policy与reference有不同职责

PPO的概率比ρ_t=πθ(a_t|s_t)/π_old(a_t|s_t)，目标为E[min(ρ_tÂ_t,clip(ρ_t,1−ε,1+ε)Â_t)]。π_old是**收集本轮rollout的策略快照**，随下一轮更新；π₀是**固定预训练参考**，用于保持分布。两者不能互相替换。value head/critic预测return，不是reference模型。

PPO提供复用本轮rollout的机制，也增加old logprob、value/advantage、终止mask和多轮更新的工程负担。clipping是surrogate，不保证对固定prior的严格KL上界，更不保证药学质量。若还要保持原模型，应独立加入reference-KL并监测实际漂移。[PPO原论文 §§2–5](https://arxiv.org/pdf/1707.06347)

适合问题：比较策略更新稳定性、value baseline对长序列终点reward的影响，或已有可靠PPO训练框架。ACEGEN为分子语言模型提供了PPO与其他算法共用组件的参考，但其结果不说明PPO天然优于REINVENT。[ACEGEN](https://arxiv.org/html/2405.04657v2)

### 4.4 图动作的value-based与policy-based方法

MolDQN用合法分子修改定义环境，再以Q(s,a)指导动作；动作数随当前分子变化，state还包括剩余步数。GCPN通过图policy选择连接位置/键类型/终止，以PPO学习。它们解决的是“设计化学合理动作空间，再学习搜索”，不是已有SMILES语言模型的小型损失替换。

迁移判断：若核心兴趣是局部结构编辑、硬价态控制，这条线有价值；若要求保留TamGen原生成架构，直接换成MolDQN/GCPN会同时改变表示、预训练和动作空间，无法把差异归因于RL算法。两篇原文中的有效结构保证只针对其规则和化学表示，不能推导可合成、稳定或实验活性。 [MolDQN Methods](https://www.nature.com/articles/s41598-019-47148-x), [GCPN](https://ai.stanford.edu/~jure/pubs/gcpn-neurips18.pdf)

### 4.5 片段级actor–critic：FREED / FREED++

FREED（NeurIPS 2021）每步选择当前分子连接点、候选片段及其连接点；以片段库和连接规则缩小动作空间，使用SAC与探索性的prioritized replay优化docking反馈。这是off-policy、maximum-entropy actor–critic路线，适合片段库内的受约束探索；“来自真实分子的片段”仍不能保证最终组合可合成或有活性。[FREED原文](https://proceedings.neurips.cc/paper/2021/file/41da609c519d77b29be442f8c1105647-Paper.pdf)

FREED++的价值还在严格复现：定位critic/action表示、target network、entropy项等实现与评估问题，简化后比较，并显示片段库选择显著影响结果。因此它很适合提醒课程项目：工程正确性和数据/动作库的影响可能比换算法名更大。论文的USP7例子是计算候选分析，结尾仍建议之后实验测活性；不能把其中引用的既有抑制剂体内证据转移到新候选。其TMLR版标12/2023，arXiv首版为2024-01，区分版本日期。[FREED++ §§3–5](https://arxiv.org/html/2401.09840v1), [官方实现](https://github.com/AIRI-Institute/FFREED)

### 4.6 DPO / 偏好学习：已可用于分子，但标准DPO不等同在线RL

对同一条件c的优选y_w与劣选y_l，DPO loss为

\[
-\log\operatorname{sigmoid}\{β[(\logπθ(y_w|c)-\logπ₀(y_w|c))-(\logπθ(y_l|c)-\logπ₀(y_l|c))]\}.
\]

标准DPO对固定偏好对做直接梯度更新，无需在线采样或训练reward model；它与KL正则reward最大化有理论联系，仍须说清偏好建模及数据覆盖假设。分子偏好可以来自计算oracle，也可以来自专家判断，**oracle派生偏好不是human feedback**。同一个靶点/条件内构对；跨不同靶点比较分数而不控制尺度，偏好可能无意义。pair构建、评过但未选中的候选也要计入oracle成本。[DPO原文 §§3–4](https://arxiv.org/html/2305.18290v3)

最新的直接例子是PolyTamGen：SFT参考模型为每个target set采样候选，按平均Vina分数与QED的组合选winner/loser，再DPO；正文明确SA未进入reward、没有antitarget控制、评价是计算proxy。它不是把2024 TamGen原checkpoint简单加DPO：采用ESM3与NatureLM的新架构。因此可作为领域发展证据，不能当“同TamGen架构的算法对照”。[PolyTamGen §2.3及limitations](https://academic.oup.com/bioinformatics/article/42/10/btag766/8911832)

一项2025分子DPO+curriculum预印本也使用SMILES偏好训练；其摘要称“target protein binding experiments”，但正文§4.2实际使用TDC JNK3/GSK3B/DRD2 oracle。不能把它引用成湿实验结合证据，亦未在本次核验中确认官方代码。[2025原文](https://arxiv.org/html/2504.01389v1)

课程意义：DPO可作为偏好学习对照或独立题目候选；本课程已明确将DPO列为示例，可以作为独立方向；需要准确解释其偏好目标与RL的关系，并用清楚问题和比较体现技术深度，不必人为追加PPO。

## 5. 多目标、多样性与样本效率：当前方法的实际分歧

### 5.1 Scalarization、约束与Pareto不是“多加几个指标”

- 固定weighted sum R=Σw_j f_j便于解释，但先统一方向/量纲/范围。某项的高分可补偿另一项不合格；mean docking可能掩盖其中一个目标完全不满足。
- geometric mean更强地惩罚短板，但接近零的分量可能导致学习信号稀疏；阈值硬门槛可能使初期全零。阈值和变换应在训练前固定，并报告未聚合的每个分量。
- Pareto dominance保留不能同时改进的trade-off，不必提前指定一个固定权重；真正选择哪些候选仍需偏好。DrugEx v2/v3把Pareto排名转成policy-gradient reward，不能说“完全不用标量反馈”。v2的WS比较还是动态权重，不是简单固定WS。
- 多目标front的指标可含hypervolume，但所有方法须用相同方向、归一化和reference point；同时报告各项全通过率、最差目标和集合多样性，防止只用一个总分。

这一段是从上述目标定义推导的评估设计建议；DrugEx提供实现案例，不证明Pareto在所有reward/预算上更优。[DrugEx v2](https://link.springer.com/article/10.1186/s13321-021-00561-9), [v3](https://link.springer.com/article/10.1186/s13321-023-00694-z)

### 5.2 多样性是集合性质，需要进入训练或选择协议

唯一canonical分子数、Bemis–Murcko scaffold数、fingerprint internal diversity、同已知配体的相似度，是不同问题。大量不同SMILES可能是同一个分子；大量不同分子也可能围绕一个scaffold。有效率/新颖性高本身不足以排除mode collapse。

可以通过entropy/先验约束、scaffold桶、重复惩罚、novelty reward或batch selection控制；这些会改变优化目标。REINVENT4的桶满后调低reward是动态记忆机制；不能把桶惩罚后的总reward与无桶模型的总reward直接比较并宣称后者更差。最终比较应另用固定raw property oracle和相同评估筛选规则。

### 5.3 Replay为何有用，也为何不能随便加

昂贵oracle通常按完整分子打分，同一分子换SMILES不应再次调用。因此重用已知分子分数能改善query efficiency。Saturn将高分memory与随机SMILES和多轮更新结合；正式版也讨论效率–多样性取舍及局部利用行为，不能只报更快达到高分。

代价是额外optimizer步骤、可能更窄分布，以及离开纯on-policy梯度解释。若diversity奖励随记忆改变，应保存raw oracle分量并用当前记忆重算shaped reward，不能将过期novelty奖励当永久真值。对比时同时给unique oracle calls、所有生成次数、cache命中率、梯度步数和wall time。省query不等于省GPU时间。[Saturn正式版](https://doi.org/10.1038/s42256-026-01200-4)

### 5.4 2024–2026值得跟进的补充范式

- **ERA（2024首版、2025修订）**：利用显式连续reward构造energy rank alignment目标，研究分子Transformer与偏好对齐；作者明确称其不需要RL。适合比较“利用全部连续分数”与只保留winner/loser排序的信息损失，不应把所有alignment都归入在线RL。[原文v3](https://arxiv.org/html/2405.12961v3)
- **GRPO / RLVR，2026分子预印本LLMol**：同一prompt采样一组候选，以组内相对reward构造advantage，避免单独critic，仍在线更新policy；研究的是性质控制/相似性约束，使用简化SELFIES，**其“targeted”主要指目标性质，不是蛋白target-aware**。去掉critic没有去掉组采样的oracle成本；组内reward相同可能没有相对学习信号。“verifiable”只意味着能核对程序规则，不意味着核对真实药效。[原文](https://arxiv.org/html/2607.19044v1)
- **GFlowNets**：目标更偏向按正reward比例采样多个解，而不是只最大化期望reward；原始工作针对多轨迹到同一终点的图生成提出流一致性学习。若核心问题是多样的高reward候选集合，这种目标很相关；接入现成TamGen并非换一个普通policy-gradient loss，课程是否值得承担取决于团队经验。[NeurIPS 2021原文](https://proceedings.neurips.cc/paper_files/paper/2021/file/e614f646836aaed9f89ce58e837e2310-Paper.pdf)

以上补充是定向检索，不冒充截至2026所有工作或统一的最佳算法。没有从“更新年份较新”推出“课程更合适”。

## 6. 文献反复暴露的证据问题

| 问题 | 会出现什么误判 | 应保留的证据/对照 |
|---|---|---|
| Reward hacking | QSAR置信度、QED或docking越高被解释为药效越强 | raw分量、结构/大小分布、独立计算评价；声明没有湿实验 |
| Mode collapse | 只有top-1很高，或不同SMILES被当不同分子 | canonical分子、scaffold、内部相似度、collapse随时间曲线 |
| Oracle overfitting | 同一个predictor既负责训练reward又“独立”证明活性 | 区分训练oracle、未用于选择的外部检查；相关proxy仍非实验真值 |
| 预算不公平 | RL搜索很多次后与少量原模型样本比best score | 相同唯一query预算、原模型等预算rejection/best-of-N、wall time |
| 训练与测试混用 | 用测试口袋选权重、β、阈值或种子 | 调参与最终测试条件分开；负结果和全部种子保留 |
| 跨架构归因 | TamGen+RL胜过REINVENT4便宣称RL更好 | 同checkpoint/条件/奖励/预算做算法消融；跨架构只做系统比较 |
| 错认新颖性 | “TamGen+RL/多目标/DPO”直接写成全新算法 | 已有REINVENT、DrugEx、ACEGEN和2026 PolyTamGen；课程可做受控实证而无需论文创新 |

PMO强调query预算和多次运行；ACEGEN在共用生成架构下揭示replay、先验约束、chemistry filters与原始分数的取舍。两者共同支持“可复现地解释何时有效”，而非“堆算法组件得到最高分”的问题表述。[PMO](https://proceedings.nips.cc/paper_files/paper/2022/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html), [ACEGEN](https://arxiv.org/html/2405.04657v2)

## 7. 课程层面的候选切入口（待用户选方向）

GPU可用，团队与资源待定。当前最有价值的决定是研究问题，不是先固定GPU时长/分子数量。以下各方向都可用已知方法形成技术正确、可分析负结果的课程作业。

| 候选问题 | 合适的方法族 | 必要对照与为什么有价值 |
|---|---|---|
| 先验约束如何改变target-aware生成的reward–validity–diversity取舍？ | REINFORCE+固定reference KL，或DAP；二者择一为主后核验目标 | 同checkpoint的原模型、naive REINFORCE、受约束更新；可回答改善来自学习还是暴力筛样 |
| 额外query应该用于探索新分子，还是replay已有高分结构？ | 同一policy-gradient基础上加replay/SMILES augmentation | 无replay与有replay，等unique query且报告额外梯度计算；直接连接近期样本效率争议 |
| 同等oracle信息，标量reward与偏好对学习有什么不同？ | 在线policy gradient vs DPO/ERA | 完全相同已评分候选池、预先定义pairs与训练预算；区分在线数据收集收益与loss收益 |
| 多目标平均分是否掩盖短板，Pareto方案是否更全面？ | Weighted scalarization vs Pareto-ranked policy gradient | raw分量/全部阈值通过率/固定评价front，多样性与预算相同；不需要声称新算法 |
| 更新幅度控制是否比简单policy gradient更稳定？ | PPO vs REINFORCE，各自有同一固定reference方案 | 同生成器、初始化和reward；监测actual KL、clip fraction/value fit与collapse，工程量相对大 |

若之后选择第一个问题，一个**可讨论的最小主线**是：REINFORCE+固定reference KL作为受约束候选，消融A去掉KL，消融B在相同KL基础上加scaffold diversity；没有任何已运行结果保证这种组合获胜。若选择replay问题，则应换成replay为核心，避免同时变化prior、reward、数据收集和训练步数。

所有方向共用的最低基线：①不更新的原始TamGen随机采样；②同一候选集合仅打分重排（与①共用样本，只隔离selection效果）；③不更新policy但花完相同oracle预算继续采样的best-of-N / rejection baseline。若可用固定公开分子库，可加库内随机抽样/筛库以检验生成是否值得。REINVENT4可作为成熟外部系统基线，但与TamGen不同架构时不能归因算法差异。

公平预算需同时固定候选产生/查询规则、预筛选规则、生成attempt上限或wall-time停止、超参开发预算和最终评估预算。canonical去重后的昂贵oracle calls作主要横轴，另报告所有attempt；否则大量重复/无效样本的模型可能依靠无限重采样规避成本。cache key应包含分子、靶点、构象/参数和oracle版本。train-time已评过的候选、偏好配对池中的落选候选、RL启动前的seed memory均需计费记账。

可交付的科学主张应限定为：“在规定计算代理目标与预算下，该策略是否改善搜索效率或候选集合的性质分布，并解释其多样性和漂移代价。”不写“发现更有效药物”，也不以原论文湿实验替代本课程验证。具体目标、训练实现、实验规模和成员分工等用户选方向后再定。

## 8. 阅读顺序与尚未核验的部分

先按field_extensions.md选择感兴趣的领域问题；算法基础可从2017 REINVENT、REINVENT4、ACEGEN和PMO入手，然后分支读合成约束/GFlowNet、3D反馈、Pareto、replay或偏好学习。TamGen与PolyTamGen仅作为显式条件语言模型分支的实例阅读。

本次核验是文献与公开接口级，没有安装或运行任何仓库。动态main分支不是已锁定实验版本；正式实现前需固定commit和依赖。部分PMC直开出现反爬页面，已使用出版者正文/作者arXiv合法公开版替代；PolyTamGen出版者直接open有重定向失败，但其出版者搜索索引返回正文，可定位§2.3；不把未下载的补充文件称为已检查。2026正式Saturn正文以PMC索引和出版记录核对；与arXiv旧稿的定量内容不合并。PolyTamGen出版者标2026-10-12，PubMed记录2026-10-02，两者冲突保留，以出版者为报告日期。


主审课程口径补充：课程公告已明确列DPO为示例，所以它可以独立构成课程认可的偏好优化方向，无需本报告额外要求先确认其是否合格。仍应准确说明离线DPO与在线RL的差别；是否再加入在线方法取决于选题，不是课程强制门槛。
