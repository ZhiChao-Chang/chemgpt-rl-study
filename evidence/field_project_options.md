# 从分子生成 RL 的领域问题到课程选题

审阅日期：2026-11-04。TamGen 只是结构条件生成的一例，不是本项目指定基础模型。本文比较问题方向，不选择靶点、模型或精确实验矩阵；GPU 已知可用，团队和具体资源待定。仅调研，没有训练、分子生成/优化、安装或权重/大数据下载。

## 先明确要解决的决策问题

分子生成包含多个不同任务：从头产生满足性质要求的分子；对已有 lead 做有限改动；在允许的片段/反应中构造分子；根据偏好改变生成分布；在昂贵评价预算下搜索多个候选；根据目标蛋白形成条件偏好。靶点、pocket、三维结构、PPO 和大语言模型都不是每个任务的必需品。课程选题应先写真实需求与可证伪问题，再选择能承载该问题的状态、动作、策略和奖励。

“为什么需要 RL”也需要回答。若只是从固定候选库选择最高分，排序、随机搜索、遗传算法或贝叶斯优化可能已足够；若可以按同一规则枚举所有合法动作，训练复杂策略未必划算。序列生成可以是 token 级有限时域 MDP，图/反应生成可用结构状态和有效动作，完整分子一次采样也可以视为 contextual bandit；不能仅因 SMILES 很长就把问题夸成复杂长期规划。对这些区别作正确建模，本身符合课程的技术深度要求。

公开原始基础覆盖很广：REINVENT4 支持从头设计、结构装饰、linker、scaffold hopping 和 molecule optimization；PMO 把多类优化器放在固定 oracle 预算下比较；SynFlowNet 把反应/原料纳入动作空间；分子 DPO 与 POLO 展示离线偏好学习和在线多轮优化；TamGen、TDC Docking、PMO-Dock 则对应部分结构/目标条件问题。[REINVENT4](https://doi.org/10.1186/s13321-024-00812-5)、[PMO](https://arxiv.org/abs/2206.12411)、[SynFlowNet](https://arxiv.org/abs/2405.01155)、[分子 DPO](https://arxiv.org/abs/2310.12304)、[POLO 官方实现](https://github.com/REAL-Lab-NU/POLO)、[TamGen](https://github.com/microsoft/TamGen)

下列假设和课程边界均是本任务提出的可选设计，不是来源论文已替我们验证的结果。

## 方向一：多性质可行域与约束优化

**真实问题。** 候选需要同时落入多个允许范围，单项很好却违反关键约束通常没有用。可研究的目标包括理化性质窗口与结构限制；若使用预测 ADMET/活性，需要明确它们是模型端点，未测量不是阴性，也不能声称改善了真实安全性。

**RL 问题。** 比较加权总奖励、带惩罚的目标、约束/可行性优先、分阶段课程学习。关注稀疏终端反馈、违反约束的探索、不同目标尺度、权重迁移。简单可表述为最大化目标收益，同时控制约束违反概率或满足全部约束的产出率；不能只用综合均分掩盖少量严重违规。

**原始基础与接口。** REINVENT4 的多组分 scorer 和 staged/curriculum 配置、GuacaMol 多性质目标、TDC 的可计算性质和预测 oracle 可作起点，不需要 pocket；它们不是真实药物有效性的替代品。[REINVENT4](https://github.com/MolecularAI/REINVENT4)、[GuacaMol](https://github.com/BenevolentAI/guacamol)、[TDC](https://tdcommons.ai/functions/oracles/)

**强基线与可证伪假设。** 固定生成器+同预算过滤/排序、简单固定权重 REINFORCE、可行性优先的搜索，都应纳入考虑。可检验假设例如：“约束处理提高满足全部条件的独特候选数，而不是只提高容易的子项”；若匹配采样/调用成本后不成立，报告稀疏反馈或边界冲突原因。

**课程边界。** 可先围绕一个明确 profile 比较一种约束策略与基础策略，再考察阈值变化时是否失效，不必同时研究大量预测端点。不预设少模型就一定简单：合理评价分母、惩罚尺度和可行域可能是主要工作。若使用拉格朗日式实现，也不自动获得约束满足/收敛保证。

**成熟组件与贡献界限。** 多性质奖励、加权和、KL 正则、hard filter 和 curriculum 都是成熟组件；“加入 QED+SA”不是方法首创。课程贡献可以是明确的比较、失败机制或约束迁移分析。

## 方向二：固定反馈预算下的样本效率与探索多样性

**真实问题。** 评价资源有限时，需要多个不同 scaffold 的合格方案，给后续筛选留选择空间，而不是反复提出同一高分结构。预算可代表昂贵计算评价；它与真实湿实验预算并不等价。

**RL 问题。** 研究探索/利用、熵/KL、replay、SMILES augmentation、diversity memory、reward shaping 的权衡。token 表述不同的同一分子不是独立发现；已缓存 reward 的多次梯度更新与新 oracle 调用要分开记账。

**原始基础与接口。** PMO 提供 oracle-budget/AUC 比较框架；REINVENT4 提供 scaffold diversity filter；MOSES 提供分布与多样性度量。SynFlowNet 所属 GFlowNet 路线目标是采样奖励相关分布，不能把它直接当 PPO 的另一实现或只以最高分比较。[PMO](https://arxiv.org/html/2206.12411v2)、[REINVENT4](https://github.com/MolecularAI/REINVENT4)、[MOSES](https://github.com/molecularsets/moses)、[SynFlowNet](https://github.com/mirunacrt/synflownet)

**强基线与可证伪假设。** 同预算 frozen best-of-N、简单图遗传算法、无 replay 的策略梯度是有解释力的比较。假设可以是“在匹配可行候选质量后，某探索机制增加不同 scaffold 的发现率”；只增加分子数但明显降低约束满足率，不能概括为更好探索。应同时观察 top-k/AUC、达到固定质量阈值的独特分子/scaffold 数、最终 policy 输出和墙钟时间。

**课程边界。** 可以只研究一个探索改动，给出多个预算前缀与重复运行；更高资源可以检验跨任务迁移，而不必立即重训更大基础模型。允许结论是简单搜索已足够强，RL 的额外成本没有换来稳定收益。

**成熟组件与贡献界限。** replay、augmentation、entropy bonus、scaffold memory、GFlowNet 均已有大量工作。严谨区分重复利用与真正新反馈、解释为何不优于 best-of-N，可成为合格课程结果。

## 方向三：合成可达性、片段和反应动作空间

**真实问题。** 一个数学上合法的高分图不代表实验室能获得。领域有两条不同路径：在自由生成后用合成 proxy/规划器筛选；直接把允许的原料、片段或反应纳入构造过程。片段拼接与有反应规则支持的路线也不是同义词。

**RL 问题。** 结构/反应构成状态，选择构件/反应/停止构成动作；合法动作 mask、终端目标、路径长度、同一分子多路径和信用分配产生实际决策问题。若比较 GFlowNet，需要解释奖励分布采样与期望奖励最大化的目标差异。

**原始基础与接口。** SynFlowNet（ICLR 2025）公开了反应模板+building blocks 的环境、mask、task/reward 接口；其原始研究使用独立 retrosynthesis 检验，而非只用 SA。REINVENT4 的 LibInvent/LinkInvent 等提供另一种结构限制入口，但不能把这种限制自动说成合成路线证明。[SynFlowNet 论文](https://openreview.net/pdf?id=uvHmnahyp1)、[官方代码](https://github.com/mirunacrt/synflownet)、[REINVENT4](https://github.com/MolecularAI/REINVENT4)

**强基线与可证伪假设。** 限定相同构件/反应集合的随机有效轨迹、可管理范围内的枚举+排序、简单策略梯度；再比较自由生成+事后筛选。假设可以是“构造约束降低失败/无路线比例且不完全牺牲可行候选质量和多样性”。若 constrained policy 优势只是搜索空间更窄，不能归功于 RL 更新。

**课程边界。** 可以选择受控的公开构件/规则子集，聚焦 action masking 或路线长度成本；不能把缩小的玩具空间当工业化学全空间。若引入大 catalog 或 retrosynthesis planner，数据权限、CPU/内存和新模型依赖会成为成本；可只报告路线规则内可达性，明确不做湿实验结论。

**成熟组件与贡献界限。** 反应动作空间、片段限制、路线生成、独立规划复核都已有工作，不是首次保证可合成。反应模板匹配只是计算可达性；不保证底物适用、区域/立体选择性、条件、产率、成本与真实采购。

## 方向四：离线偏好学习与在线反馈优化

**真实问题。** 已有一批好/差候选或排序信息时，是否还需要持续调用外部 oracle？偏好可来自化学规则、预测器、历史测量或人类，来源不同、噪声和适用范围不同。规则生成的标签不能称人类化学家意见。

**RL/学习问题。** 区分固定数据上的 preference fitting、离线策略改进、在线探索收集反馈。DPO 是直接偏好目标，不应强称其本身实现了在线 RL；它可作为在线策略梯度的强替代基线。关键问题是旧数据支持集外的探索、偏好噪声、reference-policy 正则与分布漂移。

**原始基础与接口。** Park 等 2023 分子 DPO 预印本及公开代码展示从采样/过滤标签到 preference fine-tuning 的流程，公开示例主要是 medchem filter。POLO（KDD 2026）官方说明把 trajectory-level PPO 与同一分子 oracle 得到的 turn-level preference loss 结合；这说明多轮偏好优化已是已有路线。[分子 DPO 论文](https://arxiv.org/abs/2310.12304)、[代码](https://github.com/Harmonic-Discovery/pref-opt-for-mols)、[POLO](https://github.com/REAL-Lab-NU/POLO)

**强基线与可证伪假设。** 相同初始数据上的筛选后 SFT/加权最大似然、离线 preference fitting、同先验在线 REINFORCE，以及直接 retrieval/rerank。可检验“在线新增反馈的收益能否超过同标签成本的离线学习”；必须计入建立偏好对/筛选数据的 oracle 成本，不能把线下标签当免费，再说方法更省样本。若只是研究训练计算成本，可单独命名而不是假装 oracle 成本匹配。

**课程边界。** 可以不使用大语言模型或真实化学家标注：在可审计性质上固定偏好规则，考察噪声/覆盖变化和最终候选表现。本课程公告已明确列DPO为示例，因此可独立作为被认可的偏好优化项目；无需另设确认门槛。应准确说明离线偏好训练与在线交互的差别，是否加入在线方法由研究问题决定。

**成熟组件与贡献界限。** DPO、PPO+preference loss、multi-turn editing 均非首创。合格贡献是比较信息预算、标签质量和探索能力，不能把规则偏好一致率等同更好的分子实际效用。公开分子 DPO repo 本次未见明确许可文件，不应因可阅读就假定可以直接复制发行。

## 方向五：lead 结构保持与性质改良

**真实问题。** 已有 lead 的部分结构/系列需要保留，同时改变性质、缓解某种缺点。这与从头设计不同：已有 lead 是输入约束和基准，不能用完全换骨架的更高分分子宣称成功 lead optimization。

**RL 问题。** 把 seed/保留子结构作为条件，研究编辑动作与终端收益、相似度/编辑预算约束、多步小改与一步生成的信用分配。保持指纹相似度、保持精确 scaffold、保持官能团和限制编辑数是不同问题，必须事先选定，不互相替代。

**原始基础与接口。** REINVENT4 的 Mol2Mol/结构装饰接口、POLO 的多轮 lead editing、PMO-Dock 的局部约束任务均提供基础；不必给 pocket，性质可从可计算 profile 开始。[REINVENT4](https://github.com/MolecularAI/REINVENT4)、[POLO](https://github.com/REAL-Lab-NU/POLO)、[PMO-Dock](https://arxiv.org/html/2610.19011v1)

**强基线与可证伪假设。** 原 lead 不变（零改进的必要参考）、相似分子检索+排序、固定条件生成器 best-of-N、简单局部编辑搜索。可检验“在相同保留约束和评价预算下，RL 提高每个 seed 的成功改良率”，同时报告未改动、无合法输出、失败与退化。以 seed 为评估单位，不把同一 lead 的上千个 analog 当独立样本；训练/测试要防止相同或近邻系列泄漏。

**课程边界。** 可研究一种结构约束和一个明确性质改良问题，后续再增加不同系列。若优化 bioactivity predictor，结果仅为预测器代理改良；不能把原 lead 的已知活性转移给所有 analog。只报平均增分会隐藏少数 seed 全失败，需报告覆盖和成功率。

**成熟组件与贡献界限。** similarity constraint、scaffold retention、Mol2Mol、多轮编辑和 best-of-N 已成熟。贡献可在信用分配、保持率与改良率的权衡或 per-seed 失效分析，不必重新发明完整生成器。

## 方向六：目标条件优化与结构信息是否真正有用

**真实问题。** 希望生成分布针对一个目标，而不是普遍好看/大/疏水。条件可以是目标蛋白、口袋或已知 ligand 信息；这只是整个领域的一类，TamGen 是可选 backbone 之一。

**RL 问题。** 条件策略的 target-specific feedback、昂贵/带噪 reward、训练条件保留、对新任务的适应。研究结构条件先验是否降低反馈需求，或 RL 是否把原条件信息“洗掉”。只优化 QED/SA 不能证明目标结合得到改善。

**原始基础与接口。** TamGen 是结构条件生成例子；TDC Docking 与 PMO-Dock 可提供目标相关计算反馈。是否选 TamGen、其他序列/图模型或冻结条件表征，需要在确定问题后比较，不能把环境搭建本身当 proposal 主研究问题。[TamGen](https://github.com/microsoft/TamGen)、[TDC](https://tdcommons.ai/benchmark/docking_group/overview/)、[PMO-Dock](https://arxiv.org/abs/2610.19011)

**强基线与可证伪假设。** frozen conditional generation+同预算 rerank、无条件先验优化、错配/打乱条件、尺寸/理化匹配控制。可检验“条件信息带来预算匹配后的目标 docking preference，而非纯化学分布偏移”；独立 seed/scorer、几何/strain 检查属于计算复核，不证明真实 affinity、选择性或疗效。

**课程边界。** 可以做同目标优化或少量任务迁移，不预设要复现整个 CrossDocked benchmark。依赖受体准备与 docking 的 CPU 预算，GPU 有余也不意味着整体免费。单目标结果不宣称广泛 unseen-target generalization。

**成熟组件与贡献界限。** pocket-conditioned generation、docking reward、on/off-target 分数、重打分和有限预算都已有研究。课程贡献应来自特定问题和严格对照，而非简单把 TamGen 与 PPO 拼接。

## 怎样从六个方向收敛，保持课程问题有深度

选择时可沿三条相互独立的维度讨论，而不是现在排一个“最容易模型”榜：团队对 RL 建模/理论和化学问题哪一侧更有兴趣；主要证据是便宜计算性质、预测模型还是昂贵结构评分；希望理解约束、探索、信息反馈还是局部编辑。前五类都能不依赖 pocket/TamGen。真实应用意义与计算资源可行性应分别评价：便宜的任务可有很深的 RL 问题，昂贵 docking 也可能只有浅薄的 reward 上升。

下表是比较维度，不是已决定的实验矩阵或优先级：

| 研究轴 | 主要 RL 技术深度 | 主要外部依赖 | 最需要避免的失真 |
|---|---|---|---|
| 多性质可行域 | 约束、稀疏反馈、目标尺度 | 可计算性质/可选预测器 | 总 reward 掩盖约束违反 |
| 样本效率与探索 | 探索/利用、replay、分布控制 | 标准 oracle 与去重账本 | 重复/缓存伪装新发现，弱化搜索基线 |
| 合成动作空间 | mask、层级动作、路径归因 | 规则、构件、可选规划器 | 有路径模板冒称可实验合成 |
| 离线/在线偏好 | 支持集、反馈利用、分布漂移 | 固定偏好集/在线 oracle | 离线标签成本隐去，DPO冒称在线RL |
| lead 改良 | 条件编辑、保留约束、信用分配 | 合法 seed/系列和性质端点 | 换骨架冒称结构保持，seed泄漏 |
| target-aware | 条件策略、噪声、任务适应 | 蛋白/口袋与计算反馈 | docking冒称真实affinity/选择性 |

proposal 的有力表述是“在明确条件下比较机制 X 与强基线，检验结果 Y 是否成立”；而不是“安装框架 A、接入 scorer B、最后产生高分分子”。最终可以只选择一条主轴和一条必要诊断轴，避免把约束、反应、偏好、多目标和结构全部叠进一个月。前期工程验收是实验成立的前提，但不是领域问题本身。

老师不要求 paper-level innovation；完整的复现适配、信息/计算预算公平对照、原因清楚的负结果，都可满足研究型课程目标。无提升时应解释目标冲突、先验质量、反馈噪声或搜索覆盖，不通过换目标、挑 seed 和删失败样本制造提升。具体资源与团队确定后，再把所选问题转成可执行预算、数据角色、时间线与职责。

## 本次新增原始来源与许可边界

1. **REINVENT4**，Journal of Cheminformatics 16:20（2024），DOI 10.1186/s13321-024-00812-5；[论文](https://doi.org/10.1186/s13321-024-00812-5)、[官方代码](https://github.com/MolecularAI/REINVENT4)。代码 Apache-2.0，公开 prior 单独放在 Zenodo，依赖/资产分别核许可；未下载。
2. **SynFlowNet**，ICLR 2025，arXiv 2405.01155；[原始论文](https://openreview.net/pdf?id=uvHmnahyp1)、[官方实现](https://github.com/mirunacrt/synflownet)。代码 MIT；reaction templates、building blocks 和兼容 mask 是不同输入，外部 catalog 不能由代码 MIT 推导再分发权；未下载。
3. **Preference Optimization for Molecular Language Models**，2023-10-18 arXiv:2310.12304；[论文](https://arxiv.org/abs/2310.12304)、[作者代码](https://github.com/Harmonic-Discovery/pref-opt-for-mols)。预印本，代码公开但本次根目录未见明确 LICENSE；不能默认自由复用/发行。公开示例不等于全部论文任务都可一键复现；未运行。
4. **POLO: Preference-Guided Multi-Turn Reinforcement Learning for Sample-Efficient Lead Optimization**，KDD 2026，DOI 10.1145/3770855.3819016；[官方代码及论文入口](https://github.com/REAL-Lab-NU/POLO)。主仓库 MIT，veRL/RAGEN 与模型/数据依赖独立核对；其训练/评价细节不能概括成“纯离线 DPO”。未下载/运行。

其余公共基准来源复用相邻 `evaluation_sources.json` 的 E01–E12，不重复执行检索或下载。本文所有未来实验建议均需用户选题和后续执行范围明确后再开展。
