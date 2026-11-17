# 分子生成与优化中的RL：领域问题与跨表示进展

核验日期：2026-11-04。本轮按用户纠正，将**整个领域**作为调研对象；TamGen仅为一种条件语言模型实例，不是预选模型或题目。GPU可用，团队/资源尚待确定；不据小显存预设缩题，也不预先固定RL算法。本文应先于 `rl_methods.md` 阅读；后者保留REINFORCE、DAP/KL、PPO、偏好学习及SMILES实现细节，作为技术附录。

所有内容来自原论文、出版者或作者代码/文档。以下是领域背景与待选择的研究问题，无训练、分子生成、安装、权重/大数据下载、远程操作或付费调用。课程无需论文级创新，真正有意义且受控的比较、失败诊断和负结果分析都可构成贡献。

## 1. 为什么分子领域需要RL，而不只是“训练一个生成器”

分子生成器学习“哪些结构像训练数据”；优化需要回答“哪些新候选值得有限评价预算，并同时满足设计约束”。两者目标不同。许多重要反馈只能对完整分子/结构/路线计算：预测活性、物理模拟、逆合成搜索和多属性阈值，常常不可微或代价高。RL将构造或修改分子的选择变成策略，通过评价反馈改变后续搜索分布。

但**RL不是唯一优化方式，也不自动优于搜索**。遗传算法、贝叶斯优化、固定库筛选、条件生成、best-of-N、基于梯度的guidance及偏好训练均可利用目标信息。领域的研究价值在于：指定反馈与约束后，学习策略是否更有效地发现一组有价值的解；其代价是怎样的分布漂移、候选重复、物理失真或合成困难。PMO和ACEGEN提供了预算控制与同架构算法比较的证据基础。[PMO](https://proceedings.nips.cc/paper_files/paper/2022/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html), [ACEGEN](https://arxiv.org/html/2405.04657v2)

这个领域不限于口袋生成。可以优化通用理化性质、限定结构的lead、合成路线、材料电子性质、候选集合覆盖，或显式蛋白条件下的分子分布。“targeted”有时指目标属性而非生物靶点，阅读时必须先辨别。

## 2. 跨表示的方法地图：动作空间决定了能研究什么

| 表示/决策过程 | 策略在选择什么 | 代表原始工作 | 带来的价值与主要代价 |
|---|---|---|---|
| 字符串自回归 | 下一SMILES/SELFIES token、EOS | REINVENT、DrugEx、ACEGEN；LLMol为近期组相对更新例子 | 易复用化学语言先验；语法与分子身份分离，终点credit assignment、长度与重复表示需处理 |
| 原子/键图编辑 | 加/删原子或键、连接位置、终止 | MolDQN、GCPN | 可把价态合法性写入环境；动作数随图变动，图合法仍不保证合成与物理合理 |
| 片段编辑 | 当前连接点、片段、片段连接点 | FREED / FREED++、DrugEx v3 | 缩短轨迹并引入化学片段先验；片段库本身强烈限制可探索区域，不能把所有收益归给RL |
| 反应/合成轨迹 | 反应模板、可用反应物、停止 | PGFS、REACTOR | 返回分子及虚拟路线，直接约束搜索空间；巨大的层次动作空间、模板覆盖与反应模型误差 |
| GFlowNet多解生成 | 可为图、片段或反应步骤，学习前向/后向流 | GFlowNet、SynFlowNet、RxnFlow | 目标是按正reward比例采样多解，不只追逐极大值；多路径同终点与后向policy设计增加难度 |
| 连续latent编辑 | 生成器潜变量中的位移/动作 | [MOLRL（2025）](https://www.nature.com/articles/s41598-025-99785-0) | 决策可低维且连续；解码的有效性与latent几何决定优化可达范围，不等同直接生成器token更新 |
| 3D diffusion / discrete graph flow | 带噪图/坐标的反向转移，或局部扰动再生成 | RLPF、Graph-GRPO | 直接对齐3D物理或图性质；需正确转移概率/对称性，不能照搬SMILES logprob |
| 3D flow matching / 组合流 | 连续坐标速度场、离散化学特征、结构组合/反应步骤 | 2026 FM-RL、CGFlow | 可把构象、图和路线一起建模；不同“flow”目标不相同，不能混称一种算法 |

表中的经典来源详见 `rl_method_sources.json`。下文新增来源编号F01–F13见 `field_extension_sources.json`。这些方法族可以交叉：反应空间可以用actor–critic，也可以用GFlowNet；3D模型可以在线PPO，也可以离线偏好对齐。不能只按“RNN vs Transformer”给领域分代。

## 3. 合成可达性：从“像药”走向“在给定条件下可做出来”

### 3.1 合成约束写在哪里，是两条不同路线

**By construction：把合成动作写入环境。** 从库存building block开始，选择可匹配的反应与反应物，让虚拟产物成为下一状态；最终分子自带生成轨迹。PGFS（ICML 2020）把大规模反应物选择通过连续表征与近邻环境相联系，并用TD3式actor–critic；这是层次反应选择问题，非语言模型fine-tuning。[F01，原文§3](https://proceedings.mlr.press/v119/gottipati20a/gottipati20a.pdf)

REACTOR（ACS Omega 2020）用A2C选择反应及终止，mask掉不可行反应；其报告实验用启发式选择后续反应物与多产物，并将该内部选择并入环境。因而“RL选择全部合成细节”并不准确，启发式中额外评价也应计成本。作者展示的是理论路线并承认不一定是实验首选路线。[F02，§3.2](https://arxiv.org/html/2004.14308v2)

**在reward中评价合成：保留较通用的结构生成器，调用逆合成planner为候选反馈。** 这样不必把生成空间完全限制在一套正向动作中，但需要昂贵搜索，容易遇到稀疏成功和planner偏差。2025 Chemical Science原始研究将逆合成直接纳入RL环，而非只在末尾过滤；其结果有条件性：在部分药物任务/相同wall-time下，便宜SA heuristic比每次逆合成更划算，而在材料域其相关性减弱，直接规划更有价值。不能把“更高保真oracle”自动解释成“更好预算分配”。[F03](https://pubs.rsc.org/en/content/articlepdf/2025/sc/d5sc01476j)

### 3.2 TANGO将“限定合成约束”作为独立学习问题

TANGO正式发表于Nature Computational Science，2026-03-03；前身是2024预印本。它关注：不仅planner能找到路线，还要让指定原料/中间体出现在路线中。TANGO结合Tanimoto与结构重叠，在预测路线非根节点上衡量接近指定building block的程度，将稀疏成功转换为连续反馈；再与其他性质共同用于通用生成器RL。**它是reward与约束设计，不是一种新反应动作生成器。** 路线不存在时仍无法提供同类连续反馈；文中“可合成”定义为规划器判定，不能换写为已实际合成。[F04，Results与TANGO reward function](https://www.nature.com/articles/s43588-026-00959-1), [合法公开全文索引](https://pmc.ncbi.nlm.nih.gov/articles/PMC13658032/)

这里的研究主题不是“要不要加一个SA分数”，而是：哪些约束应该成为动作规则、哪些应成为学习反馈；怎样让极稀疏目标可学习；怎样衡量对规划器的过拟合。TANGO*是另一个有关约束逆合成搜索的工作，不应与这里的生成式TANGO混为同一实现。

### 3.3 “By construction”也只是在计算模型内成立

以下是本调研的证据边界推论：反应模板的匹配可执行，不代表该底物在现实条件下有好收率/选择性；库存快照也不保证交付与成本。逆合成未找到路线可能只是搜索/模型/库存覆盖限制，不是化学上绝对不可合成。于是合理的评价应同时保留：虚拟路线、使用的反应/库存身份、步数与复杂度、独立planner可解率及失败原因；不能用“已提供路线”取代湿实验。

不同方法的可达空间也不同。若反应库、stock、最大步数同时变化，只比较最优属性不能归因RL。另一方面，强行让通用SMILES方法和反应方法共享完全相同轨迹空间也可能失去各自问题意义；应分别给系统比较和受控算法比较。

## 4. 从一个最优解到一组高质量解：GFlowNets与路线/3D结合

一般最大期望reward的最优policy可以集中在极少数解上；先验/entropy/diversity常用于缓解这一点。GFlowNets的目标不同：在相关条件与学习假设下，终点采样概率与给定正reward成比例，强调多个高reward模式。它并不保证均匀覆盖，也不使reward自动可靠；reward变换/temperature仍控制高分集中度。[GFlowNet原文](https://proceedings.neurips.cc/paper_files/paper/2021/file/e614f646836aaed9f89ce58e837e2310-Paper.pdf)

2025两个代表将这个目标带入合成空间。**SynFlowNet**把可购买原料与反应写入动作，并研究反应编码带来的后向policy困难；使用独立逆合成工具评估，而不是把虚拟路径当成最终真值。**RxnFlow**同样采用反应轨迹的GFlowNet目标，重点解决巨大的组合动作空间，通过action subsampling提升可扩展性，也区分pocket-specific与pocket-conditional生成。这说明“合成约束”“大动作空间”“多解探索”可以在一个问题中相互作用，而不只是在最终reward上加权。[F05，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7495fa446f10e9edef6e47b2d327596e-Abstract-Conference.html), [F06，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/24d36eee157559e0d2549455fba28f6a-Paper-Conference.pdf)

**CGFlow（ICML 2025）**进一步联合组合结构与连续状态，将反应路径构造同3D姿态生成结合，借助GFlowNet式目标进行reward-guided组合采样。这是路线–构象共同设计的代表，不是仅对最终3D分子重排。其证据仍是计算benchmark；“虚拟路径+好docking”不能等同实验成功。[F07，原文§3](https://arxiv.org/html/2504.08051v2), [官方代码与版本说明](https://github.com/tsa87/cgflow)

**术语提醒：** GFlowNet中的flow是构造轨迹上的概率质量流；flow matching是学习连续/离散状态演化的生成模型。二者可结合，但不是同义词，名称含flow不代表都按reward比例采样。CGFlow中的可训练组合policy、3D state-flow与reward目标需分别理解。

## 5. 在线、离线与偏好：先区分“得到什么反馈”

| 反馈设置 | 训练时能否评价新候选 | 学习内容 | 分子领域最容易混淆的地方 |
|---|---|---|---|
| Online RL | 可以，policy不断采样并调用oracle | 在线收集的轨迹/分子改变policy | oracle是QSAR/docking/force field时，online只是与计算环境交互，并非在线做湿实验 |
| Off-policy RL + replay | 可以，但还重用旧轨迹 | 当前policy从历史经验学习 | replay不自动变成offline RL；采样不来自当前policy时需说明算法/估计器依据 |
| 严格offline RL | 固定数据，训练不再新增环境交互 | 从固定transition/return经验学policy/value | 对数据支持集外动作估值可能失真；仅有结构–性质表不一定有可用轨迹，不能把普通监督模型随意叫offline RL |
| Offline preference alignment | 固定winner/loser或连续分数数据 | DPO/ERA类直接更新生成模型 | 不是仅重排；但标准DPO也没有online rollout loop，不宜直接称完整在线RL |
| Online preference alignment | 当前模型周期采样、打分、形成新偏好对 | 数据收集与偏好优化交替 | 把落选候选计入oracle预算；数据新鲜程度与loss收益分开比较 |
| Learning-free guidance / selection | 不更新生成policy参数 | 调整采样轨迹、筛选或搜索 | 改采样过程可很有用，但不能冒称已RL fine-tune |

另一个术语例子是MOLRL：它冻结encoder–decoder，用PPO训练latent-space动作policy；论文称每批后的多轮更新为“off-line update”，不意味着固定数据集的严格offline RL。其一步式优化设定也提醒我们，不能只因叫RL就默认存在长序列规划。[F13，Methods](https://www.nature.com/articles/s41598-025-99785-0)

原始DPO与ERA支持上述偏好路线的定义；本次未找到并深入核验一个足以代表全领域的“固定分子transition数据上的保守offline RL”基准，因此不硬指定CQL/IQL为分子主流。该缺口不意味着不存在相关研究。若选择offline方向，应先确定真实数据能提供何种反馈与轨迹，再选算法。[DPO](https://arxiv.org/html/2305.18290v3), [ERA](https://arxiv.org/html/2405.12961v3)

这一区分独立于表示。**AliDiff（NeurIPS 2024）**对靶点条件3D diffusion做energy preference alignment，并给出E2PO以处理偏好优化的分布匹配问题。**DecompDPO（TMLR 2025，预印本始于2024）**将分子级和子结构级偏好结合，并加入物理相关energy项。这两项说明偏好对齐已超出SMILES语言模型；其证据是计算性质与docking，不是人类偏好采集，也不是湿实验。[F08](https://proceedings.neurips.cc/paper_files/paper/2024/file/4ddfe69f164eae70abc86f0f9cbed7e8-Paper-Conference.pdf), [F09](https://arxiv.org/html/2407.13981v3)

## 6. 3D diffusion / flow的RL：研究难点已不只是“token reward”

### 6.1 把去噪视为决策过程

一种形式化是state=(噪声时间、当前坐标/原子类型、条件)，action为下一反向生成状态，最终3D分子的物理/性质反馈为terminal reward。这样可以利用policy-gradient思想；然而需要正确的转移概率、对称性、离散/连续联合分布及分子大小处理。**不能把一个普通确定性ODE更新直接当成categorical token概率来套PPO。** 这是本调研对算法接口的要求，不是声称所有3D方法使用同一个MDP。

RLPF（2025-08预印本）把denoising diffusion policy optimization延伸到分子，使用PPO式策略更新及force-field反馈，目标是更合理的平衡几何；实验包含QM9、GEOM-drug。降低残余力/应变能不等于提高靶点结合或实验活性，该问题可在不以口袋为中心时研究。[F10，§4](https://arxiv.org/html/2508.16521v1)

Graph-GRPO（ICML 2026）则针对**离散图flow模型**，推导转移概率并配合局部扰动再生成，用组相对策略优化对齐任务reward。它不是3D坐标模型，也不是另一篇同名的multi-agent通信拓扑Graph-GRPO。[F12，正式会议页](https://proceedings.mlr.press/v306/zhu26b.html), [作者代码](https://github.com/Zhubaoheng/Graph-GRPO)

### 6.2 不经轨迹likelihood的reward-weighted flow更新

2026-09 Communications Chemistry工作从SemlaFlow先验出发，用在线评估的reward对flow-matching训练loss加权，联合调整连续坐标和离散化学特征，并学习分子大小采样；还对prior附近作正则。其价值是给出了不同于直接轨迹likelihood policy gradient的fine-tuning路径，支持不可微reward及无条件/有条件场景。**不要因此把reward-weighted FM loss与PPO/clipped likelihood ratio说成同一算法，也不要把参数接近prior称为已经约束了分布KL。** [F11，Methods与Algorithm 1](https://www.nature.com/articles/s42004-026-02188-z), [官方实现](https://github.com/MolecularAI/flow_matching_rl)

### 6.3 3D评价必须拆开

以下为本调研的评估推论：图能通过RDKit不代表坐标合理；坐标经过大幅最小化后很好不代表生成时已有好姿态；redocking产生的好分数不代表原姿态好。至少区分化学有效性、几何/应变/碰撞、直接pose评分、relax后评分、redocking后评分，并保留后处理幅度和成本。优化force field、docking或其组合时，独立检查不应只重复同一个reward。3D模型并不天然比分子字符串更接近实验可信性，关键是信息与验证层次。

## 7. 独立于模型的研究问题地图

以下是可以先写入领域背景、再挑选的**问题家族**。它们是已有方法支持的课程切入口，不是假称文献尚未解决，也未锁定实现或实验规模。

| 领域问题 | 可检验的研究问题 | 候选方法/控制变量 | 应看哪些证据 |
|---|---|---|---|
| 合成可达性与可探索性 | 将约束写入动作，还是交给oracle学习，在什么情况下更好？ | 同类目标下反应动作与通用生成+规划；固定stock/模板/搜索预算；区分系统与算法比较 | predicted route可解率、性质、覆盖、route长度、实际planner/wall-time预算 |
| 极稀疏成功与credit assignment | 连续shaping或curriculum是否帮助达到原始严格成功，而不是仅抬高代理？ | 原始binary终点 vs shaping/curriculum；同policy/oracle；训练与最终成功定义分开 | time-to-first-success、严格成功分子数、失败种子比例、约束满足短板 |
| 高分多解分布 | 是否值得牺牲少量最高分换取更多不同模式？ | 普通最大期望reward、diversity正则、GFlowNet；比较各自目标而非仅top-1 | distinct合格分子/scaffold、模式覆盖、质量–多样性front、分布集中度 |
| 样本效率与计算效率 | 少量昂贵评价时，应探索、replay、学习surrogate还是筛库？ | 固定oracle预算之外记录optimizer/sample/wall-time；replay/再采样/GA/筛库 | AUC、oracle burden、cache命中、真正新高质量解数、时间与方差 |
| 多属性短板/鲁棒满足 | 总分提高是否以一个性质严重退化为代价？ | Weighted/geometric/Pareto/阈值约束，固定各项变换 | 全项通过率、最差分量、Pareto front/hypervolume、未聚合分量 |
| 物理合理性 | 把物理反馈纳入训练是否比统一后处理更有效？ | 同一3D backbone：原模型、等预算relax/filter、reward更新 | 原姿态几何、relax幅度、应变/碰撞、独立评价、有效率/多样性损失 |
| 数据支持与反馈泛化 | 在线反馈是否超过固定偏好数据？收益来自loss还是新数据？ | 同一初始数据池，offline偏好与online更新；固定新增query | held-out条件、数据距离/覆盖、独立proxy、全部训练查询成本 |
| 跨目标/跨条件泛化 | 在一个条件上会优化，是否能迁移到其他条件？ | 固定目标优化 vs 显式conditioned policy；按条件真正隔离调参与测试 | per-condition结果、失败覆盖、同源/同scaffold泄漏、对未见组合与未见实体分别说明 |

**如何从领域背景走到proposal：** 先选一项科学/决策矛盾，再确定可获得的反馈和合适表示，随后选择能隔离这个矛盾的最小算法比较，最后才确定模型、规模和资源。可以使用成熟模型/已有RL算法；贡献在于把假设、预算、失败和作用机制讲清楚，而不是先挑一个模型再寻找可加的模块。

TamGen只在“字符串自回归+显式生物条件”这一分支中作为实例。若选合成路线、无条件物理合理性、反应GFlowNet或offline反馈问题，可能完全不需要它；也不需要先决定REINFORCE、PPO或DPO谁是主算法。

## 8. 适合proposal的领域背景草稿

分子生成模型已经能够学习化学结构分布，但药物与材料设计的目标通常是满足多项相互制约的性质，并在有限评价预算下产出一组可进一步验证的候选。许多关键评价涉及不可微预测器、物理模拟或合成路线搜索，因此强化学习和奖励驱动的生成模型对齐成为连接“能够生成”与“能够按目标搜索”的重要方法。已有研究从SMILES策略梯度和图编辑，扩展到片段/反应决策、奖励分布采样的GFlowNets，以及3D diffusion/flow模型的反馈优化。

当前值得分析的难点并不是简单提高某个计算分数，而是处理高分与多样性、约束满足与探索范围、样本效率与计算成本、代理反馈与真实可验证性之间的取舍。近期合成约束与3D对齐研究进一步表明：动作空间、反馈定义和评价协议与RL算法同样影响结果。因此，本项目将从上述领域问题中选择一个明确、可验证的问题，在受控模型与预算条件下比较已有方法，分析提升、退化与失败，而不预先声称提出新药或新的通用算法。

这两段是本调研综合形成的草稿，不是对单一来源的翻译；最终选题后应为具体主张补相应文献并替换“将选择”的占位语。可优先引用PMO、ACEGEN、TANGO、SynFlowNet或所选3D方法，避免为凑年份堆砌引用。

## 9. 证据边界与交付状态

新增13条原始来源记录；未把未读补充材料或未运行代码写成已核验实现。原作者仓库存在不等于环境可跑、checkpoint兼容或课程可按时复现。TANGO及部分出版者页面直开有重定向/反爬，通过出版者检索正文/公开作者版核对，记录于JSON；PGFS原文链接的Apollo1060主要说明其案例数据/预测代码，本次不冒称已核验完整训练代码。REACTOR完整官方训练仓库未在本次确认。

本文件形成领域地图和问题候选，未开始选定题目的实验；GPU/团队具体资源不影响目前的文献覆盖范围。待用户选择问题方向后再做可复现性与预算审计。
