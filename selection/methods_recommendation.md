# 课程选题的方法侧推荐

评估日期：2026-11-04。课程proposal为11月8日，报告为12月6日；GPU已知可用，团队和具体预算待定。本文件选择研究问题，不绑定TamGen，也不因显存预设缩题。按主代理分工，本分支只做原论文/官方代码只读核验，没有安装、训练、生成或服务器操作。下列规模是可讨论的实验设计，不是已运行结果或资源承诺。

**首选：固定评价预算下，scaffold平衡的经验记忆能否比单纯高分记忆找到更多不同的合格分子？** 第二选择是多约束中的固定惩罚与自适应乘子；第三选择是固定数据上的偏好学习与新增在线反馈价值。首选不追求新算法，而以单模型、严格预算和机制消融回答一个可证伪问题。

## 1. 三个项目的取舍

| 项目 | 核心技术问题 | 推荐程度/原因 | 最主要风险 |
|---|---|---|---|
| A. 高分记忆 vs 多样记忆 | Off-policy经验复用、探索/利用、策略集中与样本效率；同一DAP/先验下改变memory保留策略 | **首选。** 原始benchmark和训练组件成熟；问题清楚，负结果容易解释；一个月可完成受控实验 | 领域竞争充分，不能包装新算法；必须匹配训练计算，不能靠额外梯度步数获胜 |
| B. 多约束的自适应惩罚 | CMDP/拉格朗日式policy-gradient；同时满足约束与保留主目标 | **第二。** RL建模/优化深度最清楚；适合团队偏理论与算法 | 自定义profile可能过易/不可行；dual更新振荡，调参容易变成挑阈值 |
| C. 离线偏好 vs 新反馈 | DPO/筛后SFT与在线policy更新；数据支持集、标签噪声、探索价值 | **第三。** 偏好学习是课程允许的路线；可避免复杂3D环境 | 两种预算口径易混；现成DPO脚本未必离线，且发现变长mask静态风险 |

不建议在这一个月同时比较所有表示、多个backbone、3D后处理、逆合成planner和大规模GFlowNet。这个判断来自研究变量与工程依赖数量，不是认为GPU不够。合成/3D可在首选成功后作为单一外部计算oracle扩展，不能成为完成主问题的前置条件。

## 2. 首选A：高分记忆是否在浪费“新发现”预算？

### 问题、假设与已有竞争

实际需求是得到多个不同的可选候选。高分replay把训练集中到已知好结构，可能让top-k提高，却反复生成同一系列；按scaffold限制记忆容量可能改善探索，但也可能降低reward和收敛速度。

**技术假设H1：** 在同一先验、reward、生成批次、唯一oracle预算和梯度训练样本数下，scaffold平衡的高分buffer相对全局top-score buffer，能提高满足固定质量阈值的distinct diverse hits；这个收益应在固定预算前缀与多数种子上出现，不能只存在于选出的最佳一次。

**辅助假设H2：** 全局高分replay可能提升top-10 AUC，却减少高质量集合的多样性；这两种指标排序会不同。H2不成立也是信息，不预设所有replay都会collapse。

最近竞争很直接：ACEGEN（2024）已消融replay/先验约束；Augmented Memory（2024）与Saturn正式版（2026）已研究augmentation/replay和效率–多样性取舍；Diverse Hits（2024）明确在sample/time预算下比较不同生成器。因此不能称“首次使用多样性记忆”。课程贡献应是**隔离memory保留策略和计算量，解释高分与多解发现的差异**。[ACEGEN](https://arxiv.org/html/2405.04657v2), [Saturn](https://doi.org/10.1038/s42256-026-01200-4), [Diverse Hits](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.4c00519)

### 成熟框架与最少修改

建议ACEGEN的同一GRU先验和REINVENT/DAP训练入口，评价参考 `diverse-hits`；这是为了隔离优化变量，不是认定GRU最好。ACEGEN提供原生replay配置和共用采样/评分接口；`divopt`提供唯一分子调用计数、时间停止及多样候选评价。正式实现前锁定commit/依赖，先核验checkpoint，不重训基础生成器。[ACEGEN官方代码](https://github.com/Acellera/acegen-open), [diverse-hits官方代码](https://github.com/ml-jku/diverse-hits)

**三种训练条件足够：**

1. 无replay：当轮样本更新，为去掉memory的消融。
2. 全局top-score buffer：按原始目标分数保留top-K，为去掉scaffold平衡的消融。
3. Scaffold-capped buffer：每个Bemis–Murcko scaffold最多m个分子，其余按同一原始分数保留，总容量仍为K；作为待检验的方法。

先只改变buffer保留规则；两类buffer都用相同采样规则，禁止一边uniform、一边优先采样而不解释。K可先固定100，m固定5作为开发起点，具体值在独立开发任务上核验后冻结。处理无环分子时，空Murcko scaffold不可简单把所有无环分子视为同一化学系列；需统一回退身份规则并报告其占比。

**为什么不把动态diversity reward作为第一版核心：** 当前ACEGEN replayer保存当时reward；历史惩罚随memory改变会过期，且对所有旧样本施加“已见分子=0”会与replay直接冲突。首版让训练reward保持固定，只改变记忆选样，机制更容易解释。若以后加动态filter，单列实验，并重算/清理旧shaped reward，不能直接复用存储分数。

### 最小实验矩阵与公平性

- 任务：优先直接复用Diverse Hits的三个公开人源靶点预测任务（JNK3、GSK3β、DRD2）及其属性过滤/阈值；如改用PMO/TDC oracle，明确这是另一protocol，不能冒称复现Diverse Hits。训练时只复用raw predictor和静态属性过滤，**关闭其默认依赖历史的距离diversity reward**，将多样性放到独立评价；因此是沿用任务/评价组件的课程衍生，不声称完整复现原论文protocol。所有“hit”是**预测器阈值命中**，不是实验活性。
- 开发/确认：选一个预先指定任务与开发种子调通并固定配置；其余任务与新的种子用于确认。若三个任务都参与调参，则诚实称benchmark开发后的重复评估，不称独立跨任务泛化。
- 主比较：3个训练条件×3任务×5预注册种子。训练预算建议10K唯一有效分子oracle调用，（有效canonical分子一旦进入统一scoring pipeline即计一次，静态性质不合格记零也计入；invalid另计attempt），并看1K/3K/10K前缀；最终规模按真实单次运行时间决定，不能把3个前缀当3个独立重复。5种子是计划，若只能3种子需在report中写明并完整显示点值。
- 必须基线：冻结相同先验的等预算best-of-N；同一个冻结候选池仅rerank/去重，不额外claim为独立实验；有现成接口时加入GraphGA或随机筛库作为系统基线，它们与模型消融分开解释。
- 计算公平：除了固定unique oracle calls，匹配每轮新生成数量、梯度步数和参与loss的token/sequence数。例如每轮都生成64个新分子；无replay用64新样本，replay用32新+32旧样本，随机选新样本子集，所有64个新分子仍进入oracle历史与候选集合。warmup不足时统一用新样本。这样避免“有replay只是多算了几倍梯度”。实际token量仍应计数。
- 记录：全部生成attempt、invalid/EOS截断、canonical重复、cache命中、新oracle调用、训练examples、实际token数、wall time。若因重复而达不到10K新分子，应在预注册attempt/time上限停止，记作该方法未用满新分子预算，不能无限运行。

**主要终点：** 固定raw scorer下，达到质量阈值且通过相同性质过滤的diverse hits曲线及其预算AUC。优先采用官方`#Circles`实现与固定距离阈值；它与buffer的Murcko分组不同，降低“训练直接优化同一个计数器”的循环性。算法通常用近似/贪心计算，须固定实现及候选遍历/排序规则，不声称精确解了最大集合。另报合格unique分子、scaffold数、top-10 raw score AUC、最终policy固定大小样本的质量/重复比例。

评价历史候选与评价最终生成policy是两个终点：前者回答“花这些预算找到了什么”，后者回答“学成了什么分布”。新sample的最终policy审计也使用oracle，单独登记统一预算，不偷偷计入某一边训练预算。

### 技术深度和失败分析

可以推导DAP与固定prior正则policy gradient的关系，并说明high-score replay带来采样分布偏差；原先的on-policy无偏解释不能直接搬给replay。再用entropy、agent/prior log-ratio、buffer scaffold集中度、重复率与首次新hit间隔，解释学习动力学。

若平衡buffer没有收益：检查是否只增加了低质量结构、buffer早已自然多样、真实探索瓶颈在prior支持集、奖励太容易饱和，或m造成不可逆损失。若全局buffer更好：保留结论，展示它在哪些预算/阈值下成立。若frozen best-of-N或GraphGA胜出：报告RL额外训练未换来新候选，而不是删掉基线。最终能交付一张预算曲线、一张质量–多样性图和一组collapse诊断即可形成完整论证。

## 3. 第二选择B：固定惩罚与自适应乘子，谁更能满足全部约束？

**具体问题：** 优化一个公开人源靶点的预测分数，同时要求MW/logP/TPSA等进入预先固定窗口；比较固定weighted penalty与自适应Lagrange multiplier，能否在相同预算下提高全约束通过率，而不只抬高总分。

**明确假设：** 当约束的难度不均时，基于当前违反程度调整乘子可减少一个短板长期不满足；但在极稀疏/不可行profile下可能振荡，甚至不如固定权重。不是承诺乘子法保证收敛或保证每个分子可行。

**方法：** 同一个REINFORCE+固定reference的训练loop，目标E[R]，cost为各性质的违反指示或归一化越界量；更新θ以增大E[R−Σλ_j(c_j−δ_j)]，更新λ_j←[λ_j+η_λ(E[c_j]−δ_j)]_+。终点约束指标与训练cost分别定义。δ是期望cost/风险预算；不能称为逐分子硬保证。baseline是同prior固定weighted penalty、原模型+过滤、可选几何平均分。权重/dual学习率开发预算对齐。

**最小设计：** 先固定一组主目标及两种约束难度profile；每profile三种策略×5种子，另报先验过滤基线。profile阈值可参考已发布设置；若根据独立先验诊断设计，必须在训练前冻结并标为课程研究衍生，不借测试结果改窗口。用全通过率、最差constraint、可行集内主目标、梯度/乘子曲线评价；不把reward曲线当主要证据。对不可行profile的负结果，用采样支持、约束冲突与dual轨迹解释。

**成熟基础与竞争：** DrugEx的scalar/Pareto多目标、REINVENT4多组件评分都已存在；本项目并非首次多目标设计，重点是受控约束优化。ACEGEN适合自定义cost和loss；REINVENT4适合成熟reward组合，但若要真正primal–dual更新，需要写清新增训练机制，不能仅改TOML组件就声称实现CMDP。[DrugEx v2](https://link.springer.com/article/10.1186/s13321-021-00561-9), [REINVENT4评分文档](https://github.com/MolecularAI/REINVENT4/blob/main/contrib/reinvent-doc/tutorials/scoring.md)

相对首选，其优点是RL数学更直接，弱点是自定义profile与hyperparameter公平性难于完全沿用公开benchmark。选择B时，建议不再同时加入Pareto/replay/PPO，以免无法解释机制。

## 4. 第三选择C：离线偏好数据是否足够，在线新反馈值不值得？

**具体问题：** 给定同一先验采样并评分的固定数据池，DPO是否比筛后SFT更有效；增加固定数量的新oracle反馈后，online更新能否带来额外收益？以一个明确可审计性质profile或公开靶点预测任务为起点。

**假设：** 偏好优化能在池内支持区域改善生成，但在初始pool稀缺高质量例子时，新增在线反馈更重要；若离线DPO只是在记住winner，训练偏好accuracy高而新候选成功率不升。不可假设偏好标签来自人类。

**分两个实验而不混淆：**

- 固定信息比较：同一pool及相同oracle调用，比较frozen prior、筛后SFT、DPO。按canonical分子先划分，再构pair；同一分子的不同SMILES不能落到train/test两边。保留score差距阈值、ties、invalid和每分子出现频次，避免pair数量膨胀冒充更多真实标签。
- 新反馈价值：另划总query预算，比较继续从固定pool训练、按当前policy新增采样的DPO、在线REINFORCE+reference。新增候选与落选候选都记query；这种比较是策略+数据收集系统比较，不再把差异全归于loss。

**最小规模：** 一个性质/预测任务、两种初始数据覆盖（随机pool与低覆盖子池）、3种固定数据方法×5种子；在线分支作为完成后的扩展，不必proposal承诺同时全部做。主要结果是独立新采样的质量/多样性、对训练pool的相似度和query效率，而不是训练pair accuracy。

**现有竞争与代码：** Park et al. 2023已直接研究分子LM的DPO；近期2026 PolyTamGen等也用oracle偏好，因此“把DPO用于SMILES”不新。可以借ACEGEN共用先验，但不能直接把其 `dpo.py`当offline入口，见下一节源码发现。课程指南已接受DPO示例，技术表述准确即可，不需要额外设审批门槛。[原始分子DPO](https://arxiv.org/html/2310.12304v1), [官方代码](https://github.com/Harmonic-Discovery/pref-opt-for-mols)

## 5. 静态源码核验：直接影响选题的可修复问题

以下是2026-11-04阅读作者main分支的静态观察，尚未运行测试；正式采用前必须固定commit再核验，不声称已修好。

1. **ACEGEN REINVENT脚本含额外高似然惩罚。** `reinvent.py`在DAP和replay loss之外无条件加high-likelihood penalty。三组比较需统一保留，或统一关闭并明确这是受控衍生。不能一边脚本默认一边手写纯DAP后归因memory差异。其replay保存当时reward，强化了首选先保持reward静态的理由。[原始代码](https://raw.githubusercontent.com/Acellera/acegen-open/main/scripts/reinvent/reinvent.py)
2. **其DPO脚本是online采样评分循环。** 每轮生成新候选，按分数分正负后更新，未提供本方案固定pool训练的直接等价实现。更关键的是 `compute_loss`定义了`neg_mask`，但negative likelihood两行乘的是`pos_mask`：正负序列长度不同可能错计token。需用不同长度样本专门核验/修复，不能以README“支持DPO”替代正确性审计。[原始代码](https://raw.githubusercontent.com/Acellera/acegen-open/main/scripts/dpo/dpo.py)
3. **生成去重设置与计数必须独立。** 当前REINVENT采样调用`remove_duplicates=True`，DPO为False；做跨算法比较时不能让它们改变统计分母和反馈成本。应在统一计数层保留全部attempt，在oracle层canonical缓存，在loss层单独定义取样。
4. **不用旧Augmented Memory仓库启动新项目。** 官方README已说明不再更新并指向Saturn。若核心是复现该旧文可固定其版本；若课程重点是memory机制与公平对照，优先维护中的公共框架，减少复原旧环境的无关工作。[作者仓库说明](https://github.com/schwallergroup/augmented_memory)

这些发现不排除ACEGEN，而是把选题从“调用现成配置”变成可审计的训练与评价。首选A所需的buffer/日志和统一loss控制比较集中；C还需固定pool入口、pair数据协议与mask修复，所以暂排第三。

## 6. 11月8日proposal前应确认什么，12月6日前如何收束

**proposal前最重要的不是完整跑出性能：** 确认A/B/C之一；固定一个框架commit和先验来源；用静态接口/最小正确性检查确认sample、logprob、EOS/mask、scorer与预算计数；写清主要终点、baseline和假设。不得把尚未运行的可行性写成已复现结果。本分支没有启动这些试验。

若选A，建议11月9–15日完成训练/计数正确性与开发任务，11月16–24日完成冻结矩阵，11月25–30日做事先规定的诊断与一次必要补实验，12月1–5日固定结果并写报告。若11月15日仍因框架阻碍不能完成最小loop，应换已验证的同类实现或缩减比较变量；不是自动换大模型/服务器。资源允许可增加任务/重复，先保留核心受控矩阵。

**最终推荐A，B作团队偏理论时的替代，C作偏好学习兴趣很强时的替代。** A的可交付结论应写成：“在固定计算proxy和预算下，记忆策略如何改变新高质量候选的发现效率、集合多样性和训练成本。”无论是否提升，都能形成技术正确、证据完整的课程报告。
