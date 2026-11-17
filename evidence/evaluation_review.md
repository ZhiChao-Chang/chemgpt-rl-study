# 分子生成 RL：评估、奖励、公开基准与可行性审阅

核查日期：2026-11-04。仅公开文献、官方代码和小型 metadata 审阅；没有生成/优化分子、训练、安装软件、下载权重/数据包或访问远程算力。依据最新用户说明，GPU 确定可用，团队与具体资源待定；以下优先说明领域问题，不预先选定靶点、算法或实验矩阵。文中“建议/推断”是本任务的方法设计判断，不是已验证结果。

## 核心判断与尚未解决的问题

有意义的课程问题不是简单地把 TamGen 接上 PPO 并报告平均 QED 上升。至少应区分三个层次：策略是否比冻结生成器的等预算搜索更有效；优化出的奖励能否在未用于训练的计算检验中保留；条件输入究竟改变了对目标口袋的偏好，还是仅改变分子大小/脂溶性。课程允许可靠负结果；完整回答其中一个问题，价值可以高于堆叠多个复杂模块。

公开评估的发展脉络是：GuacaMol/MOSES 把生成质量与分布度量标准化；PMO 强调固定 oracle 次数下的样本效率；TDC Docking 加入真实运行的 docking oracle；近期 PMO-Dock 把 hit generation、局部优化、on/off-target docking 以及开发/最终任务隔离放入同一协议。因此，“固定预算”“加 docking”“看多靶点”“加重评分”本身已不能称为首次创新。课程可以做清晰的适配、对照和失效机制研究，不需要声称发明新范式。[GuacaMol](https://github.com/BenevolentAI/guacamol)、[MOSES](https://github.com/molecularsets/moses)、[PMO](https://arxiv.org/abs/2206.12411)、[TDC Docking](https://tdcommons.ai/benchmark/docking_group/overview/)、[PMO-Dock](https://arxiv.org/abs/2610.19011)

截至本次审阅，尤其值得区分的未解问题包括：昂贵奖励下探索与模式坍缩的权衡；结构条件先验在 RL 后是否被保留；稀疏可行性约束下的信用分配；重复 SMILES/重复分子导致的虚假样本效率；有限任务上调参能否迁移；高 reward 是否利用 scorer 的盲点；尺寸和化学分布变化能否解释表观收益。它们是可研究的方向，不代表本文已经证明哪个方法最好。

## 基准到底回答什么

| 基准/资源 | 已核实的事实 | 对本项目的正确用途和限制 |
|---|---|---|
| GuacaMol | 官方接口区分 distribution-learning 与 goal-directed；标准 ChEMBL 数据按 train/validation/test/all 分文件及 hash 发布；代码 MIT | 适合明确属性目标与基本生成质量。自定义 TamGen 预训练语料不同，不能直接把结果标成严格同协议排行榜复现。 |
| MOSES | 约 1.94M 过滤后 ZINC 分子；train/test/scaffold-test；validity、uniqueness、novelty、IntDiv、SNN、FCD 等；代码 MIT | 提供生成质量度量，不能证明靶向结合。官方建议至少 30,000 samples 估计完整分布指标，课程少量 docking 样本不能把不稳定 FCD 当主要结果。 |
| PMO（NeurIPS 2022） | 23 tasks、25 algorithms、10,000 oracle queries，主要 top-10 随调用数 AUC；作者代码 MIT | 可借鉴样本效率和多 seed，但它主要是便宜 property/QSAR 目标，不是结构结合真值。 |
| TDC Docking group | 网页 DRD3 leaderboard 支持 100/500/1000/5000 calls；要求至少 3 random runs；top1/top10/top100、novelty、diversity、filter pass 等 | 便于约束预算；可不使用额外 Molecule.one API。网页的 DRD3 榜与通用 oracle 支持靶点列表必须区分。 |
| PMO-Dock（2026-10-13 v1，预印本） | 25 tasks，开发/最终任务分离；hit B=3000，lead B=1000；使用 QuickVina，并分析重打分/几何合理性；代码已公开，包声明 Apache-2.0 | 是当前评估设计的重要参照；尚不能当成熟统一共识或已由本项目复现。无需照搬全部任务/参数 sweep。 |

上述资源事实分别见 [GuacaMol README](https://github.com/BenevolentAI/guacamol)、[MOSES README](https://github.com/molecularsets/moses)、[PMO 论文](https://arxiv.org/html/2206.12411v2)、[TDC group](https://tdcommons.ai/benchmark/docking_group/overview/)、[PMO-Dock v1](https://arxiv.org/html/2610.19011v1)。

CrossDocked2020 的原始发布是多 pose/cross-docking 的结构数据资源，其官方 v1.3 全原始包包含超过 5,200 万文件；不要把这个原始包与后续生成模型的约 100k pair、100 test pocket 派生版本混称为同一下载单位。Pocket2Mol 官方 test interface 的索引为 0–99；TamGen 的 data README 列出结构目录、test-info.csv、坐标与二进制角色文件。100 pockets 是常见测试样本规模，**不是天然证明有 100 个互相独立的生物靶点**。[CrossDocked 官方说明](https://github.com/gnina/models/blob/master/data/CrossDocked2020/README.md)、[Pocket2Mol](https://github.com/pengxingang/Pocket2Mol)、[TamGen data](https://github.com/microsoft/TamGen/blob/main/data/README.md)

建议真正选择任务前建立 pocket→PDB/chains→UniProt→sequence cluster→ligand scaffold 的小型 manifest，核对原 split 而非自己推测。相同蛋白不同构象、同源家族、重复配体都影响独立性。不能默认别人的低同源 split 自动适用于新 checkpoint 的全部预训练数据，也不能把经过挑选的 1–3 口袋结果写成普遍 unseen-target 泛化。若多个口袋来自同一靶点，统计应按靶点/cluster 分组，避免把数千分子当作数千独立实验。

PDBbind 是另一资源，不是使用 TamGen checkpoint 的必选前置训练集。本次官方 PDBbind+ 页面仅返回 JavaScript 壳，旧 enroll 页面无法读取，**没有核验当前版本的完整许可协议**；因此不承诺可自由再分发、商用或公开打包。若未来选用，只在核实版本、账户条款和授权后获取；课程仓库优先公开 accession/split/hash/处理脚本，不因 PDB 结构公开就推定 PDBbind 整理包也具有相同授权。[PDBbind+ 官方入口](https://www.pdbbind-plus.org.cn/)

## 奖励含义必须先分清

| 项 | 可用于什么 | 不能据此声称什么 |
|---|---|---|
| Docking score | 给定受体准备、box、搜索算法、随机种子条件下的计算评分；更负通常更优，奖励方向可显式取负 | 不是实测 Kd/Ki/IC50，也不能由单位标签直接换算成真实 affinity 或疗效 |
| QED | 经验 drug-likeness 汇总量，适合辅助约束/诊断 | 不保证活性、ADMET 安全性或临床可开发性 |
| SA score | 片段与复杂度构成的合成难易 proxy，适合辅助筛查 | 不是可执行合成路线、实验产率、原料供应或安全保证 |
| cLogP / logP | 必须注明具体预测器，例如 RDKit Crippen MolLogP | 预测值不是实验测量，更不是水溶解度；不应无限最大化 |
| penalized logP | 特定代码定义的复合优化目标 | 不是原始 cLogP；不同实现的标准化/环惩罚可能不同 |
| QSAR / ADMET predictor | 已定义端点上的模型预测，可用于辅助研究 | 生成分子可能落在适用域外；不能与实测端点混报 |

具体例子：TDC 当前 `penalized_logp` 源码对 Crippen logP、负 SA、环惩罚分别标准化后相加，不能粗略写成原始 logP 或把未标准化公式当成实际实现；GSK3B/JNK3 返回 fingerprint 分类模型概率，也不是受体 docking。[TDC oracle 源码](https://raw.githubusercontent.com/mims-harvard/TDC/main/tdc/chem_utils/oracle/oracle.py)

建议研究靶向 RL 时保留至少一个 **target-specific** 项（docking、经审计 target predictor 等），同时把 QED/SA/大小约束明确为辅助。若只优化 QED/SA，模型可能仍接受 pocket 条件，但优化目标是 target-agnostic 的理化属性；不能据此宣称 RL 改善目标结合。是否需要 weighted sum、hard constraint 或分阶段优化，应由选定问题决定。加很多 proxy 不等于更接近真实药物发现。

建议把原始 docking、QED、SA、MW/heavy atoms、cLogP、rotatable bonds 分开报告。若 RL 更偏好大分子/疏水结构，应进一步比较相近 heavy atom/MW/cLogP 桶内结果或匹配控制集，检查 improvement 是否只是分布偏移；不能把 ligand efficiency 单独当完全校正。Vina 官方也建议针对具体靶点先评估，并在 retrospective screening 使用大小等特性相近的 decoys。[Vina manual](https://vina.scripps.edu/manual/)

## 建议的公平比较约定（研究设计，不是已运行结果）

1. **两条公平预算轴。** 主轴固定每 run 的新分子昂贵 oracle 预算 B；同时固定最大生成尝试数、墙钟/CPU、温度/top-p、长度上限、失败重试政策。按采样数、调用数、计算时长分别画图，不把“唯一分子更多”自动当每次调用更省。预算用完必须停止向 oracle 发起新请求，不能无限搜索后再挑有利前缀。
2. **两个计数器。** `n_unique_scored` 按事先固定的化学标准化身份计数；`n_actual_oracle_invocations` 记录每次真实执行（含失败、重新准备、不同 seed/conformer/target）。同时记录 invalid、disconnected、cache hit、timeout、retry。一次分子对两个靶点各三次 docking 是六次真实执行，不是“一次廉价调用”。
3. **缓存身份。** cache key 至少含 standardized isomeric molecular identity、target/receptor preparation hash、engine/version/scoring mode、box、search settings、pose/conformer/protonation policy、seed。不同 SMILES 表述不能重复计成独立发现；不同随机种子复核不应误中旧缓存。所有方法采用同一 canonicalization/tautomer/stereo 保留政策。
4. **核心基线。** 冻结 TamGen 原始采样；冻结 TamGen 的等 B best-of-N/reranking；简单 REINFORCE/带 baseline 的 policy gradient；再视研究问题加入 PPO、replay 或其他改动。等 B rerank 是识别参数更新收益的关键：RL 高分不等于优于更多采样加排序。所有 baseline 使用同 checkpoint/target 信息及相同额外评价预算。
5. **PPO 与 REINFORCE 区分。** 保持 token mask、EOS、length handling、reward、KL/reference policy、优化范围、采样和预算一致。PPO clipping/多轮更新与 value baseline 会改变优化稳定性和计算量；报告 updates/gradient steps/wall-clock，别把额外计算归为纯算法收益。可逐步比较 REINFORCE→baseline/KL→PPO，而非一次同时改多个部分。若使用 REINVENT-style augmented likelihood，不能直接把它标成 PPO 或纯 REINFORCE。
6. **过程和最终策略分开。** 一张图展示在线已评价集合的 best-so-far/top-k/AUC（去重后）；另一张表用选定 checkpoint 重新采样固定数量，检查最终策略平均质量、validity、diversity 和模式坍缩。搜索日志里的最好分子不代表最终 policy 常规输出。重采样评分也必须有独立、各方法相同的预算。
7. **多 seed 及选择纪律。** 至少 3 seeds 起步；资源允许再增加。用相同 target/task 设置成对比较，保留每 seed 数值与均值/波动，少 seed 下别过度解释显著性。开发任务调参，最终任务/预算/种子预先冻结；不能用测试结果选温度、分子过滤、reward 权重、最优 seed 或 checkpoint。
8. **质量与失败率。** 至少报告 validity、unique valid fraction、novelty、scaffold diversity、pairwise fingerprint diversity、filter pass、oracle failure fraction、top1/top10（B足够时top100）、受约束有效候选数量。给 top set 与全部输出两种分母；不能删掉失败分子后只报漂亮均值。
9. **novelty 的限定。** novelty 必须注明针对哪个已知训练/参考集合、canonicalization 和相似度定义。TamGen 预训练与后续优化数据未完整核对时，只能称“相对可获得参考集未重复”；不等于历史文献首次发现、专利新颖性或无训练泄漏。分子 novelty、scaffold novelty 与 target novelty 独立报告。

## 计算验证如何避免换名刷分

Vina **score-only、local optimization/minimize、full docking/search** 是不同操作，输入 pose 与搜索自由度不同；不得一组报 score-only、另一组报 full docking 后合并比较。固定 engine、版本、receptor/ligand preparation、box、exhaustiveness、seed、CPU 和输出选择。TamGen 属于结构条件序列生成，不能因为有 pocket 条件便假定产生了可直接作为真实结合 pose 的三维配体。[Vina manual](https://vina.scripps.edu/manual/)

可以预先保留少量 top candidates 和随机/尺寸匹配控制，在未反馈给训练的 docking seed 下重算，检验 seed 敏感性；也可选不同打分器/engine 做复核。后者有不同误差但并不一定统计独立，Vina/QVina 尤其不能当完全正交的生物实验。PoseBusters 可补充 stereochemistry、bond/angle、clash 与几何合理性检查；可另记配体 strain/力场能量差，但应冻结构象生成、力场及参考定义。这些都属于 **computational validation**，不能升级为真实结合/生物选择性证据。[PoseBusters 论文](https://doi.org/10.1039/D3SC04185A)

“同 pocket 采样→同 pocket reward”说明目标内优化；target-conditioned 对比错配/打乱 pocket 输入、同一组分子交叉打分不同口袋、匹配大小/理化控制，可以检验是否有条件依赖或计算 docking preference。未标注的 decoy 不是实验 inactive；on/off-target 分数差也不能证明真实选择性。所有控制中的额外调用按相同规则记账，不把验证信息回流到模型选择。

## 靶点示例只作为后续选项，不提前锁定

TamGen README 支持 `PDB ID + pocket center` 或由结构内 ligand 定位的自定义输入；当前证据未证明下面三者均已在它的原 test-info/pocket bundle 中。TDC main 的 metadata 当前有 **9** 个 prepared docking target entries（不是早期常说的 5 个），并给出 box 信息；网页 DRD3 leaderboard 是另一层。[TamGen](https://github.com/microsoft/TamGen)、[TDC metadata](https://raw.githubusercontent.com/mims-harvard/TDC/main/tdc/metadata.py)

| 可考虑的人源常规靶点 | 已核实入口 | 后续适用条件 |
|---|---|---|
| DRD3，3PBL | RCSB 明确 human dopamine D3 receptor；TDC prepared entry 和网页 benchmark | 适合复用已有benchmark概念；结构是 T4-lysozyme fusion，有工程突变；必须选对 receptor 部分并保留原 ligand/control 的身份说明。 |
| ADORA2A，3EML | human adenosine A2A receptor；TDC prepared entry | 有共晶配体便于以后校准；同样是融合构建体。不能凭 docking 判断 agonism/antagonism 或生理作用。 |
| ADRB2，3NY8 | human beta2 adrenergic receptor；TDC prepared entry | 有 inverse-agonist-bound 结构；存在融合/工程构建条件，单构象不能覆盖全部功能态。 |

物种和构建体证据来自 [3PBL](https://www.rcsb.org/structure/3PBL)、[3EML](https://www.rcsb.org/structure/3EML)、[3NY8](https://www.rcsb.org/structure/3NY8)。反例：TDC 常用 1IEP 实际 RCSB 标为 **Mus musculus**，不能称 human ABL1。[1IEP](https://www.rcsb.org/structure/1IEP)

三个人源示例均为 GPCR；若最终选择多个，不能把它们当跨所有蛋白家族的代表。是否改用已审计的人源酶/激酶、少量 CrossDocked held-out clusters，取决于最终科学问题与metadata审计，不应只因为现成 wrapper 就锁定选题。本文不选择病原体/毒素设计任务。

## 研究方向选项与课程阶段安排（非预定执行矩阵）

| 可选研究轴 | 可检验问题 | 最有解释力的对照 | 可报告的负结果 |
|---|---|---|---|
| 样本效率/更新规则 | 有限 oracle 下策略更新是否优于 frozen best-of-N？ | frozen / rerank / REINFORCE；PPO作为后续比较 | 相同预算下 PPO 无优势，解释稳定性/探索/计算开销 |
| 保留结构条件 | RL 后是否丢失 pocket dependence？ | shuffled/wrong-pocket conditioning，属性分布匹配 | 属性改善但目标条件差异消失 |
| 奖励可靠性/约束 | 奖励上涨是否被尺寸或 scorer 盲点解释？ | raw reward vs constraint-aware reward，held-out scoring | 训练 oracle 收益不能迁移到计算复核 |
| 多目标探索 | hit yield、quality、scaffold diversity 如何权衡？ | 同预算 scalarization/constraint方案 | 最优 reward 导致单 scaffold 坍缩 |

3–4 周的保守组织方式是：第一周复现原模型接口、核对 metadata 和 oracle 协议，冻结一个主要问题及对照；第二周完成最小比较与日志/预算核验；第三周增加 seeds、对照和未反馈的计算验证；最后一周集中误差分析、图表、复现材料和汇报。先选研究轴，再据团队/CPU/docking实际耗时决定 1 个或多个开发/最终任务。可从数百至数千 oracle 次数逐级测可行性，但这些仅为设计示例，不是本次运行授权，也不预设最小算力就是唯一合适方向。

若旧 fairseq 环境/显存使完整微调不可行，可以比较冻结生成器的 rerank 与可训练小策略模块，但必须据实写出优化变量变化；若 docking 吞吐成为瓶颈，可先将便宜属性任务用于工程验证，最终结果明确命名为属性优化；若只能在单个目标上完成，聚焦同目标 optimization 与严格 matched control，撤回跨目标泛化结论。不要通过偷偷使用 surrogate 标签、缓存旧分数、降低某方法搜索强度或少报失败来补救成本。

## 公开获取与许可核查小结

- TamGen：代码 MIT；权重/数据由 Zenodo 各自的 zip 作为下载单位，README 明列 `checkpoints.zip`、`gpt_model.zip`；权重/数据许可应由下载记录逐 artifact 核对，不能自动继承代码 MIT。本文未下载。
- CrossDocked：原始 types/full tarballs 与用于生成模型的 pocket10/split/preprocessed 包不同；原始全包文件数巨大，课程通常不需要原始全量重新处理。数据权利不能仅由周边代码许可证推定。
- GuacaMol/MOSES/PMO/Pocket2Mol/TDC/PoseBusters：相关代码 MIT；数据/第三方模型/外部 oracle 资产仍需按来源分别核对。GuacaMol 的数据下载单位是已分开的 SMILES 文件；MOSES 是预分角色数据，不必下载 Docker 镜像才可理解协议。
- AutoDock Vina：官方 Apache 许可；外部 preparation 工具各有依赖与许可。
- PMO-Dock：`pyproject.toml` 明确 Apache-2.0；其复用的算法/权重/资产应按各自上游许可检查，论文预印本 license 与代码 license 不是同一个概念。
- PDBbind：当前详细协议本次未能读取，保持待核实；不把“网上有下载”写成“可自由公开再分发”。

所有获取项都是未来研究准备信息，本次未下载安装或执行。结构/输入/真实结果的可用性需要后续单独的工程验收；公开代码存在不等于现代硬件与依赖下已可运行。
