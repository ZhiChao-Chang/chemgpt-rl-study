# 课程选题独立审查：首选多性质稀疏反馈与奖励课程

日期：2026-11-04。本文仅审阅公开代码/资料和主代理提供的资产核验信息；本代理没有进行实验、安装软件或操作服务器。用户现已允许主代理做简单试验，但本文没有把可运行性检查当作效能证据。

## 建议选定的题目

**固定评价预算下的分子多性质优化：从密集反馈到稀疏可行性的奖励课程，是否比直接稀疏训练和同预算搜索更有效？**

首选这一题。它有清楚的课程 RL 主线：有限时域序列决策、终端奖励、信用分配、稀疏反馈、先验正则、非平稳训练目标、策略改进与搜索的区别。主体可以用小型公开 SMILES GRU，不依赖 TamGen、pocket、docking 或大语言模型。即使 curriculum 无优势，仍可通过正例发现率、策略梯度信号、先验质量和约束冲突作出有内容的负结果分析。

这不是“提出一种此前没有人做过的 curriculum”。课程贡献是受控实现、信息与计算预算核对、强搜索基线和失效分析；不声称新药、更优疗效或新 RL 理论。

## 三个候选的比较

| 候选 | 教学深度 | 实现与解释优势 | 主要风险 | 独立建议 |
|---|---|---|---|---|
| 多性质约束、奖励稀疏与课程 | 可清楚解释 MDP、terminal return、baseline、KL、稀疏梯度和目标转换 | 相同模型/状态/动作，只改变可审计奖励；便宜 deterministic descriptors；负结果易解释 | dense 与 sparse 提供的信息不同；任务可能不稀疏或根本无可行输出；不保证课程有用 | **主选**，先核基本可行性，再冻结任务 |
| 样本效率、多样性与 replay | 探索/利用、经验复用、分布坍缩 | PMO、REINVENT/Augmented Memory 有现成基础，oracle 曲线直观 | naive replay 不是无偏 on-policy policy gradient；cache、SMILES augmentation、额外更新步容易混淆收益 | 第二选项；若选它，应固定已有 augmented-likelihood 算法，勿同时改成 PPO |
| DPO 与在线 PG | 支持集、偏好噪声、反馈与离线数据利用 | 可对比 offline fitting 与 online exploration | 偏好对构造成本、模型训练和反馈分布很难完全匹配；DPO 本身非在线 RL；原 molecular DPO 代码许可不明确 | 可做扩展或后续，不建议当本次主线 |

代码证据有两个关键限制：PMO 的 REINVENT 已自带经验池与 augmented-likelihood loss，不能直接称干净的 REINFORCE；其注释明确提醒 on-policy replay 的理论问题。REINVENT4 当前 staged-learning 示例虽支持多阶段，但配置内学习策略是 DAP，不是现成 PPO/REINFORCE 对照。因此，算法名应按实际损失写清，不能把成熟框架默认配置贴上课本算法标签。[PMO REINVENT run.py](https://raw.githubusercontent.com/wenhao-gao/mol_opt/main/main/reinvent/run.py)、[REINVENT4 staged learning](https://raw.githubusercontent.com/MolecularAI/REINVENT4/main/configs/staged_learning.toml)

## 可直接定义的公开任务

建议使用 **GuacaMol Fexofenadine MPO 的公开评分定义**作主要任务来源，构造显式命名的课程可行性衍生任务。它不需要真实活性标签、预测器权重或靶点结构。官方实现的三个组分是：AP 指纹对公开 reference 的相似度，经 0.8 截断归一化；TPSA 在 90 以上达到最大分；Crippen logP 在 4 以下达到最大分，另外两项用固定 Gaussian 过渡。官方 dense score 是三项几何平均。[GuacaMol standard_benchmarks.py，hard_fexofenadine](https://raw.githubusercontent.com/BenevolentAI/guacamol/master/guacamol/standard_benchmarks.py)

为了清楚研究稀疏性，定义：

- `D(x)`：上述原始三项几何平均分，保留原实现语义。
- `H_tau(x)`：合法、连通分子满足 `AP similarity >= tau`、`TPSA >= 90`、`Crippen logP <= 4` 时为 1，否则为 0。
- 主要阈值候选 `tau=0.6`；较难阈值 `tau=0.8` 只作为预先规定的压力条件。最终根据独立先验校准固定一个主要任务，不把两个阈值都扩成大矩阵。

必须写作“GuacaMol-derived sparse feasibility task”，不能把 `H_tau` 结果包装成原 GuacaMol 排行榜。`tau=0.6` 是本课程研究派生，0.8 是官方相似度项达到满分的位置。参考化合物是任务公开定义，不是本项目提出的新设计；不需要在提案里罗列或优化其具体药理用途。

这些都是理化/相似度 proxy；TPSA/logP 的阈值不是普遍临床标准，相似度也不证明结构功能保留。选择它的理由是评分可审计和多条件冲突适合研究反馈，而不是声称它代表真实药物开发的完整目标。

**先验校准是硬条件。** 在任何策略更新前，用固定小型 prior 样本检查 `H_0.6`、`H_0.8` 的正例比例和各组分瓶颈，并保存全部结果。如果没有正例或几乎所有样本都满足，不要预设任务“适度稀疏”。可以在提案提交前，根据这份公开开发校准选择并冻结上述一个阈值；该选择不看任何已训练方法的结果。若两者都完全不合适，应诚实将主体改为原 dense MPO 的奖励组合/梯度分析或换预先声明的开发任务，不能事后挑某方法最漂亮的阈值。小样本零正例只表明未观察到，不证明可行域为空。

## 数据与模型资产：不用从零预训练

主代理已提供独立核验信息：PMO 官方 commit `2da631be85af8d10a2bb43f2de76a03171166190` 的 `main/reinvent/data/Prior.ckpt`（16,697,208 bytes）与 `Voc`（206 bytes）已获得，`torch.load(weights_only=True)` 成功；结构是三层 GRU、hidden 512、embedding 128、vocab 55。**这是主代理的工程验收信息，不是本文新增的本地试验，也不代表优化已经有效。**

这个公开 prior 足以把课题集中在 RL；不必训练新基础模型，也不必下载完整 ChEMBL/MOSES。评分使用公开 reference 和 RDKit descriptors，模型初始化不需要任务标签训练矩阵。若课程希望说明预训练数据，按原发布说明记录；不能只凭语料路径推定其完整训练来源/排重都已重新核验。生成 novelty 只能相对可获得的参考集说明，不把它写成全历史未知的新化合物。[PMO official implementation](https://github.com/wenhao-gao/mol_opt)

建议复用该 checkpoint 的模型/词表适配，写一个小而可审查的 **on-policy REINFORCE + previous-batch/EMA baseline + 固定 reference 正则**训练器。状态为 SMILES prefix，动作为词表 token，EOS 截断，合法性与终端 reward 在完整序列后计算；所有方法用相同最大长度、采样温度、更新范围与 batch。只为可解释比较而实现这段小损失，不重写整个框架。baseline 必须不依赖当前动作；EOS 后 token 不进入有效 trajectory likelihood。

如实际只能稳定复用 PMO 的现成 augmented-likelihood/DAP 路线，则所有 reward 比较均固定用该算法并关闭 replay；报告准确命名它，解释与课本 policy gradient 的关系，不能改名 REINFORCE。不要为了算法齐全再加 PPO、critic、DPO 和 replay 作为同时变化的因素。具体 loss 的正确性验收是必要前提，proposal 主线仍是稀疏反馈与课程问题。

## 克制的主要实验矩阵

主任务固定后，先做 **5 组 × 3 seeds**。每组最多 `B=3000` 个新合法独特分子评价；相同种子标识用于配对展示。设统一最大生成尝试数，例如 `5B`，防止重复/非法坍缩无限占用。达到任一上限停止，并报告实际达到的 unique budget，不能给提前坍缩的 run 补足额外无限采样。

| 组别 | 更新策略/反馈 | 回答的问题 |
|---|---|---|
| Frozen prior + best-of-B / rerank | 不更新；相同 B，保留独特候选和完整性质 | RL 是否胜过同预算多采样和排序？这是核心 strong baseline |
| Graph GA | 相同新评价 B，公开 PMO 实现；无免费 target-labeled initial pool | 是否只是生成模型更新不如简单结构搜索？ |
| Sparse RL | 从头只用 `H_tau`，其他训练规则固定 | 直接稀疏奖励是否缺乏可用梯度？ |
| Dense RL | 从头只用原 `D`，其他规则固定 | 平滑反馈能否找到更多满足同一最终可行性的分子？ |
| Fixed curriculum RL | 前 1/3 新评价用 `D`，中 1/3 从 `D` 线性过渡到 `H_tau`，末 1/3 只用 `H_tau` | 是否比始终 dense 或始终 sparse 更好？ |

课程阶段按已消耗的 unique evaluations 切换，不按早停最好分数切换；重复缓存不能让训练停留在无限“免费早期阶段”。固定全局梯度更新/采样上限，避免某组因为重复多而多训练很多步。GA 的 initial population 若需要评分也计 B；可以从同一 prior 的固定、未评分样本产生初始种群，后续评价照常记账。不能用全库已知目标分数帮 GA 或 RL 预选种子。

如果 GA 适配出现超出课程范围的工程阻碍，保留 frozen best-of-B 和三个 RL 组构成 **4×3** 的完整核心实验，明确没有跨优化范式结论。第二难度/第二任务只在首矩阵完成后增加；不默认做完整 GuacaMol/PMO 全套、不预先加多分数 docking、不做新基础模型预训练。

**信息公平的准确解释。** `D` 和 `H_tau` 都由同一次相同性质计算得到，但 Sparse 组有意丢弃连续信息。这研究的是不同反馈形式的收益，不是“在相同信息量下”的算法胜负。若真实应用只提供二元反馈而拿不到连续分数，dense/curriculum 条件就不可用；报告应明示该适用前提。课程不是 potential-based shaping，训练最优策略不保证保持不变，不能声称有该理论保证。

## 评价指标与评分依据

**主结果**是统一最终 `H_tau` 下的发现与输出质量，而不是训练时各自 reward 的均值：

1. 累计独特可行候选数随新评价次数的曲线、曲线面积和预算末值；首次/前若干次命中的评价次数，未达到则右删失。
2. 用最终选定 policy 重新固定采样一小批（建议每 run 512 个尝试），报告 validity、unique-valid fraction、`H_tau` 成功率与实际分母；该额外评价预算各组相同并单列，不能回流选 checkpoint。
3. 同时报告 AP similarity、TPSA、raw Crippen logP、原 `D` 的分布与组分约束违反率；给 fingerprint diversity、scaffold 数作辅助，不把同 reference 邻域的 scaffold 多样性当无限目标。
4. unique calls / 实际性质计算 / cache hits / 总生成尝试 / invalid / 重复 / 梯度步 / 时间，逐 seed 保留。便宜描述符不意味着可以隐去训练/数据构造成本。

每 run 使用预先规定的预算末 checkpoint，而不在最终测试分子上挑最好轮次。3 seeds 报完整点值和均值/波动，可画成对差值，别靠少 seed 的显著性检验做强结论。冻结采样/预算/阈值/正则/阶段比例后才跑主矩阵；没有 oracle-budget match 的结果单独标明。

稀疏二元奖励的 best-so-far 一旦命中就饱和，因此它不能当主图；累计独特可行产出比“最高 reward=1”更能区分方法。最终 policy 的可行率与在线搜索找到过的最好分子必须分开，才能解释 curriculum 是否只是早期找到几个点，还是改善常规生成。

按老师的五项评分，题目具有比较均衡的交付：动机是可行条件与有限反馈；技术是明确 MDP/return/梯度及课程变化；评价有强基线和预算；报告能展示学习曲线/分布/负结果；小模型和有限机制便于所有成员说明自己的模块。它不需要 paper 级新方法才有完整研究价值。

## 可证伪结论与停止条件

建议预写两条假设：H1，dense 或课程相较 sparse 在相同评价预算下提高独特可行发现；H2，课程的最终 policy 可行率不低于始终 dense，并保留合理输出质量。两者可以分别失败。若只训练 reward 上升而最终 `H_tau` 不变，结论是优化了代理；若 frozen/GA 已同样好，结论是该任务不足以显示 RL 必要性；若 3 seeds 波动很大，就报告不确定，而非挑最好 seed。

在无可行性基本证据、梯度/likelihood 验收失败或主矩阵前就耗尽时间时，停止扩展。优先完成一个可解释比较，不把项目改造成多框架安装比拼。本文没有承诺以上方法一定优于 baseline，也没有把本次主代理即将做的 smoke 计入正式研究结果。

## 来源和已知限制

本次新增核查以 [GuacaMol 实际评分源码](https://raw.githubusercontent.com/BenevolentAI/guacamol/master/guacamol/standard_benchmarks.py)、[PMO REINVENT](https://raw.githubusercontent.com/wenhao-gao/mol_opt/main/main/reinvent/run.py)、[REINVENT4 staged config](https://raw.githubusercontent.com/MolecularAI/REINVENT4/main/configs/staged_learning.toml) 为主，结合已核原始 [PMO 论文](https://arxiv.org/abs/2206.12411)、[molecular DPO](https://arxiv.org/abs/2310.12304)、[POLO](https://github.com/REAL-Lab-NU/POLO)。代码许可分别按上游检查：GuacaMol/PMO MIT、REINVENT4 Apache-2.0；预训练 checkpoint、原语料和第三方组件不能只据代码许可证推出所有权利。模型资产字节数/加载成功仅引用主代理本轮核验；本文未重新加载或采样。
