# ChemGPT 分子生成的强化学习微调方案

更新日期：2026年11月4日。课程：ME2502 Reinforcement Learning。主线已收束为公开预训练ChemGPT接入REINFORCE，研究目标导向生成和先验KL约束；此前经验回放方案保留为历史候选，不再作为核心项目。

## 研究问题和预期交付

**在相同新分子评价预算下，REINFORCE能否比预训练模型直接生成再筛选，更有效地获得高分候选？相对原始模型的KL约束能否在目标改善、有效输出和多样性之间形成有用的权衡？**

模型原本学习已有分子的生成分布，我们将公开多性质评分接到生成策略上，检验反馈更新增加的能力。课程贡献定位为独立适配、正确的策略梯度实现、预算匹配的对照、正则化消融和结果分析。对“化学语言模型加RL”已有研究给予明确引用；本项目不以范式首创或超过全部SOTA作为完成门槛。

研究假设分别检验：

- H1：固定评价预算下，至少在预定任务中，RL相对冻结同一模型改善候选发现效率或最终生成分布；两种终点分开判断。
- H2：KL组相对无KL组减少分布偏移，并可能保留更多有效、独特输出；它也可能降低最高分，因此评价其权衡，而不预设全面占优。

最终交付包括可复现实验代码、冻结配置与版本、逐候选评价账本、全部随机种子结果、必要checkpoint、三张核心图及解释清楚的报告。即使RL或KL没有优势，只要实现正确、对照完整、结论有证据，项目仍能完成。

## 模型基础和已经核验的可行性

ChemGPT来自Frey等的《Neural scaling of deep chemical models》，发表于Nature Machine Intelligence 5, 1297–1305（2023）。该工作研究化学模型的规模规律，提出用于自回归分子建模的ChemGPT系列。[论文](https://doi.org/10.1038/s42256-023-00740-3)

选用作者公开的`ncfrey/ChemGPT-4.7M`，固定revision `7438a282460b3038e17a27e25b85b1376e9a23e2`。模型预训练使用PubChem10M，SELFIES预处理版本为1.0.4；本项目不从零预训练。[作者模型卡](https://huggingface.co/ncfrey/ChemGPT-4.7M/tree/7438a282460b3038e17a27e25b85b1376e9a23e2)

本轮取得70,836,865字节的发布权重并核对公开LFS SHA256。实际加载后总参数为5,099,264；模型“4.7M”名称与不含词/位置embedding的约4.75M参数口径相符，方案保留官方名称，同时报告实测总参数。

本轮已完成以下**CPU接口验收，未做ChemGPT强化学习训练**：

| 项目 | 本轮结果 | 能说明什么 |
|---|---|---|
| 同版本权重和词表加载 | 成功，无缺失/不匹配权重 | 公开资产可实际使用 |
| 固定seed的32次采样 | 32自然结束，32可解码为有效连通单分子，32独特 | 小样本解码流程可用，不推断总体100%有效 |
| rollout与teacher-forcing轨迹log概率 | 最大差约1.14e-4，FP32容差内 | 训练概率和行为分布一致 |
| 可微forward/backward | 有限、非零梯度；未调用optimizer | 可接入训练更新 |
| 两步有限MDP精确梯度检查 | KL surrogate误差<3e-17，LOO baseline误差<5e-17 | 拟用公式在可枚举环境中与目标梯度一致 |
| 三个任务reference编码 | 46/62/63 token，无UNK | 公开参考结构在现词表中可表示 |

此前GRU上的12-run预试是另一个模型的开发结果，不能作为ChemGPT优化成绩。模型卡没有明确给出权重许可证；保留来源和缺口，不将作者代码许可自动套用于权重，也不随报告重新发布checkpoint。

## 数据和公开任务

采用GuacaMol goal-directed **v2**，固定仓库commit `60ebe1f6a396f16e08b834dce448e9343d259feb`及RDKit/指纹版本。v1对部分同名任务使用算术平均，v2使用几何平均，必须固定版本。[suite源码](https://github.com/BenevolentAI/guacamol/blob/60ebe1f6a396f16e08b834dce448e9343d259feb/guacamol/benchmark_suites.py)

| 用途 | 任务 | 定义与作用 |
|---|---|---|
| 独立开发 | Perindopril MPO | ECFP4相似度与芳香环数接近2的几何平均；用于调通和检查稳定性 |
| 正式任务一 | Fexofenadine MPO | 参考结构AP相似度、TPSA、Crippen logP三个组分的几何平均 |
| 正式任务二 | Osimertinib MPO | FCFP4相似度、避免过高ECFP6相似度、TPSA、Crippen logP四组分的几何平均 |

Fexofenadine各项满分对应AP相似度≥0.8、TPSA≥90 Å²、logP≤4；Osimertinib对应FCFP4相似度≥0.8、ECFP6相似度≤0.85、TPSA≥100 Å²、logP≤1。实际评分仍按官方连续modifier计算，训练reward是其几何平均，范围[0,1]。[任务定义](https://github.com/BenevolentAI/guacamol/blob/60ebe1f6a396f16e08b834dce448e9343d259feb/guacamol/standard_benchmarks.py)

不需要另收一个大规模有标签训练矩阵：基础化学分布已在公开模型中学习；RL训练样本由策略在线生成，环境使用确定的公开函数返回reward。保存参考结构、评分公式、全部生成轨迹、canonical身份、原始性质和reward。这里改善的是计算性质及参考相似度，不据此推断实测活性或临床效果。

Fexofenadine已被旧GRU预试观察，作为带探索历史的固定benchmark；Osimertinib尚未用于生成性能选择。超参数和接口只在开发任务上确定，正式任务运行时查询reward是优化任务本身，不能据正式结果回头选配置或checkpoint。两项任务均从同一公开prior独立微调，不把结果解释成一个多任务条件模型或新靶点泛化。

## 生成环境和模型适配

ChemGPT采用SELFIES的WordPiece分词，模型动作是子词token，不等同一个原子或一个完整SELFIES符号。固定转换链为：原始token → 官方WordPiece decoder → 去空格形成SELFIES → 历史SELFIES decoder → SMILES → RDKit sanitize → 连通非空单分子 → canonical isomeric SMILES。

接口约定如下：

- 词表684维；BOS使用既有`[CLS]` ID2，终止使用`[SEP]` ID3，PAD为ID1。发布config中的50256是越界默认值，在本项目运行配置中显式纠正，不新增随机token或改变checkpoint维度。
- 共同采样分布固定屏蔽UNK0、PAD1、CLS2、MASK4，允许SEP3；三个实验组和reference完全相同。温度1，不用beam/top-k/top-p或重复惩罚。
- `token_type_ids`在rollout和teacher forcing都为全0，attention/position规则一致；不用已经包含终止SEP的“空字符串编码”当起始prompt。
- 最大256个采样动作；BOS不计入，真实SEP计入轨迹概率。到上限仍无SEP时，按时间截断结束、reward为0，不补造EOS；padding不进入loss。
- 模型保持eval模式关闭随机层，但actor保留梯度。reference复制初始权重后冻结；不随actor更新。使用FP32。
- 保留实际采样轨迹用于likelihood；canonical身份仅用于预算、cache和候选集合。非法、空、截断、多片段输出给reward0，重复输出使用cached reward，均保留在当前on-policy batch。

SELFIES使用PyPI 1.0.4 wheel并固定hash。该wheel的distribution metadata写1.0.4，内部`__version__`写1.0.3，二者均记录；不能仅靠内部字符串误判依赖。历史v1/v2语义不无声混用。

## 强化学习目标和更新方式

初始状态为BOS；状态含当前前缀与剩余步数，动作为下一个允许token或SEP。完整轨迹概率为`pθ(τ)`，固定原模型参考分布为`p0(τ)`，γ=1。

\[
J_\beta(\theta)=\mathbb{E}_{\tau\sim p_\theta}[R(\tau)]
-\beta D_{KL}(p_\theta\Vert p_0).
\]

对第i条当前策略轨迹，计算`Li=sum log qθ(action|prefix)`和`L0i=sum log q0(action|prefix)`，构造：

\[
Q_i=R_i-\beta(L_i-L_{0i}),\quad
b_i=\frac{1}{N-1}\sum_{j\ne i}Q_j,\quad
\mathcal L=-\frac1N\sum_i\operatorname{stopgrad}(Q_i-b_i)L_i.
\]

这是完整trajectory的score-function估计器。额外直接导数的期望由`E[∇log pθ]=0`消去；KL项已在return中，不能再重复添加一次KL loss。单条采样log-ratio可能为负，不clamp或取绝对值。LOO baseline仅使用其他独立轨迹；尾批N=1时使用上一批更新的EMA baseline，避免除零且不依赖本条动作。

每批从当前策略重新采样；在任何参数更新前重算可导logprob，整个batch只做一次optimizer step，不复用历史轨迹、不多epoch更新。baseline不做同批标准差whitening，不按长度除轨迹概率。采样前缀上的完整词表KL可作诊断，但不把只对local KL直接反传误称为完整轨迹KL梯度。

| 训练配置 | 固定开发起点 |
|---|---|
| actor/reference | 同一公开ChemGPT初始化；全量actor参数更新，reference冻结 |
| batch | 32条原始轨迹；预算尾部可缩小 |
| optimizer | Adam，lr=1e-5，betas=(0.9,0.999)，eps=1e-8，weight decay=0 |
| KL系数 | RF=0；RF-KL=0.01，按整条trajectory累计，非token均值 |
| gradient clipping | global norm 1.0，记录clip前范数；clip/Adam更新不宣称无偏 |
| 每批更新 | 一次，无replay/额外entropy/scheduler |
| 选择checkpoint | 固定资源预算末状态，不按正式结果挑最好轮次 |

这些数是本方案的开发起点，不是已经证实最优。开发阶段用Perindopril、两个独立seed完成三组短run，检查数值、解码、步数和时间；不因reward低就无限搜索超参数。出现非有限loss/梯度等实现失败时可修复并留记录，在正式实验前重新冻结配置。β敏感性只作为核心完成后的预先声明扩展，不能用其正式结果替换主β。

## 对照和实验规模

| 组别 | 做法 | 主要回答 |
|---|---|---|
| Frozen + best-of-B | 保持模型权重，按同一采样规则生成，记录并筛选全部已评价候选 | 单纯多采样再排序能做到什么？ |
| REINFORCE | β=0，其余训练设置相同 | 目标反馈的参数更新有无收益？ |
| REINFORCE + KL | β=0.01，固定初始模型reference | 先验正则带来什么质量和偏移权衡？ |

主矩阵为2任务×3组×3seed=18run；正式seed固定2026/2027/2028，开发seed用11/17。Frozen也重复三seed，享有相同的候选筛选预算；不能把RL的top10与Frozen的全部输出均分比较。三个组的同seed从相同初始权重和协议开始，paired seed不意味着所有后续轨迹完全相同。

可选扩展按优先顺序为：开发任务上的β敏感性、公开GraphGA系统基线、统一增加至B10000。核心完成后才考虑，均不在本轮启动。若要主张优于整个领域的现有RL算法，还需额外同协议强模型对照；当前主张限定在“接入RL及KL的增益”。

## 公平预算和停止规则

主预算B=3000个**每run首次评价的合法独特分子**，每run另外设attempt≤15000、optimizer updates≤470、训练/搜索墙钟≤1800秒，任一触发即停止。时间是拟定上限，不是已测运行时间；开发计时后如需变更，在正式任务前统一记录，不能按某组表现延长。

尾部批量设`N=min(32,B-current_unique)`。该N在采样前由既有账本决定，整个新batch都可评分且不会超出B，避免在batch内部按是否新颖筛梯度。重复会增加后续批次，但同时受attempt/update/time上限约束。N=1使用上述独立EMA baseline。无论是否首次出现，当前策略生成的完整轨迹都进入一次训练更新。

每run独立cache和逻辑查询计数；物理cache共享不免除另一run的首次逻辑预算。记录attempt、合法/非法、首次identity、duplicate/cache hit、实际性质计算和重试、tokens、更新数、时间。确定性性质cache key包含任务/版本、canonical身份、RDKit和指纹定义。

预算匹配指唯一评价机会匹配，不等于训练计算匹配；Frozen没有梯度更新，RF/KL也可能因重复不同而消耗不同attempt/updates。报告真实成本，不能将收益写成免费。计划上限为单GPU顺序执行约12小时预算（18个run各最多30分钟，余量用于开发与固定最终采样）；这不是吞吐实测或服务器使用授权。

## 评价终点与统计

**主要终点：Top10 MPO随新独特评价数变化的归一化AUC。** 每100个新identity保留点，B=3000，积分除以B。n=0时为0；不足10候选时缺位记0，维持固定top10分母。提前因资源上限结束的run将最后观察值水平延伸到B用于统一资源轴摘要，同时明确实际unique数、停止原因和observed-range曲线；延伸不代表完成了剩余评价。

**固定最终policy评价：** 每run在结束状态下重新采样1024个attempts，三组相同，不进行参数更新或checkpoint选择。报告以全部1024尝试为分母的平均reward、有效率、独特有效产出、MPO≥0.8比例、严格组分联合通过率；记录与搜索cache是否重叠，额外计分单列。

**辅助终点：** 预算末top1/top10/top100、阈值候选数、raw性质与各组分分布、固定Morgan radius2/2048bit下的内部多样性、Murcko scaffold数和最大系列占比、平均/总KL、长度、重复和实际时间。MPO≥0.8不等于每个组分达到满分，严格联合条件单独按raw阈值判断；即使其全为0也完整报告。

Top100和最终policy集合各自评价多样性，记录真实N；不靠无限重采样补足集合，不把高相似度任务的结构集中一律认作失败。两个任务分别报告各seed点、均值和标准差，以及同seed的成对差；3seed不支撑普遍显著性结论。没有完整训练语料验收时，不报告“对整个PubChem训练集的新颖性”。

## 进度和团队安排

| 时间 | 交付 |
|---|---|
| 11月4日已完成 | 固定模型与任务来源、CPU接口与数学检查、详细方案和proposal草稿 |
| 11月5日至7日拟进行 | 开发任务短run、计时、冻结最终配置，补团队姓名与分工 |
| 11月8日前 | 按课程公告提交proposal；公告未说明具体时刻，不套用其他作业23:59 |
| 11月9日至15日拟进行 | 完成两个任务的18run核心矩阵，保留全部失败/提前停止 |
| 11月16日至22日拟进行 | 独立重算、核心图、必要的一项敏感性或搜索基线 |
| 11月23日至12月5日拟进行 | 完成报告、复现入口与第13/14周展示材料，12月6日前交最终报告 |

人数最多5人，实际团队待确认。工作包为模型/采样、RL/梯度、任务/账本、评价/图表、报告/展示；少于5人可合并，每人均需理解完整问题、算法与结果。资源使用现有本地GPU及CPU，不启动CCI/ACP；GPU型号、可用时段和峰值显存仍待开发实测。

## 当前状态

本轮已把研究问题、模型身份、任务版本、目标函数、18run矩阵、预算和评价细节写明，并实际验收ChemGPT小样本生成/概率/梯度接口。**ChemGPT正式RL训练、开发调参和18run矩阵均未启动，未提交课程材料。** 剩余需要填写的是团队姓名、实际分工和可用GPU；它们不影响现在起草研究部分。

证据与复现：[evidence/interface_summary.json](evidence/interface_summary.json)、[evidence/gradient_check.json](evidence/gradient_check.json)、[REPRODUCING.md](REPRODUCING.md)。
