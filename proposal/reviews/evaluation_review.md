# ChemGPT + REINFORCE 课程项目：建议冻结的评价协议

审阅日期：2026-11-04。依据用户认可的新主线：ChemGPT-4.7M pretrained SELFIES LM；Frozen same-model best-of-B、REINFORCE、REINFORCE+KL；Fexofenadine MPO 与 Osimertinib MPO；3 seeds。本文只核查原始资料和设计评价协议，没有运行分子生成、训练、安装、资产下载或服务器操作。

## 可用于 proposal 的主问题

**有限新分子评价预算下，REINFORCE 微调能否优于同一预训练分子语言模型的直接采样与排序？固定先验 KL 正则能否缓解输出分布退化，并在分数、有效独特产出和多样性之间取得可解释的权衡？**

两项任务足以支撑一个有完整对照的课程作业，但只代表两个公开多性质计算任务；两者都有 reference similarity、TPSA 和 logP 成分，不能声称覆盖所有药物生成、结构保持或生物活性优化。优先完成 `3 methods × 2 tasks × 3 seeds = 18 runs`，不再叠加 PPO、DPO、replay、docking 或新基础模型训练。

这里的贡献是正确的策略梯度实现、强基线、预算与信息核对、KL 的实证权衡及失效分析。它不是“首次用 RL 微调化学语言模型”，也不需要证明方法达到论文级新颖性。没有提升、KL 降低峰值分数但保住有效输出、一个任务有效另一个无效，都是可交付的研究结果。

## 任务名和版本必须固定

建议直接采用 **GuacaMol goal-directed suite `v2`** 的 `hard_fexofenadine()` 与 `hard_osimertinib()` scorer，最终显示名分别为 `Fexofenadine MPO`、`Osimertinib MPO`。可选开发任务是同一 `v2` 的 `perindopril_rings()`，显示名 `Perindopril MPO`。保存仓库 commit、文件 hash、RDKit 版本、fingerprint 设定和 scoring tests；不只保存模糊字符串 `MPO`。

这点很重要：官方 suite `v1` 给 Fexofenadine/Osimertinib 显式传入 arithmetic mean，`v2` 使用默认 geometric mean。相同任务显示名不意味着跨版本分数可直接比较。[GuacaMol benchmark_suites.py](https://raw.githubusercontent.com/BenevolentAI/guacamol/master/guacamol/benchmark_suites.py)

| v2 任务 | 原始组分及满分条件 | 需要保留的实际定义 |
|---|---|---|
| Fexofenadine MPO | AP similarity >=0.8；TPSA >=90；Crippen logP <=4 | similarity 截断归一化；TPSA/logP 使用 half-Gaussian 过渡；3项几何平均 |
| Osimertinib MPO | FCFP4 similarity >=0.8；ECFP6 similarity <=0.85；TPSA >=100；Crippen logP <=1 | 两种不同 fingerprint 的“相似但不过分相似”条件；4项几何平均 |
| Perindopril MPO（dev） | ECFP4 similarity；芳香环数接近2 | 2项几何平均，环数 Gaussian(mu=2,sigma=0.5)；不需要查询测试任务 |

实际代码优先于变量名或文字注释，不能将 AP、FCFP4、ECFP6 混用，也不能把 Crippen logP 换成 penalized logP。[GuacaMol standard_benchmarks.py](https://raw.githubusercontent.com/BenevolentAI/guacamol/master/guacamol/standard_benchmarks.py)、[score_modifier.py](https://raw.githubusercontent.com/BenevolentAI/guacamol/master/guacamol/score_modifier.py)

## Dev、已观察任务与最终报告的界限

旧 GRU 已经试过 Fexofenadine，因此本项目不能再把它描述成从未看过的盲测任务。换为 ChemGPT 不会抹去人类选择任务/流程的历史。建议固定以下分工：

- **Perindopril MPO：开发与调参。** 用于学习率、KL 系数、baseline、长度/采样实现、更新与时间上限；开发预算和尝试记录单列。先用简单 `logP_benchmark(target=2.0)` 或 QED 做单元/梯度 smoke 也可以，但它们是 engineering sanity checks，不当最终任务。
- **Fexofenadine MPO：预先固定的探索性 benchmark。** 保留在主要结果里，明确有旧 GRU 探索历史；从现在起不再用其表现选超参或挑 seed。
- **Osimertinib MPO：冻结方案后的最终 benchmark。** 仅当其生成/优化结果确实尚未用于模型/超参/过滤选择时，才称 held-out final task；这不是 unseen molecule/protein 泛化。若后续先看了结果再调整方案，应如实降为探索性，并保留调整记录。

允许阅读测试 scorer 的公开定义来设定研究问题；它与依据测试输出调参不同。两项任务都是 test-time optimization：运行期间查询任务 reward 正是任务的一部分。禁止的是把最终任务试跑结果拿来反复挑超参，或把最终 policy evaluation 样本送回训练。无需硬造传统训练标签划分，重点是 **跨任务调参隔离 + 每次优化预算隔离 + 独立最终采样**。

开发推荐只在 Perindopril 上做很小的预声明 grid，例如2个学习率、2个 KL 系数；统一选定一个训练设置和一个 KL 系数，之后用于两个任务和全部 seeds。不要为每个最终任务单独找到最优 beta 后再称可迁移。若只来得及一个 dev seed，就承认超参选择稳定性有限，最终3 seeds仍完整运行。

## 预算和停止规则

**主预算建议 `B=3000` 个新合法独特分子；`B=10000` 作为预先声明的扩展预算。** 最稳妥做法是先完成完整18-run的B3000，不凭某组领先与否决定扩展。若资源允许扩展，事先统一规则，例如全部18 runs均续到10000；保留3000处 checkpoint/日志，10000不能被写成另一批独立重复。若只扩展部分方法/任务，只能标为探索性子分析，不能组成完整公平对照。

每个 run 同时执行四个上限，任一触发即停：

1. `N_unique = B`：每 run 第一次查询的合法标准化分子身份。
2. `N_attempt <= 5B`：包含非法、重复、空输出、截断失败等全部生成尝试。
3. `N_update <= 5*ceil(B/64)`：每 batch 最多一次梯度更新；不允许在重复/缓存上无限额外更新。Frozen 组没有 gradient updates，但同样受 attempts/时间限制。
4. `wall_time <= T_max`：在 dev 吞吐测试后、看最终任务结果前固定；三个方法使用相同机器和上限。可暂按B3000每run60分钟、B10000每run180分钟作为保守候选，最终以设备吞吐和总排期确定，不把这些候选数字当实测资源需求。

batch默认64。尾部采用 `batch_size=min(64, B-N_unique)`，这样整批都可得到 reward 且不跨过unique预算；重复会导致需要额外batch，但仍受更新/尝试/时间限制。禁止先计算超额新分子分数再挑前B；最后预算触发后不继续采样、打分或训练。每次恢复用原 RNG、policy/optimizer、cache和计数器，不重置预算。

不同方法不必有相同更新数，因为Frozen本来没有更新；它们共享资源上限，而实际updates、attempts、时间必须报告。较多更新带来的提升属于含计算成本的整体优化策略，不应冒称“相同计算成本”的纯算法优势。可额外给score-vs-time/attempt图，帮助判断unique-call效率是否以大量重复/计算换来。

## 非法、重复与缓存如何记账

定义并保存一条单向转换链：生成token→有效EOS截断→SELFIES字符串→SMILES→RDKit sanitize→合法连通分子→canonical isomeric SMILES identity。拒绝空分子和多片段输出；不静默取最大片段来提高validity。保留电荷与stereochemistry，固定是否保留同位素；不临时增加tautomer/去立体标准化合并。SELFIES的健壮性不意味着所有模型输出必然可解码、满足长度/单片段要求或具有高质量。[ChemGPT model card](https://huggingface.co/ncfrey/ChemGPT-4.7M)

建议每条尝试至少记录：run/task/seed、attempt序号、batch/update、原token和decode状态、identity、first-seen、reward/component值、cache_hit、异常类型与时间。主要计数如下：

- `N_attempt`：所有完整生成尝试；不因非法/重复而回退。
- `N_valid`：通过固定化学与表示检查的输出数；重复仍是valid。
- `N_unique`：本run已评价的不同合法identity；只把首次出现计入B。
- `N_cache_hit`：本run的重复合法identity；重用确定性reward，不重复当新oracle调用。
- `N_physical_eval`：实际执行性质计算的次数。共享磁盘cache可能让不同run少算一次，但每run首次逻辑查询仍计入各自B；不能让后跑的方法获得免费历史查询预算。

cache key至少包括identity、task/scorer版本、fingerprint/RDKit版本、normalization定义。Fexofenadine分数不能复用于Osimertinib；缓存raw descriptors可减少物理计算，但不免除该run逻辑评分预算。异常/retry单独记录，确定性失败不无限重试。

在线REINFORCE不能只训练新独特分子而丢掉重复轨迹：那会改变实际优化目标和采样分布。对当前batch里的重复输出使用cached reward，并按真实采样轨迹计算loss；非法输出也保留为reward0的采样事件。集合去重用于oracle账本和Top-k发现结果，不是未经说明的梯度筛选。没有replay；不把历史cache样本直接拿来额外更新。

## 奖励、KL与高相似度压力

两个v2 MPO奖励都在[0,1]，原始几何平均直接作为terminal return。保留每组分分数和raw descriptor；不在测试时作方法各自的min-max归一化，也不按每batch最高值缩放reward。不同任务分数分布可能相差很大，所以分别报告任务和配对差值；不能把“都在[0,1]”当成完全同等难度。

REINFORCE和REINFORCE+KL共享同一advantage/baseline、学习率、batch、采样和有效token mask。建议baseline用前批次EMA等不依赖当前动作的值。KL reference始终冻结为原ChemGPT，明确方向`KL(pi || pi0)`、是整条轨迹求和还是token平均、系数beta与EOS处理。二者不能混报同一个beta；token平均改变长度偏好和目标解释。KL不是对同batch跨分子分布计算相似度，也不是reward里随手减一个“logP差”。

若采样用了temperature、top-k/top-p或词表mask，必须明确它们构成的实际采样policy，并在likelihood/梯度/KL中使用相同分布；不能从截断分布采样，却用原始模型分布冒充on-policy。为第一版减少歧义，建议temperature1、无top-k/top-p截断，只保留事先固定且三组一致的必要特殊token规则。是否使用长度归一化必须事先固定，不能看结果后为KL组选有利版本。

高similarity是公开任务的压力来源：Fexo要AP>=0.8达到该项满分；Osim还要求FCFP4高而ECFP6不过高。ChemGPT可能获得不错MPO却始终达不到严格joint条件；不要把零joint成功率视为工程失败，也不要因此偷偷降低阈值。它可以支持“连续代理进步未转化为同时满足全部条件”的负结果。若决定研究新低阈值任务，必须另行命名并与预定主结果分开。

## 推荐终点和图表

**Primary：Top-10 MPO AUC vs new unique evaluations。** 按首次评价顺序维护去重集合，`T10(n)`为已评价集合中10个最高分的均值。建议统一网格`n=0,100,200,...,B`、`T10(0)=0`，梯形积分除以B；不足10个时用实际数量计算并明确，若零个则0。保存逐分子日志，能够重算而非只保留曲线图片。这是PMO-inspired预算评价；不声称与原PMO的模型、数据和所有实现细节完全一致。[PMO top_auc 实现](https://raw.githubusercontent.com/wenhao-gao/mol_opt/main/main/optimizer.py)

如果某run因attempt/update/time上限提前停止，可以把最后T10值水平延伸到B来定义固定预算轴AUC，同时突出标注其实际`N_unique`和停止原因；该延伸不表示它真的完成B个评价。另报observed-range曲线和完成率。Frozen best-of-B是真正重要的强基线，必须享有同预算独特候选的最佳10个；不能拿RL top10去比Frozen全体均分。

**Secondary分为互不混同的三类：**

1. **搜索结果：**预算末Top1/Top10/Top100原始MPO；累计`MPO>=0.8`独特分子数；每个组分达标率；严格joint满分条件的独特分子数。`MPO>=0.8`并不等于每个组分都达标，也不等于严格joint；可以某一项偏低由其他项补偿。严格joint按上表raw条件直接判定，避免浮点`score==1`误差。
2. **最终policy：**在预算末checkpoint固定采样1024个新attempts，报告全attempt为分母的validity、valid unique数、平均/分位MPO、MPO>=0.8比例、strict-joint比例、重复率。Frozen用原模型同样采样1024；不把“搜索找到过的最好分子”冒充最终policy平均表现。该evaluation是额外的相同预算，不计入训练B，不再更新/选checkpoint。结果中仍记录这些样本是否出现在该run搜索cache，不能把重复当新发现。
3. **多样性与偏移：**搜索Top100和final-policy unique-valid集合分别给Morgan/ECFP4 fingerprint内部多样性、Bemis–Murcko scaffold数/最大scaffold占比；再给MW、heavy atoms、TPSA、raw logP、相似度分布。固定fingerprint参数和集合大小；不足100如实记N，不反复重采样直到凑足。高相似度任务本来限制可允许结构范围，不能单凭scaffold少就判失败。

不建议把FCD作为1024个样本上的主要终点，也不需要为了novelty下载整个PubChem10M。ChemGPT公开模型卡说明预训练使用PubChem10M且SELFIES预处理版本1.0.4；无法核对完整训练集时只能报告“相对本run/指定可用参考集未重复”，不能声称全训练语料novelty或新化合物发现。[ChemGPT model card](https://huggingface.co/ncfrey/ChemGPT-4.7M)

最核心的图可以限制为三张：每task的Top10-vs-budget均值+逐seed点/范围；18-run预算末分数与final-policy有效独特产出的配对比较；KL/无KL的分数—多样性—重复率权衡。保留原始per-seed表。3 seeds主要用于观察波动，不以低功效p值支撑普遍结论；两个任务分别分析，简单平均只作描述性汇总，不遮盖方向相反的结果。

## 什么结果足以完成作业

- **RL稳定优于Frozen：**有限预算内参数更新有实证价值，但仅限本模型和两个代理任务；KL的额外价值另看质量/多样性。
- **KL保住validity/diversity却降低Top10：**说明正则的权衡，不能只挑其中一项称全面更好。
- **Frozen同样好或更好：**说明当前预算/先验/目标下复杂优化并非必要；有完整日志和梯度正确性证据仍是有效负结果。
- **训练Top10提高、final policy重复坍缩或strict joint仍为0：**说明“找到高分点”与“改善整体输出/联合可行性”不同；这是有解释力的主要发现。
- **高波动/早停：**报告每seed失败、实际预算与限制；不要删run或重抽seed。只工程跑通、尚未完整矩阵时，报告“可行性已验证、效能未建立”，不提前宣布课题成功。

可交付物为冻结协议/配置、模型与依赖版本、训练/评分日志、尝试及oracle账本、全部分子与组分表、必要checkpoint、逐seed指标、复现入口和解释清楚的报告。课程层面的成功标准是问题正确、比较公平、证据可追溯与结论诚实；不是保证REINFORCE或KL胜出，也不是产生可直接用于实验的药物。
