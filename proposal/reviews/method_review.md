# ChemGPT–REINFORCE课程proposal：数学与实现契约审阅

日期：2026-11-04。本分支仅阅读原论文、官方配置/源码并设计方案；没有安装、权重下载、训练、分子生成或服务器操作。模型身份来自主代理模型审计交接：`ncfrey/ChemGPT-4.7M`固定revision `7438a282460b3038e17a27e25b85b1376e9a23e2`，同revision tokenizer与checkpoint必须共同固定。主代理另行负责CPU smoke；本文不将该smoke称为已经通过。

**建议主配方：完整trajectory的on-policy REINFORCE，固定reference的sequence KL采用sampled log-ratio（k1）构造detach后的regularized return；每个全新batch只更新一次。** 不另加一个未经occupancy修正的token KL loss，不使用replay、top-k样本筛选或长度归一化。三组为frozen best-of-B、β=0的REINFORCE、β>0的REINFORCE+reference KL。

## 1. 先固定模型接口，避免公式正确而policy并非那个policy

ChemGPT是SELFIES的WordPiece化学语言模型，不能把其token直接叫SMILES token或逐原子动作。模型论文是2023-10-23的《Neural scaling of deep chemical models》；本文不声称原文已实现本项目RL。[模型原论文](https://www.nature.com/articles/s42256-023-00740-3)

本次主代理交接的固定revision包含684个原始词表ID，特殊token为UNK0、PAD1、CLS2、SEP3、MASK4；postprocessor为CLS2 + sequence + SEP3。**本项目显式使用BOS=CLS2、终止token=SEP3、padding=PAD1。** checkpoint原config中的GPT默认bos/eos50256不适用于684维输出；其他revision新增EOS684也不得混入。不得通过随机扩词表引入新EOS后仍称原checkpoint的原生先验。既有token作为任务停止规则仍需在运行前核验其实际生成与loss-mask行为。[官方固定revision](https://huggingface.co/ncfrey/ChemGPT-4.7M/tree/7438a282460b3038e17a27e25b85b1376e9a23e2)

以下策略是**应用统一采样协议后的policy**，区别于未经处理的原始logits。设允许集合A为原始684个ID中去掉UNK0、PAD1、CLS2、MASK4的其余ID，SEP3保留；这是固定token层面的特殊符号mask，不是化学语法约束。所有组（包括frozen）使用同一A、同一初始CLS、同一停止规则。

\[
q_θ(a\mid h)=\frac{\exp z_θ(h,a)}{\sum_{v\in A}\exp z_θ(h,v)},\quad a\in A.
\]

禁止的ID概率为零；在共同支持集A上计算logprob/KL，禁止把零概率项写成0×NaN。温度固定1；不使用top-k/top-p、beam、repetition penalty、no-repeat n-gram、min-length强制EOS屏蔽或生成后repair。若以后改变其中一项，必须把它纳入q的定义并让sample、agent logprob和reference logprob同步改变。

mask使本项目先验成为q₀（固定协议下的ChemGPT），并非原始unmasked softmax；三组共享这一点，因而仍能隔离RL/KL的影响。reference权重、tokenizer、mask和解码规则从初始化固定，不能随更新EMA。

## 2. 有限时域、终止、非法与截断

初始状态为固定CLS。轨迹τ=(a₁,…,a_T)，T为首次采样SEP的位置与H中的较小值。建议H=256个**实际采样动作**，CLS不计入H。若事先只用冻结先验的长度诊断发现严重截断，可在接触正式MPO结果前统一改H=512；不按任务/方法分别调H。正式实验后H保持固定。

- 若SEP出现，T包含SEP；评分文本只使用SEP之前内容，**policy likelihood必须包含这个SEP动作**。
- 若达到H仍无SEP，环境按固定time limit终止；**不给它补造一个已采样EOS，不用强制EOS替换最后动作**。主方案将其分子reward记0，即使删掉最后片段后能解出合法分子也不补救；截断轨迹仍参加policy loss和KL。
- 终止之后补PAD仅作tensor对齐，不是policy动作。mask m_it=1当且仅当它是该轨迹实际采样过的动作（含真正SEP）；BOS、padding和SEP之后均为0。EOS与PAD不能混用一个ID自动构造loss mask。
- EOS紧接CLS的空序列、WordPiece解码/SELFIES解码失败、RDKit解析/消毒失败、scorer不能合法处理的内容，reward统一0；不丢弃、不重采样到凑够“有效batch”。
- WordPiece片段解码必须复用已核验的历史词表与SELFIES兼容版本。不能先逐token去掉`##`和空格再任意拼接并声称等价；一条trajectory的logprob始终对应**实际token IDs**。

定义F(τ)为上述固定解码/有效性判定及MPO评分。若完整合法，R(τ)=官方Fexofenadine或Osimertinib MPO的分数；否则R=0。完整评分函数固定且不依赖θ/历史记忆，γ=1，无额外有效性bonus、novelty shaping或diversity reward。[GuacaMol原始评分源码](https://raw.githubusercontent.com/BenevolentAI/guacamol/master/guacamol/standard_benchmarks.py)

**分布对象：**p_θ(τ)=∏_{t≤T}q_θ(a_t|h_t)。轨迹空间同时包含提前SEP与time-limit截断两类结果；两类policy使用相同确定性环境。因此它是一个归一化有限trajectory分布。这里KL是token轨迹分布的KL，**不是canonical molecule分布的KL**；不同SELFIES/WordPiece路径可映射同一分子，不能悄悄canonicalize后换一条轨迹来算likelihood。

## 3. 精确目标与主梯度

固定reference p₀，选定β≥0，目标为

\[
J_β(θ)=E_{τ\sim p_θ}[R(τ)]-β D_{KL}(p_θ\|p₀),
\quad D_{KL}(p_θ\|p₀)=E_{p_θ}[K_θ(τ)],
\]
\[
L_θ(τ)=\sum_{t≤T}\log q_θ(a_t|h_t),\quad
K_θ(τ)=L_θ(τ)-L₀(τ).
\]

不用有歧义的“forward/reverse KL”简称，始终写清方向pθ||p₀。有限H、共同支持集、固定R保证下述交换可用。利用score-function恒等式：

\[
\begin{aligned}
∇J_β
&=E[\{R-βK_θ\}∇L_θ]-βE[∇K_θ]\\
&=E[\{R-βK_θ\}∇L_θ],
\end{aligned}
\]

因为reference不训练，∇Kθ=∇Lθ，而E_{pθ}[∇Lθ]=∇Σ_τpθ(τ)=0。**第二项是取期望后为零，不是说K对θ没有依赖。** 这解释了为什么主配方可将sampled KL作为detach后的return，而无需再加一个直接KL loss。此为针对本项目有限trajectory分布的推导，不冒充模型原论文结论。REINFORCE及KL正则policy-gradient的原始方法背景见[Williams 1992](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)、[RPG理论原文](https://complex-reasoning.github.io/RPG/RPG.pdf)。

### Leave-one-out baseline与可导surrogate

新batch含N条相互独立的完整轨迹，定义

\[
Q_i=R_i-βK_i,\qquad b_i=\frac{1}{N-1}\sum_{j\ne i}Q_j,
\]
\[
\boxed{\quad\mathcal L_{RF-KL}(θ)=-\frac1N\sum_i
\operatorname{sg}(Q_i-b_i)\,L_i\quad}.
\]

`sg`表示stop-gradient。N>1。baseline使用其他轨迹，对第i条动作条件独立；因此E[b_i∇L_i]=0。β=0的组直接复用同一式，Q_i=R_i；不为两组另写不同归一化/奖励处理。

主方案不做batch标准差whitening、advantage clipping、length normalization或仅选top样本。batch标准差依赖本条回报时并非严格无偏baseline；同batch均值包含自己时会带有限N缩放，不能未经说明称完全同一估计器。LOO每批只去除自己，不会增加oracle成本。

**k1估计量须原样保留：** 单条K_i可为负，batch均值在有限样本中也可略负；不能对K_i clamp(min=0)、abs或平方后仍说优化同一个KL目标。理论KL非负说的是总体期望。

### 为什么它是真正on-policy

采样时参数θ固定；rollout后在**任何optimizer step之前**对原token轨迹teacher forcing重算当前qθ的可导L_i，并用固定q₀计算L₀。rollout行为概率与重算概率一致。teacher-forcing采用明确移位：输入`[CLS]+actions[:-1]`的第t位置logits评价`actions[t]`，或等价地完整序列logits去掉最后位置、targets去掉CLS；loss mask跟随targets，不能错移一个token。一次完整新batch只做一次optimizer step，然后丢掉batch并重新采样。没有replay、跨更新的旧样本、多epoch复用或去重后的条件采样。

为了省显存，可在一个batch内分microbatch累积梯度，但必须先得到所有R、L₀和用于LOO的Q，并保持θ不更新，直到完整batch平均梯度完成才step。不能每个microbatch都step却称其余样本仍来自当前policy。

## 4. Exact逐token KL能做什么，不能做什么

在每个访问到的前缀h_t，词表很小，可计算

\[
d_θ(h_t)=\sum_{a\in A}q_θ(a|h_t)\log\frac{q_θ(a|h_t)}{q₀(a|h_t)}.
\]

KL chain rule给出

\[
D_{KL}(p_θ\|p₀)=E_{p_θ}\left[\sum_{t≤T}d_θ(h_t)\right].
\]

这里每个d_t对词表求和是exact的；访问哪些prefix仍是Monte Carlo。因此`mean(sum_t d_t)`可作较稳定、非负的**trajectory KL诊断估计**，不等于精确枚举了所有trajectory。主方案建议用它作日志，与mean(K_i)并列，**从总loss断开梯度**。

直接在actor采样的前缀上把`β*sum(d_t)`加进loss，只计算各前缀的local ∇d_t，遗漏前缀访问分布随θ变化的影响；不能声称它是完整trajectory KL梯度。精确处理还需要

\[
∇D=E\left[\sum_u ∇d_u+
\sum_t (∇\log q_θ(a_t|h_t))\sum_{u>t}d_u\right].
\]

第二项的u>t是因为d_t在行动a_t之前的状态上计算；a_t影响后续前缀，而本步动作分布的导数已进入local∇d_t。若真选择exact-vocabulary训练，可构造相应surrogate：

\[
\mathcal L_{exact}=-\sum_t\operatorname{sg}
\left(R-β\sum_{u>t}d_u-b_t\right)\log q_θ(a_t|h_t)
+β\sum_td_t.
\]

整batch后按trajectory数平均，局部d_t在最后一项可导；前项的d停止梯度。这是另一个一致但更复杂的方案，**不作为本课程主实现**。其价值在于明确为何只加KL loss不够。

另一个同目标的可选估计器是sampled reward-to-go：把主式中每步的Q改为R−βΣ_{u≥t}k_u，再乘本步logprob；过去k的贡献条件期望为零。它可能减少无关过去噪声，但并不保证所有模型上方差都更小。proposal只承诺一种主估计器，避免同时比较估计器而扩大问题。

## 5. 一个明确、可实现的主配置

| 项 | 主值/规则 |
|---|---|
| 训练参数 | 全量更新ChemGPT-4.7M原有参数；reference深拷贝同权重后冻结 |
| 初始状态/终止 | CLS2；SEP3终止；PAD1只作对齐 |
| 可采样ID | 原始684维输出，固定屏蔽0、1、2、4，保留3 |
| H | 256；只允许在正式任务打分前依据prior-only长度审计统一改512 |
| Batch | N=32条raw trajectory；invalid/重复/截断均占位置 |
| 温度与截断采样 | 温度1，完整允许词表categorical，无top-k/p、beam、repair |
| Reward | 原始MPO分数[0,1]，非法/空/未EOS截断=0；γ=1 |
| Baseline | 上述LOO regularized return，整体detach |
| β | RF=0；RF-KL=0.01，按**整条trajectory总KL**计，不是每token均值 |
| Optimizer | Adam lr=1e−5，betas=(0.9,0.999)，eps=1e−8，weight_decay=0 |
| 每批更新 | 一次；无schedule、warmup、replay、多个epoch、额外entropy项 |
| 数值 | FP32；显式log_softmax，reference no_grad；eval模式关dropout但保留autograd |
| 梯度 | global norm clip=1.0，记录preclip norm和触发比例；有限样本估计器的无偏结论指clip前，Adam/clip后更新不声称仍为无偏梯度 |
| 结束与选择 | 固定预注册预算末次checkpoint；不按正式任务best reward挑seed/epoch |

β=0.01与lr=1e−5是本方案给出的保守开发起点，不是文献已经证明适于ChemGPT的最佳参数。整序列KL随轨迹长度累积，故不能把别人的token-mean β直接搬来。最终报告同时给sequence KL和每token KL作诊断，**训练目标仍不按长度除**。

冻结基线使用完全相同的q₀、token接口、H和评分规则，只不更新参数；best-of-B是候选发现终点，而REINFORCE训练优化的是E[R]。提高期望reward不必提高best-of-B或不同高分分子数，proposal应把这一点写成待分析的取舍。

## 6. 受控开发，不以正式任务反馈调参

### 建议的预算与冻结过程

1. **先做无目标reward的接口验收。** 固定revision、token与SELFIES转换；tiny枚举环境检查梯度。此阶段不调用Fexofenadine/Osimertinib scorer选择H、采样温度或mask。
2. **独立开发任务：GuacaMol Amlodipine MPO。** 固定开发种子11、17；两个训练组β=0/0.01，以N32、64次更新（每run2048 raw trajectory）检查有限值、有效性和梯度。共4run、8192 raw trajectory；登记全部生成/计分，不算正式结果。
3. **不做参数网格搜索。** 若无数值失败，直接冻结主配置。若两个种子都出现预先定义的数值不稳定（非有限loss/梯度，或持续极高clip率伴随解码有效性崩溃），只允许共同将lr降至3e−6，重复相同4run一次；最大开发16384 raw trajectory。不能因reward不够高、某组不如基线或β约束太强而使用这个救援规则。
4. **β主值不选优。** 保留0.01；可在开发任务额外做预注册的0.003/0.03敏感性诊断，但如果采用，必须在看正式任务结果前锁定其范围、成本和是否影响主值。最简proposal不需要承诺此扩展。
5. **正式两任务使用相同固定lr/β与全新预注册种子。** online使用各自目标reward是优化任务本身，不是监督学习意义上的“偷偷看test label”；问题在于看这些正式结果后改算法/参数或挑种子。若正式任务已被用于pilot调参，就明确叫开发任务，并用新未参与选择的任务/评估条件作确认，不能抹掉揭盲历史。

数值“持续极高clip率”的具体阈值可在运行前写为最近20次更新≥80%触发且valid率比同配置初始prior降低至少20个百分点；这只是救援的预注册条件，不是看到曲线后临时定义。若只有低分/零分而没有数值失稳，不靠改reward或改beta解救：先检查接口/评分正确性；正确时保留稀疏reward负结果。

### Query预算与on-policy特别容易冲突的地方

训练统计分母必须是raw trajectory。canonical cache只节约oracle重复计算，**不从policy batch移除重复分子，不只用unique或valid样本更新**；这种筛选会把采样分布变成条件分布。

若最终B指unique oracle预算，建议每轮仍固定N条新trajectory，缓存内重复样本取同一R并正常更新；记录所有raw attempts、unique queries与更新数。预算末尾不要为了“恰好剩余unique槽位”在batch中挑样本训练：要么预注册允许一完整batch的≤N−1 unique overshoot并给实际B，要么把最后不完整batch只用于候选发现、不再更新，并标明规则。最容易完全匹配的是把B定义为raw scored attempt预算，同时额外报告unique oracle成本；不能把两种B口径混为一谈。

本分支建议主统计终点与主代理计数方案一致后再冻结，不擅自替其决定B单位。无论采用哪种，frozen baseline也必须使用相同采样/截断/失败规则与预算，不能只给RL保留失败attempt。

## 7. 最小理论与实现验收清单（本次仅提出，未执行）

| 验收 | 最小测试设计 | 应成立的结论 |
|---|---|---|
| 归一化与共同支持 | 两步、2–3动作的可枚举toy policy，含EOS与time-limit结果 | trajectory概率和=1；policy/ref支持一致；不能遗漏截断概率质量 |
| 主梯度正确性 | 在toy中精确枚举Jβ并autograd，与枚举期望的detach-Q score surrogate比较 | double precision下梯度逐项相等（建议误差≤1e−8）；包含β>0、非reference参数和可变长度 |
| LOO baseline不改梯度 | 枚举两条独立trajectory的joint分布，再对baseline surrogate求期望梯度 | 与不含baseline的精确∇Jβ一致；不是仅比较某个sample的梯度 |
| Exact KL诊断的chain rule | toy枚举sequence KL，比较EΣd_t和EΣk_t | 三者相等，EΣd_t≥0；单条Σk_t允许负值 |
| 反证naive KL loss | 第一步选不同分支，分支后续KL不同；比较只反传Σd_t与精确∇DKL | 应观察到遗漏prefix occupancy项；正确exact surrogate需恢复相等 |
| 初始reference一致 | θ=θ₀，同一raw trajectories对两组构造loss | K_i≈0、d_t≈0；RF与RF-KL首次gradient/update一致（容许数值误差） |
| EOS/PAD | 同一轨迹在EOS后追加任意PAD，另造未EOS长度H轨迹 | L、K、loss对padding不变；EOS计一次；截断没有额外EOS概率 |
| Sample/evaluate一致 | 冻结参数，记录逐步masked分布采样logp；与teacher-forcing重算对齐 | 同ID、同温度、同mask、相同position/attention规则，maxdiff在FP32容差内 |
| 参数/信息隔离 | 检查reference参数hash/grad，actor非零梯度与step后变化 | prior固定；actor确实更新；canonicalization不改训练轨迹 |
| 失败分母 | toy或手写fixture混入空、invalid、重复、截断 | 四类都保留batch位置；invalid不能通过排除分母提高valid率或梯度 |

这些验收不需要真实药物生成或模型训练即可先完成toy梯度部分；ChemGPT sample/evaluate与decode验收则由获授权的工程阶段完成。工程smoke通过不代表MPO性能已提升。

## 8. 可放进proposal的方法段

我们使用固定公开ChemGPT-4.7M先验及同版本SELFIES/WordPiece词表，将分子生成表述为以终止符或固定时域结束的token决策过程。策略由统一特殊token mask后的categorical分布定义。每批从当前策略独立采样完整trajectory，使用固定GuacaMol MPO作为terminal reward；所有失败、截断和重复样本均保留在策略更新中。我们比较冻结先验的预算匹配采样、REINFORCE及固定reference KL正则REINFORCE。

对于KL组，我们优化E[R]−βKL(pθ||p₀)，其中p是包含终止行为的完整token trajectory分布。采用sampled sequence log-ratio构造regularized return，并以leave-one-out baseline及停止梯度的score-function surrogate获得一致的on-policy梯度；每新batch仅更新一次。逐前缀完整词表KL仅作诊断，不将遗漏state-occupancy导数的local KL loss误称为完整trajectory KL优化。超参数在独立开发任务上按预注册的有限规则冻结，正式比较报告性能、唯一候选、有效性、多样性、分布漂移和计算成本的取舍。
