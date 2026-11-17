# TamGen：官方代码、强化学习接口与承接方向审查

访问日期：2026-11-04。仅阅读论文、GitHub 公开树/少量源码和 Zenodo 元数据；未安装环境、下载权重/数据集、训练、生成分子、优化分子、对接或使用远程资源。全文将官方事实、代码推断与待测事项分开。

## 1. 应以哪个仓库为准

论文 2024-10-29 Code availability 推荐 `SigmaGenX/TamGen`；Zenodo 2024-09-12 v1.0 也如此记录。**当前该仓库 README 已明确停止更新，并将 `microsoft/TamGen` 指为后续官方入口。** 因此论文历史复现保留前者身份，新适配先核对后者，不能把旧论文链接等同当前维护入口。[原论文](https://doi.org/10.1038/s41467-024-53632-4)、[旧仓库 README](https://github.com/SigmaGenX/TamGen/blob/858e35eb9f6d5e1f9877f14911b939bf33cf6d49/README.md)、[当前 README](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/README.md)

固定审查身份：

| 仓库 | main commit | commit 时间（UTC） | 树中 blob 数 |
|---|---|---|---:|
| SigmaGenX/TamGen | `858e35eb9f6d5e1f9877f14911b939bf33cf6d49` | 2025-06-03 01:02:00 | 405 |
| microsoft/TamGen | `9f49e6cee3a861c143600db2f1a9bc10bb1c5279` | 2025-09-24 14:39:12 | 370 |

二者并非完全等同：抽查 `transformer_3d.py`，新仓库只比旧版增加版权头，VAE/模型主体一致；VAE criterion 也只是版权头变更。`sequence_generator.py` 增加了 special token 计数及 prefix 强制生成时的数值处理；`setup_env.sh` 修复 numpy uninstall 参数、显式装 scipy、PyG 扩展采用 `--no-index`。完整树还显示其他输入处理文件不同，未作所有文件逐行语义等价承诺。固定 commit URL 及本地 SHA256 见 `tamgen_sources.json`。

## 2. 真正的输入、模型与上下文 VAE

TamGen 不是“把蛋白名称写进提示词的通用大模型”。输入是已确定口袋内的氨基酸 token 与逐残基三维坐标；输出是自回归 SMILES token。公开 center 数据入口需要 PDB/mmCIF 文件以及口袋中心 `center_x/y/z`，也提供配体定位和残基 ID 路线。`prepare_pdb_ids_center.py` 默认邻域阈值为 10 Å，交给 `process_one_pdb_given_center_coord()`；后者根据中心选择残基，使用 `simple_get_residue_positions(..., center_of_gravity=True)` 得到代表坐标，再由 `dump_center_data()` 按链与选中残基顺序拼接序列和坐标。因此不能笼统写成“完整蛋白 ESM embedding”或“只使用Cα坐标”。[数据入口](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/scripts/build_data/prepare_pdb_ids_center.py)、[构建实现](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/fairseq/molecule_utils/external/fairseq_dataset_build_utils.py#L327)

`Transformer3DEncoder.forward()` 将残基 token embedding 与坐标投影结合，通过距离注意力或标准 Transformer 编码；训练脚本启用 `--dist-attn --move-to-origin --random-rotation --use-src-coord`。`MultiheadDistanceSelfAttention` 明确计算成对距离并作距离衰减。训练脚本给出的配置是 protein encoder 4 层/256维、compound decoder 12 层/768维/12头，属于 GPT 风格 decoder + pocket cross-attention，不是直接生成三维配体位姿的模型。[模型实现](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/fairseq/models/tgt2drug/transformer_3d.py#L345)、[距离注意力](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/fairseq/modules/pointcloud/distance_attention.py)、[训练脚本](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/scripts/train.sh)

上下文 VAE 的精确代码语义（不能将其称为RL）：

- `VAEEncoder.forward(src_tokens, tgt_tokens, src_coord, ...)` 联合编码口袋及给定分子的 token，投影出逐口袋位置的均值与 log 标准差。
- `Encoder3D.forward()` 在训练时且有 `tgt_tokens` 时使用该后验；推理只有显式 `gen_vae=True` 且提供种子分子时才使用它。否则直接从标准高斯先验采样。
- 每次 encoder forward 都重新采样 latent `z`。默认与主 pocket encoder 输出相加，`concat` 分支可改为拼接，然后经 cross-attention 条件化 compound decoder。
- 训练是带 label smoothing 的 token NLL + VAE KL；该 VAE KL 对的是高斯先验，并非RL相对预训练 reference policy 的 KL。
- 数据 collater 可直接把完整 `batch['target']` 放入 `tgt_tokens`。监督学习重建中正常，但若原样用于RL rollout/replay，可能把待生成答案作为状态输入；必须区分合法种子分子与本次生成答案。[模型VAE](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/fairseq/models/tgt2drug/transformer_3d.py#L487)、[损失](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/fairseq/criterions/label_smoothed_cross_entropy_with_vae.py)、[target collater](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/fairseq/data/molecule/language_pair_coord_dataset.py#L31)

## 3. 作者的 RL 展望与当前实现边界

论文 Discussion 提出下一步可用 MCTS 或 RL 引导目标性质，短引为 “employing reinforcement learning techniques to guide molecule generation”。这是一条作者明确开放的方向，不是作者已经验证了RL增益，更不能据此认定现在做“TamGen+RL”具有首创性。论文已完成预训练、口袋条件化、VAE种子扩展及病原靶点实验验证；本课程候选应用须另行选择已有公开的常规人源靶点，病原案例只说明原论文证据。[Discussion](https://doi.org/10.1038/s41467-024-53632-4)

截至固定 commit，检查 README、模型、任务、损失、生成器、搜索及训练脚本，看到的是监督/VAE学习和 beam/diverse beam/token sampling；公开树的 criterion 路径未出现 PPO/REINFORCE 专用实现。**此次已审接口未发现现成RL训练器。** 这是范围限定的审查结果，未穷尽所有分支、issue、fork、未公开版本或后续研究，不能写成“全社区尚无人接RL”。

## 4. 能拿 logprob，但不能直接把 Demo 接上一行 PPO

| 部位 | 已有能力 | RL 接入时的实际问题 |
|---|---|---|
| `Transformer3D.forward(src_tokens, src_lengths, prev_output_tokens, **kwargs)` | teacher forcing；VAE时返回 decoder output/mean/logstd | 应将生成 token 正确右移，固定合法状态；直接重跑 encoder 会换 z |
| `model.decoder(prev_output_tokens, encoder_out=...)` | 返回每位置词表 logits；`get_normalized_probs(..., log_probs=True)` 可得可微 logprob | 可作为新 loss 的基础，需逐 token gather、EOS/PAD mask、正确对齐；无需对采样动作本身求导 |
| `SequenceGenerator.generate()` | beam、diverse beam、sampling/top-k/top-p；hypothesis 包含 tokens 和 `positional_scores` | `generate` 及 Ensemble 的 encoder/decoder 包有 `no_grad`，这些返回值不带梯度 |
| generation scores | 每位置分数、累计分数、长度归一化总分 | 返回 score 未必是采样分布下严格归一化的 logprob；受 temperature、去除特殊token、截断、min/max长度、强制prefix/EOS等影响 |
| `TamGenDemo.sample()` | 反复换 seed，调用生成，再过滤/去重，返回分子字典 | 不返回完整原始轨迹、旧policy概率和全部无效输出；过滤后再做RL会改变分布与统计分母 |
| `make_generation_fast_()` | 推理优化 | 会把模型 `.train(True)` 替换为直接报错；Demo初始化调用了它，因此训练须另建可训练模型实例/加载路径 |
| `--fix-decoder-params` | 原生支持冻结 decoder 主体、保留 cross-attention 训练 | flag 位于GPT初始化逻辑内，且后续替换attention模块；RL adapter应重新核对最终实际可训练参数，不能只相信flag |

来源：[模型forward](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/fairseq/models/tgt2drug/transformer_3d.py#L231)、[生成器](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/fairseq/sequence_generator.py)、[采样搜索](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/fairseq/search.py#L219)、[Demo](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/TamGen_Demo.py)、[推理加速锁训练](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/fairseq/models/fairseq_model.py#L98)

关键推断与待实现契约：

1. **状态一致性**：把 pocket、合法seed（若使用）、episode采样的latent z 及已生成前缀视为条件；old/current/reference都对同一条件重算。只冻结 encoder 的权重不足以固定其随机输出。若训练 encoder，保存z噪声并重建同一噪声条件；若冻结 encoder，可缓存含z的输出但记录其身份。不能让每次 teacherforce 自动重新采z。
2. **采样一致性**：现代码的默认是 beam，不是on-policy随机采样；其 `Sampling.step` 用 `torch.multinomial`，对去pad/top-k/top-p后的未归一化权重采样，却保存截断前/未重归一化概率。课程RL适配应明确定义真实动作分布并使sampling与replay完全一致，不能直接拿beam总分作PPO分母。
3. **终止语义**：EOS是动作，EOS之后/PAD不计loss；最大长度强制终止与自然EOS要区别。固定prefix不属于policy选择的动作。长度归一化是生成排序选项，不能静默替代轨迹log概率之和。
4. **随机层**：关闭或严格一致处理dropout、旋转/坐标噪声；在eval模式仍可算梯度，不能因需要反传就打开所有训练随机增强。
5. **约束与参考分布**：冻结reference应是接入前同一target条件模型；VAE KLD不能替代policy KL。PPO还需old-policy logprob/ratio及优势估计；选择actor-critic时需额外value head。REINFORCE/其他group baseline也要解决前四项。此处列接口成本，不预先锁算法。
6. **冻结策略**：候选可先冻结口袋与上下文encoder，优化decoder或末层/cross-attention；这是工程假设，不是已测最优配方。decoder embedding被VAE复用，参数共享需显式审计；本仓库不是Hugging Face `PreTrainedModel`，不能假定TRL/PEFT直接兼容。
7. **数据入口**：`TranslationCoordinatesPointsTask.build_dataset_for_inference()` 仍 `NotImplementedError`；可靠路径是现有预处理/二进制数据装载，交互式自定义tensor入口需适配。

## 5. 权重、许可与环境事实

公开 Zenodo DOI `10.5281/zenodo.13751391` 提供 `checkpoints.zip` 两类目标条件 checkpoint（README 列 `crossdock_pdb_A10` 与 `crossdocked_model`）以及 `gpt_model.zip` GPT预训练checkpoint/字典。精确metadata大小为 **2,452,735,481 B** 与 **824,353,848 B**（合计约3.28 GB十进制）。不是仅几十MB示例权重；也不能据压缩包大小推断显存或参数量。未下载或解压，尚未验证存储状态/参数数目/完整性。[Zenodo记录](https://zenodo.org/records/13751391)、[JSON API](https://zenodo.org/api/records/13751391)

官方GitHub源码MIT；Zenodo API metadata标 `cc-by-4.0`，下载发布物与代码许可分别记录，第三方训练数据仍遵原来源条款。许可元数据不是对所有训练数据权利链的独立审计。当前静态检查未从checkpoint得出精确参数量；公开训练架构支持“约百M量级”的粗略结构推算，但此报告不把它当权重实测值。未核验的Azure模型卡/二手约100M说法不作为本次精确计数证据。[源码LICENSE](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/LICENSE)

当前 README 建议 Python3.9，`setup.py` 声明fork fairseq0.8.0；**setup_env.sh 的当前实际配方是 PyTorch2.3.0/CUDA12.1、RDKit2024.03.1、NumPy1.26.4、PyG扩展对应torch2.3/cu121轮子**；PyTorch1.12.1/CUDA11.3仅列在legacy注释中。不能误写成“只能装Torch0.x”或“pip install最新版fairseq即可”。[当前安装脚本](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/setup_env.sh)、[setup.py](https://github.com/microsoft/TamGen/blob/9f49e6cee3a861c143600db2f1a9bc10bb1c5279/setup.py)

安装与迁移风险（静态推断，未实测）：

- 老fairseq fork与较新的Torch组合，部分历史DDP、checkpoint/args与旧namespace兼容逻辑需实测；直接换Torch2.6+还涉及旧式checkpoint加载行为。
- PyG编译扩展严格依赖Python/Torch/CUDA ABI，安装时须匹配已有wheel；`libbleu`还需C++编译器。若目标仅Transformer，也不能未经依赖链检查就删所有PyG导入。
- `chem` extra包含 `fy-common-ext>=0.0.3` 等包，无完整lockfile；未固定的 scipy/pandas/sacrebleu等可能造成跨时间漂移。
- 原训练脚本一并做train+generate，参数共享/数据词表与checkpoint初始化也有路径依赖；不能把原脚本直接视为RL runner。
- **GPU/CPU峰值内存、单步吞吐、oracle耗时均未知，未做任何实测。** 不能根据zip大小或可推理就承诺可PPO训练，也不应按一个假设8GB资源先否决。团队/具体GPU待定，GPU已知会有，后续按真实资源测定。

## 6. 可承接的研究问题，不急于锁任务或算法

代码与论文共同支持三类值得继续读领域文献的接口层问题：

- **有预算的目标条件策略改进**：在相同性质评估调用预算下，policy update相对“同一预训练模型采样+选优”是否仍有收益？其价值在学习效率与分布改进，不在只比较最终最好分数。现有自回归decoder使该问题可定义，但不能据此认定具有论文级新颖性。
- **性质改善与分布保持的可测权衡**：reference约束、有效率/独特性/多样性、目标条件依赖如何一起变化？是否出现模型忽略pocket或奖励投机？需要未参与优化的评价或靶点/口袋对照，不能把奖励本身作为唯一验证。
- **latent与策略学习的交互**：先验z、合法种子后验z及固定/重新采样z如何影响policy梯度方差和目标条件化？这是TamGen区别于简单无条件SMILES LM的技术特征；但应先正确固定episode条件，再谈算法比较。

以上只是可承接问题空间；具体人源靶点、oracle、算法、预算、数据划分待主报告完成领域对比后再选。不能把本任务缩为“挑一个靶点做一轮RL”，也不要求课程达到paper创新。

## 7. 条件式一周技术可行性门槛（未来另获实验授权后）

**判断：TamGen是有条件可接入、工程成本中等的优先候选，并非开箱RL模型；当前证据不足以承诺一周完成有效RL科学结论。** 无需从10M PubChem重新预训练；有完整teacherforce与公开checkpoint，主要难点是旧fork环境和正确概率/随机状态接口。

未来第一周应把完成标准设成：固定commit环境可复现；正确装载同一checkpoint和字典；在合法公共人源pocket示例上验证token与坐标对齐；建立未经推理锁定的训练实例；采样/replay token logprob逐位置一致；固定z后重复评分一致；EOS/PAD/prefix mask一致；确认仅预定参数有梯度且reference不变。通过这些，才有证据推进有限预算策略学习与科学比较。这里仅列验收目标，本次未运行。

若环境失败：先在隔离fork按官方torch2.3/cu121配方定位问题，不同时升级全部依赖；若概率接口失败：保留为负结果/实现审计，停止复杂PPO扩张，选择领域中已有可训练策略接口的公开SMILES RL基线完成课程算法部分，TamGen保留为结构条件基准或讨论对象；若oracle预算超出预期：在主报告允许的便宜公开性质任务验证RL与约束，诚实降低“target-aware binding优化”主张。具体fallback由领域比较再定，不在此提前锁死。

11月8日proposal可写清“拟研究问题+验证门槛+后备路线”，无需伪造已运行结果；12月6日报告可接受有分析的负结果。资源与分工待团队确定，不捏造人数或显存需求。

## 8. 补充：同名后续方法与角色覆盖未知

主报告另行核验的 2026 **PolyTamGen** 采用 frozen ESM3、Q-Former adaptor、setwise Transformer、NatureLM 以及 SFT/DPO；这是不同的架构与偏好优化路径，不能因为名称沿用 TamGen 就写成“在原TamGen checkpoint上直接增加DPO”。本文件关于fairseq、VAE和采样的接口审查仅针对上面固定的2024 TamGen代码，不外推到PolyTamGen。该方法的原始文献与新颖性比较以主报告证据链为准。[PolyTamGen DOI](https://doi.org/10.1093/bioinformatics/btag766)

两个仓库完整公开树均不含 CrossDocked `test-info.csv`，`data/README.md`仅列出了运行构建脚本后将产生的该文件；本次未下载原始数据或processed bundle，因此**3PBL、3EML、3NY8是否出现在TamGen benchmark test角色，仍未知**。代码树的 `tests/molecule/test_data/3ny8.cif` 只是单元测试fixture，不能当作训练/测试角色归属证据。后续若选择相应公开人源靶点，需另行对原训练/验证/测试pocket与蛋白相似性做覆盖审计；当前不声称“新靶点零泄漏”。
