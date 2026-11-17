# ChemGPT-4.7M：公开 checkpoint 与 RL 接口静态核验

核查日期：2026-11-04。用户当前优先路线为 `ncfrey/ChemGPT-4.7M`，TamGen留作文献背景。本子任务只读取公开metadata、约十KB tokenizer、配置和少量源码；**未下载weight、安装包、运行tokenizer/model、生成分子或训练**。主任务若另有实测，应作为独立证据补入，不能将本报告的静态推断称为实测。

## 已核验的版本与发布物

HF当前revision固定为 **`7438a282460b3038e17a27e25b85b1376e9a23e2`**；API `lastModified=2022-06-15T15:17:11.000Z`，公开、无gate。它是GPT-Neo `GPTNeoForCausalLM`，config记录 `transformers_version=4.10.2`；不能把该记录理解为所有其他Transformers版本已验证。[HF API](https://huggingface.co/api/models/ncfrey/ChemGPT-4.7M?blobs=true)、[固定文件树](https://huggingface.co/ncfrey/ChemGPT-4.7M/tree/7438a282460b3038e17a27e25b85b1376e9a23e2)

| 文件 | metadata字节数 | 本子任务处理 |
|---|---:|---|
| `pytorch_model.bin` | 70,836,865 | 只读LFS信息，未下载 |
| `config.json` | 1,284 | 已读 |
| `tokenizer.json` | 10,789 | 已读并用标准JSON解析 |
| `tokenizer_config.json` | 260 | 已读 |
| `special_tokens_map.json` | 2 | 已读，内容为 `{}` |
| `README.md` | 1,942 | 已读 |
| `.gitattributes` | 1,173 | 已读 |

该revision无 `generation_config.json`、无safetensors、无独立LICENSE文件。weight LFS SHA256为 `a8a39c5ea1d50c450632e003d21c90f847d815a801934b03a12b02010b74d21b`；[固定weight URL](https://huggingface.co/ncfrey/ChemGPT-4.7M/resolve/7438a282460b3038e17a27e25b85b1376e9a23e2/pytorch_model.bin)。70.8MB文件大小不等于参数内存或实际训练显存，可能包含模型buffer等；精确参数量须加载后计数，当前“4.7M”只是官方模型名。

权利信息须分别记录：HF API/model card **未给出license字段**，本次不能确认该weight的显式许可；原作者LitMatter代码仓库为MIT，不自动把这一许可外推给HF weight或PubChem每项来源数据。论文正文许可也不同于模型发布许可。这是元信息缺口，不是声称作者禁止课程研究。[模型卡](https://huggingface.co/ncfrey/ChemGPT-4.7M/blob/7438a282460b3038e17a27e25b85b1376e9a23e2/README.md)、[LitMatter LICENSE](https://github.com/ncfrey/litmatter/blob/d355ccac45e3b6299d1773c536ad7617e49a6b8a/LICENSE)

## 论文与模型用途

对应正式论文为 Frey等，*Neural scaling of deep chemical models*，Nature Machine Intelligence **2023-10-23**，5:1297–1305，DOI `10.1038/s42256-023-00740-3`；HF模型卡仍引用2022 ChemRxiv预印本，不应因此把正式论文称为未同行评审。ChemGPT用于研究化学语言模型的规模规律与预训练，其自回归分布是分子序列模型；该checkpoint不是已具备蛋白口袋条件的TamGen，也没有现成的药效/靶点保证。[正式论文](https://doi.org/10.1038/s42256-023-00740-3)

config明确：24层、hidden128、16 heads，global/local attention交替，local window256，max_position_embeddings2048，max_length512，vocab_size684，float32。模型卡明确PubChem10M与SELFIES **1.0.4**。这些足以选择标准 `AutoModelForCausalLM` 路径，无需重新安装原LitMatter的Lightning/PyG训练栈；当前具体库组合仍要smoke test。

## token冲突已定位，不能误判为整个模型不可用

**真实JSON事实**：`tokenizer.json.model`为WordPiece，vocab是连续且唯一的ID **0…683**。其 `added_tokens` 中只有五项，全部已经在base vocab内：

| token | 已存ID | 静态语义 |
|---|---:|---|
| `[UNK]` | 0 | 未知符号 |
| `[PAD]` | 1 | padding |
| `[CLS]` | 2 | single/pair序列开头 |
| `[SEP]` | 3 | single/pair序列结束/分隔 |
| `[MASK]` | 4 |保留特殊符号 |

**没有 `[EOS]`、`[BOS]`、`<|endoftext|>` 或ID684。** 若调用者看到EOS=684，须检查是否另取了tokenizer或自行新增token，不应将该值归因于本revision原文件。[固定tokenizer](https://huggingface.co/ncfrey/ChemGPT-4.7M/blob/7438a282460b3038e17a27e25b85b1376e9a23e2/tokenizer.json)

TemplateProcessing的single模板为 `[CLS] $A [SEP]`，其special token映射明确CLS2、SEP3；pair为 `[CLS] $A [SEP] $B:1 [SEP]:1`。normalizer为null，pre-tokenizer按方括号符号分割，decoder为WordPiece（`##`拼接、cleanup=true）。`special_tokens_map.json`却为空，tokenizer_config只给 `PreTrainedTokenizerFast`及作者机器历史绝对路径，未将BOS/EOS/PAD语义属性完整保存。

与此同时 `config.json` 的 **bos_token_id=eos_token_id=50256**，超过684词表。4.10.2的GPTNeoConfig源码本来就以50256为默认值；官方LitChemGPT创建GPTNeoConfig后修改vocab但未同步改bos/eos，能解释该遗留配置。[模型配置](https://huggingface.co/ncfrey/ChemGPT-4.7M/blob/7438a282460b3038e17a27e25b85b1376e9a23e2/config.json)、[GPTNeoConfig 4.10.2](https://github.com/huggingface/transformers/blob/v4.10.2/src/transformers/models/gpt_neo/configuration_gpt_neo.py)、[官方LitChemGPT](https://github.com/ncfrey/litmatter/blob/d355ccac45e3b6299d1773c536ad7617e49a6b8a/lit_models/lit_chemgpt.py)

**推断及建议契约**：在不新增词表行的前提下，显式将BOS映射既有CLS2、EOS映射既有SEP3、PAD映射既有PAD1，并验证tokenizer长度仍684、embedding/logits维度仍684。既有postprocessor支持这一解释，但本子任务没有原训练矩阵，不能声称直接核验了每个训练样本的边界。修改的是本任务adapter中的控制配置，不回写原checkpoint。

## 官方训练代码与生成实例证据

本次固定LitMatter当前main commit为 `d355ccac45e3b6299d1773c536ad7617e49a6b8a`；这是当前源码快照，不冒称2022 checkpoint发布时的唯一as-run版本。

- `lit_chemgpt.py` 加载 `pubchem10M_tokenizer/`，只显式注册PAD；设置词表大小并 `resize_token_embeddings(len(tokenizer))`，创建标准GPTNeo causal LM。
- `lit_data/lm_data.py` 使用预先tokenized的 `load_from_disk` 数据及 `DataCollatorForLanguageModeling(mlm=False)`，不在该文件里重新手工加CLS/SEP。
- 当前LitMatter `pubchem10M_tokenizer/tokenizer.json`与HF该checkpoint tokenizer逐字节相同；**不同的 `pubchem10M_selfiesv2_tokenizer/` 是另一份资产，不能替换旧版词表**。
- 已读官方 `LitHF.ipynb` 只有训练示例，未找到可直接复用的ChemGPT生成示例；HF模型卡只声明可从Transformers直接使用。普通HF网页显示的自然语言widget问题不是化学输入契约或作者实验。

来源：[数据模块](https://github.com/ncfrey/litmatter/blob/d355ccac45e3b6299d1773c536ad7617e49a6b8a/lit_data/lm_data.py)、[原tokenizer](https://github.com/ncfrey/litmatter/blob/d355ccac45e3b6299d1773c536ad7617e49a6b8a/tokenizers/pubchem10M_tokenizer/tokenizer.json)、[官方notebook](https://github.com/ncfrey/litmatter/blob/d355ccac45e3b6299d1773c536ad7617e49a6b8a/LitHF.ipynb)

未来接口验收可使用**BOS-only input_ids=[[2]]、attention_mask=[[1]]**，显式bos2/eos3/pad1及受限max_new_tokens；这是拟议调用骨架，本次未执行、不提供生成分子。不要把空串经过默认postprocessor得到的 `[CLS][SEP]` 当作未终止的空前缀；给定合法SELFIES前缀时应避免保留其末尾SEP，也避免重复加CLS。逐步记录真实EOS与长度截断标志。`generate`的完整默认设置须保存，不能依赖跨版本默认top_k等。

## SELFIES1.0.4兼容与“100%有效”的边界

此tokenizer含旧版 `expl`、`[Branch1_1]`、`[Expl=Ring1]`等符号。官方SELFIES2 changelog明确改动了这些语法；v2 decoder有 `compatible=True` 读取旧符号，但其API文档特别说明**不保证与旧主版本行为相同**。因此首轮应固定1.0.4及其语义约束；若后来使用v2兼容模式，应独立做旧样本往返与结构一致性审计，明确其版本与失败，不把可解析冒充等价。[SELFIES changelog](https://github.com/aspuru-guzik-group/selfies/blob/master/CHANGELOG.md)、[官方v2 decoder API](https://selfiesv2.readthedocs.io/en/latest/selfies.html#selfies.decoder)

WordPiece词表不等于684个完整SELFIES符号：实际还包含裸字符、`##`子词片段。任意ID序列不必组成语法完整SELFIES；tokenizer解码也可能产生空格/异常控制符。因此评价必须分层计数：原始轨迹→token控制符处理及子词重建→SELFIES解析→SMILES/RDKit有效性→其他性质。不得只留成功样本，或把SELFIES的表示理论保证直接变成此checkpoint的100%有效率承诺。当前 `[nop]` 不在词表，不宜擅自用它padding。

## 可微forward、冻结reference与RL replay

标准GPTNeoForCausalLM.forward无推理锁；输入token后输出 `[batch,length,684]` logits，取 `log_softmax` 并按下一个真实token gather可建立可微轨迹log概率。其 `labels` 分支已在内部shift，所以若只用现成cross-entropy，不能再手工shift两次。RL常需逐token/逐episode概率与mask，不能把整个batch平均 `outputs.loss` 当作优势加权轨迹loss。[4.10.2 causal LM源码](https://github.com/huggingface/transformers/blob/v4.10.2/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L899)

该config causalLM实际路径的attention/embed/resid dropout均为0；`summary_first_dropout=0.1`不属于本causal LM forward的分类summary路径。可用独立深拷贝/独立加载的reference、关闭其requires_grad并置eval；policy也可置eval同时正常求导，eval与no_grad不能混为一谈。reference要确认不同参数存储、没有进入optimizer、训练前后校验不变；同一实例变量赋值不构成reference copy。`generate`/sampling收集动作无需梯度，更新时用标准forward重算同一前缀的概率；按实现严格对齐temperature、动作mask、truncation、特殊token抑制与归一化。

**容易漏掉的token_type_ids**：GPTNeo源码在传入token_type_ids时用同一token embedding表编码并加到hidden上；因此全0 token_type_ids也不是数学上的“无影响”。tokenizer、rollout、teacherforce、reference要明确统一是否传入；已tokenized历史数据的该字段是否存在，本子任务尚未直接核验。不可一边用 `model(**tokenizer_output)`，另一边重算只传input_ids而假定概率相同。padding batch还须对齐attention_mask/position_ids；最先用单条无padding序列验证概率一致，再扩展batch。

与TamGen不同，此checkpoint没有上下文VAE和额外latent z；状态较直接，为固定模型条件/前缀与已生成token。但**它也没有原生靶点输入**，课程表述应依据所选reward与研究问题，不自动称蛋白靶点条件生成。

## 实施前最小必须核验项与当前未知

1. 固定revision和weight hash，严格加载；检查缺失/多余keys、真实参数数目、embedding/lm_head维度与权重共享、Transformers/PyTorch实际版本。
2. tokenizer反序列化后逐项核0…4，len=684；只注册既有token语义不新增行，确认CLS/SEP后处理与decode控制符。
3. 覆盖旧SELFIES符号的encode/decode小样本测试，核分子结构/立体与失败；固定SELFIES/RDKit版本，不静默升级语义。
4. BOS-only与prefix边界正确；真实SEP终止、最大长度截断、PAD与SEP之后的loss mask分别处理；BOS不是policy动作，EOS是动作。
5. 采样与teacherforce在同一token_type/position/attention/dropout/temperature/mask条件下逐token logprob一致；reference和old-policy记录身份，算法不能混淆。
6. 先测未更新模型的原始有效率、重复、长度与性质分布，才能区别checkpoint基线缺陷与RL改进/退化；相同oracle预算比较采样+选优与策略学习。
7. 本子任务的确切weight许可、实际加载可行性、有效率/终止率、梯度通路、GPU峰值内存/速度及RL收益均**尚未实测或未确认**。团队已有GPU不代表这些都已通过。

结论：**这是公开、规模小、标准可微Transformers路径的合理优先候选；现存token配置问题是可界定的适配与验收项，不足以直接否决模型，也不能被省略。** 课程方案可先立在模型分布优化与预算/约束比较，待真实加载和token验收后细化实验，不以本轮静态审查冒充已验证RL模型。
