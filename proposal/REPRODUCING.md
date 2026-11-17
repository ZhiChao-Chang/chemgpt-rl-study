# ChemGPT方案证据和复现入口

本阶段只完成方案细化和小型CPU接口/数学验收，没有开始ChemGPT RL训练或正式benchmark。入口为PLAN.md和PROPOSAL_DRAFT_ZH.md。上一阶段selection中的GRU结果和replay方案保持原样，本阶段替代其作为当前课程方案。

## 证据

- `evidence/download_manifest.json`：固定HF revision、权重/分词文件的URL、大小与SHA256；SELFIES wheel来源和hash。
- `evidence/interface_summary.json`、`interface_samples.json`：实际ChemGPT加载、32条原始输出、终止/解码/概率与梯度结果。
- `evidence/reference_encoding_check.json`：公开任务reference的旧版SELFIES编码与词表覆盖，无任务优化。
- `evidence/gradient_check.json`：可枚举两步MDP，精确目标梯度与score-function/LOO期望梯度比较。
- `reviews/`：模型、方法、评价三个独立审阅，里面“未实测”指对应只读分支；实际主代理CPU结果单列在evidence。
- `protocol.json`：拟执行的明确主配置，状态为PROPOSED，不冒称已运行。

## 依赖和隔离

接口验收只读使用现有`/home/zhangzhichao/miniconda3/envs/ddc_inference/bin/python`，torch2.7.1、transformers4.51.3、tokenizers0.21.4、RDKit；torch CPU两线程，没有调用CUDA。SELFIES1.0.4 wheel在本阶段vendor目录隔离解包并通过sys.path导入，没有pip修改共享环境。

官方config的bos/eos50256越界；运行显式用既有CLS2/SEP3/PAD1及共同特殊token mask。此适配没有改变模型维度/权重；正式输出属于这个共同协议下的policy。token_type_ids全0同时用于采样和重算，reference同样处理。

下载的是PyPI distribution1.0.4，wheel内部模块字符串1.0.3是已有发布差异；metadata和文件hash共同标识实际依赖。模型权重许可没有在模型卡中明确声明，不能将LitMatter源码许可自动推广到模型文件；本地保留下载来源，不将权重发布到公开Git。

## 命令

```bash
TASK_PYTHON=/home/zhangzhichao/miniconda3/envs/ddc_inference/bin/python
$TASK_PYTHON PATH_TO_PROPOSAL/implementation/check_gradient.py
$TASK_PYTHON PATH_TO_PROPOSAL/implementation/check_model.py
```

第二条会重新进行32次CPU采样、forward/backward并写入检查结果，不进行optimizer更新；如重跑，先复制implementation目录到新输出位置，以免覆盖已记录hash证据。脚本资产按自身目录定位：`chemgpt/`与SELFIES wheel需与脚本同级；实际存放的权重/版本见evidence/download_manifest.json。

本阶段结束时无训练后台任务，无CCI/ACP/付费调用；旧暗蛋白研究、DDC训练及其资产未操作。
