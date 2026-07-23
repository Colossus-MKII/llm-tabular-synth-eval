# 方法解读：EPIC 与 GReaT

## 1. 总体复现目标

综述把 LLM-based tabular generation 分成 prompt-based 和 fine-tuning 两类。本复现选择：

- EPIC 代表 prompt-based。原因是它不训练模型，只通过 prompt engineering 使用 LLM 的 ICL 能力，最符合低算力限制。
- GReaT 代表 fine-tuning。原因是它只需要在已有 causal LM 上做 dataset-specific fine-tuning，不需要从零预训练；也可以用 `distilgpt2` 或参数更小的本地模型做实验。

这里复现的是“实现原理”，不是照搬某个数据集脚本。原 EPIC repo 主要是 Sick 数据脚本；原 GReaT repo 是通用包。当前代码把二者抽象成可复用模块。

为什么看起来比原 repo 小很多：

- 原 repo 通常包含数据下载/清洗、多个数据集适配、API 调度、cache、日志、随机种子 sweep、baseline、ablation、paper table 复现实验等工程外壳。
- 本 repo 只保留核心方法路径：prompt construction、serialization、parsing、constraint repair、evaluation。
- 因此它能跑通“原理验证”，但不能声称已经复刻原论文所有实验结果。要复现论文表格，还需要接入真实 LLM/真实 UCI 数据、重复多 seed，并补齐原 repo 的实验配置。

## 2. EPIC 原理

EPIC 的生成逻辑可以拆成四步：

1. 目标列前置。把 label/target 放在第一列，让类别信息成为每一行的强提示。
2. CSV-style prompt。相比自然语言句子，CSV 更省 token，也更稳定。
3. Balanced grouped demonstrations。每个 set 都按类别抽相同数量的样本，即使真实训练集类别不均衡，prompt 中也强制让 LLM 看到均衡类别分布。
4. Unique variable mapping。对 categorical columns 的每个取值生成唯一 alphanumeric code。例如 `SEX=1` 和 `EDUCATION=1` 不再共享同一个表面值 `1`，而是变成不同 token，从而减少列间语义混淆。

生成后还要做 post-processing：

- 解析 LLM 返回的 CSV rows。
- 把 unique codes 映射回原始类别。
- 删除不在原始 categorical support 中的值。
- 可选地删除违反 hard constraints 的行。

代码对应：

- `tabular_synth.epic.UniqueValueMapper`
- `tabular_synth.epic.EPICPromptBuilder.build_prompt`
- `tabular_synth.epic.EPICPromptBuilder.parse_response`
- `tabular_synth.epic.EPICPromptBuilder.postprocess`

## 3. GReaT 原理

GReaT 把表格建模问题转成 causal language modeling。

一行：

```text
AGE=39, EDUCATION=2, LIMIT_BAL=200000
```

会被编码成：

```text
AGE is 39, EDUCATION is 2, LIMIT_BAL is 200000
```

训练时，每行的列顺序随机打乱：

```text
LIMIT_BAL is 200000, AGE is 39, EDUCATION is 2
```

这样模型学习的不是固定列序，而是任意 feature/value 组合后的条件分布。采样时可以只给一个 feature name：

```text
AGE is
```

也可以给多个条件：

```text
default.payment.next.month is 1, PAY_0 is 2, AGE is
```

然后让 fine-tuned causal LM 补齐剩余字段。解析阶段再用 `feature is value` 模式还原成 DataFrame。

代码对应：

- `tabular_synth.great.GReaTTextCodec.encode_frame`
- `tabular_synth.great.GReaTTextCodec.prompt_from_conditions`
- `tabular_synth.great.GReaTTextCodec.decode_texts`
- `tabular_synth.great.HuggingFaceGReaTFineTuner`

## 4. 为什么不选 TAPTAP 或 P-TA

TAPTAP 需要大规模 tabular pre-training，不符合当前算力限制。P-TA 引入 PPO，会比普通 supervised fine-tuning 更复杂，也需要 reward/model sampling loop。GReaT 的工程路径更轻：已有 LM checkpoint + tabular text serialization + supervised fine-tuning。

## 5. Hard-constraint 评价

本 repo 支持 constraint 指标：

- CVR: 所有 row-constraint 检查中违反的比例。
- CVC: 至少被违反一次的 constraint 占比。
- sCVC: 每行违反 constraint 的平均比例。

这些指标能补充传统 fidelity 和 TSTR。一个 synthetic table 可以在列分布上相似、TSTR 上有用，但仍可能生成 `AGE=-3` 或 `PAY_AMT1<0` 这类逻辑错误；hard constraints 正是用来抓这种 alignment 问题的。

## 6. Revision rate

Revision rate 衡量 post-processing/correction 对模型原始输出的改动幅度。它回答的问题是：模型自己生成的表格到底有多接近合法表格，还是主要靠后处理“救回来”？

设 correction 前表格为 `X_raw`，correction 后为 `X_fix`，逐行逐列对齐比较：

- cell revision rate = changed cells / compared cells
- row revision rate = rows with at least one changed cell / compared rows
- dropped row rate = max(raw rows - fixed rows, 0) / raw rows
- added row rate = max(fixed rows - raw rows, 0) / fixed rows

解释方式：

- CVR 高、revision rate 高：模型经常违反规则，后处理负担重。
- CVR 高、revision rate 低：可能 correction 没有覆盖所有错误。
- CVR 低、revision rate 高：后处理改动很多，但最终合法，需要检查是否扭曲了分布。
- CVR 低、revision rate 低：比较理想，模型原始输出已经较稳定。

## 7. 建议实验顺序

1. 用 `examples/run_default_credit_demo.py` 跑通离线流程。
2. 查看 `artifacts/epic_default_credit_prompt.txt`，确认 prompt 是否符合目标数据集语义。
3. 把本地 EPIC stub 换成真实 LLM 调用，先小批量生成 100-500 行。
4. 跑 constraints，若 CVR 高，优先加强 prompt 中的 schema/range 说明或启用 rejection filtering。
5. 再尝试 GReaT 的 `distilgpt2` 1-3 epoch 微调，比较 TSTR AUC、column similarity、DCR 和 CVR。

## 8. TSTR、fidelity、DCR 指标原理

TSTR 是 Train on Synthetic, Test on Real。先把真实数据分成 real train 和 real test；生成器只能看 real train，然后生成 synthetic train；再训练两个同结构预测器：一个用 real train 训练，一个用 synthetic train 训练；二者都在同一个 real test 上评估。若 synthetic-trained model 的 AUC 接近 real-trained model，说明 synthetic data 对下游任务有用。TSTR 评估的是 utility，不直接保证分布完全相同，也不直接保证 privacy。

Fidelity 衡量 synthetic table 是否保留真实数据的统计结构。本 repo 目前实现两类 column-wise fidelity：数值列使用 KS distance，比较真实列和合成列的经验 CDF 最大差异；类别列使用 total variation distance，比较类别频率分布差异。输出的 similarity 是 `1 - distance`。更完整的 fidelity 还可以加入 pair-wise correlation matrix distance、mutual information、classifier two-sample test 等。

DCR 是 Distance to Closest Record，用于粗略检查 memorization/privacy risk。对每条 synthetic row，计算它到 real train 中最近一条真实记录的距离；如果大量 synthetic rows 的 DCR 接近 0，说明生成器可能复制或几乎复制训练样本。当前实现对数值列做 min-max scaling，对类别列做 exact-match 0/1 penalty，然后取混合欧氏距离。DCR 越大通常越不容易是直接复制，但 DCR 太大也可能说明 fidelity 变差，所以它要和 TSTR/fidelity 一起看。

## 9. 从 smoke test 升级到真实规模

真实规模实验入口是 `scripts/run_real_scale_experiment.py`。需要改动的不是核心算法文件，而是把 demo 中的 proxy 环节替换成真实数据和真实模型：

1. 用 `--data-path` 加载真实 Default Credit、Adult 或 HELOC。Default Credit 支持 CSV/XLS(X)，Adult 支持 `adult.data`/`adult.test` 风格文件，HELOC 支持 `heloc_dataset_v1.csv` 风格文件并保留 `-9/-8/-7` special codes。
2. EPIC 需要传 `--llm-model` 和 `--llm-base-url`，指向 OpenAI-compatible server，例如 vLLM 或 llama.cpp 的 `/v1/chat/completions`。
3. GReaT 需要安装 `torch` 和 `transformers`，然后用 `--method great --great-model distilgpt2` 做 supervised fine-tuning。
4. 用 `--seeds 7 11 19` 做多 seed，并报告均值/方差，而不是只报一次结果。
5. 用 `--n-synthetic` 控制生成规模。论文式 TSTR 通常让 synthetic train 与 real train 同规模。
6. DCR 默认采样 5000x5000 条记录计算。若机器内存足够，可以调高 `--dcr-max-real` 和 `--dcr-max-synthetic`；传 `0` 表示不采样、做全量最近邻。

真实报告至少应包含：

- raw constraints 与 repaired constraints
- revision rate
- TSTR AUC
- column-wise fidelity
- DCR summary
- 每个 seed 的结果与均值/标准差
