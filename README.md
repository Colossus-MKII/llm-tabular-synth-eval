# LLM Tabular Synthetic Data 轻量复现

这个 repo 复现综述 `2504.16506v3` 中 LLM-based tabular data generation 的两条代表路线。这里的“复现”指复现核心实现原理和可运行评估管线，不等同于完整复刻原论文 repo 的所有数据集脚本、API 调度、日志、baseline sweep 和论文表格。

- Prompt-Based: 选择 EPIC。它只依赖 in-context learning，不更新模型参数，适合算力有限场景。
- Fine-tuning: 选择 GReaT。它不是预训练，而是在已有 causal LM checkpoint 上微调表格行文本，能够用 `distilgpt2` 这类小模型先跑通。

我把实现刻意做成低依赖版本：核心流程只需要 `pandas` 和 `numpy`。如果之后要真的微调 GReaT，再安装 optional `hf` 依赖即可。

## 为什么这样选

EPIC 的关键不是换模型，而是 prompt 结构：CSV 格式、目标类别均衡分组、重复多组 demonstrations，以及把每个 categorical value 映射成唯一 token，减少同一个值在不同列里语义冲突。原项目针对 Sick 数据写了脚本，这里抽象成通用 `EPICPromptBuilder`。

GReaT 的关键是把每行表格序列化成自然语言子句，例如 `AGE is 39, EDUCATION is 2, ...`，并在训练时随机打乱列顺序。这样 causal LM 微调后可以从任意 feature/value 前缀继续生成，实现 conditional generation。这里实现了 `GReaTTextCodec` 和可选 `HuggingFaceGReaTFineTuner`。

## 数据集与 hard constraints

主示例使用 `Default of Credit Card Clients` 风格 schema。为了不依赖网络下载，`make_default_credit_toy()` 生成一个同字段的本地 toy 数据集，不是原始 UCI 数据。它带有可验证的 hard constraints：

- `LIMIT_BAL >= 1`
- `SEX in {1, 2}`
- `EDUCATION in {0, 1, 2, 3, 4, 5, 6}`
- `MARRIAGE in {0, 1, 2, 3}`
- `18 <= AGE <= 100`
- `PAY_0, PAY_2, ..., PAY_6` 是 `[-2, 9]` 内整数
- `PAY_AMT1, ..., PAY_AMT6 >= 0`
- `default.payment.next.month in {0, 1}`

评估实现包括 CVR/CVC/sCVC、revision rate、TSTR AUC、列分布相似度和 DCR。

Revision rate 用来衡量 post-processing 或 correction 改动了多少 synthetic data：

- cell revision rate: correction 前后被修改的格子数 / 总比较格子数
- row revision rate: 至少有一个格子被修改的行数 / 总比较行数
- dropped/added row rate: 如果过滤或补行导致行数变化，会单独报告

## 快速运行

```bash
python3 examples/run_default_credit_demo.py
```

输出会写到 `artifacts/`：

- `epic_default_credit_prompt.txt`: EPIC prompt 样例
- `great_default_credit_training_texts.txt`: GReaT fine-tuning 文本样例
- `epic_raw_synthetic_default_credit.csv`: correction 前 EPIC 结果
- `epic_synthetic_default_credit.csv`: EPIC 管线生成结果，本地 LLM stub 仅用于离线验证
- `great_proxy_raw_synthetic_default_credit.csv`: correction 前 GReaT proxy 结果
- `great_proxy_synthetic_default_credit.csv`: GReaT 管线的本地 proxy 结果
- `default_credit_eval_summary.json`: 约束、TSTR、fidelity、DCR 摘要

## 真实规模实验入口

真实实验使用 `scripts/run_real_scale_experiment.py`。它要求你提供本地数据文件；脚本不会自动下载数据。

服务器环境配置见 [docs/server_setup_zh.md](docs/server_setup_zh.md)。

下载 Adult 和 HELOC：

```bash
python3 scripts/download_datasets.py --dataset adult --outdir data
python3 scripts/download_datasets.py --dataset heloc --outdir data
```

会生成：

- `data/adult.csv`
- `data/adult_train.csv`
- `data/adult_test.csv`
- `data/heloc_dataset_v1.csv`

Default Credit，真实 EPIC，接 vLLM/llama.cpp/OpenAI-compatible server：

```bash
python3 scripts/run_real_scale_experiment.py \
  --dataset default-credit \
  --data-path /path/to/default_credit.csv \
  --method epic \
  --llm-model Meta-Llama-3.1-8B-Instruct \
  --llm-base-url http://localhost:8000/v1 \
  --n-synthetic 24000 \
  --seeds 7 11 19 \
  --outdir artifacts/real_scale_epic
```

Default Credit，真实 GReaT fine-tuning：

```bash
pip install -e ".[hf]"
python3 scripts/run_real_scale_experiment.py \
  --dataset default-credit \
  --data-path /path/to/default_credit.csv \
  --method great \
  --great-model distilgpt2 \
  --great-epochs 1 \
  --great-batch-size 4 \
  --n-synthetic 24000 \
  --seeds 7 11 19 \
  --outdir artifacts/real_scale_great
```

Adult 数据：

```bash
python3 scripts/run_real_scale_experiment.py \
  --dataset adult \
  --data-path /path/to/adult.data \
  --method epic \
  --llm-model Meta-Llama-3.1-8B-Instruct \
  --llm-base-url http://localhost:8000/v1
```

HELOC 数据：

```bash
python3 scripts/run_real_scale_experiment.py \
  --dataset heloc \
  --data-path /path/to/heloc_dataset_v1.csv \
  --method epic \
  --llm-model Qwen/Qwen2.5-1.5B-Instruct \
  --llm-base-url http://localhost:8000/v1 \
  --n-synthetic 8367 \
  --seeds 7 11 19 \
  --outdir artifacts/real_scale_heloc_epic
```

HELOC 的 `-9/-8/-7` 会被保留为 FICO special codes，不会被普通非负数约束误删。

输出结构：

- `summary_all.json`: 每个 seed 的完整嵌套报告
- `summary_by_seed.csv`: 扁平化报告，便于做均值/方差
- `real_train.csv`, `real_test.csv`
- `synthetic_raw.csv`, `synthetic_repaired.csv`
- `column_similarity.csv`
- `epic_prompt_preview.txt` 或 `great_model/`

注意：`--method great-proxy` 和 EPIC 的 `--offline-proxy` 只用于 smoke test，不能作为 LLM 生成结果汇报。
DCR 默认最多用 5000 条 real 和 5000 条 synthetic 记录估计最近邻距离；传 `--dcr-max-real 0 --dcr-max-synthetic 0` 表示全量计算。

为了保持 EPIC 和 GReaT 的 base model 一致，可以用配对 wrapper：

```bash
python3 scripts/run_paired_model_experiment.py \
  --dataset heloc \
  --data-path /path/to/heloc_dataset_v1.csv \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --llm-base-url http://localhost:8000/v1 \
  --great-bf16 \
  --great-batch-size 1 \
  --great-gradient-accumulation-steps 8 \
  --n-synthetic 8367 \
  --seeds 7 11 19 \
  --outdir artifacts/paired_qwen25
```

## 接真实 EPIC LLM

把 `make_local_dataframe_llm(...)` 换成真实 LLM 调用即可：

```python
def my_llm(prompt: str) -> str:
    # 调 OpenAI、vLLM、llama.cpp 或其他 chat/completion 后端
    # 返回纯 CSV rows，不要额外解释文字
    return response_text

synthetic = epic.generate(train, llm=my_llm, n_samples=len(train), constraints=constraints)
```

## 接真实 GReaT 微调

安装依赖后：

```bash
pip install -e ".[hf]"
```

然后：

```python
from tabular_synth.great import HuggingFaceGReaTFineTuner

trainer = HuggingFaceGReaTFineTuner(
    model_name="distilgpt2",
    output_dir="artifacts/great_distilgpt2",
    epochs=1,
    batch_size=4,
)
trainer.fit(train)
synthetic = trainer.sample(100, conditions={"default.payment.next.month": 1})
```

## 参考来源

- Survey: https://arxiv.org/abs/2504.16506
- EPIC paper: https://arxiv.org/abs/2404.12404
- EPIC repo: https://github.com/seharanul17/synthetic-tabular-LLM
- GReaT paper: https://arxiv.org/abs/2210.06280
- GReaT repo/package: https://github.com/tabularis-ai/be_great
- HELOC/FICO schema reference: https://huggingface.co/datasets/mstz/heloc/blob/main/heloc.py
