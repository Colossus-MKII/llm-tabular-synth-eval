# 服务器环境配置

## 1. 建议环境拆分

建议至少分两类环境：

- `tabular-llm-synth`: 跑本 repo、GReaT fine-tuning、评估、W&B。
- `vllm-serve`: 专门跑 vLLM serving。如果服务器环境简单，也可以先放在同一个环境，但 vLLM 与训练栈偶尔会有依赖冲突。

## 2. 训练/评估环境

在服务器上进入 repo 根目录：

```bash
conda env create -f environment.yml
conda activate tabular-llm-synth
```

安装 PyTorch。NVIDIA GPU 推荐先用官方选择器确认 CUDA 版本。常见 CUDA 12.8 wheel：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

如果服务器 CUDA 版本较旧，可改为 CUDA 12.6 或 11.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# 或
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

验证：

```bash
python - <<'PY'
import torch
import transformers
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
```

## 2.1 下载实验数据

当前脚本支持自动下载 Adult 和 HELOC：

```bash
python scripts/download_datasets.py --dataset adult --outdir data
python scripts/download_datasets.py --dataset heloc --outdir data
```

输出：

```text
data/adult.csv
data/adult_train.csv
data/adult_test.csv
data/heloc_dataset_v1.csv
data/raw/adult/
data/raw/heloc/
```

Default Credit 暂时仍建议手动下载后放到 `data/default_credit.csv`，因为公开下载源格式变化更多。

## 3. vLLM serving 环境

```bash
conda create -n vllm-serve python=3.11 -y
conda activate vllm-serve
pip install vllm
```

启动 OpenAI-compatible server：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --api-key token-abc123
```

跑 EPIC 时：

```bash
conda activate tabular-llm-synth
export LLM_API_KEY=token-abc123

python scripts/run_real_scale_experiment.py \
  --dataset default-credit \
  --data-path data/default_credit.csv \
  --method epic \
  --llm-model Qwen/Qwen2.5-1.5B-Instruct \
  --llm-base-url http://localhost:8000/v1 \
  --n-synthetic 24000 \
  --seeds 7 11 19 \
  --outdir artifacts/real_scale_epic_qwen15b
```

## 4. llama.cpp server

llama.cpp 更适合 GGUF 量化模型，CPU 或低显存场景更友好。示例：

```bash
./llama-server \
  -m /path/to/model.gguf \
  -c 4096 \
  --host 0.0.0.0 \
  --port 8080
```

跑 EPIC 时：

```bash
python scripts/run_real_scale_experiment.py \
  --dataset default-credit \
  --data-path data/default_credit.csv \
  --method epic \
  --llm-model local-gguf-model \
  --llm-base-url http://localhost:8080/v1 \
  --n-synthetic 24000 \
  --seeds 7 11 19
```

## 5. GReaT fine-tuning

小模型先跑通：

```bash
python scripts/run_real_scale_experiment.py \
  --dataset default-credit \
  --data-path data/default_credit.csv \
  --method great \
  --great-model distilgpt2 \
  --great-epochs 1 \
  --great-batch-size 4 \
  --n-synthetic 24000 \
  --seeds 7 11 19 \
  --outdir artifacts/real_scale_great_distilgpt2
```

用 W&B：

```bash
export WANDB_API_KEY=你的wandb_key
wandb login

python scripts/run_real_scale_experiment.py \
  --dataset default-credit \
  --data-path data/default_credit.csv \
  --method great \
  --great-model Qwen/Qwen2.5-0.5B-Instruct \
  --great-epochs 1 \
  --great-batch-size 2 \
  --wandb-project tabular-synth-great \
  --wandb-log-model false \
  --n-synthetic 24000
```

## 6. 推荐模型顺序

先从小模型开始：

1. `distilgpt2`: 极轻，主要用于确认 GReaT 训练/采样链路。
2. `Qwen/Qwen2.5-0.5B-Instruct`: 低显存、结构化输出能力较好。
3. `Qwen/Qwen2.5-1.5B-Instruct`: 更推荐的低成本主实验模型。
4. `meta-llama/Llama-3.2-1B-Instruct`: 轻量对比模型，需确认 Hugging Face 访问授权。
5. `mistralai/Mistral-7B-Instruct-v0.3`: 7B 对比，显存压力明显更大。

当前代码支持 full fine-tuning 和 LoRA。0.5B/1.5B 可以先用 full fine-tuning；7B 建议用 LoRA。

## 7. 同模型配对对照

EPIC 和 GReaT 对照时必须使用同一个 base model，否则结果会混入模型能力差异。推荐使用配对 wrapper：

```bash
python scripts/run_paired_model_experiment.py \
  --dataset default-credit \
  --data-path data/default_credit.csv \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --llm-base-url http://localhost:8000/v1 \
  --great-bf16 \
  --great-batch-size 1 \
  --great-gradient-accumulation-steps 8 \
  --n-synthetic 24000 \
  --seeds 7 11 19 \
  --outdir artifacts/paired_qwen25
```

这会生成：

```text
artifacts/paired_qwen25/Qwen__Qwen2.5-1.5B-Instruct/epic/
artifacts/paired_qwen25/Qwen__Qwen2.5-1.5B-Instruct/great/
```

7B 配对建议：

```bash
python scripts/run_paired_model_experiment.py \
  --dataset default-credit \
  --data-path data/default_credit.csv \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --llm-base-url http://localhost:8000/v1 \
  --great-use-lora \
  --great-bf16 \
  --great-gradient-checkpointing \
  --great-batch-size 1 \
  --great-gradient-accumulation-steps 16 \
  --great-lora-r 16 \
  --great-lora-alpha 32 \
  --n-synthetic 24000 \
  --seeds 7 11 19 \
  --outdir artifacts/paired_qwen25
```

HELOC 配对建议。HELOC 原始数据约 10,459 行，80/20 split 后 train 约 8,367 行，因此 synthetic size 可以先设为 8,367：

```bash
python scripts/run_paired_model_experiment.py \
  --dataset heloc \
  --data-path data/heloc_dataset_v1.csv \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --llm-base-url http://localhost:8000/v1 \
  --great-bf16 \
  --great-batch-size 1 \
  --great-gradient-accumulation-steps 8 \
  --n-synthetic 8367 \
  --seeds 7 11 19 \
  --outdir artifacts/paired_heloc_qwen25
```

HELOC 的 `-9/-8/-7` 是 FICO special codes，当前 constraints 会把它们作为合法特殊值处理，同时检查普通有效值的整数/range 以及部分 trade-count 逻辑关系。

注意：配对 wrapper 不会自动启动或切换 vLLM。运行 EPIC 前，要确保 `--llm-base-url` 指向的 server 正在服务同一个 `--base-model`。
