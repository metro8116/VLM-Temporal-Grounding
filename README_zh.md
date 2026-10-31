<h1 align="center">VLM × Temporal Grounding</h1>

<h3 align="center">SFT-LoRA · GRPO-QLoRA · Full-GRPO</h3>

<p align="center">
  <a href="README.md">English</a> · 简体中文
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">Qwen2.5-VL-7B</a> ·
  <a href="https://huggingface.co/JungleGym/TimeLens-Qwen3-VL-8B-SFT">TimeLens-8B-SFT</a> ·
  <a href="https://huggingface.co/datasets/Boshenxx/TimeR1-Dataset">TimeRFT</a> ·
  <a href="https://huggingface.co/datasets/TencentARC/TimeLens-Bench">TimeLens-Bench</a>
</p>

<a id="项目简介"></a>

## 📌 项目简介

根据自然语言描述预测视频事件的起止时间，整理 **Time-R1 相关原生 SFT-LoRA** 和 **TimeLens GRPO** 两项研究工作，覆盖数据准备、模型微调、权重合并、推理与评测。

- **SFT-LoRA：**基于 Qwen2.5-VL-7B-Instruct，比较绝对秒数与相对百分比时间表征。
- **GRPO-QLoRA：**基于 TimeLens-8B-SFT，采用 4-bit NF4 和 LoRA 进行单卡强化微调。
- **Full-GRPO：**基于同一个 TimeLens-8B-SFT，冻结视觉编码器，更新语言模型与 merger。

Time-R1 模块使用 TimeRFT 数据及已有的原生 SFT 工程；TimeLens 模块保留其定制 GRPO trainer。模型、训练框架与本项目整理修改的归属见 [来源说明](docs/PROVENANCE.md)。

<a id="目录"></a>

## 📚 目录

- [🛠️ 安装环境](#安装环境)
- [🤗 下载模型](#下载模型)
- [📦 准备数据](#准备数据)
- [🧠 SFT-LoRA 训练](#sft-lora-训练)
- [🧠 GRPO-QLoRA 训练](#grpo-qlora-训练)
- [🧠 Full-GRPO 训练](#full-grpo-训练)
- [🚀 三种模型的评测入口](#评测入口)
- [🔄 恢复训练与 checkpoint](#恢复训练)
- [📈 实验结果](#实验结果)
- [✅ 本机检查](#本机检查)
- [🙏 致谢与许可](#致谢与许可)

<details>
<summary>📂 项目结构与文件保存位置</summary>

```text
VLM-Temporal-Grounding/
├── time_r1/                       # 原生 SFT 代码、配置及启动脚本
├── timelens/                      # TimeLens GRPO 代码、配置及预处理
├── tools/                         # 数据、训练、推理、合并、评测入口
├── vtg/                           # 共享数据、训练命令及权重加载工具
├── evaluation/                    # mIoU 与 R@1 指标
├── requirements/                  # SFT / TimeLens 独立环境
├── models/                        # 下载的底座、SFT 起点及合并模型
│   ├── Qwen2.5-VL-7B-Instruct/
│   ├── TimeLens-8B-SFT/
│   └── TimeLens-8B-Official/       # 可选：官方最终模型，用于对照评测
├── data/
│   ├── raw/                       # 下载的标注、压缩包及离线 IoU
│   ├── videos/                    # 解压后的真实视频
│   └── prepared/                  # 转换后的训练数据与固定评测清单
├── outputs/                       # 训练权重、checkpoint、预测及指标
├── results/historical/            # 已有实验的预测归档与指标
├── docs/                          # 实现细节、来源及验证记录
└── tests/                         # CPU 单元与命令行集成测试
```

`models/`、`data/raw/`、`data/videos/`、`data/prepared/` 和 `outputs/` 在使用时生成，由 `.gitignore` 排除。模型直接下载到上面的 `models/` 目录即可，无需复制进代码目录。

</details>

<a id="安装环境"></a>

## 🛠️ 安装环境

在 **Linux + NVIDIA GPU** 服务器上运行训练。以下命令均从项目根目录执行，使用 **Python 3.11、PyTorch 2.8.0 + CUDA 12.8、Transformers 4.57.1**；GRPO trainer 对应 **TRL 0.19.1**。默认采用 SDPA，无需先安装 FlashAttention。

| 流程 | Python 环境 | 依赖文件 | 参考硬件 |
| --- | --- | --- | --- |
| SFT-LoRA | `.venv-sft` | `requirements/sft.txt` | 单卡 96 GB |
| GRPO-QLoRA | `.venv-timelens` | `requirements/qlora.txt` | 单卡 A800 80 GB |
| Full-GRPO | `.venv-timelens` | 同上，包含 Full-GRPO 所需依赖 | 8 × 96 GB |

硬件列为原实验资源类别，不是最低显存承诺；当前依赖组合与 GPU 流程尚待实机验证。

Ubuntu 系统依赖（需要管理员权限）：

```bash
sudo apt-get update
sudo apt-get install -y build-essential ffmpeg unzip curl libgl1 libglib2.0-0
```

### SFT-LoRA 环境

```bash
python3.11 -m venv .venv-sft
source .venv-sft/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/sft.txt 'huggingface_hub[cli]>=0.34,<1.0'
python -m pip check
python tools/doctor.py --cuda
```

### GRPO-QLoRA 与 Full-GRPO 环境

```bash
deactivate
python3.11 -m venv .venv-timelens
source .venv-timelens/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/qlora.txt 'huggingface_hub[cli]>=0.34,<1.0'
python -m pip check
python tools/doctor.py --cuda
```

系统 NVIDIA 驱动需要支持所选 CUDA 运行时。每次切换流程时，激活对应环境；SFT 和 TimeLens 环境分开维护。

<a id="下载模型"></a>

## 🤗 下载模型

只需先下载 **两个训练起点**，保存到项目根目录的 `models/`：

| 用途 | Hugging Face 模型 | 本地目录 |
| --- | --- | --- |
| SFT-LoRA 底座 | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | `models/Qwen2.5-VL-7B-Instruct/` |
| 两种 GRPO 的共同 SFT 起点 | [JungleGym/TimeLens-Qwen3-VL-8B-SFT](https://huggingface.co/JungleGym/TimeLens-Qwen3-VL-8B-SFT) | `models/TimeLens-8B-SFT/` |
| 可选：官方最终模型对照 | [TencentARC/TimeLens-8B](https://huggingface.co/TencentARC/TimeLens-8B) | `models/TimeLens-8B-Official/` |

激活任意一个已安装的环境后下载：

```bash
mkdir -p models
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct
hf download JungleGym/TimeLens-Qwen3-VL-8B-SFT --local-dir models/TimeLens-8B-SFT
```

可选的官方最终模型：

```bash
hf download TencentARC/TimeLens-8B --local-dir models/TimeLens-8B-Official
```

每个模型目录保留完整权重、`config.json`、processor 和 tokenizer 文件。GRPO-QLoRA 与 Full-GRPO 都从 `TimeLens-8B-SFT` 开始，官方最终 RL 模型用于对照评测。

<a id="准备数据"></a>

## 📦 准备数据

### 1. SFT-LoRA：下载 TimeRFT 并生成两种监督答案

训练标注采用 Time-R1 的 `train_2k5.json`，视频来自 [TimeR1-Dataset](https://huggingface.co/datasets/Boshenxx/TimeR1-Dataset)：

```bash
source .venv-sft/bin/activate
mkdir -p data/raw/timerft data/videos/timerft
curl -fL https://raw.githubusercontent.com/xiaomi-research/Time-R1/main/dataset/timer1/annotations/train_2k5.json \
  -o data/raw/timerft/train_2k5.json
hf download Boshenxx/TimeR1-Dataset TimeRFT_data.zip \
  --repo-type dataset --local-dir data/raw/timerft
unzip -q data/raw/timerft/TimeRFT_data.zip -d data/videos/timerft
```

分别生成绝对秒数与相对百分比训练数据：

```bash
python tools/prepare_data.py --format timerft \
  --input data/raw/timerft/train_2k5.json \
  --video-root data/videos/timerft --path-mode lookup --dataset TimeRFT \
  --output data/prepared/timerft_abs_manifest.json \
  --sft-output data/prepared/train_abs.json --representation absolute

python tools/prepare_data.py --format timerft \
  --input data/raw/timerft/train_2k5.json \
  --video-root data/videos/timerft --path-mode lookup --dataset TimeRFT \
  --output data/prepared/timerft_rel_manifest.json \
  --sft-output data/prepared/train_rel.json --representation relative
```

两组使用相同事件及顺序，监督答案分别为 `12.5 to 17.8` 与 `13.4% to 48.2%`。转换器递归匹配视频文件名，重写旧服务器路径，并检查标注边界、文件存在性及非零大小。

### 2. 两种 GRPO：准备 TimeLens-100K 视频与离线 IoU

下载 [TimeLens-100K](https://huggingface.co/datasets/TencentARC/TimeLens-100K)，将分片解压到视频目录：

```bash
source .venv-timelens/bin/activate
mkdir -p data/raw/timelens_100k data/videos/timelens_100k
hf download TencentARC/TimeLens-100K --repo-type dataset --local-dir data/raw/timelens_100k
find data/raw/timelens_100k/video_shards -name '*.tar.gz' -print0 | \
  xargs -0 -P 4 -I '{}' tar -xzf '{}' -C data/videos/timelens_100k
```

下载基于同一 SFT 模型得到的[离线 IoU 预测](https://huggingface.co/datasets/JungleGym/TimeLens-Qwen3-VL-8B-filter-data)：

```bash
hf download JungleGym/TimeLens-Qwen3-VL-8B-filter-data \
  FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl \
  --repo-type dataset --local-dir data/raw/filter_data

python tools/prepare_data.py --format filtered \
  --input data/raw/filter_data/FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl \
  --video-root data/videos/timelens_100k --path-mode lookup \
  --output data/prepared/timelens_filter.json
```

训练时按时长分桶，再按离线 IoU 的高斯分布筛选约 2,500 条样本。两种 GRPO 共用这份输入，实际选中样本会保存在各自输出目录的 `selected_training_annotations.json`。需要重新生成离线 IoU 时，可使用 [数据准备说明](data/README.md) 中的完整推理与筛选入口。

### 3. SFT 评测：准备原始 Charades-STA

标注由 `lmms-lab/charades_sta` 的 `test` split 读取；视频采用 Charades 官方 480p 压缩包：

```bash
source .venv-sft/bin/activate
mkdir -p data/raw/charades_sta data/videos/charades_sta
curl -fL https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip \
  -o data/raw/charades_sta/Charades_v1_480.zip
unzip -q -j data/raw/charades_sta/Charades_v1_480.zip '*.mp4' -d data/videos/charades_sta

python tools/prepare_charades.py --from-hub \
  --video-root data/videos/charades_sta --selection original --limit 1000 \
  --output data/prepared/charades_sta_1k.json
```

同一清单用于绝对秒数、相对百分比及不同 checkpoint 的比较。需要全量评测时移除 `--limit 1000`，并使用新的清单文件名。

### 4. GRPO 评测：准备 TimeLens-Bench

```bash
source .venv-timelens/bin/activate
mkdir -p data/raw/timelens_bench data/videos/timelens_bench
hf download TencentARC/TimeLens-Bench --repo-type dataset --local-dir data/raw/timelens_bench
find data/raw/timelens_bench/video_shards -name '*.tar.gz' -print0 | \
  xargs -0 -P 4 -I '{}' tar -xzf '{}' -C data/videos/timelens_bench

for dataset in activitynet charades qvhighlights; do
  python tools/prepare_data.py --format timelens-bench \
    --input "data/raw/timelens_bench/${dataset}-timelens.json" \
    --video-root "data/videos/timelens_bench/${dataset}" \
    --dataset "${dataset}-timelens" \
    --output "data/prepared/timelens_bench/${dataset}.json"
done
```

默认生成三个数据集的全量清单：ActivityNet **4,500**、Charades **3,363**、QVHighlights **1,541** 条查询。要按旧实验建立前两者的最长视频 2,000 条子集，额外指定 `--selection longest --limit 2000` 并另存清单。

可额外解码检查媒体：

```bash
python tools/check_data.py --manifest data/prepared/charades_sta_1k.json --decode
python tools/check_data.py --manifest data/prepared/timelens_bench/qvhighlights.json --decode
```

<a id="sft-lora-训练"></a>

## 🧠 SFT-LoRA 训练

### 绝对秒数

```bash
source .venv-sft/bin/activate
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --config time_r1/configs/abs.json \
  --model "$PWD/models/Qwen2.5-VL-7B-Instruct" \
  --data data/prepared/train_abs.json \
  --output outputs/time_r1/abs
```

### 相对百分比

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --config time_r1/configs/rel.json \
  --model "$PWD/models/Qwen2.5-VL-7B-Instruct" \
  --data data/prepared/train_rel.json \
  --output outputs/time_r1/rel
```

两组配置一致，仅监督时间表征不同；视频输入及模块更新范围保持一致。

| 参数 | 配置 |
| --- | --- |
| 训练方式 | LoRA rank 64、alpha 128、dropout 0.05 |
| 更新范围 | 语言模型主体和视觉编码器冻结，LoRA 与 merger 可训练 |
| 学习率 / 预热 | `2e-5` / `0` |
| Batch / 轮数 | 单卡 batch 1 × 梯度累积 16，5 epochs |
| 视频 | 2 FPS、最多 700 帧、单帧像素 100352–109760 |
| 精度 / 优化 | BF16、SDPA、ZeRO3、梯度检查点 |
| 保存 | 每 50 步保存，最多保留 3 个 checkpoint |

单独进行低学习率百分比实验时：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --config time_r1/configs/rel_low_lr.json \
  --model "$PWD/models/Qwen2.5-VL-7B-Instruct" \
  --data data/prepared/train_rel.json \
  --output outputs/time_r1/rel_low_lr
```

该组学习率为 `1e-5`、warmup ratio 为 `0.03`，作为独立设置记录。

训练完成后，`outputs/time_r1/abs/` 或 `rel/` 保存最终 **LoRA adapter、processor 和 `non_lora_state_dict.bin`**。额外文件包含训练过的 merger，加载和合并时一并恢复。评测命令见下方 [SFT-LoRA 评测](#sft-lora-评测)。

<a id="grpo-qlora-训练"></a>

## 🧠 GRPO-QLoRA 训练

从已下载的 `models/TimeLens-8B-SFT/` 开始：

```bash
source .venv-timelens/bin/activate
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --config timelens/configs/qlora_a800.json \
  --model "$PWD/models/TimeLens-8B-SFT" \
  --data data/prepared/timelens_filter.json \
  --output outputs/timelens/grpo_qlora
```

| 参数 | 配置 |
| --- | --- |
| 量化与 LoRA | 4-bit NF4、rank 64、alpha 16、dropout 0.05 |
| 更新范围 | 语言模型主体和视觉编码器冻结，LoRA 与 merger 可训练 |
| Batch | 单卡 batch 4 × 梯度累积 8 = 32 |
| GRPO | 每个输入生成 4 个候选、纯 tIoU 奖励、`beta=0` |
| 学习率 / 步数 | `1e-6` / 100 optimizer steps |
| 视频 | 2 FPS、最多 320 帧、视觉 token 预算 10240 |
| 精度 / 优化 | BF16 计算、SDPA、ZeRO1、梯度检查点 |
| 保存 | 每 20 步保存，最多保留 3 个 checkpoint |

最终输出位于 `outputs/timelens/grpo_qlora/`，包含 **LoRA adapter、processor、`non_lora_trainables.bin`**。这是适配器输出，评测时与原始 TimeLens-8B-SFT 起点一起加载，或先合并成完整模型。评测命令见 [GRPO-QLoRA 评测](#grpo-qlora-评测)。

<a id="full-grpo-训练"></a>

## 🧠 Full-GRPO 训练

同样从 `models/TimeLens-8B-SFT/` 开始，在 8 张 GPU 上运行：

```bash
source .venv-timelens/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/train.py \
  --config timelens/configs/full_pro6000.json \
  --model "$PWD/models/TimeLens-8B-SFT" \
  --data data/prepared/timelens_filter.json \
  --output outputs/timelens/grpo_full
```

| 参数 | 配置 |
| --- | --- |
| 更新范围 | 冻结视觉编码器，更新语言模型与 merger，不使用 LoRA |
| Batch | 8 卡 × 每卡 batch 1 × 梯度累积 8 = 64 |
| GRPO | 每个输入生成 8 个候选、纯 tIoU 奖励、`beta=0` |
| 学习率 / 步数 | `1e-6` / 100 optimizer steps |
| 视频 | 2 FPS、最多 448 帧、视觉 token 预算 14336 |
| 精度 / 优化 | BF16、SDPA、ZeRO3、梯度检查点 |
| 保存 | 每 20 步保存，最多保留 3 个 checkpoint |

最终输出 `outputs/timelens/grpo_full/` 为 **完整模型与 processor**，可以直接推理，无需 LoRA 合并。两种 GRPO 的候选数、batch 和视频预算不同，结果按各自配置汇报。

<a id="评测入口"></a>

## 🚀 三种模型的评测入口

| 训练方式 | 推理入口 | 使用的权重 | 评测数据 |
| --- | --- | --- | --- |
| SFT-LoRA | `tools/infer.py --workload sft` | Qwen2.5-VL 底座 + SFT adapter，或合并模型 | 原始 Charades-STA |
| GRPO-QLoRA | `tools/infer.py --workload timelens` | TimeLens-8B-SFT + QLoRA adapter，或合并模型 | TimeLens-Bench |
| Full-GRPO | `tools/infer.py --workload timelens` | Full-GRPO 完整模型 | TimeLens-Bench |

三种方式统一通过 `tools/evaluate.py` 计算 **mIoU、R@1@0.3/0.5/0.7**。每次评测使用新的预测输出文件名，保留与训练、模型选择对应的记录。

<a id="sft-lora-评测"></a>

### 1. SFT-LoRA：推理并计算指标

绝对秒数模型：

```bash
source .venv-sft/bin/activate
CUDA_VISIBLE_DEVICES=0 python tools/infer.py --workload sft \
  --model "$PWD/models/Qwen2.5-VL-7B-Instruct" \
  --adapter outputs/time_r1/abs --representation absolute \
  --manifest data/prepared/charades_sta_1k.json \
  --fps 2 --max-frames 700 --max-new-tokens 32 \
  --output outputs/eval/sft_abs_charades.jsonl

python tools/evaluate.py \
  --predictions outputs/eval/sft_abs_charades.jsonl \
  --manifest data/prepared/charades_sta_1k.json \
  --output outputs/eval/sft_abs_charades_metrics.json
```

相对百分比模型：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/infer.py --workload sft \
  --model "$PWD/models/Qwen2.5-VL-7B-Instruct" \
  --adapter outputs/time_r1/rel --representation relative \
  --manifest data/prepared/charades_sta_1k.json \
  --fps 2 --max-frames 700 --max-new-tokens 32 \
  --output outputs/eval/sft_rel_charades.jsonl

python tools/evaluate.py \
  --predictions outputs/eval/sft_rel_charades.jsonl \
  --manifest data/prepared/charades_sta_1k.json \
  --output outputs/eval/sft_rel_charades_metrics.json
```

百分比预测在计算 IoU 前还原为秒数。评测中间 checkpoint 时，将 `--adapter` 替换为对应的 `checkpoint-*` 目录，并更换输出文件名。低学习率组同样使用 `--representation relative`。

<a id="grpo-qlora-评测"></a>

### 2. GRPO-QLoRA：三个数据集评测

下面直接加载 **SFT 起点 + adapter + 额外 merger 权重**：

```bash
source .venv-timelens/bin/activate
for dataset in activitynet charades qvhighlights; do
  CUDA_VISIBLE_DEVICES=0 python tools/infer.py --workload timelens \
    --model "$PWD/models/TimeLens-8B-SFT" \
    --adapter outputs/timelens/grpo_qlora \
    --manifest "data/prepared/timelens_bench/${dataset}.json" \
    --fps 2 --max-frames 448 --min-tokens 64 --total-tokens 14336 --round-unit 1 \
    --output "outputs/eval/grpo_qlora_${dataset}.jsonl"

  python tools/evaluate.py \
    --predictions "outputs/eval/grpo_qlora_${dataset}.jsonl" \
    --manifest "data/prepared/timelens_bench/${dataset}.json" \
    --output "outputs/eval/grpo_qlora_${dataset}_metrics.json"
done
```

<a id="full-grpo-评测"></a>

### 3. Full-GRPO：三个数据集评测

```bash
source .venv-timelens/bin/activate
for dataset in activitynet charades qvhighlights; do
  CUDA_VISIBLE_DEVICES=0 python tools/infer.py --workload timelens \
    --model "$PWD/outputs/timelens/grpo_full" \
    --manifest "data/prepared/timelens_bench/${dataset}.json" \
    --fps 2 --max-frames 448 --min-tokens 64 --total-tokens 14336 --round-unit 1 \
    --output "outputs/eval/grpo_full_${dataset}.jsonl"

  python tools/evaluate.py \
    --predictions "outputs/eval/grpo_full_${dataset}.jsonl" \
    --manifest "data/prepared/timelens_bench/${dataset}.json" \
    --output "outputs/eval/grpo_full_${dataset}_metrics.json"
done
```

两种 GRPO 在相同固定清单及相同推理预算下评测。`--round-unit 1` 沿用 TimeLens 历史评测的 1 秒取整设置。原始 Charades-STA 与重标注的 Charades-TimeLens 分别汇报。

<details>
<summary>🔗 可选：合并 LoRA，再按完整模型评测</summary>

SFT-LoRA：

```bash
source .venv-sft/bin/activate
CUDA_VISIBLE_DEVICES=0 python tools/merge_adapter.py \
  --model "$PWD/models/Qwen2.5-VL-7B-Instruct" \
  --adapter outputs/time_r1/abs --output models/SFT-Abs-Merged
```

GRPO-QLoRA：

```bash
source .venv-timelens/bin/activate
CUDA_VISIBLE_DEVICES=0 python tools/merge_adapter.py \
  --model "$PWD/models/TimeLens-8B-SFT" \
  --adapter outputs/timelens/grpo_qlora --output models/TimeLens-8B-QLoRA-Merged
```

随后把对应推理命令的 `--model` 改为合并模型目录，并移除 `--adapter`。合并过程会同时加载额外的非 LoRA 训练权重，SFT 和 QLoRA 使用各自原始训练起点。

</details>

<details>
<summary>⚖️ 可选：评测官方 TimeLens 最终模型</summary>

下载上面的可选官方模型后，以 QVHighlights 为例：

```bash
source .venv-timelens/bin/activate
CUDA_VISIBLE_DEVICES=0 python tools/infer.py --workload timelens \
  --model "$PWD/models/TimeLens-8B-Official" \
  --manifest data/prepared/timelens_bench/qvhighlights.json \
  --fps 2 --max-frames 448 --min-tokens 64 --total-tokens 14336 --round-unit 1 \
  --output outputs/eval/official_qvhighlights.jsonl

python tools/evaluate.py --predictions outputs/eval/official_qvhighlights.jsonl \
  --manifest data/prepared/timelens_bench/qvhighlights.json \
  --output outputs/eval/official_qvhighlights_metrics.json
```

</details>

所有推理结果按查询逐条保存，失败和缺失预测按预定清单计零，避免失败样本从分母消失。并行推理及分片合并见 [TimeLens 详细说明](docs/TIMELENS.md)。

<a id="恢复训练"></a>

## 🔄 恢复训练与 checkpoint

训练输出目录包含 `checkpoint-*`、最终权重、`run_config.json` 等文件。需要续训时，在原训练命令中增加 `--resume`，指向真实存在且包含优化器状态的 checkpoint。以下自动选择对应目录中最新的 checkpoint：

```bash
source .venv-sft/bin/activate
latest_checkpoint="$(find outputs/time_r1/abs -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config time_r1/configs/abs.json \
  --model "$PWD/models/Qwen2.5-VL-7B-Instruct" --data data/prepared/train_abs.json \
  --output outputs/time_r1/abs --resume "$latest_checkpoint"
```

两种 GRPO 使用同样方式，将配置、模型、数据、输出目录换成对应训练命令，并指向其最新 checkpoint。评测可以使用最终权重目录；完整续训使用保存了训练状态的 checkpoint。新实验使用新的输出目录。

<a id="实验结果"></a>

## 📈 实验结果

以下为已有预测的离线复算结果，**不是本次整理后重新训练的结果**。ActivityNet、Charades 使用共同 2,000 条查询，QVHighlights 使用全量 1,541 条。

| 模型 | ActivityNet-TimeLens | Charades-TimeLens | QVHighlights-TimeLens |
| --- | ---: | ---: | ---: |
| Qwen3-VL-8B 底座 | 43.28 | 48.14 | 59.48 |
| TimeLens-8B-SFT 起点 | 46.06 | 51.19 | 57.18 |
| GRPO-QLoRA | 46.06 | 50.99 | 57.24 |
| Full-GRPO | 50.51 | 54.59 | 65.42 |
| 官方模型本地复测 | 50.12 | 54.91 | 65.46 |

原生 SFT 备份中的绝对秒数预测重算为 **52.95%**，checkpoint 标识仍有冲突；低学习率百分比组 checkpoint-600 为 **41.37%**。两者配置不同，不能当作只改变时间表征的受控对照。新配置的训练结果需独立记录。

[历史指标 CSV](results/historical/metrics.csv) · [预测来源及哈希](results/historical/manifest.json) · [完整结果与评测口径](docs/RESULTS.md)

<a id="本机检查"></a>

## ✅ 本机检查

以下检查不下载模型、不启动 GPU 训练：

```bash
python -m unittest discover -s tests -v
python tools/check_repository.py
python tools/recompute_history.py
```

先确认训练命令，也可以追加 `--dry-run`：

```bash
python tools/train.py --config time_r1/configs/abs.json \
  --model "$PWD/models/Qwen2.5-VL-7B-Instruct" \
  --data data/prepared/train_abs.json --output outputs/time_r1/abs --dry-run
```

CPU 测试与历史指标复算已通过；GPU 训练、保存重载和推理仍需在准备好真实模型与视频的 Linux 服务器上验证，见 [验证记录](docs/VALIDATION.md)。

<a id="致谢与许可"></a>

## 🙏 致谢与许可

- [Time-R1](https://github.com/xiaomi-research/Time-R1)：TimeRFT 数据与时间定位研究基础。
- [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune)：原生 SFT 训练框架。
- [TimeLens](https://github.com/TencentARC/TimeLens)：模型、数据、定制 GRPO trainer 与评测基础。
- [Qwen](https://github.com/QwenLM/Qwen3-VL)、[Transformers](https://github.com/huggingface/transformers)、[PEFT](https://github.com/huggingface/peft)、[TRL](https://github.com/huggingface/trl)、[DeepSpeed](https://github.com/deepspeedai/DeepSpeed)：模型与训练基础设施。

各组件保留原始许可。TimeLens 许可包含学术用途及区域限制，模型和数据还需遵循各自条款；完整说明见 [LICENSE](LICENSE) 与 [第三方许可](third_party_licenses/README.md)。
