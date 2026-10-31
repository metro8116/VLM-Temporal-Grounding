<h1 align="center">VLM × Temporal Grounding</h1>

<h3 align="center">SFT-LoRA · GRPO-QLoRA · Full-GRPO</h3>

<p align="center">
  English · <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">Qwen2.5-VL-7B</a> ·
  <a href="https://huggingface.co/JungleGym/TimeLens-Qwen3-VL-8B-SFT">TimeLens-8B-SFT</a> ·
  <a href="https://huggingface.co/datasets/Boshenxx/TimeR1-Dataset">TimeRFT</a> ·
  <a href="https://huggingface.co/datasets/TencentARC/TimeLens-Bench">TimeLens-Bench</a>
</p>

<a id="overview"></a>

## 📌 Overview

Predict the start and end of a video event from a natural-language description. This repository organizes **Time-R1-related native SFT-LoRA** and **TimeLens GRPO** experiments, including data preparation, training, weight merging, inference and evaluation.

- **SFT-LoRA:** fine-tune Qwen2.5-VL-7B-Instruct with absolute-second or relative-percentage targets.
- **GRPO-QLoRA:** apply single-GPU 4-bit NF4 + LoRA reinforcement fine-tuning to TimeLens-8B-SFT.
- **Full-GRPO:** start from the same TimeLens-8B-SFT, freeze the visual encoder and update the language model and merger.

The native SFT module uses TimeRFT data and the retained Qwen-VL-Series-Finetune-based implementation. The TimeLens module retains its custom GRPO trainer. See [source provenance](docs/PROVENANCE.md).

<a id="contents"></a>

## 📚 Contents

- [🛠️ Installation](#installation)
- [🤗 Download models](#download-models)
- [📦 Prepare data](#prepare-data)
- [🧠 Train SFT-LoRA](#train-sft-lora)
- [🧠 Train GRPO-QLoRA](#train-grpo-qlora)
- [🧠 Train Full-GRPO](#train-full-grpo)
- [🚀 Evaluate all three models](#evaluation)
- [🔄 Resume training](#resume-training)
- [📈 Results](#results)
- [✅ Local checks](#local-checks)
- [🙏 Acknowledgements and licenses](#acknowledgements)

<details>
<summary>📂 Project structure and artifact locations</summary>

```text
VLM-Temporal-Grounding/
├── time_r1/                       # Native SFT source, configs and launchers
├── timelens/                      # TimeLens GRPO source, configs and preprocessing
├── tools/                         # Data, train, infer, merge and evaluate CLIs
├── vtg/                           # Shared data, launch and weight-loading utilities
├── evaluation/                    # Temporal IoU and recall
├── requirements/                  # Independent SFT and TimeLens environments
├── models/                        # Downloaded starting points and merged models
│   ├── Qwen2.5-VL-7B-Instruct/
│   ├── TimeLens-8B-SFT/
│   └── TimeLens-8B-Official/       # Optional final model for reference evaluation
├── data/
│   ├── raw/                       # Downloaded annotations, archives and offline IoU
│   ├── videos/                    # Extracted video files
│   └── prepared/                  # Training data and fixed evaluation manifests
├── outputs/                       # Checkpoints, final weights, predictions and metrics
├── results/historical/            # Archived predictions and reconstructed metrics
├── docs/                          # Implementation, provenance and validation details
└── tests/                         # CPU unit and CLI integration tests
```

`models/`, `data/raw/`, `data/videos/`, `data/prepared/` and `outputs/` are generated locally and ignored by Git. Download models directly into the listed `models/` directories.

</details>

<a id="installation"></a>

## 🛠️ Installation

Train on **Linux with NVIDIA GPUs**. Run all commands from the repository root. The candidate stack uses **Python 3.11, PyTorch 2.8.0 + CUDA 12.8, Transformers 4.57.1**, and **TRL 0.19.1** for GRPO. The default attention backend is SDPA.

| Workflow | Environment | Requirements | Historical hardware class |
| --- | --- | --- | --- |
| SFT-LoRA | `.venv-sft` | `requirements/sft.txt` | 1 × 96 GB |
| GRPO-QLoRA | `.venv-timelens` | `requirements/qlora.txt` | 1 × A800 80 GB |
| Full-GRPO | `.venv-timelens` | Same, includes full training dependencies | 8 × 96 GB |

These are historical resource classes, not verified memory minima. The candidate GPU environment and model paths still require hardware validation.

Ubuntu system dependencies (administrator access required):

```bash
sudo apt-get update
sudo apt-get install -y build-essential ffmpeg unzip curl libgl1 libglib2.0-0
```

### SFT-LoRA environment

```bash
python3.11 -m venv .venv-sft
source .venv-sft/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/sft.txt 'huggingface_hub[cli]>=0.34,<1.0'
python -m pip check
python tools/doctor.py --cuda
```

### GRPO-QLoRA and Full-GRPO environment

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

The NVIDIA driver must support the chosen CUDA runtime. Activate the corresponding environment when switching workflows.

<a id="download-models"></a>

## 🤗 Download models

Download the **two training starting points** into `models/`:

| Purpose | Hugging Face model | Local directory |
| --- | --- | --- |
| SFT-LoRA base | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | `models/Qwen2.5-VL-7B-Instruct/` |
| Shared GRPO SFT initialization | [JungleGym/TimeLens-Qwen3-VL-8B-SFT](https://huggingface.co/JungleGym/TimeLens-Qwen3-VL-8B-SFT) | `models/TimeLens-8B-SFT/` |
| Optional official final model | [TencentARC/TimeLens-8B](https://huggingface.co/TencentARC/TimeLens-8B) | `models/TimeLens-8B-Official/` |

With either installed environment active:

```bash
mkdir -p models
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct
hf download JungleGym/TimeLens-Qwen3-VL-8B-SFT --local-dir models/TimeLens-8B-SFT
```

Optional official final model:

```bash
hf download TencentARC/TimeLens-8B --local-dir models/TimeLens-8B-Official
```

Keep the complete weights, `config.json`, processor and tokenizer in each directory. Both GRPO branches start from **TimeLens-8B-SFT**; the official final RL model is an evaluation reference.

<a id="prepare-data"></a>

## 📦 Prepare data

### 1. SFT-LoRA: download TimeRFT and create targets

Use Time-R1's `train_2k5.json` annotations and videos from [TimeR1-Dataset](https://huggingface.co/datasets/Boshenxx/TimeR1-Dataset):

```bash
source .venv-sft/bin/activate
mkdir -p data/raw/timerft data/videos/timerft
curl -fL https://raw.githubusercontent.com/xiaomi-research/Time-R1/main/dataset/timer1/annotations/train_2k5.json \
  -o data/raw/timerft/train_2k5.json
hf download Boshenxx/TimeR1-Dataset TimeRFT_data.zip \
  --repo-type dataset --local-dir data/raw/timerft
unzip -q data/raw/timerft/TimeRFT_data.zip -d data/videos/timerft
```

Generate absolute-second and relative-percentage training examples:

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

The two sets share event order. Targets are formatted as `12.5 to 17.8` or `13.4% to 48.2%`. The converter recursively resolves unique video basenames, rewrites old server paths and checks annotation boundaries and nonempty files.

### 2. GRPO: prepare TimeLens-100K videos and offline IoU

Download [TimeLens-100K](https://huggingface.co/datasets/TencentARC/TimeLens-100K) and extract its video shards:

```bash
source .venv-timelens/bin/activate
mkdir -p data/raw/timelens_100k data/videos/timelens_100k
hf download TencentARC/TimeLens-100K --repo-type dataset --local-dir data/raw/timelens_100k
find data/raw/timelens_100k/video_shards -name '*.tar.gz' -print0 | \
  xargs -0 -P 4 -I '{}' tar -xzf '{}' -C data/videos/timelens_100k
```

Download [offline IoU predictions](https://huggingface.co/datasets/JungleGym/TimeLens-Qwen3-VL-8B-filter-data) from the same SFT starting point and remap the video paths:

```bash
hf download JungleGym/TimeLens-Qwen3-VL-8B-filter-data \
  FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl \
  --repo-type dataset --local-dir data/raw/filter_data

python tools/prepare_data.py --format filtered \
  --input data/raw/filter_data/FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl \
  --video-root data/videos/timelens_100k --path-mode lookup \
  --output data/prepared/timelens_filter.json
```

The training dataset samples approximately 2,500 events using duration buckets and Gaussian IoU weighting. Both GRPO branches use this input file; each run saves its actual sample selection as `selected_training_annotations.json`. To regenerate offline IoU predictions, see the complete optional workflow in [data preparation](data/README.md).

### 3. SFT evaluation: original Charades-STA

Use the `test` annotations from `lmms-lab/charades_sta` and the official Charades 480p video archive:

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

Use the same manifest for absolute/relative models and checkpoint comparisons. For a full evaluation, omit `--limit 1000` and choose a separate manifest filename.

### 4. GRPO evaluation: TimeLens-Bench

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

By default the manifests contain the full benchmarks: ActivityNet **4,500**, Charades **3,363**, and QVHighlights **1,541** queries. To construct a longest-video 2,000-query subset, explicitly add `--selection longest --limit 2000` and save it separately.

Optionally decode media during validation:

```bash
python tools/check_data.py --manifest data/prepared/charades_sta_1k.json --decode
python tools/check_data.py --manifest data/prepared/timelens_bench/qvhighlights.json --decode
```

<a id="train-sft-lora"></a>

## 🧠 Train SFT-LoRA

### Absolute seconds

```bash
source .venv-sft/bin/activate
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --config time_r1/configs/abs.json \
  --model "$PWD/models/Qwen2.5-VL-7B-Instruct" \
  --data data/prepared/train_abs.json \
  --output outputs/time_r1/abs
```

### Relative percentages

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --config time_r1/configs/rel.json \
  --model "$PWD/models/Qwen2.5-VL-7B-Instruct" \
  --data data/prepared/train_rel.json \
  --output outputs/time_r1/rel
```

The two recipes match training settings and differ in target representation.

| Setting | Value |
| --- | --- |
| LoRA | rank 64, alpha 128, dropout 0.05 |
| Trainable modules | LoRA and merger; language backbone and visual encoder frozen |
| Learning rate / warmup | `2e-5` / `0` |
| Batch / epochs | 1 GPU × batch 1 × accumulation 16; 5 epochs |
| Video | 2 FPS, at most 700 frames, 100352–109760 pixels per frame |
| Precision / optimization | BF16, SDPA, ZeRO3, gradient checkpointing |
| Saving | every 50 steps, at most 3 checkpoints retained |

Separate low-learning-rate relative experiment:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --config time_r1/configs/rel_low_lr.json \
  --model "$PWD/models/Qwen2.5-VL-7B-Instruct" \
  --data data/prepared/train_rel.json \
  --output outputs/time_r1/rel_low_lr
```

The low-LR recipe uses `1e-5` and warmup ratio `0.03` and is recorded separately.

Final `outputs/time_r1/abs/` or `rel/` contains the **LoRA adapter, processor and `non_lora_state_dict.bin`**. The extra file contains trained merger weights and is loaded together with the adapter. See [SFT-LoRA evaluation](#evaluate-sft-lora).

<a id="train-grpo-qlora"></a>

## 🧠 Train GRPO-QLoRA

Start from the downloaded `models/TimeLens-8B-SFT/`:

```bash
source .venv-timelens/bin/activate
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --config timelens/configs/qlora_a800.json \
  --model "$PWD/models/TimeLens-8B-SFT" \
  --data data/prepared/timelens_filter.json \
  --output outputs/timelens/grpo_qlora
```

| Setting | Value |
| --- | --- |
| Quantization / LoRA | 4-bit NF4; rank 64, alpha 16, dropout 0.05 |
| Trainable modules | LoRA and merger; language backbone and visual encoder frozen |
| Batch | 1 GPU × batch 4 × accumulation 8 = 32 |
| GRPO | 4 candidates per input, pure tIoU reward, `beta=0` |
| Learning rate / steps | `1e-6` / 100 optimizer steps |
| Video | 2 FPS, at most 320 frames, 10240 visual tokens |
| Precision / optimization | BF16 compute, SDPA, ZeRO1, gradient checkpointing |
| Saving | every 20 steps, at most 3 checkpoints retained |

Final `outputs/timelens/grpo_qlora/` contains the **adapter, processor and `non_lora_trainables.bin`**. Load it with the original SFT starting point, or merge it into a complete model. See [GRPO-QLoRA evaluation](#evaluate-grpo-qlora).

<a id="train-full-grpo"></a>

## 🧠 Train Full-GRPO

Use the same `models/TimeLens-8B-SFT/` initialization on eight GPUs:

```bash
source .venv-timelens/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/train.py \
  --config timelens/configs/full_pro6000.json \
  --model "$PWD/models/TimeLens-8B-SFT" \
  --data data/prepared/timelens_filter.json \
  --output outputs/timelens/grpo_full
```

| Setting | Value |
| --- | --- |
| Trainable modules | Full language model and merger; visual encoder frozen; no LoRA |
| Batch | 8 GPUs × batch 1 × accumulation 8 = 64 |
| GRPO | 8 candidates per input, pure tIoU reward, `beta=0` |
| Learning rate / steps | `1e-6` / 100 optimizer steps |
| Video | 2 FPS, at most 448 frames, 14336 visual tokens |
| Precision / optimization | BF16, SDPA, ZeRO3, gradient checkpointing |
| Saving | every 20 steps, at most 3 checkpoints retained |

Final `outputs/timelens/grpo_full/` contains a **complete model and processor**, ready for inference without adapter merging. The two GRPO recipes differ in candidate count, batch and visual input budget as well as parameter updates.

<a id="evaluation"></a>

## 🚀 Evaluate all three models

| Training method | Inference entry | Weights | Benchmark |
| --- | --- | --- | --- |
| SFT-LoRA | `tools/infer.py --workload sft` | Qwen2.5-VL base + adapter, or merged model | Original Charades-STA |
| GRPO-QLoRA | `tools/infer.py --workload timelens` | TimeLens-8B-SFT + adapter, or merged model | TimeLens-Bench |
| Full-GRPO | `tools/infer.py --workload timelens` | Complete Full-GRPO model | TimeLens-Bench |

`tools/evaluate.py` computes **mIoU and R@1 at 0.3/0.5/0.7** for all methods. Use a new prediction filename for every run.

<a id="evaluate-sft-lora"></a>

### 1. SFT-LoRA inference and metrics

Absolute-second model:

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

Relative-percentage model:

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

Percentages are mapped back to seconds before IoU computation. To evaluate an intermediate checkpoint, replace `--adapter` with its `checkpoint-*` directory and choose a new output filename. The low-LR recipe also uses `--representation relative`.

<a id="evaluate-grpo-qlora"></a>

### 2. GRPO-QLoRA on all three datasets

Load the **SFT starting point, adapter and extra trained merger weights**:

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

<a id="evaluate-full-grpo"></a>

### 3. Full-GRPO on all three datasets

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

Both GRPO models are evaluated with the same fixed manifests and inference budget. `--round-unit 1` follows historical TimeLens evaluation rounding. Original Charades-STA and reannotated Charades-TimeLens are reported separately.

<details>
<summary>🔗 Optional: merge adapters and evaluate complete models</summary>

SFT-LoRA:

```bash
source .venv-sft/bin/activate
CUDA_VISIBLE_DEVICES=0 python tools/merge_adapter.py \
  --model "$PWD/models/Qwen2.5-VL-7B-Instruct" \
  --adapter outputs/time_r1/abs --output models/SFT-Abs-Merged
```

GRPO-QLoRA:

```bash
source .venv-timelens/bin/activate
CUDA_VISIBLE_DEVICES=0 python tools/merge_adapter.py \
  --model "$PWD/models/TimeLens-8B-SFT" \
  --adapter outputs/timelens/grpo_qlora --output models/TimeLens-8B-QLoRA-Merged
```

Change the corresponding inference command's `--model` to the merged directory and remove `--adapter`. Merging restores the extra non-LoRA trainables as well as LoRA deltas. Use the exact original training base for each adapter.

</details>

<details>
<summary>⚖️ Optional: evaluate the official final TimeLens model</summary>

Download the optional official model above, then evaluate QVHighlights as follows:

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

Inference writes a record for every query. Failed and missing predictions count as zero against the expected manifest. See [the detailed TimeLens guide](docs/TIMELENS.md) for parallel inference and explicit shard merging.

<a id="resume-training"></a>

## 🔄 Resume training

Each training output contains `checkpoint-*`, final weights and `run_config.json`. Add `--resume` to the original training command and point it to an existing checkpoint with optimizer state. This example selects the latest checkpoint:

```bash
source .venv-sft/bin/activate
latest_checkpoint="$(find outputs/time_r1/abs -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config time_r1/configs/abs.json \
  --model "$PWD/models/Qwen2.5-VL-7B-Instruct" --data data/prepared/train_abs.json \
  --output outputs/time_r1/abs --resume "$latest_checkpoint"
```

For either GRPO method, use its corresponding config, model, data and output paths and select its latest checkpoint in the same way. Final inference weights and resumable training checkpoints have different purposes. Use a new output directory for a new experiment.

<a id="results"></a>

## 📈 Results

These mIoU values are reconstructed from **historical predictions**, not freshly trained models from this packaging session. ActivityNet and Charades use matched 2,000-query sets; QVHighlights uses all 1,541 queries.

| Model | ActivityNet-TimeLens | Charades-TimeLens | QVHighlights-TimeLens |
| --- | ---: | ---: | ---: |
| Qwen3-VL-8B base | 43.28 | 48.14 | 59.48 |
| TimeLens-8B-SFT start | 46.06 | 51.19 | 57.18 |
| GRPO-QLoRA | 46.06 | 50.99 | 57.24 |
| Full-GRPO | 50.51 | 54.59 | 65.42 |
| Official model, local evaluation | 50.12 | 54.91 | 65.46 |

The retained native SFT absolute-second record recomputes to **52.95%**, with an unresolved checkpoint identifier; the low-LR percentage checkpoint-600 record gives **41.37%**. Their settings differ, so these are not a controlled timestamp-only pair. Record new recipe results separately.

[Metrics CSV](results/historical/metrics.csv) · [Prediction provenance and hashes](results/historical/manifest.json) · [Full results and evaluation protocol](docs/RESULTS.md)

<a id="local-checks"></a>

## ✅ Local checks

These commands do not download models or start GPU training:

```bash
python -m unittest discover -s tests -v
python tools/check_repository.py
python tools/recompute_history.py
```

Inspect a training command by appending `--dry-run`:

```bash
python tools/train.py --config time_r1/configs/abs.json \
  --model "$PWD/models/Qwen2.5-VL-7B-Instruct" \
  --data data/prepared/train_abs.json --output outputs/time_r1/abs --dry-run
```

CPU tests and historical metric reconstruction have passed. Training, weight save/reload and inference still need validation on Linux with real models and videos; see [validation status](docs/VALIDATION.md).

<a id="acknowledgements"></a>

## 🙏 Acknowledgements and licenses

- [Time-R1](https://github.com/xiaomi-research/Time-R1): TimeRFT data and temporal grounding research.
- [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune): native SFT training framework.
- [TimeLens](https://github.com/TencentARC/TimeLens): models, data, custom GRPO trainer and evaluation foundation.
- [Qwen](https://github.com/QwenLM/Qwen3-VL), [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), and [DeepSpeed](https://github.com/deepspeedai/DeepSpeed): model and training infrastructure.

Each component retains its original license. TimeLens includes academic-use and regional restrictions; models and datasets also retain their own terms. See [LICENSE](LICENSE) and [third-party notices](third_party_licenses/README.md).
