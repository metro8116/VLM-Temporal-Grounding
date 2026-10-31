# TimeLens GRPO comparison

Both branches start from **TimeLens-8B-SFT**, built on Qwen3-VL-8B. The downloaded
official final RL model is used only as an evaluation reference.

## Recipes

| Config | GPUs | Update | Batch | Generations | Max frames | Visual token budget |
|---|---:|---|---:|---:|---:|---:|
| `qlora_a800.json` | 1 | 4-bit NF4 + LoRA | 32 | 4 | 320 | 10240 |
| `full_pro6000.json` | 8 | full language model + merger | 64 | 8 | 448 | 14336 |

Both freeze the visual encoder, use 2 FPS, BF16 computation, LR 1e-6, pure temporal
IoU reward, beta 0 and 100 optimizer steps. Full GRPO uses ZeRO3/SDPA; QLoRA uses
ZeRO1, LoRA rank 64/alpha 16/dropout 0.05. The packaged QLoRA attention default is
also SDPA; `--attention flash_attention_2` restores that backend when compatible.
The names describe historical hardware class, not a verified memory minimum.

The generation count, batch and input budget differ between the recipes. Results
cannot isolate the effect of quantization or trainable parameter count alone.

## Prepare and train

See [the data guide](../data/README.md) for the SFT checkpoint, actual videos and
offline IoU annotations. The converter fixes media paths but does not pre-sample
training events. The training dataset applies the retained duration buckets and
Gaussian IoU sampling (target approximately 2,500, mean 0.05, std 0.2).

```bash
python tools/train.py --config timelens/configs/qlora_a800.json \
  --model models/TimeLens-8B-SFT --data data/prepared/timelens_filter.json \
  --output outputs/grpo_qlora
python tools/train.py --config timelens/configs/full_pro6000.json \
  --model models/TimeLens-8B-SFT --data data/prepared/timelens_filter.json \
  --output outputs/grpo_full
```

The launcher checks batch/generation divisibility. `--gpus` overrides must remain
consistent with the recipe. For a one-GPU full-parameter smoke test, a possible
override is `--gpus 1 --set num_generations=2 --set per_device_train_batch_size=2`,
with `--set max_steps=1 --set fps_max_frames=16 --set total_tokens=512` and a short
smoke dataset. Memory feasibility must be checked on that GPU; this is not an 8 GB recipe.

Each run records effective arguments, installed package versions, actually selected
training events and trainable parameter names. The trainer restores merger gradients
after quantization/PEFT and rejects an unintended freeze policy.

## Weights and resume

- Full GRPO saves a complete model and processor.
- QLoRA saves its adapter, processor and `non_lora_trainables.bin`, including during
  checkpoint saves. ZeRO gather operations are called by every participating rank.
- Resume is explicit and requires optimizer state; final inference weights alone
  are not a full training checkpoint.
- Trainer checkpoint rotation retains the latest checkpoint; the old manual cleanup
  that could delete the newest checkpoint was removed.

Merge QLoRA into the **same SFT initialization**, not the original Qwen3 base:

```bash
python tools/merge_adapter.py --model models/TimeLens-8B-SFT \
  --adapter outputs/grpo_qlora --output models/TimeLens-8B-QLoRA-Merged
```

## Inference and evaluation

Use the three fixed TimeLens-Bench manifests prepared in the data guide. Example:

```bash
python tools/infer.py --workload timelens --model outputs/grpo_full \
  --manifest data/prepared/qvhighlights_manifest.json \
  --output outputs/grpo_full_qv.jsonl
python tools/evaluate.py --predictions outputs/grpo_full_qv.jsonl \
  --manifest data/prepared/qvhighlights_manifest.json --output outputs/grpo_full_qv_metrics.json
```

For QLoRA, pass `--model models/TimeLens-8B-SFT --adapter outputs/grpo_qlora`, or a
merged full model. The official final model can be evaluated with the same manifest.
TimeLens historical evaluation rounds predictions to one-second units; the new
TimeLens inference defaults to `--round-unit 1`. Record an override as a protocol change.

For multi-GPU inference, launch independent processes with per-process
`CUDA_VISIBLE_DEVICES`, the same manifest, and `--num-shards N --shard-index I`.
Merge only explicitly named prediction files:

```bash
python tools/merge_predictions.py --inputs outputs/qv_0.jsonl outputs/qv_1.jsonl \
  --manifest data/prepared/qvhighlights_manifest.json --output outputs/qv_merged.jsonl
```

Duplicate ids are rejected. Missing outputs become recorded zero-score failures;
error records remain separate entries rather than being concatenated as annotations.
