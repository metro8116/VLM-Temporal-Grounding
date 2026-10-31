# Native SFT and time representation experiments

The runnable SFT implementation is based on the retained April 15 native project.
It uses TimeRFT's approximately 2,500 events and Qwen2.5-VL-7B-Instruct. It is not
an implementation of Time-R1's complete original RL pipeline.

## Training recipes

| Config | Target | LR | Warmup | Role |
|---|---|---:|---:|---|
| `time_r1/configs/abs.json` | absolute seconds | 2e-5 | 0 | matched native comparison |
| `time_r1/configs/rel.json` | relative percentages | 2e-5 | 0 | matched native comparison |
| `time_r1/configs/rel_low_lr.json` | relative percentages | 1e-5 | 0.03 | separate migrated low-LR setting |

Common settings: LoRA rank 64 / alpha 128 / dropout 0.05, language backbone frozen,
vision encoder frozen, merger trainable, BF16, five epochs, effective batch 16,
2 FPS, maximum 700 frames, pixel budget 100352–109760, ZeRO3.
The packaged default uses SDPA and disables optional Liger; these are recorded
adaptations rather than an exact historical environment claim.

Prepare data using [the data guide](../data/README.md), then run from the repo root:

```bash
python tools/train.py --config time_r1/configs/abs.json \
  --model models/Qwen2.5-VL-7B-Instruct --data data/prepared/train_abs.json \
  --output outputs/sft_abs
python tools/train.py --config time_r1/configs/rel.json \
  --model models/Qwen2.5-VL-7B-Instruct --data data/prepared/train_rel.json \
  --output outputs/sft_rel
```

Use a separate output directory and `rel_low_lr.json` for that additional experiment.
`bash time_r1/scripts/train_abs.sh ...` is an equivalent thin launcher.
The Windows host can inspect every command with `--dry-run`; training requires Linux/CUDA.

For a minimal GPU smoke run, use a few short videos and explicit overrides:

```bash
python tools/train.py --config time_r1/configs/abs.json \
  --model models/Qwen2.5-VL-7B-Instruct --data data/prepared/train_abs_smoke.json \
  --output outputs/sft_smoke --set max_steps=1 --set save_steps=1 \
  --set video_max_frames=16 --set max_seq_length=16384
```

These smoke overrides change the input budget and are not a reported benchmark recipe.
Use `--resume outputs/sft_abs/checkpoint-50` only for a checkpoint with optimizer state.
Neither old output directories nor checkpoints trigger automatic resume.

## Complete adapter loading

The trained merger is stored in `non_lora_state_dict.bin`, including intermediate
checkpoints. Loading only the LoRA adapter drops trained parameters. The shared loader
requires and maps these extra parameters against actual model keys.

```bash
python tools/merge_adapter.py --model models/Qwen2.5-VL-7B-Instruct \
  --adapter outputs/sft_abs/checkpoint-50 --output models/sft_abs_merged
```

Use `--adapter-only` only for a run that intentionally froze every non-LoRA parameter.
Adapter merging is also a GPU/dependency-dependent path requiring validation.

## Inference and evaluation

```bash
python tools/infer.py --workload sft --model models/Qwen2.5-VL-7B-Instruct \
  --adapter outputs/sft_abs/checkpoint-50 --representation absolute \
  --manifest data/prepared/charades_sta_1k.json --max-new-tokens 32 \
  --output outputs/sft_abs_predictions.jsonl
python tools/evaluate.py --predictions outputs/sft_abs_predictions.jsonl \
  --manifest data/prepared/charades_sta_1k.json --output outputs/sft_abs_metrics.json
```

For relative targets, select the relative adapter and `--representation relative`.
The parser converts percentages back to seconds before computing IoU. Evaluate all
checkpoints on the same saved query manifest; write each checkpoint to its own output.

Do not infer a mechanism solely from this experiment: changes in preprocessing,
input budget, learning rate or module freezing must be recorded separately.
