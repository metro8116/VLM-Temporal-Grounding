# Results and evaluation protocol

## Historical prediction reconstruction

`results/historical/manifest.json` indexes 17 compressed prediction files. Each entry
records the original local relative path, uncompressed SHA-256, query count and model.
`metrics.csv` stores recalculated mIoU and R@1 percentages.

```bash
python tools/recompute_history.py
```

This verifies decompressed hashes and recomputes all metrics from the stored predictions;
it does not load a model, rerun inference or repeat training.

### TimeLens

| Model | ActivityNet N=2000 | Charades N=2000 | QVHighlights N=1541 |
|---|---:|---:|---:|
| Qwen3-VL-8B base | 43.28 | 48.14 | 59.48 |
| TimeLens-8B-SFT start | 46.06 | 51.19 | 57.18 |
| GRPO-QLoRA | 46.06 | 50.99 | 57.24 |
| Full GRPO | 50.51 | 54.59 | 65.42 |
| Official final model, local evaluation | 50.12 | 54.91 | 65.46 |

The five models have exactly matching query keys within each of these dataset groups.
The official row comes from saved local predictions, not a paper table. QVHighlights
uses the full benchmark; the ActivityNet/Charades rows are common 2,000-query subsets.
Some old evaluation entry points selected longest videos, so these are not labeled
random subsets. The exact records, not just the counts, define this comparison.

QLoRA/full configurations also change generation count, visual budget and batch.
The observed difference does not independently establish a universal superiority of
full-parameter training. In both cases the configured visual encoder remains frozen.

### Native SFT records

| Preserved record | N | Recomputed mIoU | Provenance |
|---|---:|---:|---|
| Absolute seconds backup | 1000 | 52.9521 | `eval_results_abs_ckp100.json` |
| Relative, low LR, checkpoint-600 | 1000 | 41.3680 | retained April 15 prediction JSON |

The absolute record's filename/loading script identifies checkpoint-100, whereas
the textual report heading says checkpoint-785. Checkpoint identity remains unresolved.
The low-LR group changes more than the timestamp representation. These records are
therefore not presented as a controlled absolute/relative pair.

The earlier summary's 54.23%/44.60% belongs to a different historical training path.
It is not assigned to a newly packaged native run. The new matched `abs.json` and
`rel.json` recipes have no newly measured results yet.

## New evaluation protocol

1. Prepare an explicit query manifest once, including id, GT span, query, duration and video path.
2. Use that same manifest across compared checkpoints/models. Record selection order,
   subset/full status, FPS, frames/pixels/tokens, prompt, rounding and package versions.
3. Predict one interval per query. For multiple returned intervals, use the first parsed interval.
4. Convert relative outputs back to seconds. Do not silently clamp predicted spans or
   truncate videos without recording a changed setting.
5. New inference writes a record even when preprocessing/generation fails. Missing,
   invalid and failed predictions contribute zero; the denominator is the expected manifest.
6. Compute mIoU and R@1 at 0.3/0.5/0.7. The CLI returns percentages.

For historical files evaluated without an expected manifest, the denominator is only
the saved records. Failures that the old scripts silently omitted cannot be recovered
from those files. The evaluation report explicitly states this limitation.

## New runs

Store newly generated models/logs/predictions under ignored `outputs/`. Publish reviewed
summaries separately from historical records. Keep the training run config, actual
sample selection, expected query ids, model/adapter identity and software environment.
