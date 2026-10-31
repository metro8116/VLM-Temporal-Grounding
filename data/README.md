# Data and model preparation

Models, videos and full external datasets are not included in Git. Put downloads in
`data/raw/`, prepared annotations in `data/prepared/`, videos in `data/videos/` and
weights in `models/` (all ignored). Use absolute paths for a training server.
The standard manifest schema is a list/JSONL of:

```json
{"id":"stable-query-id","video_path":"/data/videos/example.mp4","duration":20.0,"query":"open the door","span":[5.0,10.0]}
```

All canonical spans are seconds. `id` is generated from dataset, video basename,
query and GT; new inference carries it through unchanged. Invalid GT, duplicate ids,
missing files and zero-byte videos are errors. `--skip-video-check` is only for
annotation preparation without media, not an assertion that training can run.
`tools/check_data.py --decode` additionally decodes a frame per distinct video.

## Native SFT resources

- Base model: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- TimeRFT data/media: https://huggingface.co/datasets/Boshenxx/TimeR1-Dataset
- Time-R1 data instructions: https://github.com/xiaomi-research/Time-R1/blob/main/docs/DATA.md
- Original Charades-STA test annotations: `lmms-lab/charades_sta`, split `test`.
  Corresponding Charades videos must be obtained separately under their data terms.

The original `train_2k5.json` is a list with `video`, `duration`, `timestamp`, `sentence`.
For an existing local archive, this source file can be provided directly to the converter.

```bash
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct
python tools/prepare_data.py --format timerft --input /data/annotations/train_2k5.json \
  --video-root /data/timerft_data --dataset TimeRFT \
  --output data/prepared/timerft_abs_manifest.json \
  --sft-output data/prepared/train_abs.json --representation absolute
python tools/prepare_data.py --format timerft --input /data/annotations/train_2k5.json \
  --video-root /data/timerft_data --dataset TimeRFT \
  --output data/prepared/timerft_rel_manifest.json \
  --sft-output data/prepared/train_rel.json --representation relative
```

The converters keep event order unchanged by default, so the two training datasets
use the same events. Their conversation answers differ, not the underlying GT.

## TimeLens resources

- SFT initialization: https://huggingface.co/JungleGym/TimeLens-Qwen3-VL-8B-SFT
- Offline IoU predictions: https://huggingface.co/datasets/JungleGym/TimeLens-Qwen3-VL-8B-filter-data
- Training media/annotations: https://huggingface.co/datasets/TencentARC/TimeLens-100K
- Evaluation media/annotations: https://huggingface.co/datasets/TencentARC/TimeLens-Bench

Use the SFT initialization for both GRPO branches; the official final RL model is
an evaluation reference, not a substitute for this starting point.

```bash
hf download JungleGym/TimeLens-Qwen3-VL-8B-SFT --local-dir models/TimeLens-8B-SFT
hf download JungleGym/TimeLens-Qwen3-VL-8B-filter-data \
  FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl \
  --repo-type dataset --local-dir data/raw/filter_data
```

Download and extract the actual TimeLens videos according to the upstream instructions.
Training videos include source subdirectories. Rewrite old server paths using a unique
basename lookup, which rejects ambiguous media names:

```bash
python tools/prepare_data.py --format filtered \
  --input data/raw/filter_data/FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl \
  --video-root /data/TimeLens-100K/videos --path-mode lookup \
  --output data/prepared/timelens_filter.json
```

For already correct relative paths, use `--path-mode relative`; it preserves source
subdirectories and rejects traversal/old absolute paths. The default `basename`
mapping is appropriate only when all media are directly under `--video-root`.

To regenerate offline IoU annotations instead of downloading them:

```bash
python tools/prepare_data.py --format timelens-100k \
  --input /data/TimeLens-100K/timelens-100k.jsonl --exclude-audio \
  --video-root /data/TimeLens-100K/videos --path-mode relative \
  --output data/prepared/timelens_train_manifest.json
python tools/infer.py --workload timelens --model models/TimeLens-8B-SFT \
  --manifest data/prepared/timelens_train_manifest.json --round-unit 0 \
  --output outputs/filter_predictions.jsonl
python tools/filter_data.py --manifest data/prepared/timelens_train_manifest.json \
  --predictions outputs/filter_predictions.jsonl --output data/prepared/timelens_filter.json
```

This optional full-dataset inference is expensive. Failed offline predictions are
excluded from training annotations and recorded separately. Difficulty/duration
sampling remains in the TimeLens training dataset, not the converter.

## Prepare fixed evaluation sets

Original Charades-STA, for the SFT study:

```bash
python tools/prepare_charades.py --from-hub --video-root /data/Charades_v1_480 \
  --output data/prepared/charades_sta_1k.json --limit 1000
```

This selects the first 1,000 specified queries rather than silently replacing failed
ones with later queries. It therefore does not promise identical membership to an
old evaluation that skipped errors. A local JSON export can be supplied via `--input`.

TimeLens-Bench files are dictionaries keyed by video, with `duration`, `queries`, `spans`:

```bash
python tools/prepare_data.py --format timelens-bench \
  --input /data/TimeLens-Bench/qvhighlights-timelens.json \
  --video-root /data/TimeLens-Bench/videos/qvhighlights --dataset QVHighlights-TimeLens \
  --output data/prepared/qvhighlights_manifest.json
```

Repeat for ActivityNet and Charades. Use full manifests for full-benchmark evaluation.
To create a longest-video subset, explicitly add `--selection longest --limit 2000`;
do not describe that as random sampling. Save the manifest and its metadata per run.
