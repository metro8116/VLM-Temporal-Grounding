# Source provenance

## Native SFT

Local source: the retained `2026-04-15_框架转换+相对百分比SFT-LoRA/Time-R1(SFT2)`
project. Its shell headers identify Qwen-VL-Series-Finetune as the implementation
base. This repository retains the SFT dependency closure, removes unrelated
classification/DPO/GRPO entry points, and organizes TimeRFT-based timestamp experiments.

The Time-R1 upstream recipe/data references are credited separately; this repository
does not claim to author Qwen, the upstream training framework or TimeRFT data.

## TimeLens

Local source: `TimeLens-train/TimeLens-train`. The custom GRPO trainer, reward parser,
data processing and required helper modules are retained with their original headers.
The pure tIoU experiment is the supported GRPO scope.

## Exact file records

`source_manifest.json` records each copied original file and its source SHA-256.
Paths are relative to the original research workspace, not hard-coded runtime paths.
The original directories were not edited. Upstream Git revisions were not present
in the backup; local hashes establish the precise source snapshots used here.

`results/historical/manifest.json` separately records the original prediction files.
The compressed archives decompress to byte-identical original content.

The Qwen-VL-Series-Finetune Apache-2.0 license was fetched from the upstream master
branch during packaging; TimeLens LICENSE/NOTICE and Time-R1 LICENSE were copied
from their local distributions.

See [CHANGES.md](CHANGES.md) for modifications applied to the new copies.
