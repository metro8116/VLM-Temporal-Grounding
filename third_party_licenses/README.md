# Attribution and licenses

- **TimeLens:** https://github.com/TencentARC/TimeLens . The retained local `train`
  code snapshot supplies the custom trainer, data pipeline, parser and related helpers.
  The complete local LICENSE and NOTICE are preserved. Its license includes academic
  use and regional limitations; source redistribution must retain its conditions.
- **Qwen-VL-Series-Finetune:** https://github.com/2U1/Qwen-VL-Series-Finetune . The native
  SFT backup identifies this codebase in its training scripts. The Apache-2.0 LICENSE
  was retrieved from the upstream `master` branch while organizing this repository.
- **Time-R1:** https://github.com/xiaomi-research/Time-R1 . The local Time-R1 distribution supplies
  the Apache-2.0 license and references for the TimeRFT recipe/data. The main SFT code
  in this repository is the retained native adaptation, not a claimed new base model.
- The retained TimeLens trainer includes TRL-derived implementation and original
  imports/attribution. Qwen/Transformers model implementations are external dependencies.

Local snapshots did not contain upstream Git revisions. `docs/source_manifest.json`
records exact local source paths and SHA-256 hashes, and the modified copies remain
separate from the originals. Model weights and training videos are not redistributed.
