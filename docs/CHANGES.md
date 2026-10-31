# Packaging and implementation changes

## Native SFT

- Retained only the native SFT modules and their actual dependencies; removed unused
  DPO/GRPO/classification imports from package initialization and argument classes.
- Exposed the video frame limit as `video_max_frames` instead of hard-coding 700.
- Reject sequences exceeding `max_seq_length` rather than silently slicing visual tokens.
- Omit Qwen3-only `mm_token_type_ids` when collating the Qwen2.5-only experiment.
- Replace automatic output-directory resume with explicit `resume_checkpoint`.
- Restore saved non-LoRA trainables when resuming an adapter run.
- Gather/clone ZeRO-partitioned weights before leaving the gather context.

## TimeLens

- Resolve model classes through config rather than requiring particular directory names.
- Support locating the visual tower at the current Qwen nested model path; fail when
  the requested freeze policy cannot be established.
- Restore trainable merger parameters after quantization and PEFT wrapping, then
  validate/record the actual trainable names after constructing the trainer.
- Record the actual duration/IoU-selected training annotations per run.
- Save non-LoRA merger parameters at each adapter checkpoint as well as final save.
- Execute distributed ZeRO gathers on all ranks; restrict file writing to the saving rank.
- Remove manual cleanup that could delete the newest checkpoint; use Trainer rotation.
- Keep resume explicit and preserve optimizer state in the provided recipes.
- Reject empty training selections.

## Shared utilities

- Explicit JSON configs and a torchrun command builder with batch/generation checks.
- Canonical annotation manifests, controlled path remapping and zero-byte media checks.
- Complete adapter + extra-weight loading, strict weight-key matching and merge CLI.
- Model-specific video preprocessing with explicit FPS/budgets and rounding protocol.
- Fixed-id inference with error records, duplicate-aware shard merging and expected-set metrics.
- Safe historical parsing via `ast.literal_eval`, gzip hash verification and metric reconstruction.

These edits are covered by CPU tests where applicable. They have not established
runtime compatibility or quality of the GPU model paths; see VALIDATION.md.

## Complete README workflows

- Reorganized both READMEs with centered titles, navigation, model/data locations and complete training/evaluation commands.
- Added direct download/extraction instructions for the verified model and dataset resources.
- Documented separate SFT-LoRA, GRPO-QLoRA and Full-GRPO prediction and metric entry points.
- Resolve an existing local model path before the launcher changes its working directory; preserve Hugging Face model IDs.
- Validated CLI arguments, training configs, Bash syntax and bilingual command consistency without executing GPU workloads.
