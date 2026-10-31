# Validation status

The local packaging environment is Windows. The original model weights, complete
training/benchmark videos and original server environment exports are unavailable.
Training and inference are intended for a separately prepared Linux CUDA host.

## Checks available locally

```bash
python -m unittest discover -s tests -v
python tools/recompute_history.py
python tools/check_repository.py
```

The tests exercise temporal interval metrics, relative coordinate conversion, invalid
predictions, missing/error denominators, duplicate ids, safe legacy parsing, annotation
validation, video-path mapping, extra-weight key mapping, training command construction
and CLI preparation/inference dry runs. They do not substitute for model execution.

Historical reconstruction verifies all 17 prediction hashes and recomputes mIoU/recall.
The repository checker parses Python/JSON files and validates documented local links.
The source manifest permits checking that originals retain their recorded hashes.

## Required GPU validation before claiming an executable training reproduction

1. Obtain the correct base/SFT models and actual decodable videos; validate query manifests.
2. Install one candidate environment and run `pip check` and `tools/doctor.py --cuda`.
3. Run one short-video SFT update, save, reload adapter + merger, merge, and compare
   direct-adapter versus merged-model inference on the same sample.
4. Do the same for single-GPU QLoRA; inspect trainable merger parameters and gradients.
5. Run a multi-GPU full-GRPO smoke update and verify that model/optimizer saves finish
   on all ranks and that an explicit resumed run restores its step/state.
6. Run inference for fixed manifests on all benchmark datasets, preserving every failure.
7. Only then run the complete experiment recipes and publish newly measured scores.

Full training costs are not incurred by the CPU checks or `--dry-run`. Historic resource
classes were one 80/96 GB GPU and eight 96 GB GPUs; no 8 GB training claim is made.
