# TimeLens inference entry

Use `python tools/infer.py --workload timelens` from the repository root, or
`bash timelens/scripts/infer.sh ...`. Model-specific preprocessing is reused from
`timelens/training/data/inference_collator.py`. The shared entry additionally handles
complete adapter loading, fixed manifests, errors, sharding and evaluation metadata.
