"""Launch native SFT or TimeLens GRPO from a versioned JSON configuration."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtg.data import read_records, validate_sample, write_records
from vtg.launch import build_command, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True, help="Same base/SFT model for training and adapter reload")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpus", type=int)
    p.add_argument("--attention", choices=["sdpa", "flash_attention_2"])
    p.add_argument("--resume", help="Explicit resumable checkpoint; existing outputs never resume automatically")
    p.add_argument("--set", action="append", default=[], metavar="ARG=JSON_VALUE")
    p.add_argument("--dry-run", action="store_true", help="Print command only; does not create outputs or load models")
    args = p.parse_args()
    config = load_config(args.config)
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    cmd, env, cwd, effective = build_command(config, args.model, args.data, args.output,
                                            args.gpus, args.attention, args.resume, overrides)
    print(shlex.join(cmd))
    print(f"Working directory: {cwd}")
    if args.dry_run:
        return
    if sys.platform == "win32":
        p.error("Training requires Linux with CUDA; use --dry-run on Windows")
    rows = read_records(args.data)
    if not isinstance(rows, list) or not rows:
        p.error("Training data must be a non-empty list/JSONL")
    for row in rows:
        if config["workload"] == "sft":
            video = Path(row["video"])
            if not video.is_file() or video.stat().st_size == 0:
                p.error(f"Missing/empty video: {video}")
            if len(row.get("conversations", [])) != 2:
                p.error("SFT requires one human turn and one assistant target")
        else:
            validate_sample(row, check_video=True)
            if not isinstance(row.get("iou"), (int, float)) or not 0 <= row["iou"] <= 1:
                p.error("Filtered GRPO data must have a finite IoU in [0,1]")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        p.error("Output is nonempty; choose a new run directory or explicitly --resume")
    if args.resume:
        checkpoint = Path(args.resume)
        if not checkpoint.is_dir() or not (checkpoint / "trainer_state.json").is_file():
            p.error("--resume must point to a Trainer checkpoint, not final model-only weights")
        if not list(checkpoint.glob("*optim*")) and not list(checkpoint.glob("global_step*")):
            p.error("Checkpoint has no optimizer state; start a new run from model weights instead")
    output.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ["torch", "transformers", "accelerate", "peft", "trl", "deepspeed", "qwen-vl-utils"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    write_records(output / "run_config.json", {"recipe": config, "effective_arguments": effective,
                  "command": cmd, "python": sys.version, "versions": versions,
                  "resume": args.resume, "training_queries": len(rows)})
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    main()
