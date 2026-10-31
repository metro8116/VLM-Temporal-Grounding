"""Build training commands without importing CUDA/model dependencies."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if config["workload"] not in {"sft", "grpo"}:
        raise ValueError("workload must be sft or grpo")
    return config


def build_command(config, model, data, output, gpus=None, attention=None, resume=None, overrides=None):
    count = gpus if gpus is not None else config["num_processes"]
    if count < 1:
        raise ValueError("gpus must be positive")
    arguments = dict(config["trainer_args"])
    arguments.update(overrides or {})
    if attention is not None:
        arguments["disable_flash_attn2"] = attention == "sdpa"
    workload = config["workload"]
    module = ROOT / ("time_r1" if workload == "sft" else "timelens")
    entry = module / ("src/train/train_sft.py" if workload == "sft"
                      else "training/train/train_grpo_timelens.py")
    # Child processes run inside the experiment directory, so resolve an existing
    # local model against the caller's working directory before changing cwd.
    local_model = Path(model).expanduser()
    model_reference = str(local_model.resolve()) if local_model.exists() else model
    arguments["model_id" if workload == "sft" else "model_name_or_path"] = model_reference
    arguments["data_path" if workload == "sft" else "raw_anno_path"] = str(Path(data).resolve())
    arguments["output_dir"] = str(Path(output).resolve())
    arguments["resume_checkpoint"] = str(Path(resume).resolve()) if resume else None
    ds = arguments.get("deepspeed")
    if ds:
        arguments["deepspeed"] = str((ROOT / ds).resolve())
        if not Path(arguments["deepspeed"]).is_file():
            raise ValueError(f"Missing DeepSpeed config: {ds}")
    if "global_batch_size" in config:
        divisor = count * arguments["per_device_train_batch_size"]
        if config["global_batch_size"] % divisor:
            raise ValueError("global_batch_size must divide by GPU count × batch per GPU")
        arguments["gradient_accumulation_steps"] = config["global_batch_size"] // divisor
    if workload == "grpo":
        generation_batch = count * arguments["per_device_train_batch_size"] * arguments.get("steps_per_generation", 1)
        if generation_batch % arguments["num_generations"]:
            raise ValueError("Generation batch must be divisible by num_generations")
        if arguments["lora_enable"] and not arguments["freeze_llm"]:
            raise ValueError("LoRA training must freeze the language backbone")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           f"--nproc_per_node={count}", str(entry)]
    for key, value in arguments.items():
        if value is None:
            continue
        if isinstance(value, list):
            value = repr(value)
        cmd.extend([f"--{key}", str(value)])
    env = os.environ.copy()
    prefix = module / "src" if workload == "sft" else module
    env["PYTHONPATH"] = os.pathsep.join([str(prefix), str(ROOT), env.get("PYTHONPATH", "")])
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return cmd, env, module, arguments
