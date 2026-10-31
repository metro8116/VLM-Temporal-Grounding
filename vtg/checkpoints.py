"""Load trained merger weights together with LoRA adapters, with strict key checks."""
from __future__ import annotations

from pathlib import Path

EXTRA_NAMES = ("non_lora_state_dict.bin", "non_lora_trainables.bin")


def remap_extra_keys(state, target_keys):
    result = {}
    for key, tensor in state.items():
        candidates = [key]
        for prefix in ("base_model.model.", "base_model."):
            if key.startswith(prefix):
                candidates.append(key[len(prefix):])
        for candidate in list(candidates):
            if candidate.startswith("model.model."):
                candidates.append(candidate[len("model."):])
            if candidate.startswith("visual."):
                candidates.append("model." + candidate)
            if candidate.startswith("model.visual."):
                candidates.append(candidate[len("model."):])
        matches = [candidate for candidate in candidates if candidate in target_keys]
        if not matches:
            raise ValueError(f"Extra trainable parameter has no matching base-model key: {key}")
        if matches[0] in result:
            raise ValueError(f"Duplicate remapped parameter: {matches[0]}")
        result[matches[0]] = tensor
    return result


def load_extra_trainables(model, directory, required=True):
    import torch
    files = [Path(directory) / name for name in EXTRA_NAMES if (Path(directory) / name).is_file()]
    if len(files) > 1:
        raise ValueError("Ambiguous extra weight files; keep one explicitly selected version")
    if not files:
        if required:
            raise FileNotFoundError("Adapter is missing its trained merger/non-LoRA weights; "
                                    "use --adapter-only only for an intentionally pure LoRA run")
        return 0
    state = torch.load(files[0], map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Expected a weight state dictionary")
    mapped = remap_extra_keys(state, set(model.state_dict()))
    if required and not mapped:
        raise ValueError("Extra trainable weights are empty despite a trainable merger policy")
    result = model.load_state_dict(mapped, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected extra parameters: {result.unexpected_keys}")
    return len(mapped)


def load_model(model_path, adapter=None, attention="sdpa", dtype="bfloat16", adapter_only=False):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if adapter is None and (Path(model_path) / "adapter_config.json").is_file():
        raise ValueError("Adapter directory supplied as a full model; pass its base with --model and --adapter")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=getattr(torch, dtype), attn_implementation=attention, device_map="auto")
    processor_source = model_path
    if adapter:
        from peft import PeftModel
        loaded = load_extra_trainables(model, adapter, required=not adapter_only)
        print(f"Loaded {loaded} extra trained parameters")
        model = PeftModel.from_pretrained(model, adapter)
        if any((Path(adapter) / name).exists() for name in ["processor_config.json", "preprocessor_config.json"]):
            processor_source = adapter
    processor = AutoProcessor.from_pretrained(processor_source, padding_side="left")
    return model.eval(), processor
