"""
Minimal load_model.py for Qwen2.5-VL SFT only.
No monkey patches - use original model forward.
"""
from __future__ import annotations
from typing import Any

from transformers import AutoConfig, AutoModelForImageTextToText, PretrainedConfig

_GENERATION_MODEL_TYPES = {"qwen2_5_vl"}


def get_qwen_vl_generation_backbone(model):
    if not hasattr(model, "model"):
        raise TypeError(f"Unsupported generation model wrapper: {type(model)!r}")
    return model.model


def load_qwen_vl_generation_model(
    model_name_or_path: str, *, config: PretrainedConfig | None = None, **kwargs: Any
):
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path)
    if config.model_type not in _GENERATION_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type: {config.model_type}. Supported: {list(_GENERATION_MODEL_TYPES)}")
    return AutoModelForImageTextToText.from_pretrained(
        model_name_or_path, config=config, **kwargs,
    )
