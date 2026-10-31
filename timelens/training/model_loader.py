"""Resolve generation classes from saved model configurations."""
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


def get_model_class(model_path):
    return AutoModelForImageTextToText


def get_config_class(model_path):
    return AutoConfig


def get_processor_class(model_path):
    return AutoProcessor
