"""Merge an adapter and its extra trainables into the exact training base model."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtg.checkpoints import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Qwen2.5-VL base or TimeLens SFT starting model")
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--adapter-only", action="store_true", help="Only if no merger parameters were trained")
    p.add_argument("--attention", choices=["sdpa", "flash_attention_2"], default="sdpa")
    args = p.parse_args()
    target = Path(args.output)
    if target.exists() and any(target.iterdir()):
        p.error("Use an empty output directory")
    model, processor = load_model(args.model, args.adapter, args.attention, adapter_only=args.adapter_only)
    model = model.merge_and_unload()
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    processor.save_pretrained(target)
    (target / "merge_provenance.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
