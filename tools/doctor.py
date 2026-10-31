"""Report optional training dependencies and CUDA without installing anything."""
import argparse
import importlib.metadata
import json
import platform
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuda", action="store_true", help="Also import torch and query CUDA devices")
    args = p.parse_args()
    result = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for name in ["torch", "torchvision", "transformers", "accelerate", "peft", "trl", "deepspeed", "bitsandbytes", "qwen-vl-utils", "decord", "nncore"]:
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    if args.cuda:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_version"] = torch.version.cuda
        result["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
