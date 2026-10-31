"""Check syntax, configs, local documentation links and optional original-file hashes."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-root", help="Optional original research directory for hash comparison")
    args = p.parse_args()
    python_files = [path for path in ROOT.rglob("*.py") if ".git" not in path.parts]
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for path in ROOT.rglob("*.json"):
        if not any(part in {"outputs", ".git"} for part in path.parts):
            json.loads(path.read_text(encoding="utf-8-sig"))
    links = 0
    for path in ROOT.rglob("*.md"):
        if any(part in {".git", "third_party_licenses"} for part in path.parts):
            continue
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in link or link.startswith("#"):
                continue
            target = link.split("#", 1)[0]
            if target and not (path.parent / target).exists():
                raise ValueError(f"Broken documentation link: {path.relative_to(ROOT)} -> {link}")
            links += 1
    if args.original_root:
        original = Path(args.original_root)
        source = json.loads((ROOT / "docs/source_manifest.json").read_text(encoding="utf-8"))
        for record in source:
            path = original / record["source"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["source_sha256"]:
                raise ValueError(f"Original changed: {record['source']}")
        print(f"Original file hashes unchanged: {len(source)}")
    print(f"Python syntax OK: {len(python_files)}; local documentation links OK: {links}")


if __name__ == "__main__":
    main()
