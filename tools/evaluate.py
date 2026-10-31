"""Compute metrics from saved predictions and an optional expected manifest."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import score_records
from vtg.data import read_records, write_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--manifest", help="Required to account for predictions missing from the output file")
    p.add_argument("--output")
    args = p.parse_args()
    report = score_records(read_records(args.predictions),
                           read_records(args.manifest) if args.manifest else None)
    if not args.manifest:
        report["limitation"] = "No expected manifest: failures omitted by historical runs cannot be recovered"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        write_records(args.output, report)


if __name__ == "__main__":
    main()
