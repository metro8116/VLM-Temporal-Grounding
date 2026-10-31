"""Verify compressed historical prediction hashes and recompute all bundled metrics."""
import argparse
import csv
import gzip
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.metrics import score_records
from vtg.data import read_records, write_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", help="Optional JSON report, otherwise print only")
    args = p.parse_args()
    manifest = read_records(ROOT / "results/historical/manifest.json")
    with (ROOT / "results/historical/metrics.csv").open(encoding="utf-8") as handle:
        expected = {(row["model"], row["dataset"]): row for row in csv.DictReader(handle)}
    reports = []
    for entry in manifest:
        path = ROOT / entry["file"]
        with gzip.open(path, "rb") as handle:
            checksum = hashlib.sha256(handle.read()).hexdigest()
        if checksum != entry["source_sha256"]:
            raise ValueError(f"Historical prediction content changed: {path}")
        report = score_records(read_records(path))
        target = expected[(entry["model"], entry["dataset"])]
        if report["expected"] != int(target["queries"]) or abs(report["mIoU"] - float(target["mIoU"])) > 1e-9:
            raise ValueError(f"Metric mismatch for {path}")
        for threshold in ("0.3", "0.5", "0.7"):
            if abs(report["R@1"][threshold] - float(target[f"R@1_{threshold}"])) > 1e-9:
                raise ValueError(f"Recall mismatch for {path}")
        reports.append(dict(model=entry["model"], dataset=entry["dataset"], **report))
        print(f"{entry['model']:24} {entry['dataset']:16} N={report['expected']:4} mIoU={report['mIoU']:.4f}")
    print(f"Verified {len(reports)} historical prediction files (not a model inference rerun)")
    if args.output:
        write_records(args.output, reports)


if __name__ == "__main__":
    main()
