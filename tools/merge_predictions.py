"""Merge explicit shard paths; reject duplicates and keep every error record."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import score_records
from vtg.data import read_records, write_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        p.error("Output exists; choose a new file")
    rows = [row for path in args.inputs for row in read_records(path)]
    expected = read_records(args.manifest)
    report = score_records(rows, expected)
    by_id = {row["id"]: row for row in rows}
    ordered = [by_id.get(row["id"], {"id": row["id"], "span": row["span"],
               "prediction": None, "error": "shard did not produce this query",
               "error_type": "missing_prediction"}) for row in expected]
    write_records(args.output, ordered)
    write_records(str(args.output) + ".metrics.json", report)


if __name__ == "__main__":
    main()
