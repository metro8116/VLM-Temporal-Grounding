"""Convert offline grounding predictions into TimeLens filtering annotations."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import normalize_predictions, temporal_iou
from vtg.data import read_records, write_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    expected = read_records(args.manifest)
    rows = normalize_predictions(read_records(args.predictions))
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        p.error("Duplicate prediction ids")
    expected_ids = {row["id"] for row in expected}
    if set(by_id) - expected_ids:
        p.error("Prediction ids outside the manifest")
    kept, failed = [], []
    for sample in expected:
        pred = by_id.get(sample["id"])
        if pred is None or pred.get("error") or pred.get("prediction") is None:
            failed.append({"id": sample["id"], "error": pred.get("error") if pred else "missing prediction"})
            continue
        row = dict(sample)
        row.update(iou=temporal_iou(pred["prediction"], row["span"]), pred=[pred["prediction"]],
                   answer=pred.get("answer", ""), source=row.get("source", "TimeLens-100K"))
        kept.append(row)
    if not kept:
        p.error("No valid offline predictions to train from")
    write_records(args.output, kept)
    write_records(str(args.output) + ".failures.json", failed)
    print(f"Kept {len(kept)}; excluded and recorded {len(failed)} failed offline predictions")


if __name__ == "__main__":
    main()
