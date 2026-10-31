"""Temporal IoU with an explicit expected sample set and failure accounting."""
from __future__ import annotations

import ast
from collections import Counter
import math
import re

from vtg.data import first_span, sample_id


def temporal_iou(prediction, target):
    try:
        a, b = first_span(prediction), first_span(target)
    except (TypeError, ValueError):
        return 0.0
    if a[0] < 0 or a[0] >= a[1] or b[0] < 0 or b[0] >= b[1]:
        return 0.0
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    return intersection / union if union > 0 else 0.0


def parse_prediction(text, representation="absolute", duration=None):
    number = r"(-?\d+(?:\.\d+)?)"
    if representation == "relative":
        pattern = number + r"\s*%?\s*(?:to|-)\s*" + number + r"\s*%?"
    else:
        pattern = number + r"\s*(?:to|-)\s*" + number
    match = re.search(pattern, str(text), flags=re.IGNORECASE)
    if not match:
        return None
    start, end = map(float, match.groups())
    if not all(math.isfinite(v) for v in (start, end)) or start < 0 or start >= end:
        return None
    if representation == "relative":
        if duration is None or float(duration) <= 0 or end > 100:
            return None
        start, end = start / 100 * float(duration), end / 100 * float(duration)
    return [start, end]


def normalize_predictions(records):
    """Read canonical or historical TimeLens keyed predictions without eval()."""
    rows = []
    for record in records:
        if "id" in record:
            rows.append(record)
            continue
        if "video" in record and "query" in record:
            target = record.get("gt", record.get("gt_seconds"))
            prediction = record.get("pred", record.get("pred_seconds"))
            rows.append(dict(id=sample_id(record["video"], record["query"], target),
                             video_path=record["video"], query=record["query"], span=first_span(target),
                             prediction=prediction, answer=record.get("raw_output", "")))
            continue
        if len(record) != 1:
            raise ValueError("Historical prediction must contain exactly one sample key")
        key, value = next(iter(record.items()))
        video, query, target = key.rsplit(">>>", 2)
        span = first_span(ast.literal_eval(target))
        rows.append(dict(id=sample_id(video, query, span), video_path=video, query=query,
                         span=span, prediction=(value.get("timestamps") or [None])[0],
                         answer=value.get("answers", ""), error=value.get("error")))
    return rows


def score_records(records, expected=None):
    rows = normalize_predictions(records)
    by_id = {}
    for row in rows:
        if row["id"] in by_id:
            raise ValueError(f"Duplicate prediction id: {row['id']}")
        by_id[row["id"]] = row
    if expected is None:
        expected = rows
    expected_ids = [row["id"] for row in expected]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected manifest contains duplicate ids")
    extras = set(by_id) - set(expected_ids)
    if extras:
        raise ValueError(f"Predictions contain {len(extras)} ids outside expected manifest")
    errors, scores = Counter(), []
    for item in expected:
        row = by_id.get(item["id"])
        if row is None:
            errors["missing_prediction"] += 1
            scores.append(0.0)
        elif row.get("error"):
            errors[str(row.get("error_type", "inference_error"))] += 1
            scores.append(0.0)
        else:
            prediction = row.get("prediction")
            try:
                valid = first_span(prediction)
                if valid[0] < 0 or valid[0] >= valid[1]:
                    raise ValueError("Invalid prediction")
            except (TypeError, ValueError):
                errors["invalid_prediction"] += 1
                scores.append(0.0)
                continue
            scores.append(temporal_iou(prediction, item["span"]))
    n = len(scores)
    return {"expected": n, "received": len(rows), "failed": sum(errors.values()),
            "errors": dict(errors), "mIoU": sum(scores) / n * 100 if n else 0.0,
            "R@1": {str(t): sum(s >= t for s in scores) / n * 100 if n else 0.0
                    for t in (0.3, 0.5, 0.7)},
            "denominator": "all expected queries; errors and missing predictions count as zero",
            "manifest_supplied": expected is not rows}
