"""Explicit, dependency-free annotation conversion and validation."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from functools import lru_cache

ABS_PROMPT = ('To accurately pinpoint the event "{query}" in the video, determine the precise '
              'time period of the event. Provide the start and end times (in seconds, precise '
              'to one decimal places) in the format "start time to end time". '
              'For example: "12.5 to 17.8".')
REL_PROMPT = ('To accurately pinpoint the event "{query}" in the video, determine the precise '
              'relative time period of the event. Represent this segment as a percentage of '
              'the total video duration (from 0.0% to 100.0%, precise to one decimal place). '
              'You MUST output strictly in the format "start percentage to end percentage". '
              'For example: "13.4% to 48.2%".')
TL_PROMPT = ("Please find the visual event described by the sentence '{query}', determining its "
             "starting and ending times. The format should be: "
             "'The event happens in <start time> - <end time> seconds'.")


def read_records(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        if ".jsonl" in path.name:
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def write_records(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        if path.suffix == ".jsonl":
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def first_span(value):
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Expected [start, end], got {value!r}")
    if any(isinstance(x, bool) for x in value):
        raise ValueError("Boolean timestamps are invalid")
    span = [float(x) for x in value]
    if not all(math.isfinite(x) for x in span):
        raise ValueError("Timestamps must be finite")
    return span


def sample_id(video_path, query, span, dataset=""):
    name = PurePosixPath(str(video_path).replace("\\", "/")).name
    payload = json.dumps([dataset, name, query, first_span(span)], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def validate_sample(row, check_video=False):
    duration = float(row["duration"])
    start, end = first_span(row["span"])
    if not math.isfinite(duration) or duration <= 0 or not 0 <= start < end <= duration:
        raise ValueError(f"Invalid duration/span: duration={duration}, span={row['span']}")
    if not isinstance(row["query"], str) or not row["query"].strip():
        raise ValueError("Empty query")
    if not isinstance(row["video_path"], str) or not row["video_path"]:
        raise ValueError("Missing video_path")
    if check_video:
        video = Path(row["video_path"])
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError(f"Missing or empty video: {video}")
    return row


@lru_cache(maxsize=8)
def _video_index(root):
    index = {}
    for path in Path(root).rglob("*"):
        if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov"}:
            index.setdefault(path.name, []).append(path)
    return index


def resolve_video(value, video_root, mode="basename"):
    """Never keep an old server prefix; the mapping policy is explicit."""
    value = str(value).replace("\\", "/")
    root = Path(video_root).resolve()
    if mode == "lookup":
        matches = _video_index(str(root)).get(PurePosixPath(value).name, [])
        if len(matches) != 1:
            raise ValueError(f"Expected one media match for {value}, found {len(matches)}")
        return str(matches[0].resolve())
    if mode == "basename":
        rel = PurePosixPath(value).name
    else:
        if PurePosixPath(value).is_absolute() or ":" in value or ".." in PurePosixPath(value).parts:
            raise ValueError(f"Relative mapping requires a safe relative path: {value}")
        rel = value.removeprefix("./")
    target = (root / rel).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Mapped video escapes video root")
    return str(target)


def normalize_annotations(raw, kind, video_root=None, dataset="", path_mode="basename"):
    rows = []
    if kind == "timelens-bench":
        for video, item in raw.items():
            if len(item["spans"]) != len(item["queries"]):
                raise ValueError(f"Mismatched query/span counts for {video}")
            for span, query in zip(item["spans"], item["queries"]):
                rows.append(dict(video_path=video + ".mp4", duration=item["duration"],
                                 query=" ".join(query.split()).strip(". "), span=first_span(span)))
    else:
        for item in raw:
            if kind == "timelens-100k":
                for event in item["events"]:
                    rows.append(dict(video_path=item["video_path"], duration=item["duration"],
                                     query=" ".join(event["query"].split()).strip(". "),
                                     span=first_span(event["span"]), source=item["source"]))
                continue
            if kind == "timerft":
                row = dict(video_path=item["video"], duration=item["duration"],
                           query=item["sentence"], span=first_span(item["timestamp"]))
            elif kind in {"canonical", "filtered"}:
                row = dict(item)
                row["span"] = first_span(row["span"])
            else:
                raise ValueError(f"Unsupported annotation format: {kind}")
            rows.append(row)
    seen = set()
    for row in rows:
        if video_root is not None:
            row["video_path"] = resolve_video(row["video_path"], video_root, path_mode)
        row["duration"] = float(row["duration"])
        row.setdefault("dataset", dataset)
        row.setdefault("id", sample_id(row["video_path"], row["query"], row["span"], row["dataset"]))
        validate_sample(row)
        if row["id"] in seen:
            raise ValueError(f"Duplicate sample id: {row['id']}")
        seen.add(row["id"])
    return rows


def as_sft(row, representation):
    start, end = first_span(row["span"])
    if representation == "relative":
        start, end = start / row["duration"] * 100, end / row["duration"] * 100
        prompt, answer = REL_PROMPT, f"{start:.1f}% to {end:.1f}%"
    elif representation == "absolute":
        prompt, answer = ABS_PROMPT, f"{start:.1f} to {end:.1f}"
    else:
        raise ValueError("representation must be absolute or relative")
    return {"video": row["video_path"], "conversations": [
        {"from": "human", "value": "<video>\n" + prompt.format(query=row["query"])},
        {"from": "gpt", "value": answer}]}
