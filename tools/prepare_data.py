"""Convert and validate annotations without importing model libraries."""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtg.data import (as_sft, normalize_annotations, read_records, validate_sample, write_records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--format", choices=["timerft", "timelens-100k", "timelens-bench", "filtered", "canonical"], required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--path-mode", choices=["basename", "relative", "lookup"], default="basename")
    p.add_argument("--dataset", default="")
    p.add_argument("--output", required=True, help="Canonical manifest; all times in seconds")
    p.add_argument("--sft-output", help="Optional native SFT conversation JSON")
    p.add_argument("--representation", choices=["absolute", "relative"], default="absolute")
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--selection", choices=["original", "random", "longest"], default="original")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-audio", action="store_true", help="TimeLens gemini_refined_data query filtering")
    p.add_argument("--skip-video-check", action="store_true", help="Annotation-only preparation; media still required for execution")
    args = p.parse_args()
    rows = normalize_annotations(read_records(args.input), args.format, args.video_root,
                                 args.dataset, args.path_mode)
    if args.exclude_audio:
        words = {"hear", "heard", "hears", "hearing", "sound", "sounded", "sounds", "sounding", "audio"}
        rows = [row for row in rows if not words.intersection(row["query"].strip("?").lower().split())]
    if args.selection == "random":
        random.Random(args.seed).shuffle(rows)
    elif args.selection == "longest":
        rows.sort(key=lambda r: r["duration"], reverse=True)
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("Empty annotation selection")
    for row in rows:
        validate_sample(row, check_video=not args.skip_video_check)
        if args.format == "filtered":
            if not isinstance(row.get("iou"), (int, float)) or not 0 <= row["iou"] <= 1:
                raise ValueError(f"Invalid or absent offline IoU for {row['id']}")
            row.setdefault("source", args.dataset or "TimeLens-100K")
            row["span"] = [row["span"]]
    write_records(args.output, rows)
    if args.sft_output:
        if args.format == "filtered":
            raise ValueError("Use TimeRFT/canonical annotations for the SFT study")
        write_records(args.sft_output, [as_sft(row, args.representation) for row in rows])
    write_records(str(args.output) + ".meta.json", {
        "count": len(rows), "format": args.format, "selection": args.selection,
        "limit": args.limit, "seed": args.seed, "media_checked": not args.skip_video_check,
        "representation": args.representation if args.sft_output else "seconds"})
    print(json.dumps({"queries": len(rows), "manifest": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
