"""Prepare a fixed original Charades-STA test manifest from HF or a local export."""
import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtg.data import normalize_annotations, read_records, write_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="JSON/JSONL export with video, timestamp, caption, and optional duration")
    source.add_argument("--from-hub", action="store_true", help="Download lmms-lab/charades_sta test annotations")
    p.add_argument("--video-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--selection", choices=["original", "random", "longest"], default="original")
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.from_hub:
        from datasets import load_dataset
        raw = load_dataset("lmms-lab/charades_sta", split="test")
    else:
        raw = read_records(args.input)
    rows, durations = [], {}
    for item in raw:
        video = Path(args.video_root).resolve() / item["video"]
        if not video.is_file() or video.stat().st_size == 0:
            p.error(f"Missing/empty video: {video}; preparation never silently skips queries")
        duration = item.get("duration")
        if duration is None:
            if video not in durations:
                import cv2
                capture = cv2.VideoCapture(str(video))
                fps = capture.get(cv2.CAP_PROP_FPS)
                durations[video] = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
                capture.release()
            duration = durations[video]
        rows.append(dict(video_path=str(video), duration=duration, query=item["caption"], span=item["timestamp"]))
    rows = normalize_annotations(rows, "canonical", dataset="Charades-STA")
    if args.selection == "random":
        random.Random(args.seed).shuffle(rows)
    elif args.selection == "longest":
        rows.sort(key=lambda row: row["duration"], reverse=True)
    if args.limit > 0:
        rows = rows[:args.limit]
    write_records(args.output, rows)
    write_records(str(args.output) + ".meta.json", {"queries": len(rows), "selection": args.selection,
                  "seed": args.seed, "limit": args.limit, "dataset": "original Charades-STA"})
    print(f"Prepared {len(rows)} fixed queries")


if __name__ == "__main__":
    main()
