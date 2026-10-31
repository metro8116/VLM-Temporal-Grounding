"""Check manifest integrity and media; optionally decode frames with OpenCV."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtg.data import read_records, validate_sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--decode", action="store_true", help="Actually open one frame per distinct video; requires opencv-python")
    args = p.parse_args()
    rows = read_records(args.manifest)
    seen = set()
    if not rows:
        p.error("Empty manifest")
    for row in rows:
        validate_sample(row, check_video=True)
        if args.decode and row["video_path"] not in seen:
            import cv2
            capture = cv2.VideoCapture(row["video_path"])
            success, frame = capture.read()
            capture.release()
            if not success or frame is None:
                raise ValueError(f"Undecodable video: {row['video_path']}")
        seen.add(row["video_path"])
    print(f"Validated {len(rows)} queries and {len(seen)} nonempty videos; decoded={args.decode}")


if __name__ == "__main__":
    main()
