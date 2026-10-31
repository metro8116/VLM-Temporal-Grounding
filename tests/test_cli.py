import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CliIntegrationTests(unittest.TestCase):
    def run_tool(self, name, *args):
        return subprocess.run([sys.executable, str(ROOT / "tools" / name), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)

    def test_prepare_then_dry_run_without_any_ml_dependencies(self):
        with tempfile.TemporaryDirectory(prefix="vtg space ") as temp:
            root = Path(temp)
            raw = root / "source.json"
            raw.write_text(json.dumps([{"video": "/old/a.mp4", "duration": 20,
                                       "timestamp": [5, 10], "sentence": "open door"}]), encoding="utf-8")
            manifest, sft = root / "manifest.json", root / "train.json"
            self.run_tool("prepare_data.py", "--input", raw, "--format", "timerft",
                          "--video-root", root, "--output", manifest, "--sft-output", sft,
                          "--skip-video-check")
            result = self.run_tool("train.py", "--config", ROOT / "time_r1/configs/abs.json",
                                   "--model", "missing model", "--data", sft,
                                   "--output", root / "output", "--dry-run")
            self.assertIn("torch.distributed.run", result.stdout)
            self.assertFalse((root / "output").exists())
            self.run_tool("infer.py", "--workload", "sft", "--model", "missing model",
                          "--manifest", manifest, "--output", root / "pred.jsonl", "--dry-run")
            self.assertFalse((root / "pred.jsonl").exists())

    def test_merge_preserves_missing_samples_in_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([{"id": "a", "span": [0, 2]}, {"id": "b", "span": [0, 2]}]))
            shard = root / "shard.jsonl"
            shard.write_text(json.dumps({"id": "a", "prediction": [0, 2]}) + "\n")
            merged = root / "merged.jsonl"
            self.run_tool("merge_predictions.py", "--inputs", shard, "--manifest", manifest, "--output", merged)
            report = root / "report.json"
            self.run_tool("evaluate.py", "--predictions", merged, "--manifest", manifest, "--output", report)
            self.assertEqual(json.loads(report.read_text())["mIoU"], 50)


if __name__ == "__main__":
    unittest.main()
