from pathlib import Path
import tempfile
import unittest

from vtg.checkpoints import remap_extra_keys
from vtg.data import as_sft, normalize_annotations, read_records, resolve_video, sample_id, validate_sample


class AnnotationTests(unittest.TestCase):
    def sample(self):
        return {"video_path": "a.mp4", "duration": 20, "query": "open door", "span": [5, 10]}

    def test_same_events_get_different_sft_answers(self):
        row = self.sample()
        absolute = as_sft(row, "absolute")
        relative = as_sft(row, "relative")
        self.assertEqual(absolute["conversations"][1]["value"], "5.0 to 10.0")
        self.assertEqual(relative["conversations"][1]["value"], "25.0% to 50.0%")
        self.assertEqual(absolute["video"], relative["video"])

    def test_invalid_annotations_are_rejected(self):
        for span in ([5, 5], [-1, 4], [3, 21]):
            with self.assertRaises(ValueError):
                validate_sample(dict(self.sample(), span=span))

    def test_timelens_query_alignment(self):
        raw = {"a": {"duration": 20, "queries": ["open door"], "spans": [[5, 10]]}}
        rows = normalize_annotations(raw, "timelens-bench", dataset="test")
        self.assertEqual(rows[0]["span"], [5, 10])
        raw["a"]["queries"].append("close door")
        with self.assertRaises(ValueError):
            normalize_annotations(raw, "timelens-bench")

    def test_zero_byte_video_is_not_a_valid_asset(self):
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "a.mp4"
            video.touch()
            with self.assertRaises(ValueError):
                validate_sample(dict(self.sample(), video_path=str(video)), check_video=True)

    def test_path_mapping_and_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(Path(resolve_video("/old/server/a.mp4", temp)).name, "a.mp4")
            with self.assertRaises(ValueError):
                resolve_video("../a.mp4", temp, mode="relative")

    def test_ids_are_stable_across_server_paths(self):
        self.assertEqual(sample_id("/server/a.mp4", "open", [0, 2]),
                         sample_id("C:\\videos\\a.mp4", "open", [0, 2]))

    def test_truncated_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "broken.jsonl"
            path.write_text('{"video":', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_records(path)

    def test_extra_weight_key_mapping_is_strict(self):
        result = remap_extra_keys({"base_model.model.model.visual.merger.weight": "tensor"},
                                 {"model.visual.merger.weight"})
        self.assertEqual(result, {"model.visual.merger.weight": "tensor"})
        with self.assertRaises(ValueError):
            remap_extra_keys({"unknown.weight": "tensor"}, {"model.visual.merger.weight"})

    def test_nested_media_lookup_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "source").mkdir()
            video = root / "source/a.mp4"
            video.write_bytes(b"fixture")
            self.assertEqual(Path(resolve_video("/old/a.mp4", root, "lookup")), video.resolve())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "one").mkdir()
            (root / "two").mkdir()
            (root / "one/a.mp4").touch()
            (root / "two/a.mp4").touch()
            with self.assertRaises(ValueError):
                resolve_video("a.mp4", root, "lookup")

    def test_timelens_100k_event_flattening(self):
        raw = [{"video_path": "source/a.mp4", "duration": 20, "source": "activitynet",
                "events": [{"query": "open door.", "span": [5, 10]}]}]
        rows = normalize_annotations(raw, "timelens-100k")
        self.assertEqual(rows[0]["source"], "activitynet")
        self.assertEqual(rows[0]["query"], "open door")


if __name__ == "__main__":
    unittest.main()
