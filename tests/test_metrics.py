import unittest

from evaluation.metrics import normalize_predictions, parse_prediction, score_records, temporal_iou


class TemporalMetricTests(unittest.TestCase):
    def test_overlap_and_disjoint(self):
        self.assertAlmostEqual(temporal_iou([1, 5], [3, 7]), 1 / 3)
        self.assertEqual(temporal_iou([0, 1], [2, 3]), 0)
        self.assertEqual(temporal_iou([0, 3], [0, 3]), 1)

    def test_invalid_predictions(self):
        for value in (None, [2, 1], [1, 1], [-1, 2], [float("nan"), 2]):
            self.assertEqual(temporal_iou(value, [0, 3]), 0)

    def test_relative_coordinates_return_to_seconds(self):
        self.assertEqual(parse_prediction("13.4% to 48.2%", "relative", 100), [13.4, 48.2])
        self.assertEqual(parse_prediction("25.0% to 50.0%", "relative", 80), [20, 40])
        self.assertIsNone(parse_prediction("10% to 110%", "relative", 80))

    def test_answer_format_and_invalid_answer(self):
        self.assertEqual(parse_prediction("The event happens in 12.5 - 17.8 seconds."), [12.5, 17.8])
        self.assertIsNone(parse_prediction("could not locate the event"))
        self.assertIsNone(parse_prediction("-1 to 4"))

    def test_expected_denominator_includes_failure_and_missing(self):
        expected = [{"id": str(i), "span": [0, 2]} for i in range(3)]
        predictions = [{"id": "0", "prediction": [0, 2]},
                       {"id": "1", "prediction": None, "error": "OOM"}]
        report = score_records(predictions, expected)
        self.assertAlmostEqual(report["mIoU"], 100 / 3)
        self.assertEqual(report["failed"], 2)
        self.assertEqual(report["errors"]["missing_prediction"], 1)

    def test_duplicates_and_unexpected_ids_are_rejected(self):
        row = {"id": "a", "span": [0, 2], "prediction": [0, 2]}
        with self.assertRaises(ValueError):
            score_records([row, row])
        with self.assertRaises(ValueError):
            score_records([row], [{"id": "b", "span": [0, 2]}])

    def test_historical_timelens_and_native_sft(self):
        legacy = {"a.mp4>>>open door>>>[0, 2]": {"timestamps": [[0, 2]], "answers": "0 - 2"}}
        self.assertEqual(score_records([legacy])["mIoU"], 100)
        native = {"video": "a.mp4", "query": "open door", "gt_seconds": [0, 2], "pred_seconds": [0, 2]}
        self.assertEqual(normalize_predictions([legacy])[0]["id"], normalize_predictions([native])[0]["id"])

    def test_no_eval_for_historical_ground_truth(self):
        malicious = {"a.mp4>>>x>>>__import__('os').getcwd()": {"timestamps": [[0, 2]]}}
        with self.assertRaises((ValueError, SyntaxError)):
            normalize_predictions([malicious])


if __name__ == "__main__":
    unittest.main()
