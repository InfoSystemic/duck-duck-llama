import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import ab_profiles as runner


class MeasurementGates(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bench = self.root / "baseline-before" / "benchmark"
        self.bench.mkdir(parents=True)
        self.summary = {
            "model": "Qwen3.8-Flash-Next", "passed": True, "finished_utc": "2026-09-12T00:00:00Z",
            "quality": [{"passed": True}] * 3,
            "rows": [{"prompt": name, "rep": 0, "passed": True,
                      "timings": {"predicted_n": 192, "cache_n": 0}} for name in ("prose", "code", "analysis")],
        }
        self.write(self.bench / "summary.json", self.summary)
        for name, text in (("arithmetic", "17 * 23 = 391."), ("fact", "Canberra."), ("state", "10 marbles.")):
            self.write(self.bench / f"quality-{name}.json",
                       {"response": {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}})
        for name in ("prose", "code", "analysis"):
            self.write(self.bench / f"{name}-0.json",
                       {"request": {"prompt": name, "n_predict": 192},
                        "response": {"content": f"Valid {name} content.",
                                     "timings": {"predicted_per_second": 20.0, "predicted_n": 192, "prompt_n": 30}}})

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data))

    def test_full_length_correct_uncached_sample_passes(self):
        self.assertTrue(runner.check_benchmark(self.bench, 192)["passed"])

    def test_short_answer_cannot_be_used_as_throughput(self):
        self.summary["rows"][0]["timings"]["predicted_n"] = 8
        self.write(self.bench / "summary.json", self.summary)
        with self.assertRaisesRegex(RuntimeError, "short completion"):
            runner.check_benchmark(self.bench, 192)

    def test_cached_prompt_is_not_cold_measurement(self):
        self.summary["rows"][1]["timings"]["cache_n"] = 16
        self.write(self.bench / "summary.json", self.summary)
        with self.assertRaisesRegex(RuntimeError, "prompt-cache reuse"):
            runner.check_benchmark(self.bench, 192)

    def test_numeric_substring_is_not_correct_answer(self):
        self.write(self.bench / "quality-state.json",
                   {"response": {"choices": [{"message": {"content": "100 marbles."}, "finish_reason": "stop"}]}})
        with self.assertRaisesRegex(RuntimeError, "exact answer token"):
            runner.check_benchmark(self.bench, 192)

    def test_negated_answer_is_not_correct(self):
        self.write(self.bench / "quality-arithmetic.json",
                   {"response": {"choices": [{"message": {"content": "It is not 391."}, "finish_reason": "stop"}]}})
        with self.assertRaisesRegex(RuntimeError, "negates"):
            runner.check_benchmark(self.bench, 192)

    def test_output_mismatch_blocks_following_arm(self):
        dest = self.root / "tuned" / "benchmark"
        dest.mkdir(parents=True)
        for path in self.bench.glob("*.json"):
            (dest / path.name).write_bytes(path.read_bytes())
        raw = json.loads((dest / "prose-0.json").read_text())
        raw["response"]["content"] = "Different content."
        self.write(dest / "prose-0.json", raw)
        with self.assertRaisesRegex(RuntimeError, "no subsequent arm"):
            runner.compare_to_first(self.root, "tuned")

    def test_identical_outputs_with_changed_request_are_rejected(self):
        dest = self.root / "tuned" / "benchmark"
        dest.mkdir(parents=True)
        for path in self.bench.glob("*.json"):
            (dest / path.name).write_bytes(path.read_bytes())
        raw = json.loads((dest / "prose-0.json").read_text())
        raw["request"]["n_predict"] = 256
        self.write(dest / "prose-0.json", raw)
        with self.assertRaisesRegex(RuntimeError, "request differs"):
            runner.compare_to_first(self.root, "tuned")


class ProcessGates(unittest.TestCase):
    def test_missing_start_identity_is_never_live(self):
        with mock.patch.object(runner, "identity", return_value=None):
            self.assertFalse(runner.alive({"pid": 123, "start_ticks": None}))

    def test_reused_pid_is_never_live(self):
        with mock.patch.object(runner, "identity", return_value="456"):
            self.assertFalse(runner.alive({"pid": 123, "start_ticks": "455"}))

    def test_foreign_executable_cannot_be_signalled(self):
        with mock.patch.object(runner, "alive", return_value=True), \
             mock.patch.object(runner, "require_server", side_effect=RuntimeError("wrong executable")), \
             mock.patch.object(runner.os, "pidfd_open") as pidfd:
            with self.assertRaisesRegex(RuntimeError, "wrong executable"):
                runner.stop_owned({"pid": 123, "start_ticks": "455"}, {}, 18131)
            pidfd.assert_not_called()


if __name__ == "__main__":
    unittest.main()
