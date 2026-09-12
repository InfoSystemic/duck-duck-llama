#!/usr/bin/env python3
"""Tests that drive the shipped quality_probe and decode_bench functions."""
import unittest

import decode_bench
import quality_probe


class TestShippedQualityProbe(unittest.TestCase):
    def test_build_probe_payload_cache_prompt_false(self):
        payload = quality_probe.build_probe_payload("GLM-5.3-Flash", "What is 17 * 23?")
        self.assertIs(payload["cache_prompt"], False)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["seed"], 42)

    def test_math_fact_logic_validators(self):
        names = [p[0] for p in quality_probe.PROBES]
        self.assertEqual(names, ["math", "fact", "logic"])
        self.assertTrue(quality_probe.PROBES[0][2]("391"))
        self.assertTrue(quality_probe.PROBES[1][2]("Canberra"))
        self.assertFalse(quality_probe.is_degenerate_completion("391"))
        self.assertTrue(quality_probe.is_degenerate_completion("////"))


class TestShippedDecodeBench(unittest.TestCase):
    def test_extract_decode_tps_from_llama_timings(self):
        data = {"timings": {"predicted_per_second": 19.90, "predicted_n": 192}}
        self.assertAlmostEqual(decode_bench.extract_decode_tps(data), 19.90)

    def test_extract_decode_tps_from_deepseek_timings(self):
        data = {
            "usage": {"completion_tokens": 10},
            "timings": {"decode_seconds": 5.0},
        }
        self.assertAlmostEqual(decode_bench.extract_decode_tps(data), 1.8)

    def test_workloads_and_validate_output(self):
        self.assertIn("prose", decode_bench.WORKLOADS)
        self.assertIn("code", decode_bench.WORKLOADS)
        ok, _ = decode_bench.validate_output(
            "structured", " ".join(map(str, range(1, 61)))
        )
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
