#!/usr/bin/env python3
"""Unit tests for the shipped-entry harness. Drive the real helper functions."""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = Path("/home/kwebb/InfoSystemic/AI-Server/engines/llama-llama-duck/tools")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TOOLS))

import decode_bench
import quality_probe
import shipped_entry_harness as harness


LLAMA_FLASH_RESPONSE = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "**17 × 23 = 391**\n\nA quick way to see it: 17 × 23 = 17 × (20 + 3) = 340 + 51 = 391.",
                "reasoning_content": "17*23 = 391",
            },
            "finish_reason": "stop",
        }
    ],
    "timings": {
        "predicted_n": 52,
        "predicted_per_second": 16.473,
        "prompt_per_second": 17.7,
    },
}

DEEPSEEK_RESPONSE = {
    "choices": [
        {
            "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 6, "completion_tokens": 10, "total_tokens": 16},
    "timings": {
        "total_seconds": 7.87,
        "prefill_seconds": 2.89,
        "decode_seconds": 5.0,
        "downloaded_bytes": 0,
    },
}

GARBAGE_RESPONSE = {
    "choices": [{"message": {"content": "////////////////////"}, "finish_reason": "length"}],
    "timings": {"predicted_n": 20, "predicted_per_second": 40.0},
}


class TestDegenerate(unittest.TestCase):
    def test_quality_probe_rejects_slash_garbage(self):
        self.assertTrue(quality_probe.is_degenerate_completion("////////////////////"))
        self.assertTrue(quality_probe.is_degenerate_completion(""))
        self.assertTrue(quality_probe.is_degenerate_completion("   \n  "))
        self.assertFalse(quality_probe.is_degenerate_completion("**17 × 23 = 391**"))
        self.assertFalse(quality_probe.is_degenerate_completion("Hello! How can I help you today?"))

    def test_decode_bench_same_garbage_rule(self):
        self.assertTrue(decode_bench.is_degenerate_completion("////////////////////"))
        self.assertFalse(decode_bench.is_degenerate_completion("The capital of Australia is Canberra."))


class TestExtractDecodeTps(unittest.TestCase):
    def test_llama_predicted_per_second(self):
        tps = decode_bench.extract_decode_tps(LLAMA_FLASH_RESPONSE)
        self.assertAlmostEqual(tps, 16.473)

    def test_deepseek_decode_seconds(self):
        tps = decode_bench.extract_decode_tps(DEEPSEEK_RESPONSE)
        # 9 decode tokens after one prefill-produced token, 5.0 seconds
        self.assertAlmostEqual(tps, 9 / 5.0)

    def test_missing_timings(self):
        self.assertIsNone(decode_bench.extract_decode_tps({"choices": []}))


class TestQualityProbePayloadAndValidators(unittest.TestCase):
    def test_payload_is_temp0_cache_prompt_false(self):
        payload = quality_probe.build_probe_payload("GLM-5.3-Flash", "What is 17 * 23?")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertIs(payload["cache_prompt"], False)
        self.assertEqual(payload["messages"][0]["content"], "What is 17 * 23?")

    def test_evaluate_flash_math(self):
        math_validator = quality_probe.PROBES[0][2]
        ok, text, rate, detail = quality_probe.evaluate_probe_response(
            LLAMA_FLASH_RESPONSE, math_validator
        )
        self.assertTrue(ok)
        self.assertIn("391", text)
        self.assertAlmostEqual(rate, 16.473)
        self.assertEqual(detail, "ok")

    def test_evaluate_rejects_garbage_even_with_high_tps(self):
        math_validator = quality_probe.PROBES[0][2]
        ok, text, rate, detail = quality_probe.evaluate_probe_response(
            GARBAGE_RESPONSE, math_validator
        )
        self.assertFalse(ok)
        self.assertEqual(detail, "degenerate")
        self.assertTrue(quality_probe.is_degenerate_completion(text))

    def test_house_probes(self):
        self.assertTrue(quality_probe.PROBES[0][2]("391"))
        self.assertTrue(quality_probe.PROBES[1][2]("Canberra is the capital"))
        self.assertTrue(quality_probe.PROBES[2][2]("Yes, all Bloops are Lazzies"))
        self.assertFalse(quality_probe.PROBES[0][2]("392"))


class TestDecodeBenchValidate(unittest.TestCase):
    def test_structured_and_novel(self):
        ok, _ = decode_bench.validate_output("structured", " ".join(str(i) for i in range(1, 61)))
        self.assertTrue(ok)
        bad, _ = decode_bench.validate_output("structured", "1 2 3")
        self.assertFalse(bad)
        ok2, detail = decode_bench.validate_output(
            "novel", "One. Two. Three. Four. Five. Six."
        )
        self.assertTrue(ok2)


class TestBaselineComparison(unittest.TestCase):
    def test_reads_this_host_baselines_file(self):
        baselines = harness.load_baselines()
        self.assertIn("glm-5.3-flash", baselines["models"])
        self.assertAlmostEqual(baselines["models"]["glm-5.3-flash"]["baseline_tps"], 15.20)
        self.assertAlmostEqual(baselines["models"]["qwen3.8-flash-next"]["baseline_tps"], 19.90)
        self.assertAlmostEqual(baselines["models"]["glm-5.3-full"]["baseline_tps"], 7.89)
        self.assertAlmostEqual(baselines["models"]["deepseek-v4.1-flash"]["baseline_tps"], 1.800)
        # Do not hard-code a live measured rate here; only the baseline file values.

    def test_meets_baseline_spread(self):
        self.assertTrue(harness.meets_baseline(15.20, 15.20, 0.05))
        self.assertTrue(harness.meets_baseline(16.47, 15.20, 0.05))
        self.assertTrue(harness.meets_baseline(14.50, 15.20, 0.05))  # within 5%
        self.assertFalse(harness.meets_baseline(8.33, 19.90, 0.05))  # silent half-speed pin
        self.assertFalse(harness.meets_baseline(None, 15.20, 0.05))

    def test_harness_extracts_from_real_response_shapes(self):
        self.assertAlmostEqual(harness.extract_decode_tps(LLAMA_FLASH_RESPONSE), 16.473)
        self.assertAlmostEqual(harness.extract_decode_tps(DEEPSEEK_RESPONSE), 1.8)
        self.assertTrue(harness.is_degenerate_completion("////////////////////"))
        self.assertFalse(harness.is_degenerate_completion(harness.completion_text(LLAMA_FLASH_RESPONSE)))


class TestLaunchScriptsWidenAffinity(unittest.TestCase):
    def test_qwen_restore_full_use_taskset_0_127(self):
        """Spare-core inheritance is a silent ~4 tok/s pin; launchers must widen."""
        files = [
            HERE / "launch-qwen.sh",
            HERE / "restore-baseline-0911.sh",
            HERE / "launch-glmfull-dev2.sh",
            HERE / "finish.sh",
        ]
        for path in files:
            text = path.read_text()
            self.assertIn("taskset -c 0-127", text, msg=str(path))

    def test_full_launcher_uses_glm_dsa_tensor_split_runtime(self):
        text = (HERE / "launch-glmfull-dev2.sh").read_text()
        self.assertIn("llama.cpp-sr950-glm/build-dev2/bin/llama-server", text)
        self.assertIn("--split-mode tensor", text)
        self.assertIn("draft-mtp", text)
        self.assertIn('SPLIT=${GLM_FULL_TENSOR_SPLIT:-1,1,1,1}', text)
        self.assertIn("FULL_NUMA_REPACK_OVERRIDE", text)
        self.assertNotIn("ngram-mod,draft-mtp", text)
        self.assertNotIn("glm-flash-q8-r8-ordered-k-runtime-0908", text)


class TestPayloadCachePrompt(unittest.TestCase):

    def test_probe_once_payload_builder(self):
        payload = harness.build_probe_payload("Qwen3.8-Flash-Next", "What is 17 * 23?", max_tokens=128)
        self.assertIs(payload["cache_prompt"], False)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["max_tokens"], 128)


if __name__ == "__main__":
    unittest.main()
