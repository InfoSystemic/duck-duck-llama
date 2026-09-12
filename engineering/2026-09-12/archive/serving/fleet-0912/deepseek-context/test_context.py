"""Small no-checkpoint tests for context bounds, native Engram bytes and state."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'fleet-0903'))
import deepseek_v41_cpu_reference_0910 as cpu
from bounded_engram import BoundedEngram
from context_policy import BoundedPrefill, resize_context

torch.set_num_threads(1)
torch.set_default_device('cpu')


class ContextTests(unittest.TestCase):
    def test_prefill_positions_reset_and_bounds(self):
        class Sequence:
            max_seq_len = 16384
            def __call__(self, ids, position):
                if position == 0:
                    self.position, self.total = 0, 0
                else:
                    self_test.assertEqual(ids.shape[1], 1)
                self_test.assertEqual(position, self.position)
                self.position += ids.shape[1]
                self.total += int(ids.sum())
                return self.total
        self_test = self
        model = Sequence()
        proxy = BoundedPrefill(model, 256)
        for length in (1, 257, 4096, 16384, 7):
            ids = torch.arange(length)[None, :]
            self.assertEqual(proxy(ids, 0), int(ids.sum()))
            self.assertEqual(model.position, length)
            self.assertEqual(proxy.prefill_calls, 1 + max(0, length - 256))
        with self.assertRaises(ValueError):
            proxy(torch.ones((1, 2), dtype=torch.int64), 7)
        with self.assertRaises(ValueError):
            proxy(torch.ones((1, 1), dtype=torch.int64), 16384)

    def test_native_engram_rows_memory_and_persistence_are_bounded(self):
        with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(2) as pool:
            root = Path(tmp)
            legacy = root / 'legacy'
            legacy.mkdir()
            prefix = 'layers.1.engram.embed'
            weight = torch.arange(12 * 256).remainder(31).float().reshape(12, 256).to(torch.float8_e4m3fn)
            scale = torch.ones((12, 8)).to(torch.float8_e8m0fnu)
            raw = {'weight': weight.view(torch.uint8).numpy().tobytes(),
                   'scale': scale.view(torch.uint8).numpy().tobytes()}
            store = SimpleNamespace(rows_dir=legacy, pool=pool, downloaded_bytes=0, network_seconds=0.)
            store.catalog = SimpleNamespace(tensors={prefix + '.weight': {'shape': [12, 256], 'kind': 'weight'},
                                                       prefix + '.scale': {'shape': [12, 8], 'kind': 'scale'}})
            store._fetch = lambda meta, offset, size: raw[meta['kind']][offset:offset + size]
            rows = BoundedEngram(store, root / 'new', memory_rows=3, persistent_rows=2, fetch_rows=2)
            ids = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 2, 11, 0]])
            self.assertTrue(torch.equal(rows.rows(prefix, ids), weight.float().bfloat16()[ids]))
            self.assertLessEqual(len(rows.memory), 3)
            self.assertEqual(rows.persisted, 2)
            self.assertEqual(len(list((root / 'new').glob('*.bin'))), 2)
            self.assertFalse(list(legacy.iterdir()))
            self.assertGreater(rows.evicted, 0)
            self.assertGreater(rows.unpersisted, 0)
            restarted = BoundedEngram(store, root / 'new', memory_rows=3, persistent_rows=2, fetch_rows=2)
            before = store.downloaded_bytes
            self.assertTrue(torch.equal(restarted.rows(prefix, torch.tensor([0, 1])), weight.float().bfloat16()[:2]))
            self.assertEqual(before, store.downloaded_bytes)
            path = next((root / 'new').glob('*.bin'))
            path.write_bytes(bytes(264))
            restarted.memory.clear()
            with self.assertRaises(RuntimeError):
                restarted.rows(prefix, torch.tensor([0, 1]))

    def test_official_engram_position_and_reset(self):
        module = cpu.load_official_model('deepseek_v41_context_hash_oracle')
        state = module.NgramHashState.__new__(module.NgramHashState)
        torch.nn.Module.__init__(state)
        state.layout = SimpleNamespace(max_ngram_size=4)
        state.pad_id = 0
        state.register_buffer('token_map', torch.arange(128))
        state.register_buffer('cache', torch.zeros((1, 16384), dtype=torch.int64))
        state.register_buffer('multipliers', torch.tensor([[3, 7, 11, 13]]))
        state.register_buffer('primes', torch.tensor([[[103, 107], [109, 113], [127, 131]]]))
        state.register_buffer('offsets', torch.tensor([[0, 103, 210, 319, 432, 559]]))
        ids = torch.arange(16384).remainder(128)[None, :]
        full = state(ids, 0)
        for position in (255, 256, 4095, 4096, 16383):
            actual = state(ids[:, position:position + 1], position)
            self.assertTrue(torch.equal(actual, full[:, position:position + 1]))
        small = torch.tensor([[9, 8, 7]])
        reset = state(small, 0)
        state.cache.fill_(0)
        self.assertTrue(torch.equal(reset, state(small, 0)))

    def test_resize_source_caches_and_rope(self):
        module = cpu.load_official_model('deepseek_v41_context_resize_oracle')
        config = SimpleNamespace(max_seq_len=256, rope_head_dim=32, original_seq_len=65536,
                                 compress_rope_theta=160000, rope_theta=10000,
                                 rope_factor=16, beta_fast=32, beta_slow=1)
        layers = []
        for ratio in (0, 2, 1):
            indexer = None if ratio == 0 else SimpleNamespace(
                owns_k=True, k_cache=torch.ones((1, 256 // ratio, 32)), freqs_cis='stale')
            attn = SimpleNamespace(compress_ratio=ratio, is_kv_source=bool(ratio),
                                   window_kv_cache=torch.ones((1, 128, 32)), indexer=indexer)
            if ratio:
                attn.compress_kv_cache = torch.ones((1, 256 // ratio, 32))
                attn.compressor = SimpleNamespace(kv_state=torch.ones((1, 2, 32)),
                                                   score_state=torch.ones((1, 2, 32)))
            layers.append(SimpleNamespace(attn=attn))
        model = SimpleNamespace(max_seq_len=256, layers=layers,
                                 engram_hash=SimpleNamespace(cache=torch.ones((1, 256), dtype=torch.int64)))
        for context in (4096, 16384):
            resize_context(module, config, model, context)
            self.assertEqual(config.max_seq_len, context)
            self.assertEqual(model.max_seq_len, context)
            self.assertEqual(model.engram_hash.cache.shape, (1, context))
            for layer in layers:
                attn = layer.attn
                self.assertEqual(attn.freqs_cis.shape[0], context)
                self.assertTrue(torch.isfinite(attn.freqs_cis[context - 1]).all())
                self.assertFalse(attn.window_kv_cache.any())
                if attn.compress_ratio:
                    self.assertEqual(attn.compress_kv_cache.shape[1], context // attn.compress_ratio)
                    self.assertEqual(attn.indexer.k_cache.shape[1], context // attn.compress_ratio)
                    self.assertIsNone(attn.indexer.freqs_cis)
            self.assertIsNone(module.shared_attn.candidates)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContextTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    files = [HERE / n for n in ('context_policy.py', 'bounded_engram.py', 'context_server.py', 'test_context.py')]
    files += [Path(cpu.__file__), cpu.OFFICIAL / 'model.py', cpu.OFFICIAL / 'engram.py']
    report = dict(passed=result.wasSuccessful(), tests=result.testsRun, full_checkpoint_loaded=False,
                  model_context_validated=False, network_requests=0,
                  limitations='Policy/native-row/hash/cache tests only; full-model extended-context logits untested',
                  sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (HERE / 'test-result.json').write_text(json.dumps(report, indent=2) + '\n')
    sys.exit(0 if result.wasSuccessful() else 1)
