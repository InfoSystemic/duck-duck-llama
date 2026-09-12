"""Fault-injection checks for the persistent native Engram row quota."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from bounded_engram import BoundedEngram

torch.set_num_threads(1)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.legacy, self.directory = self.root / 'legacy', self.root / 'context'
        self.legacy.mkdir()
        self.pool = ThreadPoolExecutor(2)
        self.addCleanup(self.pool.shutdown)
        self.prefix = 'layers.1.engram.embed'
        self.weight = torch.arange(4 * 256).remainder(31).float().reshape(4, 256).to(torch.float8_e4m3fn)
        raw = {'weight': self.weight.view(torch.uint8).numpy().tobytes(),
               'scale': torch.ones((4, 8)).to(torch.float8_e8m0fnu).view(torch.uint8).numpy().tobytes()}
        self.store = SimpleNamespace(rows_dir=self.legacy, pool=self.pool,
                                     downloaded_bytes=0, network_seconds=0.,
                                     catalog=SimpleNamespace(tensors={
                                         self.prefix + '.weight': {'shape': [4, 256], 'kind': 'weight'},
                                         self.prefix + '.scale': {'shape': [4, 8], 'kind': 'scale'}}))
        self.store._fetch = lambda meta, offset, size: raw[meta['kind']][offset:offset + size]

    def cache(self):
        cache = BoundedEngram(self.store, self.directory, memory_rows=2, persistent_rows=1, fetch_rows=1)
        if hasattr(cache, 'close'):
            self.addCleanup(cache.close)
        return cache

    def read(self, cache, row):
        value = cache.rows(self.prefix, torch.tensor([row]))
        self.assertTrue(torch.equal(value[0], self.weight[row].float().bfloat16()))

    def fail_before_rename(self, cache):
        with patch.object(Path, 'rename', side_effect=OSError('injected interruption before commit')):
            with self.assertRaisesRegex(OSError, 'injected interruption'):
                self.read(cache, 0)
        self.assertEqual(len(list(self.directory.glob('*.part'))), 1)

    def test_restart_recovery_counts_committed_partial_against_limit(self):
        cache = self.cache()
        self.fail_before_rename(cache)
        if hasattr(cache, 'close'):
            cache.close()
        restarted = self.cache()
        downloaded = self.store.downloaded_bytes
        self.read(restarted, 0)
        self.assertEqual(self.store.downloaded_bytes, downloaded)
        self.read(restarted, 1)
        self.assertEqual(len(list(self.directory.glob('*.bin'))), 1)
        self.assertEqual(restarted.persisted, 1)
        self.assertFalse(list(self.legacy.iterdir()))

    def test_same_process_recovery_does_not_allow_a_second_persistent_row(self):
        cache = self.cache()
        self.fail_before_rename(cache)
        self.read(cache, 0)
        self.read(cache, 1)
        self.assertEqual(len(list(self.directory.glob('*.bin'))), 1)
        self.assertEqual(cache.persisted, 1)

    def test_unmanifested_partial_is_refetched_and_reclaims_its_slot(self):
        cache = self.cache()
        partial = self.directory / f'{self.prefix}.0.part'
        partial.write_bytes(b'incomplete row')
        if hasattr(cache, 'close'):
            cache.close()
        restarted = self.cache()
        self.read(restarted, 0)
        self.assertFalse(partial.exists())
        self.assertEqual(restarted.persisted, 1)
        self.assertEqual(len(list(self.directory.glob('*.bin'))), 1)

    def test_torn_manifest_without_committed_row_is_refetched(self):
        cache = self.cache()
        self.fail_before_rename(cache)
        (self.directory / f'{self.prefix}.0.json').write_text('{"prefix":')
        self.read(cache, 0)
        self.assertEqual(cache.persisted, 1)
        self.assertEqual(len(list(self.directory.glob('*.bin'))), 1)

    def test_second_cache_writer_cannot_bypass_the_directory_quota(self):
        cache = self.cache()
        with self.assertRaisesRegex(RuntimeError, 'already in use'):
            self.cache()
        cache.close()
        self.read(self.cache(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
