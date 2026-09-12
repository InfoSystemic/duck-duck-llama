"""Bound Engram memory and new persistent rows while preserving native row bytes."""
from collections import OrderedDict
from concurrent.futures import as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import torch


class BoundedEngram:
    def __init__(self, store, directory, memory_rows=8192, persistent_rows=8192, fetch_rows=128):
        if not 1 <= fetch_rows <= memory_rows <= 100000:
            raise ValueError('Expected 1 <= fetch_rows <= memory_rows <= 100000')
        if not 0 <= persistent_rows <= 100000:
            raise ValueError('Persistent row limit must be between 0 and 100000')
        self.store, self.directory = store, Path(directory)
        self.memory_rows, self.persistent_rows, self.fetch_rows = memory_rows, persistent_rows, fetch_rows
        if self.directory == store.rows_dir:
            raise ValueError('Context row directory must differ from the selected legacy row directory')
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink() or self.directory.stat().st_uid != os.getuid():
            raise ValueError('Expected an owned nonsymlink row directory')
        marker = self.directory / 'context-cache-owner.json'
        owner = {'task': 'deepseek-v41-context-0912', 'revision': 'fb2764a5cf321eaa5070ca8f9e892818f477c16d'}
        if marker.exists():
            if json.loads(marker.read_text()) != owner:
                raise ValueError('Context cache owner mismatch')
        elif any(self.directory.iterdir()):
            raise ValueError('A new context cache directory must be empty')
        else:
            marker.write_text(json.dumps(owner) + '\n')
        self.persisted = sum(1 for _ in self.directory.glob('*.bin'))
        if self.persisted > persistent_rows:
            raise ValueError('Existing context rows exceed the requested persistent limit')
        self.memory = OrderedDict()
        self.hits = self.disk_hits = self.fetched = self.evicted = self.unpersisted = 0

    def _read(self, directory, prefix, row):
        path = directory / f'{prefix}.{row}.bin'
        record = path.with_suffix('.json')
        if not record.exists():
            if path.exists():
                raise RuntimeError('Unmanifested Engram row')
            return None
        if path.is_symlink() or record.is_symlink():
            raise RuntimeError('Symlinked Engram cache record')
        metadata = json.loads(record.read_text())
        partial = path.with_suffix('.part')
        if not path.exists() and directory == self.directory and partial.exists():
            if partial.is_symlink():
                raise RuntimeError('Symlinked partial Engram row')
            raw = partial.read_bytes()
            if len(raw) != 264 or hashlib.sha256(raw).hexdigest() != metadata['sha256']:
                raise RuntimeError('Invalid partial Engram row')
            partial.rename(path)
        raw = path.read_bytes()
        if (metadata.get('prefix') != prefix or metadata.get('row') != row or len(raw) != 264
                or hashlib.sha256(raw).hexdigest() != metadata.get('sha256')):
            raise RuntimeError('Engram row integrity mismatch')
        return raw

    def _persist(self, prefix, row, raw):
        if self.persisted >= self.persistent_rows:
            self.unpersisted += 1
            return
        path = self.directory / f'{prefix}.{row}.bin'
        partial, record = path.with_suffix('.part'), path.with_suffix('.json')
        if path.exists() or partial.exists() or record.exists():
            raise RuntimeError('Unexpected existing context cache output')
        with partial.open('xb') as handle:
            handle.write(raw)
        with record.open('x') as handle:
            json.dump({'prefix': prefix, 'row': row, 'sha256': hashlib.sha256(raw).hexdigest()}, handle)
        partial.rename(path)
        self.persisted += 1

    def _decode(self, prefix, row, raw):
        data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        value = (data[:256].view(torch.float8_e4m3fn).float().reshape(8, 32)
                 * data[256:].view(torch.float8_e8m0fnu).float()[:, None]).flatten().bfloat16()
        if not torch.isfinite(value).all():
            raise RuntimeError('Nonfinite Engram row')
        while len(self.memory) >= self.memory_rows:
            self.memory.popitem(last=False)
            self.evicted += 1
        self.memory[prefix, row] = value

    def rows(self, prefix, ids):
        store = self.store
        w, s = store.catalog.tensors[prefix + '.weight'], store.catalog.tensors[prefix + '.scale']
        if w['shape'][1] != 256 or s['shape'] != [w['shape'][0], 8]:
            raise ValueError('Unexpected native Engram tensor shape')
        flat = ids.flatten().tolist()
        if any(type(row) is not int or not 0 <= row < w['shape'][0] for row in flat):
            raise ValueError('Engram row outside checkpoint')
        result = torch.empty((len(flat), 256), dtype=torch.bfloat16)
        began = time.perf_counter()
        try:
            for begin in range(0, len(flat), self.fetch_rows):
                batch = flat[begin:begin + self.fetch_rows]
                values, missing = {}, []
                for row in dict.fromkeys(batch):
                    key = prefix, row
                    if key in self.memory:
                        self.hits += 1
                        self.memory.move_to_end(key)
                        values[row] = self.memory[key]
                        continue
                    raw = self._read(self.directory, prefix, row)
                    if raw is None:
                        raw = self._read(store.rows_dir, prefix, row)
                    if raw is None:
                        missing.append(row)
                    else:
                        self.disk_hits += 1
                        self._decode(prefix, row, raw)
                        values[row] = self.memory[key]
                pending, parts = {}, {}
                try:
                    for row in missing:
                        for meta, width, kind in ((w, 256, 'weight'), (s, 8, 'scale')):
                            pending[store.pool.submit(store._fetch, meta, row * width, width)] = row, kind
                    for future in as_completed(pending):
                        row, kind = pending[future]
                        raw = future.result()
                        if len(raw) != (256 if kind == 'weight' else 8):
                            raise RuntimeError('Incomplete native Engram range')
                        parts[row, kind] = bytes(raw)
                        store.downloaded_bytes += len(raw)
                finally:
                    for future in pending:
                        future.cancel()
                for row in missing:
                    raw = parts[row, 'weight'] + parts[row, 'scale']
                    self._persist(prefix, row, raw)
                    self._decode(prefix, row, raw)
                    values[row] = self.memory[prefix, row]
                    self.fetched += 1
                result[begin:begin + len(batch)] = torch.stack([values[row] for row in batch])
        finally:
            store.network_seconds += time.perf_counter() - began
        return result.reshape(*ids.shape, 256)

    def metrics(self):
        return dict(memory_rows=len(self.memory), memory_limit=self.memory_rows,
                    persisted_rows=self.persisted, persistent_limit=self.persistent_rows,
                    hits=self.hits, disk_hits=self.disk_hits, fetched=self.fetched,
                    evicted=self.evicted, unpersisted=self.unpersisted)
