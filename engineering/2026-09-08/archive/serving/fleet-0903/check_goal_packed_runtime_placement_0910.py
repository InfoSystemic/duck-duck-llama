#!/usr/bin/env python3
"""Small owned-buffer fixture; does not import Torch or load/run any model."""
import ctypes
import gc
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import weakref
import goal_packed_runtime_placement_0910 as helper


class Tensor:
    def __init__(self, size):
        self.data = (ctypes.c_uint8 * size)()
        self.device = SimpleNamespace(type='cpu')
        self.shape = (size,)
        self.dtype = 'uint8'
        self.is_meta = False
    def data_ptr(self): return ctypes.addressof(self.data)
    def numel(self): return len(self.data)
    def element_size(self): return 1


def main():
    cache, refs = {}, []
    for index in range(32):
        weight, scale = Tensor(16384), Tensor(128)
        packed = SimpleNamespace(packed=Tensor(16384), scales=Tensor(128))
        refs += [weakref.ref(weight), weakref.ref(scale), weakref.ref(packed.packed), weakref.ref(packed.scales)]
        cache[(weight.data_ptr(), scale.data_ptr(), weight.shape, scale.shape)] = (weight, scale, packed)
    del weight, scale, packed
    before = [(bytes(value[0].data), bytes(value[2].packed.data)) for value in cache.values()]
    stop, ready = threading.Event(), threading.Event()
    def worker():
        ready.set(); stop.wait()
    thread = threading.Thread(target=worker)
    thread.start(); ready.wait()
    try:
        first = helper.capture_placement(cache, max_entries=12, label='owned_fixture')
        second = helper.capture_placement(cache, max_entries=12, previous=first)
        assert first['cache_entries'] == 32 and first['sampled_entries'] == 12
        assert [x['cache_index'] for x in first['entries']] == [i * 31 // 11 for i in range(12)]
        assert first['query_page_count'] == 12 * 4 * 3
        assert first['threads']['total_threads'] >= 2
        assert any('cpu_tick_delta' in row for row in second['threads']['records'])
        assert all('smaps' in mapping for mapping in first['mappings'].values())
        assert before == [(bytes(value[0].data), bytes(value[2].packed.data)) for value in cache.values()]
        encoded = json.dumps(first)
        assert len(encoded) < 200000, len(encoded)
        # Empty cache still reports the existing process workers and OMP mappings.
        empty = helper.capture_placement({}, max_entries=12)
        assert empty['sampled_entries'] == empty['query_page_count'] == 0
        for value in [0, 25]:
            try: helper.capture_placement(cache, max_entries=value)
            except ValueError: pass
            else: raise AssertionError('Invalid sampling bound accepted')
    finally:
        stop.set(); thread.join()
    cache.clear(); gc.collect()
    assert all(ref() is None for ref in refs), 'Helper retained source/packed tensors'
    result = dict(passed=True, cache_entries=32, sampled_entries=12, queried_pages=first['query_page_count'],
        existing_threads_observed=first['threads']['total_threads'], query_result=first['move_pages'],
        output_json_bytes=len(encoded), unchanged_owned_buffers=True, no_tensor_retention=True,
        source_sha256=first['source_sha256'], full_checkpoint_loaded=False, model_tok_s_measured=False)
    out = Path(__file__).resolve().parent / 'results/goal-packed-runtime-placement-0910'
    out.mkdir(exist_ok=True)
    (out / 'check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
