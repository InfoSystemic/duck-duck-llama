#!/usr/bin/env python3
"""Query sampled NUMA residency of owned, verified tmpfs shards; move no pages."""
from collections import Counter
import ctypes
import fcntl
import json
import mmap
import os
from pathlib import Path
import time

from extract_glm_flash_q8_mtp_0908 import verified_sources

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/glm-flash-q8-page-placement-0908.json'


def main():
    assert not OUT.exists()
    state = verified_sources()
    library = ctypes.CDLL('libnuma.so.1', use_errno=True)
    move = library.move_pages
    move.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p),
                     ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    move.restype = ctypes.c_long
    result = dict(started=time.time(), moved_pages=0, sample_stride_bytes=2 << 20, files=[])
    for record in state['records']:
        if record['location'] != 'tmpfs':
            continue
        path = Path(record['path'])
        with path.open('rb') as stream:
            mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
            try:
                first = ctypes.c_char.from_buffer(mapping)
                start = ctypes.addressof(first)
                positions = range(0, record['bytes'], result['sample_stride_bytes'])
                # Read only: populate a sparse sample of page-table entries.
                checksum = 0
                for offset in positions:
                    checksum ^= mapping[offset]
                addresses = (ctypes.c_void_p * len(positions))(*(start + offset for offset in positions))
                status = (ctypes.c_int * len(positions))(*([-999] * len(positions)))
                code = move(0, len(positions), addresses, None, status, 0)
                if code < 0:
                    raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
                nodes = Counter(status)
                row = dict(path=str(path), bytes=record['bytes'], samples=len(positions), nodes=dict(nodes), checksum=checksum)
                result['files'].append(row)
                print(json.dumps(row), flush=True)
                del first
            finally:
                mapping.close()
    verified_sources()
    result['finished'] = time.time()
    OUT.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    with (BASE / 'results/glm-flash-q8-download-0908/download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
