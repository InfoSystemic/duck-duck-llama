#!/usr/bin/env python3
"""Functional contract for the Codex-generated isolated LRU implementation."""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('generated_lru', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
C = m.LRUCache
try:
    C(-1)
except ValueError:
    pass
else:
    raise AssertionError('negative capacity must fail')
c = C(0)
c.put('a', 1)
assert c.get('a') == -1
c = C(2)
assert c.get('missing') == -1
c.put('a', 1); c.put('b', 2)
assert c.get('a') == 1
c.put('c', 3)
assert c.get('b') == -1
assert c.get('a') == 1 and c.get('c') == 3
c.put('a', 11); c.put('d', 4)
assert c.get('c') == -1 and c.get('a') == 11
c.put('null', None)
assert c.get('null') is None
c.put('minus', -1)
assert c.get('minus') == -1
c.put('later', 5)
assert c.get('null') == -1
assert c.get('later') == 5
print('PASS: misses, recency, eviction, update, zero/negative capacity, None and -1 values')
