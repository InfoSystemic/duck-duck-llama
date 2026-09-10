#!/usr/bin/env python3
"""End-to-end hash/FP8 lookup oracle, full-sized sparse mappings, and bounded real rows."""
import ast
from concurrent.futures import ThreadPoolExecutor
import copy
import ctypes as C
import hashlib
import importlib.util
import json
import mmap
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from audit_deepseek_v41_tensors_0910 import read_range
from check_deepseek_v41_engram_hash_0910 import TokenizerAdapter, sha

BASE = Path(__file__).resolve().parent
OFFICIAL = BASE / 'results/deepseek-v41-intake-0910/official/inference'
U8P, U16P, U32P, U64P, I64P = [C.POINTER(t) for t in [C.c_uint8, C.c_uint16, C.c_uint32, C.c_uint64, C.c_int64]]


class Config(C.Structure):
    _fields_ = [('token_map', U32P), ('vocabulary', C.c_int64), ('compressed_vocabulary', C.c_int64),
        ('pad_token', C.c_int64), ('moduli', U64P), ('multipliers', U64P), ('table_rows', U64P)]


class Shard(C.Structure):
    _fields_ = [('layer', C.c_int32), ('first_row', C.c_uint64), ('rows', C.c_uint64),
        ('weight', U8P), ('weight_bytes', C.c_uint64), ('scale', U8P), ('scale_bytes', C.c_uint64)]


def main():
    out, library = map(Path, sys.argv[1:])
    assert not (out / 'correctness.json').exists()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location('lookup_reference_engram', OFFICIAL / 'engram.py')
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    args = SimpleNamespace(**json.loads((OFFICIAL / 'config.json').read_text()), max_batch_size=3, max_seq_len=512)
    layout = official.EngramLayout.from_args(args)
    reference = official.NgramHashState(args, layout, TokenizerAdapter())
    token_map = reference.token_map.numpy().astype('<u4')
    moduli = reference.primes.numpy().astype('<u8').reshape(-1)
    multipliers = reference.multipliers.numpy().astype('<u8').reshape(-1)
    rows = np.array(layout.num_embeddings, dtype='<u8')
    assert rows.tolist() == [384006168, 384016682]
    prior = json.loads((BASE / 'results/deepseek-v41-engram-hash-0910/correctness.json').read_text())
    assert hashlib.sha256(token_map.tobytes()).hexdigest() == prior['token_map_sha256']
    config = Config(token_map.ctypes.data_as(U32P), len(token_map), 99092, args.engram_pad_id,
        moduli.ctypes.data_as(U64P), multipliers.ctypes.data_as(U64P), rows.ctypes.data_as(U64P))

    # Execute only this reviewed, CPU-compatible class from the pinned model source.
    # Compact reference tables contain exactly the selected rows, remapped to dense IDs.
    tree = ast.parse((OFFICIAL / 'model.py').read_text())
    node, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ParallelEngramEmbedding']
    namespace = dict(torch=torch, nn=nn, F=F, world_size=1, rank=0, fp8_block_size=32,
        scale_dtype=torch.float8_e8m0fnu)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(OFFICIAL / 'model.py'), 'exec'), namespace)
    embedding = namespace['ParallelEngramEmbedding']

    lib = C.CDLL(str(library))
    lib.deepseek_v41_lookup_create.argtypes = [C.POINTER(Config), C.POINTER(Shard), C.c_int64,
        C.c_int64, C.c_int64, C.c_int, C.c_int]
    lib.deepseek_v41_lookup_create.restype = C.c_void_p
    lib.deepseek_v41_lookup_destroy.argtypes = [C.c_void_p]
    lib.deepseek_v41_lookup_destroy.restype = None
    lib.deepseek_v41_lookup_clone.argtypes = [C.c_void_p]
    lib.deepseek_v41_lookup_clone.restype = C.c_void_p
    lib.deepseek_v41_lookup_length.argtypes = [C.c_void_p]
    lib.deepseek_v41_lookup_length.restype = C.c_int64
    lib.deepseek_v41_lookup_rewind.argtypes = [C.c_void_p, C.c_int64]
    lib.deepseek_v41_lookup_rewind.restype = C.c_int
    lib.deepseek_v41_lookup_forward.argtypes = [C.c_void_p, I64P, U8P, C.c_int64, C.c_int64,
        U16P, C.c_uint64, U64P, C.c_uint64]
    lib.deepseek_v41_lookup_forward.restype = C.c_int

    maps, pointers, handles = [], [], []
    # Linux MAP_NORESERVE reserves the official address spans without committing
    # 202.8 GB. Only explicitly populated fixture rows touch physical pages.
    for n in rows:
        layer_maps, layer_pointers = [], []
        for width in [256, 8]:
            region = mmap.mmap(-1, int(n) * width,
                flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | 0x4000,
                prot=mmap.PROT_READ | mmap.PROT_WRITE)
            layer_maps.append(region)
            layer_pointers.append(C.addressof(C.c_uint8.from_buffer(region)))
        maps.append(layer_maps)
        pointers.append(layer_pointers)

    def descriptors(parts):
        result = []
        for layer, n in enumerate(rows.tolist()):
            boundaries = [n * i // parts for i in range(parts + 1)]
            for begin, end in zip(boundaries, boundaries[1:]):
                result.append(Shard(layer, begin, end - begin,
                    C.cast(pointers[layer][0] + begin * 256, U8P), (end - begin) * 256,
                    C.cast(pointers[layer][1] + begin * 8, U8P), (end - begin) * 8))
        # The API must sort descriptors, not depend on the caller's ordering.
        return (Shard * len(result))(*reversed(result))

    def create(parts, default_reciprocal, default_vectorize, **changes):
        values = dict(config=C.pointer(config), shards=descriptors(parts), shard_count=parts * 2,
            max_sequence=512, max_chunk=512, reciprocal=default_reciprocal, vectorize=default_vectorize)
        values.update(changes)
        handle = lib.deepseek_v41_lookup_create(*values.values())
        if handle: handles.append(handle)
        return handle

    states = {(p, r, v): [create(p, r, v) for _ in range(3)]
        for p in [1, 4] for r in [0, 1] for v in [0, 1]}
    assert all(h for hs in states.values() for h in hs)
    fixture_rows, real_rows = {}, {}
    def materialize(layer, row):
        key = (layer, row)
        if key not in fixture_rows:
            w = ((np.arange(256, dtype=np.uint64) * 17 + row * 131 + (row >> 9) + layer * 29) & 255).astype(np.uint8)
            w[(w & 127) == 127] ^= 1
            s = (112 + ((np.arange(8, dtype=np.uint64) * 7 + row + (row >> 16) + layer * 3) % 16)).astype(np.uint8)
            fixture_rows[key] = (w.tobytes(), s.tobytes())
        w, s = fixture_rows[key]
        maps[layer][0][row * 256:(row + 1) * 256] = w
        maps[layer][1][row * 8:(row + 1) * 8] = s
        return w, s

    def expected_values(ids):
        layers = []
        for layer in range(2):
            unique, inverse = np.unique(ids[:, :, layer, :], return_inverse=True)
            raw = [materialize(layer, int(row)) for row in unique]
            w = np.frombuffer(b''.join(v[0] for v in raw), dtype=np.uint8).copy().reshape(-1, 256)
            s = np.frombuffer(b''.join(v[1] for v in raw), dtype=np.uint8).copy().reshape(-1, 8)
            model = embedding(len(unique), 256)
            model.weight = nn.Parameter(torch.from_numpy(w).view(torch.float8_e4m3fn), requires_grad=False)
            model.scale = nn.Parameter(torch.from_numpy(s).view(torch.float8_e8m0fnu), requires_grad=False)
            values = model(torch.from_numpy(inverse.astype(np.int64))).view(torch.uint16).numpy()
            layers.append(values.reshape(*ids.shape[:2], 24, 256))
        return np.stack(layers, axis=2)

    def native(original_handle, input_tokens, input_mask, original_start, omit_ids=False, **changes):
        tokens = np.ascontiguousarray(input_tokens, dtype=np.int64)
        mask = None if input_mask is None else np.ascontiguousarray(input_mask, dtype=np.uint8)
        count = len(tokens)
        output = np.full(count * 48 * 256 + 2, 0xa53c, dtype=np.uint16)
        ids = np.full(count * 48 + 2, 0xdecafbad12345678, dtype=np.uint64)
        values = dict(handle=original_handle, tokens=tokens.ctypes.data_as(I64P),
            mask=None if mask is None else mask.ctypes.data_as(U8P), count=count, start=original_start,
            output=C.cast(output.ctypes.data + 2, U16P), output_elements=count * 48 * 256,
            ids_output=None if omit_ids else C.cast(ids.ctypes.data + 8, U64P), ids_elements=0 if omit_ids else count * 48)
        values.update(changes)
        status = lib.deepseek_v41_lookup_forward(*values.values())
        assert output[0] == output[-1] == 0xa53c and ids[0] == ids[-1] == 0xdecafbad12345678
        return status, output[1:-1].reshape(count, 2, 24, 256).copy(), ids[1:-1].reshape(count, 2, 24).copy()

    cases, exact_bf16, exact_ids = [], 0, 0
    def compare(label, tokens, mask=None, start=0, ref=None, chosen=None, omit_ids=False):
        nonlocal exact_bf16, exact_ids
        tokens = np.ascontiguousarray(tokens, dtype=np.int64)
        if tokens.ndim == 1: tokens = tokens[None, :]
        if mask is not None: mask = np.ascontiguousarray(mask, dtype=bool).reshape(tokens.shape)
        expected_ids = (reference if ref is None else ref)(torch.from_numpy(tokens), start,
            None if mask is None else torch.from_numpy(mask)).numpy().astype(np.uint64)
        expected = expected_values(expected_ids)
        selected = states if chosen is None else chosen
        for mode, mode_handles in selected.items():
            for batch in range(len(tokens)):
                status, actual, actual_ids = native(mode_handles[batch], tokens[batch],
                    None if mask is None else mask[batch], start, omit_ids=omit_ids)
                assert status == 0, (label, mode, status)
                assert np.array_equal(actual, expected[batch]), (label, mode, 'BF16')
                exact_bf16 += actual.size
                if not omit_ids:
                    assert np.array_equal(actual_ids, expected_ids[batch]), (label, mode, 'row IDs')
                    exact_ids += actual_ids.size
        cases.append(dict(label=label, batch=len(tokens), tokens=tokens.shape[1], start=start,
            modes=[list(m) for m in selected], exact=True, ids_returned=not omit_ids,
            expected_bf16_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
            expected_ids_sha256=hashlib.sha256(expected_ids.tobytes()).hexdigest()))

    rng = np.random.default_rng(4100910)
    try:
        for count in [1, 4, 17, 128, 512]:
            compare(f'full-{count}', rng.integers(0, 129280, (3, count), dtype=np.int64))
        for pattern in range(16):
            compare(f'image-mask-{pattern}', [1, 23, 456, 7890], [(pattern >> i) & 1 for i in range(4)])
        tokens = rng.integers(0, 129280, 17, dtype=np.int64)
        mask = np.ones(17, dtype=bool); mask[[0, 5, 12]] = False
        for split in range(1, 17):
            compare(f'split-{split}-prefill', tokens[:split], mask[:split])
            compare(f'split-{split}-decode', tokens[split:], mask[split:], split)
        tokens = rng.integers(0, 129280, (3, 128), dtype=np.int64)
        mask = rng.integers(0, 2, tokens.shape, dtype=np.uint8)
        start = 0
        for length in [7, 1, 19, 3, 64, 34]:
            compare(f'variable-chunk-{start}', tokens[:, start:start+length], mask[:, start:start+length], start)
            start += length
        compare('optional-ids-omitted', [7, 9, 11], omit_ids=True)
        clones = {m: [lib.deepseek_v41_lookup_clone(hs[0])] for m, hs in states.items()}
        handles.extend(h for hs in clones.values() for h in hs)
        assert all(h for hs in clones.values() for h in hs)
        cloned_reference = copy.deepcopy(reference)
        compare('clone-append', [15, 19], start=3, ref=cloned_reference, chosen=clones)
        compare('original-independent-append', [17, 23], start=3)
        for hs in clones.values(): assert lib.deepseek_v41_lookup_rewind(hs[0], 2) == 0
        compare('clone-rewind-append', [100, 101, 102], start=2, ref=cloned_reference, chosen=clones)
        compare('overwrite-suffix', [88], start=1)
        compare('append-after-truncation', [99, 100], start=2)
        compare('request-reset', [6789])
        compare('append-after-reset', [5432], start=1)

        invalid = []
        for mode, hs in states.items():
            handle = hs[0]
            length = lib.deepseek_v41_lookup_length(handle)
            clone = lib.deepseek_v41_lookup_clone(handle); assert clone; handles.append(clone)
            bad = [dict(count=-1), dict(count=513), dict(start=-1), dict(start=length+1),
                dict(count=512, start=length), dict(output_elements=1), dict(ids_elements=1),
                dict(output=None), dict(tokens=None), dict(ids_output=None), dict(handle=None)]
            for changes in bad:
                status, values, selected = native(handle, [1, 2, 3], None, length, **changes)
                assert status == -1 and np.all(values == 0xa53c) and np.all(selected == 0xdecafbad12345678)
                assert lib.deepseek_v41_lookup_length(handle) == length
                invalid.append(dict(mode=list(mode), changes=list(changes), rejected=True, unchanged=True))
            for ids, mask in [([1, -1, 3], None), ([1, 129280, 3], None), ([1, 2, 3], [1, 2, 1])]:
                status, values, selected = native(handle, ids, mask, length)
                assert status == -1 and np.all(values == 0xa53c) and np.all(selected == 0xdecafbad12345678)
                assert lib.deepseek_v41_lookup_length(handle) == length
                invalid.append(dict(mode=list(mode), changes=['invalid token or mask'], rejected=True, unchanged=True))
            before = lib.deepseek_v41_lookup_length(handle)
            assert native(handle, [], None, 0)[0] == 0 and lib.deepseek_v41_lookup_length(handle) == before
            assert lib.deepseek_v41_lookup_rewind(handle, -1) == -1
            assert lib.deepseek_v41_lookup_rewind(handle, before + 1) == -1
            # A matching append proves the token history also survived all failures.
            # Populate both possible paths before calling the native gather.
            expected_values(reference(torch.tensor([[55, 66]]), length).numpy().astype(np.uint64))
            a, b = native(handle, [55, 66], None, length), native(clone, [55, 66], None, length)
            assert a[0] == b[0] == 0 and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])

        rejected = []
        changes = [dict(config=None), dict(shards=None), dict(shard_count=1), dict(shard_count=129),
            dict(max_chunk=0), dict(max_chunk=513), dict(max_sequence=0), dict(reciprocal=2), dict(vectorize=2)]
        for change in changes:
            assert not create(1, 0, 0, **change); rejected.append(list(change))
        for field, value in [('layer', 2), ('first_row', 1), ('rows', 0), ('rows', (1 << 64)-1),
                ('weight', None), ('scale', None), ('weight_bytes', 1), ('scale_bytes', 1)]:
            parts = descriptors(1); setattr(parts[0], field, value)
            assert not create(1, 0, 0, shards=parts); rejected.append([field])
        for adjustment in [-1, 1]:
            parts = descriptors(4); parts[0].first_row += adjustment
            assert not create(4, 0, 0, shards=parts); rejected.append(['overlap or gap'])
        assert lib.deepseek_v41_lookup_length(None) == -1
        assert not lib.deepseek_v41_lookup_clone(None) and lib.deepseek_v41_lookup_rewind(None, 0) == -1
        print(json.dumps(dict(completed='synthetic-reference-and-lifecycle', cases=len(cases),
            exact_bf16_values=exact_bf16, invalid_calls=len(invalid))), flush=True)

        # Fetch exactly the 48 rows selected by one real token at sequence start.
        # Original public URLs and byte extents are retained; signed redirects are not.
        real_ids = reference(torch.tensor([[42]]), 0).numpy().astype(np.uint64)
        audit_path = BASE / 'results/deepseek-v41-tensor-audit-0910/result.json'
        audit = json.loads(audit_path.read_text()); assert audit['passed']
        jobs = []
        for layer, model_layer in enumerate([1, 14]):
            for suffix, width in [('weight', 256), ('scale', 8)]:
                tensor_name = f'layers.{model_layer}.engram.embed.{suffix}'
                sample = next(s for s in audit['samples'] if s['tensor'] == tensor_name)
                for shard in audit['shards']:
                    header_path = audit_path.parent / 'headers' / (shard['file'] + '.json')
                    header = json.loads(header_path.read_text())
                    if tensor_name not in header: continue
                    assert sha(header_path) == shard['header_sha256']
                    tensor = header[tensor_name]
                    assert tensor['shape'] == sample['shape'] == [int(rows[layer]), width]
                    assert tensor['data_offsets'][1] - tensor['data_offsets'][0] == int(rows[layer]) * width
                    for row in real_ids[0, 0, layer].tolist():
                        begin = 8 + shard['header_bytes'] + tensor['data_offsets'][0] + row * width
                        jobs.append(dict(layer=layer, row=row, suffix=suffix, bytes=width,
                            tensor=tensor_name, source_url=shard['source_url'], start=begin, end=begin+width-1,
                            total=shard['size'], header_sha256=shard['header_sha256'],
                            publisher_whole_shard_sha256=shard['publisher_sha256']))
                    break
                else: raise AssertionError(tensor_name)
        assert len(jobs) == 96
        sample_dir = out / 'real-rows'; sample_dir.mkdir()
        def fetch(job):
            try: raw = read_range(job['source_url'], job['start'], job['end'], job['total'])
            except Exception as error: raise RuntimeError('Bounded row download failed: ' + type(error).__name__) from None
            path = sample_dir / f"layer{job['layer']}-row{job['row']}-{job['suffix']}.bin"
            path.write_bytes(raw)
            return dict(job, file=str(path), sha256=sha(path))
        records = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for item in pool.map(fetch, jobs):
                records.append(item)
                if len(records) % 16 == 0:
                    print(json.dumps(dict(downloaded_row_slices=len(records), total=96)), flush=True)
        for layer in range(2):
            for row in real_ids[0, 0, layer].tolist():
                raw = [Path(next(x['file'] for x in records if x['layer'] == layer and x['row'] == row and x['suffix'] == suffix)).read_bytes()
                    for suffix in ['weight', 'scale']]
                fixture_rows[layer, row] = tuple(raw)
                real_rows[layer, row] = tuple(raw)
        compare('actual-pinned-token42-lookups', [42])
        real_manifest = dict(revision=audit['revision'], token_ids=[42], start=0, mask=None,
            selected_rows=real_ids.tolist(), records=records, retrieved_bytes=sum(x['bytes'] for x in records),
            whole_shard_hashes_verified=False)
        (out / 'real-rows.json').write_text(json.dumps(real_manifest, indent=2) + '\n')
        report = dict(passed=True, finished=time.time(), cases=cases, invalid_calls=invalid,
            rejected_configurations=rejected, exact_bf16_values_compared=exact_bf16,
            exact_row_ids_compared=exact_ids, native_variants=8, sparse_virtual_table_bytes=sum(int(n) * 264 for n in rows),
            populated_rows=len(fixture_rows), actual_pinned_rows=len(real_rows), actual_retrieved_bytes=real_manifest['retrieved_bytes'],
            official_reference_hash_and_embedding=True, official_embedding_class_ast_sha256=hashlib.sha256(ast.dump(node).encode()).hexdigest(),
            real_manifest_sha256=sha(out / 'real-rows.json'),
            input_sha256={str(p): sha(p) for p in [Path(__file__), library, OFFICIAL / 'engram.py', OFFICIAL / 'model.py',
                OFFICIAL / 'config.json', BASE / 'audit_deepseek_v41_tensors_0910.py', audit_path,
                BASE / 'check_deepseek_v41_engram_hash_0910.py',
                BASE / 'results/deepseek-v41-engram-hash-0910/correctness.json']},
            full_model_loaded=False, imc_bandwidth_measured=False,
            scope='Exact CPU hash to native FP8/BF16 lookup only. Full-sized address mappings contain bounded fixture rows; no full table, projection/gate, model graph, NUMA placement, or model quality claim.')
        (out / 'correctness.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(passed=True, cases=len(cases), exact_bf16_values=exact_bf16,
            exact_row_ids=exact_ids, actual_pinned_rows=len(real_rows), retrieved_bytes=real_manifest['retrieved_bytes'])), flush=True)
    finally:
        for handle in handles: lib.deepseek_v41_lookup_destroy(handle)
        for layer in maps:
            for region in layer: region.close()


if __name__ == '__main__':
    main()
