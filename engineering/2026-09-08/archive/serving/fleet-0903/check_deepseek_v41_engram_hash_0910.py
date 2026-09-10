#!/usr/bin/env python3
"""Compare native Engram row IDs with the reviewed, pinned publisher CPU reference."""
import copy
import ctypes as C
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from tokenizers import Tokenizer

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'results/deepseek-v41-intake-0910/official/inference/engram.py'
CONFIG = BASE / 'results/deepseek-v41-intake-0910/official/inference/config.json'
SETUP = BASE / 'results/deepseek-v41-hash-reference-setup-0910'
U64P, I64P, U32P, U8P = [C.POINTER(t) for t in [C.c_uint64, C.c_int64, C.c_uint32, C.c_uint8]]


def sha(path):
    assert '.private.' not in str(path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TokenizerAdapter:
    def __init__(self):
        self.backend_tokenizer = Tokenizer.from_file(str(SETUP / 'tokenizer.json'))

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


def main():
    out, library = map(Path, sys.argv[1:])
    destination = out / 'correctness.json'
    assert not destination.exists()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    setup = json.loads((SETUP / 'result.json').read_text())
    assert setup['passed']
    for item in setup['retrieved']:
        assert sha(item['file']) == item['sha256']
    # This 192-line module was inspected before use: tokenizer normalization,
    # prime layout, seeded integer multipliers, and tensor-only cache/hash logic.
    spec = importlib.util.spec_from_file_location('deepseek_v41_pinned_engram', SOURCE)
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    config = json.loads(CONFIG.read_text())
    args = SimpleNamespace(**config, max_batch_size=3, max_seq_len=129280)
    tokenizer = TokenizerAdapter()
    layout = official.EngramLayout.from_args(args)
    reference = official.NgramHashState(args, layout, tokenizer)
    assert reference.token_map.max().item() + 1 == args.engram_compressed_vocab_size == 99092
    assert len(tokenizer) == args.vocab_size == 129280
    token_map = reference.token_map.numpy().astype('<u4')
    moduli = reference.primes.numpy().astype('<u8').reshape(-1)
    multipliers = reference.multipliers.numpy().astype('<u8').reshape(-1)
    table_rows = np.array(layout.num_embeddings, dtype='<u8')
    assert np.array_equal(moduli.reshape(2, 24).sum(axis=1), table_rows)
    map_bytes = token_map.tobytes()
    (out / 'compressed-token-map.bin').write_bytes(map_bytes)
    config_bytes = struct.pack('<III', len(tokenizer), 99092, args.engram_pad_id)
    config_bytes += moduli.tobytes() + multipliers.tobytes() + table_rows.tobytes() + map_bytes
    (out / 'hash-config.bin').write_bytes(config_bytes)
    lib = C.CDLL(str(library))
    lib.deepseek_v41_hash_create.argtypes = [U32P, C.c_int64, C.c_int64, C.c_int64, U64P, U64P, U64P, C.c_int64, C.c_int]
    lib.deepseek_v41_hash_create.restype = C.c_void_p
    lib.deepseek_v41_hash_destroy.argtypes = [C.c_void_p]
    lib.deepseek_v41_hash_destroy.restype = None
    lib.deepseek_v41_hash_length.argtypes = [C.c_void_p]
    lib.deepseek_v41_hash_length.restype = C.c_int64
    lib.deepseek_v41_hash_clone.argtypes = [C.c_void_p]
    lib.deepseek_v41_hash_clone.restype = C.c_void_p
    lib.deepseek_v41_hash_rewind.argtypes = [C.c_void_p, C.c_int64]
    lib.deepseek_v41_hash_rewind.restype = C.c_int
    lib.deepseek_v41_hash_forward.argtypes = [C.c_void_p, I64P, U8P, C.c_int64, C.c_int64, U64P, C.c_uint64]
    lib.deepseek_v41_hash_forward.restype = C.c_int
    lib.deepseek_v41_hash_remainder.argtypes = [C.c_uint64, C.c_uint64, U64P]
    lib.deepseek_v41_hash_remainder.restype = C.c_int
    handles = []
    def create(default_mode, **changes):
        values = dict(token_map=token_map, vocabulary=len(tokenizer), compressed=99092, pad=args.engram_pad_id,
            moduli=moduli, multipliers=multipliers, rows=table_rows, max_sequence=args.max_seq_len, mode=default_mode)
        values.update(changes)
        handle = lib.deepseek_v41_hash_create(values['token_map'].ctypes.data_as(U32P), values['vocabulary'],
            values['compressed'], values['pad'], values['moduli'].ctypes.data_as(U64P),
            values['multipliers'].ctypes.data_as(U64P), values['rows'].ctypes.data_as(U64P), values['max_sequence'], values['mode'])
        if handle: handles.append(handle)
        return handle
    def forward(handle, ids, mask, start, capacity=None):
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        mask = None if mask is None else np.ascontiguousarray(mask, dtype=np.uint8)
        count = len(ids)
        # Output is naturally 8-byte aligned but offset from a larger allocation.
        guard = np.full(count * 48 + 2, 0xdecafbad12345678, dtype=np.uint64)
        ptr = C.cast(guard.ctypes.data + 8, U64P)
        status = lib.deepseek_v41_hash_forward(handle, ids.ctypes.data_as(I64P),
            None if mask is None else mask.ctypes.data_as(U8P), count, start, ptr,
            count * 48 if capacity is None else capacity)
        assert guard[0] == guard[-1] == 0xdecafbad12345678
        return status, guard[1:-1].reshape(count, 2, 24).copy()
    states = {mode: [create(mode) for _ in range(3)] for mode in [0, 1]}
    assert all(h for values in states.values() for h in values)
    rng = np.random.default_rng(20260910)
    cases = []
    total_values = 0
    def compare(label, ids, mask=None, start=0, ref=None, selected=None):
        nonlocal total_values
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        if ids.ndim == 1: ids = ids[None, :]
        if mask is not None:
            mask = np.ascontiguousarray(mask, dtype=bool).reshape(ids.shape)
        expected = (reference if ref is None else ref)(torch.from_numpy(ids), start,
            None if mask is None else torch.from_numpy(mask)).numpy().astype(np.uint64)
        selected = states if selected is None else selected
        for mode, mode_states in selected.items():
            actual = []
            for batch, handle in enumerate(mode_states[:len(ids)]):
                status, values = forward(handle, ids[batch], None if mask is None else mask[batch], start)
                assert status == 0, (label, mode, batch)
                assert lib.deepseek_v41_hash_length(handle) == start + ids.shape[1]
                actual.append(values)
            actual = np.stack(actual)
            assert np.array_equal(actual, expected), (label, mode, np.argwhere(actual != expected)[:5].tolist())
            total_values += int(actual.size)
        digest = hashlib.sha256(expected.astype('<u8').tobytes()).hexdigest()
        cases.append(dict(label=label, batch=len(ids), tokens=ids.shape[1], start=start,
                          implementations=list(selected), output_sha256=digest, exact=True))
        return expected
    try:
        compare('entire-vocabulary-in-sequence', np.arange(len(tokenizer)))
        for length in [1, 2, 3, 4, 7, 31, 129, 4096]:
            for batch in [1, 3]:
                ids = rng.integers(0, len(tokenizer), (batch, length), dtype=np.int64)
                compare(f'random-{batch}x{length}-unmasked', ids)
                mask = rng.integers(0, 2, ids.shape, dtype=np.uint8).astype(bool)
                compare(f'random-{batch}x{length}-image-boundaries', ids, mask)
        for bits in range(16):
            ids = rng.integers(0, len(tokenizer), (1, 4), dtype=np.int64)
            mask = [[bool(bits & (1 << shift)) for shift in range(4)]]
            compare(f'all-four-token-masks-{bits}', ids, mask)
        ids = rng.integers(0, len(tokenizer), (3, 17), dtype=np.int64)
        mask = rng.integers(0, 2, ids.shape, dtype=np.uint8).astype(bool)
        whole = compare('split-reference', ids, mask)
        for split in range(1, 17):
            first = compare(f'split-{split}-prefill', ids[:, :split], mask[:, :split])
            second = compare(f'split-{split}-decode', ids[:, split:], mask[:, split:], split)
            assert np.array_equal(np.concatenate([first, second], axis=1), whole)
        ids = rng.integers(0, len(tokenizer), (3, 512), dtype=np.int64)
        mask = rng.integers(0, 11, ids.shape) != 0
        whole = compare('variable-chunks-reference', ids, mask)
        chunks, pos = [], 0
        while pos < ids.shape[1]:
            size = min(int(rng.integers(1, 37)), ids.shape[1] - pos)
            chunks.append(compare(f'variable-chunks-{pos}', ids[:, pos:pos + size], mask[:, pos:pos + size], pos))
            pos += size
        assert np.array_equal(np.concatenate(chunks, axis=1), whole)

        # Branches start from the same image-containing prefix and then diverge.
        prefix = rng.integers(0, len(tokenizer), (3, 25), dtype=np.int64)
        prefix_mask = np.ones(prefix.shape, dtype=bool)
        prefix_mask[:, 22] = False
        compare('fork-prefix', prefix, prefix_mask)
        fork_reference = copy.deepcopy(reference)
        clones = {mode: [lib.deepseek_v41_hash_clone(h) for h in values] for mode, values in states.items()}
        assert all(h for values in clones.values() for h in values)
        handles.extend(h for values in clones.values() for h in values)
        branch_a = rng.integers(0, len(tokenizer), (3, 12), dtype=np.int64)
        branch_b = rng.integers(0, len(tokenizer), (3, 5), dtype=np.int64)
        compare('fork-original', branch_a, start=25)
        compare('fork-clone', branch_b, start=25, ref=fork_reference, selected=clones)
        for values in states.values():
            for handle in values: assert lib.deepseek_v41_hash_rewind(handle, 25) == 0
        replay = compare('rollback-and-replay', branch_b, start=25)
        replay_reference = fork_reference(torch.from_numpy(branch_b), 25).numpy().astype(np.uint64)
        assert np.array_equal(replay, replay_reference)
        # Position-zero replacement exercises request reuse without a full cache clear.
        compare('request-reset-after-longer-sequence', np.array([[2, 129264, 0], [1, 2, 3], [4, 5, 6]]),
                [[True, False, True], [True, True, True], [False, False, True]])
        # Replacing an earlier suffix must shrink the valid length.
        compare('replace-populated-suffix', [[7], [8], [9]], start=1)

        invalid = []
        for mode in [0, 1]:
            handle = states[mode][0]
            for label, values, mask_value, start, capacity in [
                ('negative-token', [7, -1], None, 0, None), ('out-of-vocabulary', [7, len(tokenizer)], None, 0, None),
                ('invalid-masked-token', [7, len(tokenizer)], [1, 0], 0, None), ('invalid-mask', [7, 8], [1, 2], 0, None),
                ('gap-in-history', [7], None, 3, None), ('negative-start', [7], None, -1, None),
                ('short-output', [7], None, 0, 47), ('sequence-overrun', [7] * (args.max_seq_len + 1), None, 0, None)]:
                before = lib.deepseek_v41_hash_length(handle)
                status, output = forward(handle, values, mask_value, start, capacity)
                assert status == -1 and np.all(output == 0xdecafbad12345678)
                assert lib.deepseek_v41_hash_length(handle) == before
                invalid.append(dict(mode=mode, case=label, unchanged=True))
            assert lib.deepseek_v41_hash_rewind(handle, -1) == -1
            assert lib.deepseek_v41_hash_rewind(handle, 3) == -1
            status, _ = forward(handle, [], None, 0)
            assert status == 0 and lib.deepseek_v41_hash_length(handle) == 2
        # Rejected calls must preserve history bytes as well as its length.
        compare('state-after-rejected-calls', [[11], [12], [13]], start=2)
        bad_map = token_map.copy(); bad_map[-1] = 99092
        bad_mul = multipliers.copy(); bad_mul[0] = 2
        overflow_mul = multipliers.copy(); overflow_mul[0] = (1 << 64) - 1
        bad_rows = table_rows.copy(); bad_rows[0] += 1
        bad_moduli = moduli.copy(); bad_moduli[0] = 1
        rejected_configs = [dict(token_map=bad_map), dict(multipliers=bad_mul), dict(multipliers=overflow_mul),
            dict(rows=bad_rows), dict(moduli=bad_moduli), dict(pad=-1), dict(pad=len(tokenizer)),
            dict(max_sequence=0), dict(compressed=1), dict(vocabulary=0), dict(mode=2)]
        for values in rejected_configs: assert not create(0, **values)
        assert lib.deepseek_v41_hash_length(None) == -1
        assert not lib.deepseek_v41_hash_clone(None) and lib.deepseek_v41_hash_rewind(None, 0) == -1

        # Independent Python arbitrary-precision division tests the reciprocal shortcut.
        arithmetic_cases = 0
        remainder = C.c_uint64()
        for modulus in sorted(set(moduli.tolist() + [2, 3, 5, 7, (1 << 32) - 1])):
            probes = [0, 1, modulus - 1, modulus, modulus + 1, modulus * modulus - 1, modulus * modulus,
                (1 << 63) - 1, 1 << 63, (1 << 64) - 1]
            probes += [int(x) for x in rng.integers(0, (1 << 64) - 1, 1024, dtype=np.uint64)]
            if modulus in [2, 3, int(moduli[0]), int(moduli[-1])]: probes += list(range(65536))
            for value in probes:
                assert lib.deepseek_v41_hash_remainder(value, modulus, C.byref(remainder)) == 0
                assert remainder.value == value % modulus
                arithmetic_cases += 1
        report = dict(passed=True, finished=time.time(), official_reference_executed_on='CPU',
            sources={str(p): sha(p) for p in [Path(__file__), SOURCE, CONFIG, SETUP / 'result.json',
                SETUP / 'tokenizer.json', SETUP / 'tokenizer_config.json', library]},
            versions={p: importlib.metadata.version(p) for p in ['numpy', 'sympy', 'tokenizers', 'torch']},
            original_vocabulary=len(tokenizer), compressed_vocabulary=99092, compressed_pad_id=reference.pad_id,
            token_map_bytes=len(map_bytes), token_map_sha256=sha(out / 'compressed-token-map.bin'),
            config_bytes=len(config_bytes), config_sha256=sha(out / 'hash-config.bin'),
            layer_ids=list(layout.layer_ids), moduli=moduli.reshape(2, 3, 8).tolist(),
            multipliers=multipliers.reshape(2, 4).tolist(), table_rows=table_rows.tolist(),
            exact_hash_values_compared=total_values, cases=cases, invalid_calls=invalid,
            rejected_configurations=len(rejected_configs), reciprocal_arithmetic_cases=arithmetic_cases,
            full_model_loaded=False, runtime_promoted=False,
            scope='Two native implementations match official CPU integer row IDs, including all vocabulary IDs, batches via independent handles, image masks, chunks, clones, rewinds, and request reuse. No projection/gate, full graph, model quality, or throughput claim.')
        destination.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(passed=True, exact_hash_values=total_values, sequence_cases=len(cases),
                              reciprocal_arithmetic_cases=arithmetic_cases, compressed_vocabulary=99092)), flush=True)
    finally:
        for handle in handles: lib.deepseek_v41_hash_destroy(handle)


if __name__ == '__main__':
    main()
