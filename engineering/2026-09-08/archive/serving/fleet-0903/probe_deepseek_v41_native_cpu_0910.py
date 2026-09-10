#!/usr/bin/env python3
"""Validate native formats and time an isolated Engram CPU candidate."""
import ctypes
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import struct
import subprocess
import time

from benchmark_flash_q4_selected_0910 import background
from deepseek_v41_native_formats_0910 import repack_mxfp4_rows, unpack_mxfp4_rows
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent
AUDIT = BASE / 'results/deepseek-v41-tensor-audit-0910/result.json'
OUT = BASE / 'results/deepseek-v41-native-cpu-0910'


def fp8(code):
    magnitude, sign = code & 127, -1.0 if code & 128 else 1.0
    if magnitude == 127:
        return math.nan
    exponent, mantissa = magnitude >> 3, magnitude & 7
    return sign * (math.ldexp(mantissa, -9) if exponent == 0 else math.ldexp(8 + mantissa, exponent - 10))


def reference_bf16(code, scale):
    value = fp8(code) * (math.nan if scale == 255 else math.ldexp(1.0, scale - 127))
    if math.isnan(value):
        return 0x7fc0
    try:
        bits, = struct.unpack('<I', struct.pack('<f', value))
    except OverflowError:
        bits = 0xff800000 if math.copysign(1.0, value) < 0 else 0x7f800000
    return ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffffffff) >> 16


def validate_engram(library, audit):
    lib = ctypes.CDLL(str(library))
    gather = lib.deepseek_v41_engram_gather_bf16
    u8p, i64p, u16p = ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint16)
    gather.argtypes = [u8p, u8p, ctypes.c_int64, i64p, ctypes.c_int64, u16p, ctypes.c_int]
    gather.restype = ctypes.c_int
    assert lib.deepseek_v41_engram_has_avx512()
    def run(weights, scales, ids, mode):
        assert len(weights) % 256 == 0 and len(scales) * 32 == len(weights)
        w = (ctypes.c_uint8 * len(weights)).from_buffer_copy(weights)
        s = (ctypes.c_uint8 * len(scales)).from_buffer_copy(scales)
        indices = (ctypes.c_int64 * len(ids))(*ids)
        out = (ctypes.c_uint16 * (len(ids) * 256))(*([0x1234] * (len(ids) * 256)))
        status = gather(w, s, len(weights) // 256, indices, len(ids), out, mode)
        return status, list(out)
    exhaustive_weights = bytes(range(256)) * 256
    exhaustive_scales = b''.join(bytes([exponent]) * 8 for exponent in range(256))
    expected = [reference_bf16(code, scale) for scale in range(256) for code in range(256)]
    checks = []
    for mode in [0, 1]:
        status, values = run(exhaustive_weights, exhaustive_scales, list(range(256)), mode)
        assert status == 0 and values == expected
        checks.append(dict(mode=mode, kind='all E4M3 x E8M0 byte pairs', values=len(values), exact=True,
                           nan_policy='canonical quiet BF16 NaN; signs/payloads not preserved for NaNs'))
    weights, scales = bytearray(), bytearray()
    for item in audit['samples']:
        if not item['tensor'].endswith('engram.embed.weight'):
            continue
        scale = next(x for x in audit['samples'] if x['tensor'] == item['tensor'].replace('.weight', '.scale') and x['first_row'] == item['first_row'])
        weights.extend(Path(item['file']).read_bytes())
        scales.extend(Path(scale['file']).read_bytes())
    rows = len(weights) // 256
    # Real rows, reverse order, repeated IDs, and boundary IDs use the same path as random access.
    ids = list(range(rows - 1, -1, -1)) + [0, rows - 1, 0, rows // 2]
    expected = [reference_bf16(weights[row * 256 + col], scales[row * 8 + col // 32]) for row in ids for col in range(256)]
    assert all((value & 0x7f80) != 0x7f80 for value in expected), 'Nonfinite real sample'
    for mode in [0, 1]:
        status, values = run(weights, scales, ids, mode)
        assert status == 0 and values == expected
        checks.append(dict(mode=mode, kind='actual pinned Engram rows and duplicate/reversed IDs',
                           values=len(values), exact=True))
        for bad_ids in [[-1], [rows], [0, -1], [rows - 1, rows]]:
            status, values = run(weights, scales, bad_ids, mode)
            assert status == -1 and values == [0x1234] * (len(bad_ids) * 256)
        status, values = run(weights, scales, [], mode)
        assert status == 0 and values == []
    return dict(checks=checks, real_table_rows=rows, invalid_id_cases=8, empty_cases=2, passed=True)


def validate_fp4(audit):
    runtime = json.loads((BASE / 'results/qwen-get-rows-runtime-0909/result.json').read_text())
    lib = ctypes.CDLL(runtime['base'])
    dequant = lib.dequantize_row_mxfp4
    dequant.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    dequant.restype = None
    cases = []
    for item in audit['samples']:
        if '.ffn.experts.' not in item['tensor'] or not item['tensor'].endswith('.weight'):
            continue
        scale = next(x for x in audit['samples'] if x['tensor'] == item['tensor'].replace('.weight', '.scale') and x['first_row'] == item['first_row'])
        packed, scales = Path(item['file']).read_bytes(), Path(scale['file']).read_bytes()
        rows, cols = item['rows'], item['row_bytes'] * 2
        raw = repack_mxfp4_rows(packed, scales, rows, cols)
        assert unpack_mxfp4_rows(raw, rows, cols) == (packed, scales)
        output = (ctypes.c_float * (rows * cols))()
        data = ctypes.create_string_buffer(raw)
        dequant(data, output, rows * cols)
        fp4_values = [0, .5, 1, 1.5, 2, 3, 4, 6]
        expected = []
        for index in range(rows * cols):
            code = (packed[index // 2] >> (4 * (index % 2))) & 15
            value = math.ldexp(fp4_values[code & 7], scales[index // 32] - 127)
            expected.append(-value if code & 8 else value)
        assert list(output) == expected and all(math.isfinite(x) for x in expected)
        cases.append(dict(tensor=item['tensor'], first_row=item['first_row'], rows=rows, cols=cols,
                          values=rows * cols, packed_codes_and_scales_roundtrip_exact=True,
                          finite_values_match_ggml=True,
                          signed_zero_note='ggml numeric comparison treats positive/negative zero equally'))
    # Cover every possible packed byte at every nibble position, independent of published samples.
    for value in range(256):
        packed, scales = bytes([value]) * 16, bytes([value])
        assert unpack_mxfp4_rows(repack_mxfp4_rows(packed, scales, 1, 32), 1, 32) == (packed, scales)
    rejected = 0
    for args in [(b'', b'', 0, 32), (b'', b'', 1, 31), (b'', b'', 1, 32), (bytes(16), b'', 1, 32)]:
        try:
            repack_mxfp4_rows(*args)
        except ValueError:
            rejected += 1
    assert rejected == 4 and len(cases) == 4
    return dict(passed=True, samples=cases, exhaustive_packed_byte_cases=256, rejected_shapes=rejected,
                decoder_library=runtime['base'], decoder_sha256=sha256(runtime['base']))


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    audit = json.loads(AUDIT.read_text())
    assert audit['passed']
    for item in audit['samples']:
        assert sha256(item['file']) == item['sha256']
    manager = Manager()
    peer = manager.validate_current()
    guard = ModelMeasurementGuard(peer['pid'], {peer['pid']: 18131}, inference_snapshot)
    cpp = BASE / 'deepseek-v41-engram-cpu-0910.cpp'
    inputs = {str(p): sha256(p) for p in [Path(__file__), cpp, AUDIT,
        BASE / 'deepseek_v41_native_formats_0910.py', BASE / 'model_measurement_guard.py']}
    result = dict(started=time.time(), passed=False, input_sha256=inputs, peer_pid=peer['pid'],
                  model_loaded=False, runtime_promoted=False, steps=[], runs=[])
    owned = None
    def save(): atomic_json(OUT / 'result.json', result)
    def cancel(*_): raise InterruptedError('Stop the owned DeepSeek component only')
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]: signal.signal(sig, cancel)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir()
        def run(command, label):
            nonlocal owned
            guard.assert_idle()
            log = OUT / (label + '.log')
            with log.open('w') as handle:
                owned = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + 180
                while owned.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline
                    time.sleep(.25)
            result['steps'].append(dict(label=label, command=command, exit_code=owned.returncode,
                                        log_sha256=sha256(log)))
            save()
            assert owned.returncode == 0, (label, owned.returncode)
            return log.read_text()
        save()
        try:
            flags = ['c++', '-O3', '-std=c++17', '-march=x86-64-v3', '-ffp-contract=off', '-fno-fast-math', '-Wall', '-Wextra']
            library, binary = OUT / 'libdeepseek-v41-engram.so', OUT / 'deepseek-v41-engram-bench'
            run(flags + ['-fPIC', '-shared', str(cpp), '-o', str(library)], 'compile-library')
            run(flags + ['-DDEEPSEEK_V41_ENGRAM_BENCH', str(cpp), '-o', str(binary)], 'compile-benchmark')
            result['library_sha256'], result['binary_sha256'] = sha256(library), sha256(binary)
            result['engram_correctness'] = validate_engram(library, audit)
            result['fp4_correctness'] = validate_fp4(audit)
            save()
            print(json.dumps({'engram_exact': True, 'fp4_roundtrip_and_ggml_values': True}), flush=True)
            for rows in [1024, 1048576]:
                expected = None
                for index, mode in enumerate([0, 1, 1, 0]):
                    before = background(peer['pid'])
                    label = f'rows{rows}-arm{index}-mode{mode}'
                    log = run(['taskset', '-c', '48', str(binary), str(mode), str(rows)], label)
                    measurement = json.loads(log)
                    if expected is None: expected = measurement['checksum']
                    assert measurement['checksum'] == expected
                    result['runs'].append(dict(measurement, arm=index, background_before=before,
                                               background_after=background(peer['pid'])))
                    save()
                    print(json.dumps(measurement), flush=True)
            result['component_comparison'] = []
            for rows in [1024, 1048576]:
                group = [v for v in result['runs'] if v['table_rows'] == rows]
                scalar = statistics.mean(v['median_us'] for v in group if v['mode'] == 0)
                vector = statistics.mean(v['median_us'] for v in group if v['mode'] == 1)
                result['component_comparison'].append(dict(table_rows=rows, scalar_mean_median_us=scalar,
                    avx512_mean_median_us=vector, scalar_over_avx512=scalar / vector,
                    scope='single worker synthetic table; not model tok/s, full-table residency, or IMC bandwidth'))
            assert all(sha256(p) == digest for p, digest in inputs.items())
            manager.validate_current()
            result.update(passed=True, peer_preserved=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                try: owned.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(owned.pid, signal.SIGKILL); owned.wait(timeout=10)
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    os.umask(0o077)
    main()
