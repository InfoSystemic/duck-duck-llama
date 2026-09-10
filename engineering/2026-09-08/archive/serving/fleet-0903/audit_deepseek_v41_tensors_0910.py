#!/usr/bin/env python3
"""Audit pinned public safetensors headers without downloading full model shards."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import urllib.request

from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent
INTAKE = BASE / 'results/deepseek-v41-intake-0910'
OUT = BASE / 'results/deepseek-v41-tensor-audit-0910'


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def read_range(url, start, end, total):
    # A separate cache key prevents an intermediary reusing a different range.
    request = urllib.request.Request(
        url + f'?download=true&range_start={start}&range_end={end}',
        headers={'Range': f'bytes={start}-{end}',
                 'User-Agent': 'llama-llama-duck-v41-tensor-audit'},
    )
    assert 0 <= start <= end < total and end - start < 8 * 1024 * 1024
    with urllib.request.urlopen(request, timeout=30) as response:
        # Abort before reading if the server ignores Range. Never save signed URLs.
        assert response.status == 206, ('range unsupported', response.status)
        assert response.headers['Content-Range'] == f'bytes {start}-{end}/{total}'
        raw = response.read(end - start + 2)
    assert len(raw) == end - start + 1
    return raw


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    meta = json.loads((INTAKE / 'model-metadata.json').read_text())
    index = json.loads((INTAKE / 'official/model.safetensors.index.json').read_text())
    sources = json.loads((INTAKE / 'sources.json').read_text())
    for source in sources:
        assert sha256(source['file']) == source['sha256']
    revision = meta['sha']
    manager = Manager()
    peer = manager.validate_current()
    guard = ModelMeasurementGuard(peer['pid'], {peer['pid']: 18131}, inference_snapshot)
    result = dict(started=time.time(), passed=False, revision=revision,
                  model_id=meta['id'], peer_pid=peer['pid'], shards=[], samples=[],
                  input_sha256={str(Path(__file__)): sha256(__file__),
                                str(INTAKE / 'sources.json'): sha256(INTAKE / 'sources.json')},
                  full_weights_downloaded=False, model_loaded=False)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir()
        (OUT / 'headers').mkdir()
        (OUT / 'samples').mkdir()
        all_tensors = {}
        try:
            for sibling in meta['siblings']:
                name = sibling['rfilename']
                if not name.endswith('.safetensors'):
                    continue
                guard.assert_idle()
                url = f'https://huggingface.co/{meta["id"]}/resolve/{revision}/{name}'
                size = sibling['size']
                length, = struct.unpack('<Q', read_range(url, 0, 7, size))
                assert 2 <= length < 8 * 1024 * 1024
                header_raw = read_range(url, 8, 7 + length, size)
                header = json.loads(header_raw)
                tensors = {k: v for k, v in header.items() if k != '__metadata__'}
                intervals = []
                for tensor, entry in tensors.items():
                    assert tensor not in all_tensors and index['weight_map'][tensor] == name
                    start, end = entry['data_offsets']
                    assert 0 <= start < end <= size - length - 8
                    intervals.append((start, end))
                    all_tensors[tensor] = dict(entry, shard=name)
                intervals.sort()
                assert intervals[0][0] == 0 and intervals[-1][1] == size - length - 8
                assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
                path = OUT / 'headers' / (name + '.json')
                path.write_bytes(header_raw)
                result['shards'].append(dict(file=name, source_url=url, size=size,
                    publisher_sha256=sibling['lfs']['sha256'], header_bytes=length,
                    header_sha256=sha256(path), tensors=len(tensors)))
                write_json(OUT / 'result.json', result)
                if len(result['shards']) % 8 == 0:
                    print(json.dumps({'headers': len(result['shards']), 'tensors': len(all_tensors)}), flush=True)
            assert set(all_tensors) == set(index['weight_map'])
            counts = {}
            payload_bytes = 0
            for name, tensor in all_tensors.items():
                size = tensor['data_offsets'][1] - tensor['data_offsets'][0]
                payload_bytes += size
                family = ('engram_tables' if '.engram.embed.' in name else
                          'engram_other' if '.engram.' in name else
                          'draft' if name.startswith('mtp.') else
                          'routed_experts' if '.ffn.experts.' in name else
                          'vision' if name.startswith(('vision.', 'aligner.', 'image_')) else
                          'other_backbone')
                group = counts.setdefault(family, {'bytes': 0, 'tensors': 0, 'dtypes': {}})
                group['bytes'] += size
                group['tensors'] += 1
                group['dtypes'][tensor['dtype']] = group['dtypes'].get(tensor['dtype'], 0) + size
            assert payload_bytes == index['metadata']['total_size']
            # Small real rows support bit-preserving format and decode checks.
            for name in ['layers.0.ffn.experts.0.w1.weight', 'layers.0.ffn.experts.0.w1.scale',
                         'layers.0.ffn.experts.0.w2.weight', 'layers.0.ffn.experts.0.w2.scale',
                         'layers.1.engram.embed.weight', 'layers.1.engram.embed.scale',
                         'layers.14.engram.embed.weight', 'layers.14.engram.embed.scale']:
                guard.assert_idle()
                tensor = all_tensors[name]
                shard = next(v for v in result['shards'] if v['file'] == tensor['shard'])
                rows = tensor['shape'][0]
                row_bytes = (tensor['data_offsets'][1] - tensor['data_offsets'][0]) // rows
                assert row_bytes * rows == tensor['data_offsets'][1] - tensor['data_offsets'][0]
                for start_row in [0, rows // 2]:
                    n_rows = 32 if '.engram.embed.' in name else 4
                    offset = 8 + shard['header_bytes'] + tensor['data_offsets'][0] + start_row * row_bytes
                    raw = read_range(shard['source_url'], offset, offset + n_rows * row_bytes - 1, shard['size'])
                    sample = OUT / 'samples' / f'{name}-row{start_row}.bin'
                    sample.write_bytes(raw)
                    result['samples'].append(dict(tensor=name, dtype=tensor['dtype'], shape=tensor['shape'],
                        first_row=start_row, rows=n_rows, row_bytes=row_bytes, file=str(sample),
                        sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
            result.update(passed=True, tensor_count=len(all_tensors), payload_bytes=payload_bytes,
                shard_bytes=sum(s['size'] for s in result['shards']), families=counts,
                retrieved_bytes=sum(s['header_bytes'] + 8 for s in result['shards']) + sum(s['bytes'] for s in result['samples']))
            assert sha256(__file__) == result['input_sha256'][str(Path(__file__))]
            manager.validate_current()
            result['peer_preserved'] = True
        except BaseException as error:
            # URL exceptions can contain signed redirects; publish only the error class.
            result['error_type'] = type(error).__name__
            raise RuntimeError('Tensor audit failed: ' + type(error).__name__) from None
        finally:
            result['finished'] = time.time()
            write_json(OUT / 'result.json', result)
        print(json.dumps({k: result[k] for k in ['passed', 'tensor_count', 'payload_bytes', 'families', 'retrieved_bytes']}), flush=True)


if __name__ == '__main__':
    os.umask(0o077)
    main()
