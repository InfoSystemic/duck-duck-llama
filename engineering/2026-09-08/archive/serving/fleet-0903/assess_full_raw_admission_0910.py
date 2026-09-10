#!/usr/bin/env python3
"""Refresh Full's raw-target admission estimate against the selected Flash process."""
import fcntl
import json
import os
from pathlib import Path
import time

from glm_flash_q8_trial import memory_status, node_memory_status
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json, port_available, unit_state
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/full-raw-admission-0910.json'


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    assets_path = BASE / 'results/full-bandwidth250-assets-0910/result.json'
    simulation_path = BASE / 'results/full-memory-plan-0910b/result.json'
    assets, simulation = [json.loads(p.read_text()) for p in [assets_path, simulation_path]]
    assert assets['passed'] and simulation['passed']
    assert all(sha256(path) == digest for path, digest in assets['libraries'].items())
    for item in assets['target_shards']:
        stat = Path(item['path']).stat()
        assert (stat.st_size, stat.st_ino, stat.st_dev, stat.st_mtime_ns) == (item['size'], item['inode'], item['device'], item['mtime_ns'])
    # The prior no-allocation simulation remains applicable only while its helper,
    # loader sources, configuration and exact linked libraries are unchanged.
    assert all(sha256(path) == digest for path, digest in simulation['source_sha256'].items())
    estimate = simulation['allocation_estimate']
    assert estimate['no_alloc'] and not estimate['load_mtp'] and estimate['context'] == 32768
    manager = Manager()
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        peer = manager.validate_current()
        ModelMeasurementGuard(peer['pid'], {peer['pid']: 18131}, inference_snapshot).assert_idle()
        assert set(inference_snapshot()) == {str(peer['pid'])}
        assert unit_state()['ActiveState'] == 'inactive' and port_available(18161)
        credits = {str(i): 0 for i in range(4)}
        for line in Path(f'/proc/{peer["pid"]}/numa_maps').read_text().splitlines():
            fields = line.split()
            if any(f.startswith('anon=') for f in fields) and not any(f.startswith('file=') for f in fields):
                for field in fields:
                    if field.startswith(tuple('N' + str(i) + '=' for i in range(4))):
                        key, value = field.split('=')
                        credits[key[1:]] += int(value) * os.sysconf('SC_PAGE_SIZE')
        memory, nodes = memory_status(), node_memory_status()
        projected_global = memory['MemAvailable'] + sum(credits.values())
        projected_nodes = {i: nodes['node' + i]['estimated_available'] + credits[i] for i in credits}
        global_required = estimate['total_bytes'] + (32 << 30)
        distributed = sum(row['model_bytes'] + row['context_bytes'] + row['compute_bytes']
                          for row in estimate['buffers'] if row['name'].startswith('Meta('))
        host = estimate['total_bytes'] - distributed
        # Allow the entire host allocation on any one node, in addition to that
        # node's quarter of distributed storage and an 8 GiB running reserve.
        node_required = (distributed + 3) // 4 + host + (8 << 30)
        result = dict(passed=True, finished=time.time(), peer_pid=peer['pid'], peer_start=peer['info']['start'],
            source_sha256={str(p): sha256(p) for p in [Path(__file__), assets_path, simulation_path]},
            target_shards=assets['target_shards'], allocation_estimate=estimate, memory=memory, nodes=nodes,
            anonymous_credit_per_node=credits, projected_global_available=projected_global,
            projected_node_available=projected_nodes, global_admission_bytes=global_required,
            node_admission_bytes=node_required,
            estimated_admission_passes=projected_global > global_required and all(v > node_required for v in projected_nodes.values()),
            full_payload_hashes_verified=False, full_model_loaded=False, selected_flash_preserved=True,
            limitations=['Projection credits only anonymous, non-file-backed Flash mappings. Actual post-stop admission must be rechecked.',
                'The frozen no-allocation simulation is an allocation estimate, not peak residency; live reserve guards remain necessary.',
                'File identities and historical engine hashes are verified. Full 467 GB payload hashes are not recomputed.',
                'Raw target only, context 32768, Q8 KV, 15 workers per socket. No MTP model is included.'])
        atomic_json(OUT, result)
        print(json.dumps({k: result[k] for k in ['passed', 'estimated_admission_passes', 'projected_global_available',
            'projected_node_available', 'global_admission_bytes', 'node_admission_bytes', 'selected_flash_preserved']}))


if __name__ == '__main__':
    os.umask(0o077)
    main()
