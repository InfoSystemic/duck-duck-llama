#!/usr/bin/env python3
"""Inspect Full's existing assets and loader without starting a model."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from glm_flash_q8_trial import memory_status, node_memory_status
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import port_available, unit_state
from qwen_split_trial import inference_snapshot, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/full-bandwidth250-assets-0910'


def record(path):
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, inode=stat.st_ino, device=stat.st_dev, mtime_ns=stat.st_mtime_ns)


def main():
    assert os.sched_getaffinity(0) == {127}
    reference_path = BASE / 'results/glm53-full-current-bandwidth-0906/result.json'
    audit_path = reference_path.parent / 'validation-audit.json'
    reference, audit = [json.loads(path.read_text()) for path in (reference_path, audit_path)]
    assert reference['finished'] and reference['input_integrity_verified'] and not reference.get('error')
    assert audit['result_sha256'] == sha256(reference_path)
    assert all(sha256(path) == digest for path, digest in audit['verified_full_binary_sha256'].items())
    command = reference['server_command']
    binary = Path(command[0])
    model = Path(command[command.index('--model')+1])
    draft = Path(command[command.index('--spec-draft-model')+1])
    template = Path(command[command.index('--chat-template-file')+1])
    shards = sorted(model.parent.glob('GLM-5.3-UD-Q4_K_XL-*-of-00011.gguf'))
    assert len(shards) == 11 and shards[0] == model
    assert process_info(1219506)['start'] == '103969952' and set(inference_snapshot()) == {'1219506'}
    assert unit_state()['ActiveState'] == 'inactive' and port_available(18161)
    guard = ModelMeasurementGuard(1219506, {1219506: 18095}, inference_snapshot)
    environment = {key: value for key, value in os.environ.items()
                   if key not in runtime_environment(os.environ) and not key.startswith('LLAMA_ARG_') and key != 'LD_PRELOAD'}
    environment.update(reference['runtime_env'], LD_LIBRARY_PATH=str(binary.parent))
    libraries = {str(path.resolve()): sha256(path) for path in binary.parent.glob('lib*.so*') if path.is_file()}
    paths = [Path(__file__).resolve(), reference_path, audit_path, binary, template,
             BASE / 'model_measurement_guard.py', BASE / 'qwen_split_trial.py',
             BASE / 'glm_flash_q8_trial.py', BASE / 'qwen_high_quant_trial.py']
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        result = dict(started=time.time(), passed=False, controller_pid=os.getpid(),
                      input_sha256={str(path): sha256(path) for path in paths},
                      reference_command=command, reference_runtime_env=reference['runtime_env'],
                      runtime_env={**reference['runtime_env'], 'LD_LIBRARY_PATH':str(binary.parent)},
                      libraries=libraries, target_shards=[record(path) for path in shards],
                      target_file_bytes=sum(path.stat().st_size for path in shards),
                      draft_present=draft.is_file(), draft=record(draft) if draft.is_file() else dict(path=str(draft)),
                      model_loaded=False, services_changed=False, steps=[])
        try:
            for name, args in (('devices', ['--list-devices']), ('help', ['--help']), ('version', ['--version'])):
                guard.assert_idle()
                checked = subprocess.run(['taskset', '-c', '0-127', str(binary), *args], env=environment,
                                         cwd=BASE, capture_output=True, text=True, timeout=60)
                text = checked.stdout+checked.stderr
                path = OUT / (name+'.log')
                path.write_text(text)
                assert checked.returncode == 0, name
                if name == 'devices':
                    assert all('CPU-NUMA'+str(i) in text for i in range(4))
                if name == 'help':
                    assert all(flag in text for flag in ('--spec-type', '--spec-draft-n-default', '--no-cache-prompt', '--list-devices'))
                result['steps'].append(dict(name=name, exit_code=0, log_sha256=sha256(path)))
            result['memory_available'] = memory_status()['MemAvailable']
            result['node_available'] = {key: row['estimated_available'] for key, row in node_memory_status().items()}
            status = Path('/proc/1219506/status').read_text().splitlines()
            result['peer_memory_kb'] = {line.split(':',1)[0]: int(line.split()[1]) for line in status
                                      if line.split(':',1)[0] in ('VmRSS', 'RssAnon', 'RssFile', 'RssShmem')}
            assert all(sha256(path) == digest for path, digest in result['input_sha256'].items())
            guard.assert_idle()
            result.update(passed=True, full_inactive=unit_state()['ActiveState'] == 'inactive',
                          peer_preserved=process_info(1219506)['start'] == '103969952',
                          full_payload_hashes_verified=False,
                          scope='The current Full binary exposes all four NUMA devices and required launch options without loading weights. Existing CPU/base library hashes match the historical audit. Model records are fresh file identities, not complete payload verification or model performance evidence.')
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            (OUT / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({key: result.get(key) for key in ('passed', 'error', 'target_file_bytes', 'draft_present',
                  'memory_available', 'node_available', 'peer_memory_kb', 'model_loaded', 'services_changed')}, indent=2))


if __name__ == '__main__':
    os.umask(0o077)
    main()
