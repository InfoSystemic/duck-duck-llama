#!/usr/bin/env python3
"""Qualify packed Q6 batching with distinct activations and a CPU gate per case."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import time

from benchmark_qwen_q6 import host_cpu, wait_background
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parent.parent / 'engines/llama.cpp-q4e-goal-0904'
PEER = 1219506


def aggregate_cpu_seconds():
    values = [int(value) for value in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
    return (sum(values[:3]) + sum(values[5:8])) / os.sysconf('SC_CLK_TCK')


def children_cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    args = parser.parse_args()
    assert re.fullmatch(r'qwen-q6-packed-batch-component-[A-Za-z0-9_-]+',args.label)
    out = BASE / 'results' / args.label
    out.mkdir(exist_ok=False)
    proof_path = BASE / 'results/qwen-q6-packed-batch-proof-0908/result.json'
    proof = json.loads(proof_path.read_text())
    assert proof['passed'] and all(sha256(path) == digest for path,digest in proof['source_sha256'].items())
    cpu = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/libggml-cpu.so.0.22.0'
    source = BASE / 'benchmark-qwen-q6-packed-batch-0908.cpp'
    command = list(proof['compile_command'])
    command[command.index(str(BASE / 'check-qwen-q6-packed-batch-0908.cpp'))] = str(source)
    command[command.index('-o')+1] = str(out / 'benchmark')
    command.insert(1,'-DNDEBUG')
    inputs = [Path(__file__).resolve(),source,BASE / 'qwen-q6-packed-batch-0908.h',proof_path,cpu,
              BASE / 'benchmark_qwen_q6.py',BASE / 'model_measurement_guard.py',BASE / 'qwen_split_trial.py',
              BASE / 'inference_contention_guard.py']
    result = dict(started=time.time(),controller_pid=os.getpid(),passed=False,cases=[],compile_command=command,
                  input_sha256={str(path):sha256(path) for path in inputs},
                  scope='Single reserved-core component comparison, 2560-wide Q6 tiles, distinct activation rows and shuffled matrix order. No model tok/s or IMC utilization claim.')
    owned = None

    def save(stage):
        result['stage'] = stage
        (out / 'result.json').write_text(json.dumps(result,indent=2)+'\n')

    def interrupt(signum,frame):
        raise InterruptedError('Release only the owned component benchmark')

    def gate_timeout(signum,frame):
        raise TimeoutError('No quiet CPU window within 300 seconds; model service preserved')

    signal.signal(signal.SIGALRM,gate_timeout)
    for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        signal.signal(signum,interrupt)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save('compiling')
        try:
            with (out / 'compile.log').open('w') as log:
                compiled = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            result['compile_exit'] = compiled.returncode
            assert compiled.returncode == 0
            for path in inputs[:3]:
                (out / path.name).write_bytes(path.read_bytes())
            peer = process_info(PEER)
            assert peer['start'] == '103969952' and set(inference_snapshot()) == {str(PEER)}
            result['peer'] = peer
            guard = ModelMeasurementGuard(PEER,{PEER:18095},inference_snapshot)
            result['idle_gate'] = guard.wait_idle(out / 'idle.json',quiet_seconds=30)
            result['l3_size'] = Path('/sys/devices/system/cpu/cpu127/cache/index3/size').read_text().strip()
            environment = dict(os.environ,LD_LIBRARY_PATH=str(cpu.parent)+':'+str(ENGINE / 'validated-iq-batch3-bin'))
            # 64/32 rows match the full and tail tiles in the fused expert path.
            cases = [(32,512,2)]
            for rows,matrices,activations in cases:
                case = dict(rows=rows,matrices=matrices,activations=activations,attempts=[])
                result['cases'].append(case)
                for attempt in range(2):
                    name = f'{rows}-{matrices}-{activations}-attempt{attempt+1}'
                    save('waiting for CPU gate: '+name)
                    signal.alarm(300)
                    try:
                        gated = wait_background(guard,PEER,4,out / (name+'-background.json'))
                    finally:
                        signal.alarm(0)
                    guard.assert_idle()
                    before = host_cpu()
                    cpu_before = aggregate_cpu_seconds()
                    child_before = children_cpu_seconds()
                    started = time.monotonic()
                    save('measuring '+name)
                    with (out / (name+'.log')).open('w') as log:
                        owned = subprocess.Popen(['taskset','-c','0-127',str(out / 'benchmark'),str(rows),str(matrices),str(activations),'1.5'],
                                                 env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        while owned.poll() is None:
                            guard.assert_idle()
                            time.sleep(.2)
                    assert owned.returncode == 0, (name,owned.returncode)
                    elapsed = time.monotonic() - started
                    aggregate_background = (aggregate_cpu_seconds() - cpu_before - (children_cpu_seconds() - child_before)) / elapsed
                    after = host_cpu()
                    loads = []
                    for pid,value in after.items():
                        old = before.get(pid)
                        if pid == owned.pid or old is None or old[1] != value[1]:
                            continue
                        cores = (value[0]-old[0])/os.sysconf('SC_CLK_TCK')/elapsed
                        if cores > 0:
                            loads.append(dict(pid=pid,name=value[2],cores=cores))
                    measured = json.loads((out / (name+'.log')).read_text())
                    assert measured['activations_per_matrix'] == activations and measured['shuffled_matrix_order']
                    assert measured['output_equality_checked'] and measured['cpu'] == 127
                    assert measured['weight_storage_unchanged'] and Path(measured['cpu_library']).resolve() == cpu.resolve()
                    native = [row['matrices_per_second'] for row in measured['runs'] if row['variant']=='native']
                    packed_batch = [row['matrices_per_second'] for row in measured['runs'] if row['variant']=='packed_batch']
                    assert len(native) == len(packed_batch) == 2
                    process_background = sum(row['cores'] for row in loads)
                    cores = max(process_background,aggregate_background)
                    measured.update(background_gate=gated,other_host_cores=cores,
                                    process_background_cores=process_background,aggregate_background_cores=aggregate_background,
                                    largest_background=sorted(loads,key=lambda row:-row['cores'])[:6],
                                    packed_batch_speed_ratio=sum(packed_batch)/sum(native),
                                    background_within_gate=cores <= 4)
                    case['attempts'].append(measured)
                    print(json.dumps(dict(case=name,other_host_cores=cores,speed_ratio=measured['packed_batch_speed_ratio'],valid=cores<=4)),flush=True)
                    save('case measured: '+name)
                    if cores <= 4:
                        case['accepted_attempt'] = attempt
                        break
            assert all(sha256(path) == digest for path,digest in result['input_sha256'].items())
            result['passed'] = all('accepted_attempt' in case for case in result['cases'])
            result['binary_sha256'] = sha256(out / 'benchmark')
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                owned.terminate()
                owned.wait(timeout=10)
            result['finished'] = time.time()
            try:
                result['peer_preserved'] = process_info(PEER)['start'] == '103969952'
                result['peer_after'] = read_service(18095)
            except Exception as error:
                result['peer_preserved'] = False
                result['peer_error'] = repr(error)
            save('finished; owned component released')
            print(json.dumps(dict(passed=result['passed'],cases=len(result['cases']),error=result.get('error'))),flush=True)


if __name__ == '__main__':
    main()
