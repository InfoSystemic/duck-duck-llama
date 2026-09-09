#!/usr/bin/env python3
"""Profile the loaded Qwen Q6 configuration with queue-aware guards."""
import argparse
import hashlib
import os
import re
import signal
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

from guarded_inference_request import stream_completion
from inference_contention_guard import activity
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import process_info, process_environment, runtime_environment, identity_matches

parser=argparse.ArgumentParser()
parser.add_argument('label')
parser.add_argument('--pid',type=int,required=True)
parser.add_argument('--port',type=int,required=True)
parser.add_argument('--alias',required=True)
parser.add_argument('--drafts',default='2')
parser.add_argument('--allowed-idle-pids',default='4005448')
parser.add_argument('--after',type=Path,required=True)
parser.add_argument('--counter-events',help='Collect perf stat totals for these events instead of cycle samples')
parser.add_argument('--record-event',default='cycles:u',help='Event to sample when not collecting stat totals')
parser.add_argument('--sample-frequency',type=int,default=199)
parser.add_argument('--sample-addresses',action='store_true')
parser.add_argument('--tokens',type=int,default=512)
options=parser.parse_args()
assert re.fullmatch(r"[A-Za-z0-9_-]+", options.label)
assert options.port == 18095 and options.alias == 'qwen-q6-trial' and options.allowed_idle_pids == ''
assert options.sample_frequency == 199 and options.record_event == 'cycles:u'
assert not options.counter_events and not options.sample_addresses
assert 32 <= options.tokens <= 1024
base=Path(__file__).resolve().parent
out=base/'results'/options.label
out.mkdir(exist_ok=False)
result=dict(config={**vars(options),'after':str(options.after)},started=time.time(),profiles=[],completed=False,checks=[])

def save():
    temporary = out / 'result.json.tmp'
    temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(out / 'result.json')

def snapshot():
    state={}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            name=(proc/'comm').read_text().strip()
            if name!='llama-server' and not name.startswith(('glm-mtp-head','qwen-mtp-head')):continue
            fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
            args=(proc/'cmdline').read_bytes().split(b'\0')
            ports=[args[i+1].decode() for i,x in enumerate(args[:-1]) if x==b'--port']
            state[proc.name]=(int(fields[11])+int(fields[12]),fields[19],ports[-1] if ports else None)
        except (OSError,ValueError,IndexError):pass
    return state

def get(path):
    with urllib.request.urlopen(f'http://127.0.0.1:{options.port}/'+path,timeout=3) as response:
        return response.read().decode()

def queued():
    values=[float(line.split()[-1]) for line in get('metrics').splitlines()
            if not line.startswith('#') and 'requests_deferred' in line]
    assert len(values)==1,'Queue metric missing'
    return values[0]

def host_snapshot():
    values = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            stat = (proc / 'stat').read_text()
            fields = stat.rsplit(')', 1)[1].split()
            values[proc.name] = (int(fields[11]) + int(fields[12]), fields[19], stat[stat.find('(') + 1:stat.rfind(')')])
        except (OSError, ValueError, IndexError): pass
    return values


def thread_snapshot():
    values = {}
    for proc in Path(f'/proc/{options.pid}/task').iterdir():
        try:
            stat = (proc / 'stat').read_text()
            fields = stat.rsplit(')', 1)[1].split()
            values[proc.name] = dict(ticks=int(fields[11]) + int(fields[12]), start=fields[19],
                name=stat[stat.find('(') + 1:stat.rfind(')')], affinity=sorted(os.sched_getaffinity(int(proc.name))))
        except (OSError, ValueError, IndexError): pass
    return values


def signal_abort(signum, frame):
    raise InterruptedError(f'Received signal {signum}; release only this HTTP request')


for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(signum, signal_abort)
trial = json.loads((base / 'results/qwen-q6-trial-0907/state.json').read_text())
current = trial['current']
assert trial['full_stopped'] and not current['original']
assert current['pid'] == options.pid and str(current['drafts']) == options.drafts
plan = dict(protected={str(options.pid): dict(start_ticks=current['info']['start'], exe=current['info']['exe'])},
            original_command=current['command'], original_environment=current['runtime_env'])
plan_path = out / 'plan.json'
plan_path.write_text(json.dumps(plan, indent=2) + '\n')
result['protected_before'] = {}
for pid, expected in plan['protected'].items():
    info = process_info(int(pid))
    assert identity_matches(info, expected)
    result['protected_before'][pid] = dict(info=info, environment=runtime_environment(process_environment(int(pid))))
assert result['protected_before'][str(options.pid)]['info']['command'] == plan['original_command']
assert result['protected_before'][str(options.pid)]['environment'] == plan['original_environment']
source_paths = [Path(__file__).resolve(), base / 'model_measurement_guard.py', base / 'guarded_inference_request.py',
                base / 'qwen_split_trial.py', base / 'inference_contention_guard.py', plan_path]
result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
for p in source_paths:
    (out / p.name).write_bytes(p.read_bytes())
guard = ModelMeasurementGuard(options.pid, {str(options.pid): 18095}, snapshot)
result['note'] = 'Cycle profiles of the unchanged launched MTP configuration. Sampling can affect throughput; these are not independent bandwidth or completed-task speed benchmarks.'
save()
try:
    prior = json.loads(options.after.read_text())
    assert prior.get('finished') and not prior.get('error'), 'Required completed baseline is absent or invalid'
    result['baseline_sha256'] = hashlib.sha256(options.after.read_bytes()).hexdigest()
    result['effective_numa_threads'] = int(Path(current['runtime_env']['GGML_CPU_NUMA_THREADS_FILE']).read_text())
    args=Path(f'/proc/{options.pid}/cmdline').read_bytes().split(b'\0')
    ports=[int(args[i+1]) for i,x in enumerate(args[:-1]) if x==b'--port']
    assert ports and ports[-1]==options.port and b'llama-server' in args[0]
    result['server_command']=[x.decode() for x in args if x]
    allowed=tuple(set(options.allowed_idle_pids.split(','))|{str(options.pid)})
    result['idle_gate'] = guard.wait_idle(out/'waiting-for-idle.json')
    for prompt, expected in [('What is 17 * 23? Reply with only the number.', '391'),
                             ('Name the capital city of France. Reply with one word.', 'Paris')]:
        guard.assert_idle()
        guard.reset_activity()
        check = dict(prompt=prompt, expected=expected, abort=[])
        result['checks'].append(check)
        body = dict(model=options.alias, messages=[dict(role='user', content=prompt)], temperature=0, seed=42,
                    max_tokens=512, cache_prompt=False, stream=True, chat_template_kwargs={'enable_thinking':False})
        chunks = stream_completion(options.port, body, guard.abort_reason, check['abort'], interval=0.5)
        content = ''.join(c.get('delta', {}).get('content') or '' for x in chunks for c in x.get('choices', []))
        check.update(chunks=chunks, content=content, passed=content.strip().rstrip('.') == expected)
        save()
        assert not check['abort'] and check['passed']
        print('short check', check['passed'], content[:80], flush=True)
    prompts = [('prose', 'Explain how a refrigerator moves heat. Give a detailed explanation in plain English.'),
               ('code', 'Write a Python function that merges two sorted lists. Include an explanation of its time complexity.')]
    for kind, prompt in prompts[:1]:
        draft = int(options.drafts)
        guard.assert_idle()
        guard.reset_activity()
        directory=out/f'{kind}-draft{draft}'
        directory.mkdir()
        entry=dict(kind=kind, draft_n=draft, started=time.time(), abort=[])
        result['profiles'].append(entry)
        previous=snapshot(); previous_time=time.monotonic()
        def check_abort():
            return guard.abort_reason()
        perf=None;first=None;last=None;chunks=[];perf_started=None;perf_ended=[];perf_waiter=None
        if options.counter_events:
            command=['sudo','-n','perf','stat','--per-thread','--no-inherit','-x,','-e',options.counter_events,
                     '-p',str(options.pid),'--','sleep','8']
        else:
            command=['sudo','-n','perf','record','--per-thread','--no-inherit','--clockid','mono',
                     '--timestamp','--sample-cpu','-F',str(options.sample_frequency),'-e',options.record_event,
                     '-p',str(options.pid),'-o',str(directory/'perf.data'),'--','sleep','8']
            if options.sample_addresses:command.insert(command.index('--'),'-d')
        with (directory/'record.log').open('w') as log:
            def on_chunk(chunk):
                global perf,first,last,perf_started,perf_waiter
                chunks.append(chunk)
                if any(any(c.get('delta',{}).get(k) for k in ('content','reasoning','reasoning_content')) for c in chunk.get('choices',[])):
                    last=time.monotonic()
                    if first is None:
                        first=last
                        perf_started=time.monotonic()
                        entry['host_before'] = host_snapshot()
                        entry['threads_before'] = thread_snapshot()
                        entry['inference_before'] = snapshot()
                        perf=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        entry['owned_perf_pid'] = perf.pid
                        def collect_end(process=perf):
                            code=process.wait()
                            ended = time.monotonic()
                            entry['host_after'] = host_snapshot()
                            entry['threads_after'] = thread_snapshot()
                            entry['inference_after'] = snapshot()
                            perf_ended.append((code,ended))
                        perf_waiter=threading.Thread(target=collect_end,daemon=True)
                        perf_waiter.start()
                        print('profiling', options.alias, kind, 'launched MTP', flush=True)
            payload=dict(model=options.alias,messages=[dict(role='user',content=prompt)],
                         temperature=0,seed=42,max_tokens=options.tokens,cache_prompt=False,stream=True,
                         chat_template_kwargs={'enable_thinking':False})
            entry['payload'] = payload
            try:
                stream_completion(options.port,payload,check_abort,entry['abort'],on_chunk)
            finally:
                if perf is not None:
                    perf_waiter.join(timeout=15)
                    if perf_waiter.is_alive():
                        # This process group belongs only to this capture.
                        subprocess.run(['sudo', '-n', 'kill', '-INT', '--', '-' + str(perf.pid)], check=True, timeout=5)
                        perf_waiter.join(timeout=10)
                    assert not perf_waiter.is_alive(), 'Owned perf capture did not finish'
                    entry['perf_exit']=perf_ended[0][0]
                    entry['collection_window_monotonic']=[perf_started,perf_ended[0][1]]
                (directory/'chunks.json').write_text(json.dumps(chunks,indent=2)+'\n')
                entry.update(first_content_monotonic=first,last_content_monotonic=last,command=command)
                save()
        assert not entry['abort'], entry['abort']
        assert first is not None and last-first>=10,'Profile was not enclosed by a sufficiently long decode window'
        start, end = entry['collection_window_monotonic']
        assert first <= start <= end <= last
        duration = end - start
        loads, churn = activity(entry['host_before'], entry['host_after'], duration, options.pid)
        entry['other_host_cpu'] = sorted(loads, key=lambda x: -x['cpu_percent'])[:20]
        entry['host_process_churn'] = churn
        entry['other_inference'], entry['inference_churn'] = activity(entry['inference_before'], entry['inference_after'], duration, options.pid)
        assert not entry['inference_churn'] and not any(x['cpu_percent'] > 20 for x in entry['other_inference'])
        assert entry['perf_exit']==0
        if options.counter_events:
            start,end=entry['collection_window_monotonic']
            assert first<=start<=end<=last,'CPU counter collection escaped the observed decode window'
            entry['counter_log']=str(directory/'record.log')
            entry['counter_validation']='Pending event completeness and running-time analysis of the raw CSV'
        else:
            sample_times=subprocess.run(['sudo','-n','perf','script','-F','time',
                                         '-i',str(directory/'perf.data')],capture_output=True,text=True)
            (directory/'sample-time-errors.txt').write_text(sample_times.stderr)
            assert sample_times.returncode==0
            timestamps=[float(line.strip().rstrip(':')) for line in sample_times.stdout.splitlines() if line.strip()]
            assert timestamps and first<=min(timestamps)<=max(timestamps)<=last,'Event samples escaped the observed decode window'
            entry['sample_window_monotonic']=[min(timestamps),max(timestamps)]
        timings=[c['timings'] for c in chunks if c.get('timings')]
        assert timings
        entry['timings']=timings[-1]
        assert bool(timings[-1].get('draft_n',0))==bool(draft),'Unexpected speculative mode'
        if options.counter_events:
            entry['finished']=time.time()
            print('CPU counters collected',options.alias,'draft',draft,'events',options.counter_events,flush=True)
        else:
            report=subprocess.run(['sudo','-n','perf','report','--stdio','--no-children','--sort','dso,symbol',
                                   '--percent-limit','1','-i',str(directory/'perf.data')],capture_output=True,text=True)
            (directory/'report.txt').write_text(report.stdout+report.stderr)
            entry.update(report_exit=report.returncode,finished=time.time())
            assert report.returncode==0
            with (directory / 'samples-by-thread.txt').open('w') as samples_file:
                script = subprocess.run(['sudo', '-n', 'perf', 'script', '-F', 'pid,tid,cpu,time,period,ip,sym,dso',
                    '-i', str(directory / 'perf.data')], stdout=samples_file, stderr=subprocess.PIPE, text=True, timeout=60)
                assert script.returncode == 0, script.stderr
            entry['report_sha256'] = hashlib.sha256((directory / 'report.txt').read_bytes()).hexdigest()
            entry['samples_sha256'] = hashlib.sha256((directory / 'samples-by-thread.txt').read_bytes()).hexdigest()
            print(json.dumps(dict(kind=kind, profiled_tok_s=entry['timings']['predicted_per_second'], sample_window=entry['sample_window_monotonic'], other_host_cpu=entry['other_host_cpu'][:5])), flush=True)
        save()
    guard.assert_idle()
    result['protected_after'] = {}
    for pid, expected in result['protected_before'].items():
        info = process_info(int(pid))
        environment = runtime_environment(process_environment(int(pid)))
        assert identity_matches(info, plan['protected'][pid])
        assert info['command'] == expected['info']['command'] and info['affinity'] == expected['info']['affinity']
        assert environment == expected['environment']
        result['protected_after'][pid] = dict(info=info, environment=environment)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in result['source_sha256'].items())
    result['completed'] = True
except BaseException as error:
    result['error']=repr(error)
    raise
finally:
    result['finished']=time.time()
    save()
