#!/usr/bin/env python3
"""Measure an already loaded model; never signal or reconfigure its server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request

from dram_bandwidth import PerfDramRecorder, summarize_samples
from guarded_inference_request import stream_completion
from inference_contention_guard import activity
from model_measurement_guard import ModelMeasurementGuard

p = argparse.ArgumentParser()
p.add_argument('label')
p.add_argument('--port',type=int,required=True)
p.add_argument('--pid',type=int,required=True)
p.add_argument('--alias',required=True)
p.add_argument('--drafts',default='0,2')
p.add_argument('--tokens',type=int,default=512)
p.add_argument('--request-timeout-seconds',type=float,default=600)
p.add_argument('--bandwidth-capacity-gb-s',type=float,default=380)
p.add_argument('--bandwidth-target-gb-s',type=float,default=285)
p.add_argument('--chat-template-kwargs',type=json.loads,default={'enable_thinking':False})
p.add_argument('--reasoning-budget-tokens',type=int,default=None)
p.add_argument('--check-reasoning-budget-tokens',type=int,default=None)
p.add_argument('--allowed-idle-pids',default='4005448',help='Other loaded inference PIDs permitted only while idle')
p.add_argument('--skip-idle-gate',action='store_true',help='Caller already gated and owns an isolated Flash server')
a = p.parse_args()
assert 0 < a.bandwidth_target_gb_s <= a.bandwidth_capacity_gb_s
assert isinstance(a.chat_template_kwargs,dict)
assert a.reasoning_budget_tokens is None or a.reasoning_budget_tokens >= 0
assert a.check_reasoning_budget_tokens is None or a.check_reasoning_budget_tokens >= 0
assert re.fullmatch(r'[A-Za-z0-9_-]+', a.label)
base = Path(__file__).resolve().parent
out = base/'results'/a.label
out.mkdir(exist_ok=False)
input_paths=[Path(__file__).resolve(),base/'model_measurement_guard.py',base/'guarded_inference_request.py',
             base/'dram_bandwidth.py',base/'inference_contention_guard.py']
input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
for path in input_paths:
    (out/path.name).write_bytes(path.read_bytes())
args = Path(f'/proc/{a.pid}/cmdline').read_bytes().split(b'\0')
ports=[int(args[i+1]) for i,x in enumerate(args[:-1]) if x==b'--port']
assert ports and ports[-1] == a.port
assert 'llama-server' in args[0].decode()
url = f'http://127.0.0.1:{a.port}/'

def request(path,payload=None,timeout=600):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url+path,data=body,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.load(response)

def cpu_snapshot(inference_only=False):
    result = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            stat = (proc/'stat').read_text(); name = stat[stat.find('(')+1:stat.rfind(')')]
            if inference_only and name != 'llama-server' and not name.startswith(('glm-mtp-head','qwen-mtp-head')): continue
            fields = stat[stat.rfind(')')+2:].split()
            result[proc.name] = (int(fields[11])+int(fields[12]),fields[19],name)
        except (OSError,ValueError,IndexError): pass
    return result

def queued_requests():
    with urllib.request.urlopen(url+'metrics',timeout=3) as response: text=response.read().decode()
    values = [float(line.split()[-1]) for line in text.splitlines()
              if not line.startswith('#') and 'requests_deferred' in line]
    assert len(values)==1, 'Missing queue metric'
    return values[0]

def check_idle():
    return guard.assert_idle()

ports_by_pid={str(a.pid):a.port}
for pid in set(filter(None,a.allowed_idle_pids.split(',')))-{str(a.pid)}:
    peer_args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    peer_ports=[int(peer_args[i+1]) for i,arg in enumerate(peer_args[:-1]) if arg==b'--port']
    assert peer_ports, f'No port found for allowed peer {pid}'
    ports_by_pid[pid]=peer_ports[-1]
guard=ModelMeasurementGuard(a.pid,ports_by_pid,lambda:cpu_snapshot(True))
idle_status=None
if not a.skip_idle_gate:
    idle_status=guard.wait_idle(out/'waiting-for-idle.json')
check_idle()
runtime_env={}
for value in Path(f'/proc/{a.pid}/environ').read_bytes().split(b'\0'):
    if b'=' not in value:continue
    key,value=value.split(b'=',1)
    if key.startswith((b'GGML_',b'LLAMA_GRAPH_PHASE',b'LLAMA_MTP_',b'OMP_',b'GOMP_')) or key==b'LD_LIBRARY_PATH':
        runtime_env[key.decode()]=value.decode()
result = dict(config=vars(a),target_pid=a.pid,started=time.time(),checks=[],measurements=[],
              protected_server=True,bandwidth_target_gb_s=a.bandwidth_target_gb_s,
              bandwidth_capacity_gb_s=a.bandwidth_capacity_gb_s,
              server_command=[x.decode() for x in args if x],runtime_env=runtime_env,
              note='System-wide IMC traffic; background-subtracted values are attribution estimates.',
              input_sha256=input_sha256,idle_gate=idle_status,guarded_ports=ports_by_pid)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
prompts = [('prose','Explain how a refrigerator moves heat. Give a detailed explanation in plain English.'),
           ('code','Write a Python function that merges two sorted lists. Include an explanation of its time complexity.')]

def payload(prompt,draft,stream=False,limit=None):
    body = dict(model=a.alias,messages=[dict(role='user',content=prompt)],temperature=0,seed=42,
                max_tokens=limit or a.tokens,cache_prompt=False,stream=stream,
                chat_template_kwargs=a.chat_template_kwargs,
                **{'speculative.n_max':draft,'speculative.p_min':0})
    if a.reasoning_budget_tokens is not None:
        body['reasoning_budget_tokens'] = a.reasoning_budget_tokens
    return body

def stream_request(body,abort,on_chunk=None):
    guard.reset_activity()
    return stream_completion(a.port,body,guard.abort_reason,abort,on_chunk,interval=0.5,
                             max_elapsed=a.request_timeout_seconds)

try:
    for prompt,expected in [('What is 17 * 23? Reply with only the number.','391'),
                            ('Name the capital city of France. Reply with one word.','Paris')]:
        check_idle()
        short_payload=payload(prompt,0,stream=True,limit=512)
        if a.check_reasoning_budget_tokens is not None:
            short_payload['reasoning_budget_tokens'] = a.check_reasoning_budget_tokens
        check=dict(prompt=prompt,expected=expected,abort=[],chunks=[],request=short_payload)
        result['checks'].append(check)
        chunks=stream_request(short_payload,check['abort'],check['chunks'].append)
        content=''.join(c.get('delta',{}).get('content') or '' for x in chunks for c in x.get('choices',[]))
        passed=content.strip().rstrip('.')==expected
        check.update(chunks=chunks,content=content,pass_check=passed)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print('short check',passed,content[:80],flush=True)
        assert passed, 'Short correctness check failed'
    for draft in [int(n) for n in a.drafts.split(',')]:
        for kind,prompt in prompts:
            check_idle()
            directory=out/f'{kind}-draft{draft}'
            recorder=PerfDramRecorder(directory).start()
            entry=dict(kind=kind,draft_n=draft,prompt=prompt,started=time.time())
            result['measurements'].append(entry)
            chunks=[]; first=None; last=None; abort=[]
            before_cpu=cpu_snapshot(); before_inference=cpu_snapshot(True); cpu_started=time.monotonic()
            try:
                baseline_start=time.monotonic();guard.pause_idle(5);baseline_end=time.monotonic()
                check_idle()
                print('measuring',kind,'draft',draft,flush=True)
                def on_chunk(chunk):
                    global first,last
                    chunks.append(chunk)
                    for choice in chunk.get('choices',[]):
                        delta=choice.get('delta',{})
                        if delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning'):
                            now=time.monotonic()
                            if first is None: first=now
                            last=now
                stream_request(payload(prompt,draft,True),abort,on_chunk)
                assert not abort, abort
                assert first is not None and last-first>=5, 'Insufficient streamed generation window'
                after_start=time.monotonic();guard.pause_idle(5);after_end=time.monotonic()
            finally:
                samples,metadata=recorder.stop()
                (directory/'chunks.json').write_text(json.dumps(chunks,indent=2)+'\n')
                entry.update(counter_metadata=metadata,abort=abort,first_content_monotonic=first,last_content_monotonic=last)
            assert metadata['exit_code']==0 and metadata['valid'], 'Invalid hardware counter capture'
            before=summarize_samples(samples,baseline_start,baseline_end)
            after=summarize_samples(samples,after_start,after_end)
            decode=summarize_samples(samples,first+0.5,last-0.5)
            assert before['valid'] and after['valid'] and decode['valid'] and decode['sampled_seconds']>=4
            background=max(before['total_gb_s'],after['total_gb_s'])
            net=max(0,decode['total_gb_s']-background)
            timings=[c['timings'] for c in chunks if c.get('timings')]
            assert timings, 'No authoritative generation timing returned by server'
            timings=timings[-1]
            assert draft != 0 or timings.get('draft_n',0)==0, 'Requested raw decode was not raw'
            assert draft == 0 or timings.get('draft_n',0)>0, 'Requested speculative decode produced no draft tokens'
            elapsed=time.monotonic()-cpu_started;after_cpu=cpu_snapshot()
            loads=[]
            for pid,v in after_cpu.items():
                old=before_cpu.get(pid)
                if pid==str(a.pid) or old is None or old[1]!=v[1]:continue
                percent=100*(v[0]-old[0])/os.sysconf('SC_CLK_TCK')/elapsed
                if percent>=5:loads.append(dict(pid=int(pid),name=v[2],cpu_percent=percent))
            other,churn=activity(before_inference,cpu_snapshot(True),elapsed,a.pid)
            answer=''.join(choice.get('delta',{}).get('content') or '' for chunk in chunks for choice in chunk.get('choices',[]))
            reasoning=''.join(choice.get('delta',{}).get('reasoning_content') or choice.get('delta',{}).get('reasoning') or '' for chunk in chunks for choice in chunk.get('choices',[]))
            finish_reasons=[x.get('finish_reason') for c in chunks for x in c.get('choices',[]) if x.get('finish_reason')]
            entry.update(baseline_before=before,baseline_after=after,decode=decode,
                         background_subtracted_gb_s=net,
                         background_subtracted_utilization=net/a.bandwidth_capacity_gb_s,
                         timings=timings,other_host_cpu=sorted(loads,key=lambda x:-x['cpu_percent'])[:20],
                         other_inference=other,inference_churn=churn,
                         finish_reasons=finish_reasons,answer_characters=len(answer),reasoning_characters=len(reasoning),
                         completed_answer=bool(answer.strip()) and finish_reasons==['stop'],
                         reasoning_budget_tokens=a.reasoning_budget_tokens,
                         finished=time.time())
            (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(kind=kind,draft=draft,tok_s=timings['predicted_per_second'],
                                  total_gb_s=decode['total_gb_s'],background_gb_s=background,
                                  adjusted_gb_s=net,sockets=decode['sockets'])),flush=True)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in input_sha256.items())
    result['input_integrity_verified']=True
except BaseException as e:
    result['error']=repr(e)
    raise
finally:
    result['finished']=time.time()
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
