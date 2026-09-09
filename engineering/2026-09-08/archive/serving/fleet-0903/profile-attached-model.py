#!/usr/bin/env python3
"""Profile request-scoped decode on an idle existing server; never signal it."""
import argparse
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

from guarded_inference_request import stream_completion
from inference_contention_guard import activity, wait_for_idle

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
options=parser.parse_args()
assert options.sample_frequency>0
base=Path(__file__).resolve().parent
out=base/'results'/options.label
out.mkdir(exist_ok=False)
result=dict(config={**vars(options),'after':str(options.after)},started=time.time(),profiles=[])

def save():
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')

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

save()
try:
    while True:
        try:
            prior=json.loads(options.after.read_text())
            if 'finished' in prior:
                assert not prior.get('error'),'Preceding measurement failed'
                break
        except (FileNotFoundError,ValueError):pass
        time.sleep(2)
    args=Path(f'/proc/{options.pid}/cmdline').read_bytes().split(b'\0')
    ports=[int(args[i+1]) for i,x in enumerate(args[:-1]) if x==b'--port']
    assert ports and ports[-1]==options.port and b'llama-server' in args[0]
    result['server_command']=[x.decode() for x in args if x]
    allowed=tuple(set(options.allowed_idle_pids.split(','))|{str(options.pid)})
    wait_for_idle(snapshot,out/'waiting-for-idle.json',allowed_idle_pids=allowed)
    for draft in map(int,options.drafts.split(',')):
        assert not any(s['is_processing'] for s in json.loads(get('slots')))
        assert queued()==0
        directory=out/f'draft{draft}'
        directory.mkdir()
        entry=dict(draft_n=draft,started=time.time(),abort=[])
        result['profiles'].append(entry)
        previous=snapshot();previous_time=time.monotonic();busy_count=0
        def check_abort():
            global previous,previous_time,busy_count
            now=time.monotonic();current=snapshot()
            other,churn=activity(previous,current,now-previous_time,options.pid)
            busy_count=busy_count+1 if any(x['cpu_percent']>100 for x in other) else 0
            previous,previous_time=current,now
            reason=None
            if churn or busy_count>=2:reason='Other inference became active or changed'
            if queued()>0:reason='User request queued; releasing profile request'
            if reason:return dict(reason=reason,other_inference=other,churn=churn)
        perf=None;first=None;last=None;chunks=[];perf_started=None;perf_ended=[];perf_waiter=None
        if options.counter_events:
            command=['sudo','-n','perf','stat','--per-thread','--no-inherit','-x,','-e',options.counter_events,
                     '-p',str(options.pid),'--','sleep','6']
        else:
            command=['sudo','-n','perf','record','--per-thread','--no-inherit','--clockid','mono',
                     '--timestamp','--sample-cpu','-F',str(options.sample_frequency),'-e',options.record_event,
                     '-p',str(options.pid),'-o',str(directory/'perf.data'),'--','sleep','6']
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
                        perf=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                        if options.counter_events:
                            def collect_end(process=perf):
                                code=process.wait()
                                perf_ended.append((code,time.monotonic()))
                            perf_waiter=threading.Thread(target=collect_end,daemon=True)
                            perf_waiter.start()
                        print('profiling',options.alias,'draft',draft,flush=True)
            payload=dict(model=options.alias,messages=[dict(role='user',content='Explain how a refrigerator moves heat. Give a detailed explanation in plain English.')],
                         temperature=0,seed=42,max_tokens=256,cache_prompt=False,stream=True,
                         chat_template_kwargs={'enable_thinking':False},
                         **{'speculative.n_max':draft,'speculative.p_min':0})
            try:
                stream_completion(options.port,payload,check_abort,entry['abort'],on_chunk)
            finally:
                if perf is not None:
                    if options.counter_events:
                        perf_waiter.join(timeout=15)
                        assert not perf_waiter.is_alive(),'CPU counter collection did not finish'
                        entry['perf_exit']=perf_ended[0][0]
                        entry['collection_window_monotonic']=[perf_started,perf_ended[0][1]]
                    else:
                        entry['perf_exit']=perf.wait(timeout=15)
                (directory/'chunks.json').write_text(json.dumps(chunks,indent=2)+'\n')
                entry.update(first_content_monotonic=first,last_content_monotonic=last,command=command)
                save()
        assert first is not None and last-first>=8,'Profile was not enclosed by a sufficiently long decode window'
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
            print(report.stdout[-10000:],flush=True)
        save()
except Exception as error:
    result['error']=repr(error)
    raise
finally:
    result['finished']=time.time()
    save()
