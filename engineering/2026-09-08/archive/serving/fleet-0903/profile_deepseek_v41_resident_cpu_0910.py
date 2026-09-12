#!/usr/bin/env python3
"""Sample CPU cycles inside cached decode on the selected DeepSeek endpoint."""
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import process_info,sha256,inference_snapshot
from model_measurement_guard import ModelMeasurementGuard
from select_flash_q4_0910c import Manager

BASE=Path(__file__).resolve().parent
OUT=BASE/'results/deepseek-v41-resident-cpu-profile-0910'


def main():
    assert os.sched_getaffinity(0)=={127} and not OUT.exists()
    selected=json.loads((BASE/'deepseek-v41-selected.json').read_text());pid=selected['pid']
    assert process_info(pid)['start']==selected['start']
    assert all(sha256(p)==h for p,h in selected['source_sha256'].items())
    peer=Manager().validate_current();guard=ModelMeasurementGuard(peer['pid'],{peer['pid']:18131},inference_snapshot)
    with urllib.request.urlopen('http://127.0.0.1:18170/health',timeout=5) as response:assert not json.load(response)['busy']
    request_record=Path(selected['request_record'])
    before=json.loads(request_record.read_text());guard.assert_idle();OUT.mkdir()
    result=dict(passed=False,started=time.time(),server_pid=pid,server_start=selected['start'],
        source_sha256=sha256(__file__),selected_sha256=sha256(BASE/'deepseek-v41-selected.json'),imc_bandwidth_measured=False)
    perf=None;perf_end=[];watcher=None;first=None;last=None;chunks=[];stop=threading.Event();failures=[]
    def cancel(*_):raise InterruptedError('Cancel only the owned DeepSeek CPU profile')
    for s in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(s,cancel)
    def monitor():
        while not stop.wait(.5):
            try:guard.assert_idle()
            except Exception as e:failures.append(type(e).__name__);return
    threading.Thread(target=monitor,daemon=True).start()
    try:
        command=['sudo','-n','perf','record','--per-thread','--no-inherit','--clockid','mono','--timestamp','--sample-cpu',
            '-F','199','-e','cycles:u','-p',str(pid),'-o',str(OUT/'perf.data'),'--','sleep','5']
        payload=dict(model='DeepSeek-V4.1-Flash',messages=[dict(role='user',content='Hi.')],temperature=0,max_tokens=16,stream=True)
        request=urllib.request.Request(selected['endpoint']+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        done=False
        with (OUT/'record.log').open('w') as log,urllib.request.urlopen(request,timeout=120) as response:
            assert response.status==200
            for raw in response:
                assert not failures,failures
                if not raw.startswith(b'data: '):continue
                raw=raw[6:].strip()
                if raw==b'[DONE]':done=True;break
                chunk=json.loads(raw);assert 'error' not in chunk;chunks.append(chunk)
                delta=chunk['choices'][0]['delta'].get('content','')
                if delta:
                    last=time.monotonic()
                    if first is None:
                        first=last
                        perf=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        result['owned_perf_pid']=perf.pid;result['command']=command
                        def collect():perf_end.append((perf.wait(),time.monotonic()))
                        watcher=threading.Thread(target=collect);watcher.start()
                        print(json.dumps(dict(profiling_cached_decode=pid,perf_pid=perf.pid)),flush=True)
        assert done and ''.join(c['choices'][0]['delta'].get('content','') for c in chunks)=='Hello! How can I help you today?'
        assert watcher is not None;watcher.join(timeout=30);assert not watcher.is_alive() and perf_end[0][0]==0
        after=json.loads(request_record.read_text());assert after['completed']==before['completed']+1
        assert after['timings']['downloaded_bytes']==0 and after['usage']['completion_tokens']==10
        times=subprocess.run(['sudo','-n','perf','script','-G','-F','time','-i',str(OUT/'perf.data')],capture_output=True,text=True,timeout=30)
        assert times.returncode==0
        stamps=[float(l.strip().rstrip(':')) for l in times.stdout.splitlines() if l.strip()]
        assert stamps and first<=min(stamps)<=max(stamps)<=last,(first,min(stamps),max(stamps),last)
        report=subprocess.run(['sudo','-n','perf','report','--stdio','--no-children','--sort','dso,symbol',
            '--call-graph','none','--percent-limit','0.5','-i',str(OUT/'perf.data')],capture_output=True,text=True,timeout=60)
        assert report.returncode==0;(OUT/'report.txt').write_text(report.stdout+report.stderr)
        with (OUT/'samples.txt').open('w') as samples:
            decoded=subprocess.run(['sudo','-n','perf','script','-G','-F','pid,tid,cpu,time,period,ip,sym,dso',
                '-i',str(OUT/'perf.data')],stdout=samples,stderr=subprocess.PIPE,text=True,timeout=60)
        assert decoded.returncode==0
        assert process_info(pid)['start']==selected['start'] and not failures
        result.update(passed=True,server_preserved=True,response_matches=True,zero_downloads=True,request=after,
            first_content_monotonic=first,last_content_monotonic=last,profile_sample_min=min(stamps),profile_sample_max=max(stamps),
            samples=len(stamps),all_samples_inside_decode=True,report_sha256=sha256(OUT/'report.txt'),samples_sha256=sha256(OUT/'samples.txt'))
    except BaseException as e:result['error']=repr(e);raise
    finally:
        stop.set()
        if perf is not None and perf.poll() is None:
            subprocess.run(['sudo','-n','kill','-INT','--','-'+str(perf.pid)],check=True,timeout=5)
            if watcher:watcher.join(timeout=15)
            assert perf.poll() is not None
        result['finished']=time.time();atomic_json(OUT/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
