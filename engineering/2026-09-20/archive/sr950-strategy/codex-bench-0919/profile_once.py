#!/usr/bin/env python3
"""One explicitly profiled request; its timings are not a speed benchmark."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import threading
import time
import urllib.request
from capture_runtime import snapshot

HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pid',type=int,required=True)
    ap.add_argument('--tag',required=True)
    ap.add_argument('--port',type=int,choices=(18131,18141),default=18141)
    args=ap.parse_args()
    state=snapshot(args.pid,args.port)
    assert state['health']=={'status':'ok'}
    assert not any(s['is_processing'] for s in state['slots'])
    arm=Path(state['environment']['GGML_CPU_OP_PROFILE_ARM_FILE'])
    assert arm==Path('/dev/shm/flash-optrace.arm')
    assert state['environment']['GGML_CPU_OP_PROFILE_COUNT']=='8'
    rng=random.Random(91)
    words='memory bandwidth socket kernel scheduler page node thread barrier expert router token cache state vector matrix'.split()
    prompt='Context glossary: '+' '.join(rng.choice(words) for _ in range(3900))+'\nNow write a Python LRU cache using OrderedDict. Explain its operations.'
    body={'prompt':prompt,'n_predict':96,'temperature':0,'seed':42,'cache_prompt':False,'stream':True}
    result={'pid':args.pid,'port':args.port,'profiled':True,'not_a_speed_benchmark':True,
            'started_at':datetime.now(timezone.utc).isoformat(),
            'runtime':state,'request':body}
    done=threading.Event()
    def heartbeat():
        start=time.monotonic()
        while not done.wait(30): print(f'{args.tag} operation profile active {time.monotonic()-start:.0f}s',flush=True)
    assert not arm.exists(), 'Another profiling session is already armed'
    armed=False
    chunks=[]
    started=time.monotonic()
    try:
        threading.Thread(target=heartbeat,daemon=True).start()
        req=urllib.request.Request(f'http://127.0.0.1:{args.port}/completion',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=600) as stream:
            for line in stream:
                if not line.startswith(b'data: '):
                    continue
                data=line[6:].strip()
                if data==b'[DONE]':
                    break
                item=json.loads(data)
                if item.get('error'):
                    raise RuntimeError(item['error'])
                chunks.append(item)
                if item.get('content') and not armed:
                    # Wait until prefill has finished, so the eight graphs are decode work.
                    fd=os.open(arm,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
                    os.close(fd)
                    armed=True
                    result['armed_after_seconds']=time.monotonic()-started
                    result['armed_after_content_chars']=sum(len(c.get('content','')) for c in chunks)
                    print('First generated text received; bounded decode profiling armed',flush=True)
                if item.get('stop'):
                    result['response']=item
        assert armed, 'No generation arrived; profiling never armed'
        assert 'response' in result, 'No completed response'
    except BaseException as exc:
        result['error']=repr(exc)
        raise
    finally:
        done.set()
        if armed:
            arm.unlink()
        result['chunks']=chunks
        result['output_text']=''.join(c.get('content','') for c in chunks)
        result['wall_seconds']=time.monotonic()-started
        (HERE/('profile-'+args.tag+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('Operation profile complete; profiling disarmed',flush=True)

if __name__=='__main__': main()
