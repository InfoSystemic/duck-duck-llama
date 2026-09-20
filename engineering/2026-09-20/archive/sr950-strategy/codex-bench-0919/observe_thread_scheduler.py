#!/usr/bin/env python3
"""Read-only scheduler snapshots during the staged cached Codex comparisons."""
from pathlib import Path
import json,os,time,urllib.request
HERE=Path(__file__).resolve().parent
PID=3145266
LIMIT=HERE/'thread-window-limit.txt'
REPORT=HERE/'thread-window-report.json'
def threads():
    result={}
    for p in Path(f'/proc/{PID}/task').iterdir():
        try:
            values=[int(v) for v in (p/'schedstat').read_text().split()]
            result[p.name]={'affinity':list(os.sched_getaffinity(int(p.name))),
                            'run_ns':values[0],'wait_ns':values[1]}
        except (FileNotFoundError,ProcessLookupError):pass
    return result
for count in [15,16]:
    target=HERE/f'thread-scheduler-codex{count}.json'
    assert not target.exists()
    deadline=time.monotonic()+1200
    last_notice=0
    while time.monotonic()<deadline:
        assert Path(f'/proc/{PID}').exists(), 'Private model stopped before the requested observation'
        state=json.loads(REPORT.read_text())
        if state.get('codex_cold_control') and LIMIT.read_text().strip()==str(count):
            slots=json.load(urllib.request.urlopen('http://127.0.0.1:18141/slots',timeout=5))
            if any(s['is_processing'] and s.get('n_prompt_tokens',0)>4003 for s in slots):break
        if time.monotonic()-last_notice>30:
            print(f'Waiting for cached Codex at {count} workers; private PID {PID} is live',flush=True)
            last_notice=time.monotonic()
        time.sleep(.5)
    else:raise TimeoutError('Requested observation did not arrive; no inference was started')
    start=time.monotonic();a=threads();time.sleep(3);z=threads();elapsed=time.monotonic()-start
    rows=[]
    for tid,v in z.items():
        if tid in a:
            run=(v['run_ns']-a[tid]['run_ns'])/1e9
            wait=(v['wait_ns']-a[tid]['wait_ns'])/1e9
            if run>.02 or wait>.05:rows.append({'tid':int(tid),'affinity':v['affinity'],'cpu_seconds':run,'runqueue_seconds':wait})
    result={'pid':PID,'workers':count,'control_unchanged':LIMIT.read_text().strip()==str(count),
            'sample_seconds':elapsed,'total_threads':len(z),'thread_deltas':rows,
            'total_cpu_seconds':sum(x['cpu_seconds'] for x in rows),
            'total_runqueue_seconds':sum(x['runqueue_seconds'] for x in rows)}
    assert result['control_unchanged']
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='thread_deltas'}),flush=True)
