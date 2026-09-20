"""Read-only per-thread CPU/runqueue observer, started at first model delta."""
import collections,json,os,threading,time
from pathlib import Path

def system_ticks():
    return {v[0]:list(map(int,v[1:])) for line in Path('/proc/stat').read_text().splitlines() if (v:=line.split())[0].startswith('cpu')}

def thread_rows(pid):
    rows={}
    for p in Path(f'/proc/{pid}/task').iterdir():
        try:
            v=list(map(int,(p/'schedstat').read_text().split()))
            stat=(p/'stat').read_text().rsplit(')',1)[1].split()
            rows[int(p.name)]={'run_ns':v[0],'wait_ns':v[1],'slices':v[2],'state':stat[0],'last_cpu':int(stat[36]),
                              'user_ticks':int(stat[11]),'system_ticks':int(stat[12]),'wchan':(p/'wchan').read_text().strip()}
        except (FileNotFoundError,ProcessLookupError):pass
    return rows

class SchedulerProbe:
    def __init__(self,pid):
        self.pid=pid;self.started=False;self.done=threading.Event();self.samples=[];self.error=None
        self.ident=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        self.meta={}
        for p in Path(f'/proc/{pid}/task').iterdir():
            try:self.meta[int(p.name)]={'comm':(p/'comm').read_text().strip(),'affinity':sorted(os.sched_getaffinity(int(p.name)))}
            except (FileNotFoundError,ProcessLookupError):pass
    def sample(self):
        start=time.monotonic();rows=thread_rows(self.pid)
        return {'time':time.monotonic(),'collection_seconds':time.monotonic()-start,'rows':rows}
    def start(self):
        assert not self.started;self.started=True;self.start_cpu=time.process_time();self.first_ticks=system_ticks()
        self.samples=[self.sample()]
        self.worker=threading.Thread(target=self.loop,daemon=True);self.worker.start()
    def loop(self):
        try:
            os.sched_setaffinity(0,{127}) # Spare physical core63's sibling; outside pinned model workers.
            while not self.done.wait(.5):self.samples.append(self.sample())
        except BaseException as exc:self.error=repr(exc);self.done.set()
    def finish(self):
        if not self.started:return {'started':False}
        self.done.set();self.worker.join(timeout=5);assert not self.worker.is_alive()
        self.samples.append(self.sample());last_ticks=system_ticks()
        assert self.ident==Path(f'/proc/{self.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        a,z=self.samples[0],self.samples[-1];elapsed=z['time']-a['time'];rows=[]
        for tid,v in z['rows'].items():
            if tid not in a['rows']:continue
            u=a['rows'][tid];meta=self.meta.get(tid,{'affinity':sorted(os.sched_getaffinity(tid))})
            affinity=meta['affinity'];group='pinned' if len(affinity)==1 else 'unbound'
            states=collections.Counter(s['rows'][tid]['state'] for s in self.samples if tid in s['rows'])
            waits=collections.Counter(s['rows'][tid]['wchan'] for s in self.samples if tid in s['rows'])
            rows.append({'tid':tid,**meta,'group':group,'cpu_seconds':(v['run_ns']-u['run_ns'])/1e9,
                         'runqueue_seconds':(v['wait_ns']-u['wait_ns'])/1e9,'slices':v['slices']-u['slices'],
                         'user_seconds':(v['user_ticks']-u['user_ticks'])/os.sysconf('SC_CLK_TCK'),
                         'system_seconds':(v['system_ticks']-u['system_ticks'])/os.sysconf('SC_CLK_TCK'),
                         'sampled_states':dict(states),'sampled_wait_channels':dict(waits)})
        groups={}
        for g in ['pinned','unbound']:
            rr=[x for x in rows if x['group']==g];cpu=sum(x['cpu_seconds'] for x in rr);wait=sum(x['runqueue_seconds'] for x in rr)
            groups[g]={'threads':len(rr),'active_threads_over_20ms':sum(x['cpu_seconds']>.02 for x in rr),'cpu_seconds':cpu,'runqueue_seconds':wait,'average_cpu_cores':cpu/elapsed,'runqueue_to_cpu_ratio':wait/cpu if cpu else None}
        per_core={}
        for r in rows:
            if r['group']=='pinned':
                c=str(r['affinity'][0]);x=per_core.setdefault(c,{'cpu_seconds':0,'runqueue_seconds':0,'threads':[]})
                x['cpu_seconds']+=r['cpu_seconds'];x['runqueue_seconds']+=r['runqueue_seconds'];x['threads'].append(r['tid'])
        return {'started':True,'pid':self.pid,'proc_start_ticks':self.ident,'sample_seconds':elapsed,'samples':len(self.samples),
                'max_collection_seconds':max(s['collection_seconds'] for s in self.samples),'observer_process_cpu_seconds':time.process_time()-self.start_cpu,
                'observer_thread_cpu':127,'groups':groups,'per_pinned_core':per_core,'threads':rows,'error':self.error,
                'system_tick_deltas':{k:[v-u for u,v in zip(self.first_ticks[k],z)] for k,z in last_ticks.items() if k in self.first_ticks}}
