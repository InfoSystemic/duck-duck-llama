#!/usr/bin/env python3
"""Exercise the actual handoff error paths with fake processes and fake environments."""
import copy
import io
import json
from pathlib import Path
import signal
import tempfile
from unittest.mock import patch

import probe_qwen_repeatability_0910c as trial
from qwen_split_trial import sha256


def check(case):
    original = dict(pid=1001,info=dict(start='111'),command=['fake-flash','--port','18131'],
        runtime_env={},affinity=list(range(128)),log='fake-original.log',quant='UD-Q4_K_XL',drafts=2)
    env = {'PROFILE_FIXTURE_ONLY':'true'}
    live, events, callbacks = {1001:dict(env=env,command=original['command'],cwd='/fake')}, [], {}
    clock = [0.0]
    class FakeManager:
        def __init__(self):self.state={'current':copy.deepcopy(original),'events':[]}
        def validate_current(self):
            current = self.state['current']
            assert current and current['pid'] in live
            return current
        def stop_flash(self):
            events.append('stop-flash');live.pop(1001);self.state['current']=None
            if case=='cancel-after-stop':callbacks[signal.SIGINT](signal.SIGINT,None)
        def record(self,*a,**k):events.append('record-restore')
    manager = FakeManager()
    class Guard:
        def __init__(self,*a):pass
        def assert_idle(self):pass
    class Proc:
        next_pid=2000
        def __init__(self,args,env=None,cwd=None,**kw):
            assert args[:3]==['taskset','-c','0-127']
            self.pid=Proc.next_pid;Proc.next_pid+=1
            self.returncode = 7 if case=='qwen-load-exits' and args[3]=='fake-qwen' else None
            events.append('spawn-'+args[3])
            if self.returncode is None:live[self.pid]=dict(env=env,command=args[3:],cwd=str(cwd))
        def poll(self):return self.returncode
        def terminate(self):self.returncode=0;live.pop(self.pid,None)
        def kill(self):self.terminate()
        def wait(self,timeout=None):return self.returncode
    def info(pid):
        row=live[pid]
        return dict(pid=pid,start='111' if pid==1001 else '222',command=row['command'],cwd=row['cwd'],affinity=list(range(128)))
    def memory(global_bytes,node_bytes):
        if global_bytes==220_000_000_000 and case in {'memory-after-stop','restore-reserve-fails'}:
            raise AssertionError('fixture Qwen admission reserve')
        if global_bytes==260_000_000_000 and case=='restore-reserve-fails':
            raise AssertionError('fixture Flash restoration reserve')
        return {}
    def sleep(seconds):clock[0]+=seconds
    with tempfile.TemporaryDirectory(prefix='qwen-profile-lifecycle-') as directory:
        out=Path(directory)
        peer=copy.deepcopy(original)
        if case=='changed-identity':peer['info']['start']='wrong'
        plan=dict(peer=peer,peer_libraries=[],arm_file=str(out/'arm'),command=['fake-qwen'],runtime_env={},
            source_sha256={str(trial.SELECTED):'fixture-sha'})
        (out/'plan.json').write_text(json.dumps(plan))
        changes={
            'OUT':out,'Manager':lambda:manager,'verify_sources':lambda p:None,
            'inference_snapshot':lambda:{str(pid):{} for pid in live},'port_available':lambda p:True,
            'process_environment':lambda pid:live[pid]['env'],'process_info':info,
            'mapped_libraries':lambda pid:set(),'sha256':lambda p:'fixture-sha',
            'memory_gate':memory,'ModelMeasurementGuard':Guard,'background':lambda pid:{},
            'read_service':lambda port:{'status':'ok'}}
        error = None
        with patch.multiple(trial,**changes),patch.object(trial.subprocess,'Popen',Proc),\
             patch.object(trial.signal,'signal',lambda sig,fn:callbacks.update({sig:fn})),\
             patch.object(trial.time,'sleep',sleep),patch.object(trial.time,'monotonic',lambda:clock[0]),\
             patch.object(trial.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'{"status":"ok"}')):
            try:trial.execute(-1)
            except (AssertionError,InterruptedError) as caught:error=str(caught)
        assert error is not None,case
        path=out/'result.json'
        result=json.loads(path.read_text()) if path.exists() else None
        if case=='changed-identity':
            assert not events and set(live)=={1001} and result is None
        elif case=='restore-reserve-fails':
            assert not live and result['restore_error'] and result['finished'] and not result['passed']
        else:
            assert result['restored'] and result['finished'] and not result['passed']
            assert len(live)==1 and 1001 not in live
            restored=next(iter(live.values()))
            assert restored==dict(env=env,command=original['command'],cwd='/fake')
            assert events.count('stop-flash')==events.count('spawn-fake-flash')==1
        return dict(case=case,passed=True,restored=bool(result and result.get('restored')),events=events)


if __name__=='__main__':
    destination=trial.BASE/'results/qwen-repeatability-checks-0910c.json'
    assert not destination.exists()
    rows=[check(case) for case in ['changed-identity','cancel-after-stop','memory-after-stop','qwen-load-exits','restore-reserve-fails']]
    result=dict(passed=True,checks=rows,sources={str(p):sha256(p) for p in [Path(__file__),Path(trial.__file__)]},
        scope='Fake processes, synthetic environment, temporary files. Executes actual handoff failure paths; no real inference, signal, download, or service mutation.')
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
