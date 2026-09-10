"""Bounded native-tensor downloads with contiguous coalescing and HTTP backoff.

The tensor format and graph are unchanged. Adjacent requested tensor spans share
one range request (at most 32 MiB), reducing resolver traffic for the six tensors
in each expert. Successful files use the original checkpoint store manifest.
"""
from concurrent.futures import as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import httpx
from deepseek_v41_checkpoint_0910 import Store,REVISION
from qwen_split_trial import sha256


class CoalescedStore(Store):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.client=httpx.Client(follow_redirects=True,timeout=60,
            limits=httpx.Limits(max_connections=8,max_keepalive_connections=8))
        self.rate_lock=threading.Lock();self.next_request=0.;self.cooldown=0.
        self.retry_events=[];self.requests=0

    def close(self):
        super().close();self.client.close()
        (self.output/'transport.json').write_text(json.dumps(dict(requests=self.requests,retries=self.retry_events,
            max_range_bytes=32<<20,minimum_request_interval_seconds=.14,coalescing_preserves_native_tensor_bytes=True),indent=2)+'\n')

    def _fetch(self,meta,offset,size):
        start=meta['start']+offset;end=start+size-1
        assert 0<size<=32<<20 and 0<=start<=end<meta['total']
        failure='none'
        for attempt in range(12):
            with self.rate_lock:
                slot=max(time.monotonic(),self.next_request,self.cooldown)
                self.next_request=slot+.14
            wait=slot-time.monotonic()
            if wait>0:time.sleep(wait)
            query=f'?download=true&range_start={start}&range_end={end}'
            if attempt:query+=f'&retry_nonce={time.time_ns()}'
            status=None
            try:
                with self.client.stream('GET',meta['url']+query,
                    headers={'Range':f'bytes={start}-{end}','User-Agent':'llama-llama-duck-v41-native-cpu'}) as response:
                    status=response.status_code
                    with self.rate_lock:self.requests+=1
                    if status==206:
                        assert response.headers.get('Content-Range')==f'bytes {start}-{end}/{meta["total"]}'
                        data=bytearray()
                        for chunk in response.iter_bytes():
                            assert len(data)+len(chunk)<=size
                            data.extend(chunk)
                        assert len(data)==size
                        return data
                    assert status in (403,408,429,500,502,503,504),('Range response rejected',status)
                    raw_retry=response.headers.get('Retry-After','')
                    delay=min(60,max(1,int(raw_retry))) if raw_retry.isdigit() else min(30,2**attempt)
                    failure='HTTP '+str(status)
            except (httpx.TimeoutException,httpx.TransportError) as e:
                failure=type(e).__name__;delay=min(30,2**attempt)
            with self.rate_lock:
                if status==429:self.cooldown=max(self.cooldown,time.monotonic()+delay)
                self.retry_events.append(dict(time=time.time(),attempt=attempt+1,status=status,failure=failure,delay_seconds=delay))
            time.sleep(delay)
        raise RuntimeError('Checkpoint range retries exhausted: '+failure)

    def ensure(self,names):
        names=list(dict.fromkeys(names));new=[]
        for name in names:
            if name in self.existing_projection:
                item=self.existing_projection[name]
                if name not in self.verified:
                    assert sha256(item['file'])==item['sha256'];self.verified.add(name)
                continue
            path=self._path(name);meta=self.catalog.tensors[name]
            if name in self.records:
                record=self.records[name]
                assert record['bytes']==meta['bytes'] and path.stat().st_size==meta['bytes']
                if name not in self.verified:
                    assert sha256(path)==record['sha256'];self.verified.add(name)
            else:
                assert not path.exists(),('Unexpected unmanifested cache file',str(path));new.append(name)
        if not new:return
        amount=sum(self.catalog.tensors[n]['bytes'] for n in new)
        assert self.stored_bytes+amount<=self.cap_bytes,('RAM cache limit',self.stored_bytes,amount,self.cap_bytes)
        available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
        assert available>amount+(100<<30),('RAM reserve',available,amount)
        stat=os.statvfs(self.root);assert stat.f_bavail*stat.f_frsize>amount+(8<<30)
        spans=sorted((self.catalog.tensors[n]['url'],self.catalog.tensors[n]['start'],self.catalog.tensors[n]['bytes'],n) for n in new)
        groups=[]
        for url,start,size,name in spans:
            if groups and groups[-1]['url']==url and groups[-1]['end']==start:
                groups[-1]['end']+=size;groups[-1]['tensors'].append(name)
            else:groups.append(dict(url=url,start=start,end=start+size,tensors=[name]))
        files={};pending={};began=time.perf_counter()
        try:
            for name in new:
                meta=self.catalog.tensors[name];partial=self._path(name).with_suffix('.part')
                if partial.exists():
                    assert partial.is_file() and not partial.is_symlink();partial.unlink()
                fd=os.open(partial,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600);os.ftruncate(fd,meta['bytes'])
                files[name]=(fd,partial,[])
            for group in groups:
                meta=dict(self.catalog.tensors[group['tensors'][0]],start=group['start'])
                for start in range(group['start'],group['end'],32<<20):
                    size=min(32<<20,group['end']-start)
                    pending[self.pool.submit(self._fetch,meta,start-group['start'],size)]=(group,start,size)
            for future in as_completed(pending):
                group,start,size=pending.pop(future);data=future.result();assert len(data)==size
                view=memoryview(data)
                for name in group['tensors']:
                    meta=self.catalog.tensors[name];a=max(start,meta['start']);z=min(start+size,meta['start']+meta['bytes'])
                    if a>=z:continue
                    fd,_,ranges=files[name];piece=view[a-start:z-start]
                    assert os.pwrite(fd,piece,a-meta['start'])==z-a
                    ranges.append((a-meta['start'],z-a,hashlib.sha256(piece).hexdigest()))
                self.downloaded_bytes+=size
            for name in new:
                fd,partial,ranges=files[name];os.close(fd);files[name]=(-1,partial,ranges)
                meta=self.catalog.tensors[name];ranges.sort();cursor=0
                for offset,size,_ in ranges:assert offset==cursor;cursor+=size
                assert cursor==meta['bytes']
                path=self._path(name);partial.rename(path)
                record=dict(name=name,bytes=meta['bytes'],sha256=sha256(path),revision=REVISION,
                    shard=meta['shard'],start=meta['start'],ranges=ranges,whole_shard_sha_verified=False)
                with self.manifest.open('a') as handle:handle.write(json.dumps(record,separators=(',',':'))+'\n')
                self.records[name]=record;self.verified.add(name);self.stored_bytes+=meta['bytes']
        finally:
            for future in pending:future.cancel()
            for fd,_,_ in files.values():
                if fd>=0:os.close(fd)
            self.network_seconds+=time.perf_counter()-began
