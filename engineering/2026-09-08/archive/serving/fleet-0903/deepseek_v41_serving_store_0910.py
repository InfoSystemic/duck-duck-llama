"""Reusable, bounded cache for the native CPU server; only task-owned files evict."""
from concurrent.futures import as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import time
import torch
from deepseek_v41_checkpoint_transport_0910b import CoalescedStore
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256


def finish_eviction(root):
    """Complete a recorded eviction after normal execution or a process interruption."""
    root=Path(root);journal=root/'eviction.json'
    if not journal.exists():return
    assert root.is_dir() and root.stat().st_uid==os.getuid() and not root.is_symlink()
    assert json.loads((root/'owner.json').read_text())['task']=='deepseek-v41-native-0910'
    victims=json.loads(journal.read_text())['victims']
    manifest=root/'tensors.jsonl'
    records=[json.loads(x) for x in manifest.read_text().splitlines()]
    known={r['name']:r for r in records}
    names=set()
    for item in victims:
        name=item['name']
        assert re.fullmatch(r'layers\.\d+\.ffn\.experts\.\d+\.w[123]\.(weight|scale)',name)
        assert name not in names;names.add(name)
        if name in known:assert known[name]['sha256']==item['sha256'] and known[name]['bytes']==item['bytes']
        path=root/(name+'.bin')
        if path.exists():
            stat=path.stat()
            assert path.is_file() and not path.is_symlink() and stat.st_uid==os.getuid() and stat.st_nlink==1
            assert stat.st_size==item['bytes'] and sha256(path)==item['sha256']
            path.unlink()
    temporary=manifest.with_suffix('.next')
    with temporary.open('w') as handle:
        for item in records:
            if item['name'] not in names:handle.write(json.dumps(item,separators=(',',':'))+'\n')
        handle.flush();os.fsync(handle.fileno())
    temporary.replace(manifest);journal.unlink()


class ServingStore(CoalescedStore):
    def __init__(self,root,*args,**kwargs):
        finish_eviction(root)
        super().__init__(root,*args,**kwargs)
        self.touched={n:0 for n in self.records};self.tick=0;self.evicted_bytes=0
        self.rows_dir=self.root/'engram-rows';self.rows_dir.mkdir(mode=0o700,exist_ok=True)
        assert not self.rows_dir.is_symlink()

    def ensure(self,names):
        names=list(dict.fromkeys(names));self.tick+=1
        amount=sum(self.catalog.tensors[n]['bytes'] for n in names if n not in self.records and n not in self.existing_projection)
        needed=self.stored_bytes+amount-self.cap_bytes
        if needed>0:
            groups={}
            protected={n.rsplit('.',2)[0] for n in names if '.ffn.experts.' in n}
            for n,item in self.records.items():
                if '.ffn.experts.' not in n:continue
                prefix=n.rsplit('.',2)[0]
                if prefix not in protected:groups.setdefault(prefix,[]).append(item)
            order=sorted(groups,key=lambda p:max(self.touched.get(r['name'],0) for r in groups[p]))
            victims=[];reclaimed=0
            for prefix in order:
                victims+=groups[prefix];reclaimed+=sum(r['bytes'] for r in groups[prefix])
                if reclaimed>=max(needed,4<<30):break
            assert reclaimed>=needed,'Insufficient evictable expert cache'
            atomic_json(self.root/'eviction.json',dict(victims=[{k:r[k] for k in ['name','bytes','sha256']} for r in victims]))
            finish_eviction(self.root)
            for item in victims:
                name=item['name'];del self.records[name];self.verified.discard(name);self.touched.pop(name,None)
            self.stored_bytes-=reclaimed;self.evicted_bytes+=reclaimed
        super().ensure(names)
        self.touched.update({n:self.tick for n in names})

    def rows(self,prefix,ids):
        flat=ids.flatten().tolist();w=self.catalog.tensors[prefix+'.weight'];s=self.catalog.tensors[prefix+'.scale']
        assert w['shape'][1]==256 and s['shape']==[w['shape'][0],8]
        missing=[];raw_rows={}
        for row in sorted(set(flat)):
            assert 0<=row<w['shape'][0]
            if (prefix,row) in self.row_cache:continue
            path=self.rows_dir/f'{prefix}.{row}.bin';record=path.with_suffix('.json')
            if record.exists():
                metadata=json.loads(record.read_text())
                partial=path.with_suffix('.part')
                if not path.exists() and partial.exists():
                    assert not partial.is_symlink() and sha256(partial)==metadata['sha256'];partial.rename(path)
                assert metadata['prefix']==prefix and metadata['row']==row and path.stat().st_size==264
                assert not path.is_symlink() and sha256(path)==metadata['sha256']
                raw_rows[row]=path.read_bytes()
            else:
                assert not path.exists(),('Unmanifested Engram row',str(path));missing.append(row)
        assert len(self.row_cache)+len(missing)+len(raw_rows)<=100000
        pending={};parts={};began=time.perf_counter()
        for row in missing:
            for meta,width,kind in [(w,256,'weight'),(s,8,'scale')]:
                pending[self.pool.submit(self._fetch,meta,row*width,width)]=(row,kind)
        for future in as_completed(pending):
            row,kind=pending.pop(future);raw=future.result();parts[row,kind]=raw;self.downloaded_bytes+=len(raw)
        for row in missing:
            raw=bytes(parts[row,'weight'])+bytes(parts[row,'scale']);assert len(raw)==264
            path=self.rows_dir/f'{prefix}.{row}.bin'
            partial=path.with_suffix('.part')
            assert not partial.is_symlink()
            with partial.open('wb') as handle:handle.write(raw)
            atomic_json(path.with_suffix('.json'),dict(prefix=prefix,row=row,sha256=hashlib.sha256(raw).hexdigest()))
            partial.rename(path)
            raw_rows[row]=raw
        for row,raw in raw_rows.items():
            data=torch.frombuffer(bytearray(raw),dtype=torch.uint8)
            value=(data[:256].view(torch.float8_e4m3fn).float().reshape(8,32)*
                data[256:].view(torch.float8_e8m0fnu).float()[:,None]).flatten().bfloat16()
            assert torch.isfinite(value).all();self.row_cache[prefix,row]=value
        self.network_seconds+=time.perf_counter()-began
        return torch.stack([self.row_cache[prefix,r] for r in flat]).reshape(*ids.shape,256)
