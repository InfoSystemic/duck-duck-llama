"""Demand-load the native released text checkpoint into a bounded, owned RAM cache.

No quantization or tensor substitution: sparse routing fetches the selected real
experts, and Engram fetches the selected real rows from the pinned public weights.
The cache is temporary across reboots. HTTP cold-load time is not inference speed.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import types
import torch

from audit_deepseek_v41_tensors_0910 import read_range
from check_deepseek_v41_engram_hash_0910 import TokenizerAdapter
import deepseek_v41_cpu_reference_0910 as cpu
from qwen_split_trial import sha256

BASE=Path(__file__).resolve().parent
AUDIT=BASE/'results/deepseek-v41-tensor-audit-0910/result.json'
REVISION='fb2764a5cf321eaa5070ca8f9e892818f477c16d'
DTYPES={'BF16':torch.bfloat16,'F32':torch.float32,'F8_E4M3':torch.float8_e4m3fn,
        'F8_E8M0':torch.float8_e8m0fnu,'I8':torch.int8}


class Catalog:
    def __init__(self):
        self.audit=json.loads(AUDIT.read_text())
        assert self.audit['passed'] and self.audit['revision']==REVISION
        self.tensors={}
        for shard in self.audit['shards']:
            path=AUDIT.parent/'headers'/(shard['file']+'.json')
            assert sha256(path)==shard['header_sha256']
            for name,meta in json.loads(path.read_text()).items():
                if name=='__metadata__':continue
                begin,end=meta['data_offsets']
                self.tensors[name]=dict(meta,start=8+shard['header_bytes']+begin,bytes=end-begin,
                    url=shard['source_url'],total=shard['size'],shard=shard['file'],publisher_sha256=shard['publisher_sha256'])
        assert len(self.tensors)==96085


class Store:
    def __init__(self,root,output,cap_bytes=90<<30,workers=8):
        self.root=Path(root);self.output=Path(output);self.cap_bytes=cap_bytes
        assert self.root.is_dir() and not self.root.is_symlink()
        assert self.root.stat().st_uid==os.getuid()
        marker=self.root/'owner.json'
        assert json.loads(marker.read_text())=={'task':'deepseek-v41-native-0910','revision':REVISION}
        self.catalog=Catalog();self.records={};self.verified=set();self.row_cache={}
        self.downloaded_bytes=0;self.network_seconds=0;self.tensor_loads=0
        self.pool=ThreadPoolExecutor(max_workers=workers)
        self.manifest=self.root/'tensors.jsonl'
        if self.manifest.exists():
            for line in self.manifest.read_text().splitlines():
                item=json.loads(line);assert item['name'] not in self.records
                self.records[item['name']]=item
        self.stored_bytes=sum(x['bytes'] for x in self.records.values())
        assert self.stored_bytes<=self.cap_bytes
        self.existing_projection={x['tensor']:x for x in json.loads((BASE/'results/deepseek-v41-engram-project-0910/weight-manifest.json').read_text())['records']}

    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)

    def _path(self,name):
        assert re.fullmatch(r'[A-Za-z0-9_.]+',name) and name in self.catalog.tensors
        path=self.root/(name+'.bin');assert not path.is_symlink()
        return path

    def _fetch(self,meta,offset,size):
        for attempt in range(4):
            try:return read_range(meta['url'],meta['start']+offset,meta['start']+offset+size-1,meta['total'])
            except Exception as e:
                if attempt==3:raise RuntimeError('Pinned checkpoint range failed: '+type(e).__name__) from None
        raise AssertionError('unreachable')

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
                assert not path.exists(),('Unexpected unmanifested cache file',str(path))
                new.append(name)
        if not new:return
        amount=sum(self.catalog.tensors[n]['bytes'] for n in new)
        assert self.stored_bytes+amount<=self.cap_bytes,('RAM cache limit',self.stored_bytes,amount,self.cap_bytes)
        available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
        assert available>amount+(100<<30),('RAM reserve',available,amount)
        stat=os.statvfs(self.root);assert stat.f_bavail*stat.f_frsize>amount+(8<<30)
        files={};pending={};began=time.perf_counter()
        try:
            for name in new:
                meta=self.catalog.tensors[name];partial=self._path(name).with_suffix('.part')
                # Only an interrupted partial in this task-owned cache is replaceable.
                if partial.exists():
                    assert partial.is_file() and not partial.is_symlink();partial.unlink()
                fd=os.open(partial,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600);os.ftruncate(fd,meta['bytes'])
                files[name]=(fd,partial,[])
                for offset in range(0,meta['bytes'],8<<20):
                    size=min(8<<20,meta['bytes']-offset)
                    pending[self.pool.submit(self._fetch,meta,offset,size)]=(name,offset,size)
            for future in as_completed(pending):
                name,offset,size=pending.pop(future);data=future.result();assert len(data)==size
                fd,partial,ranges=files[name]
                assert os.pwrite(fd,data,offset)==size
                ranges.append((offset,size,hashlib.sha256(data).hexdigest()))
                self.downloaded_bytes+=size
            for name in new:
                fd,partial,ranges=files[name];os.close(fd);files[name]=(-1,partial,ranges)
                meta=self.catalog.tensors[name];ranges.sort()
                assert sum(s for _,s,_ in ranges)==meta['bytes']
                path=self._path(name);partial.rename(path)
                record=dict(name=name,bytes=meta['bytes'],sha256=sha256(path),revision=REVISION,
                    shard=meta['shard'],start=meta['start'],ranges=ranges,whole_shard_sha_verified=False)
                with self.manifest.open('a') as handle:handle.write(json.dumps(record,separators=(',',':'))+'\n')
                self.records[name]=record;self.verified.add(name);self.stored_bytes+=meta['bytes']
        finally:
            for future in pending:future.cancel()
            # Running workers hold no output files; only this thread writes them.
            for fd,_,_ in files.values():
                if fd>=0:os.close(fd)
            self.network_seconds+=time.perf_counter()-began

    def tensor(self,name):
        self.ensure([name]);meta=self.catalog.tensors[name]
        path=Path(self.existing_projection[name]['file']) if name in self.existing_projection else self._path(name)
        raw=torch.from_file(str(path),shared=False,size=meta['bytes'],dtype=torch.uint8)
        data=raw.view(DTYPES[meta['dtype']]).reshape(meta['shape'])
        if meta['dtype']=='I8':data=data.view(torch.float4_e2m1fn_x2)
        self.tensor_loads+=1
        return data

    def rows(self,prefix,ids):
        flat=ids.flatten().tolist();rows=sorted(set(flat))
        w=self.catalog.tensors[prefix+'.weight'];s=self.catalog.tensors[prefix+'.scale']
        assert w['shape'][1]==256 and s['shape']==[w['shape'][0],8]
        missing=[r for r in rows if (prefix,r) not in self.row_cache]
        assert len(self.row_cache)+len(missing)<=100000
        began=time.perf_counter();pending={};parts={}
        for row in missing:
            assert 0<=row<w['shape'][0]
            for meta,width,kind in [(w,256,'weight'),(s,8,'scale')]:
                pending[self.pool.submit(self._fetch,meta,row*width,width)]=(row,kind)
        for future in as_completed(pending):
            row,kind=pending.pop(future);raw=future.result();parts[row,kind]=raw;self.downloaded_bytes+=len(raw)
        for row in missing:
            weight=torch.frombuffer(bytearray(parts[row,'weight']),dtype=torch.uint8).view(torch.float8_e4m3fn)
            scale=torch.frombuffer(bytearray(parts[row,'scale']),dtype=torch.uint8).view(torch.float8_e8m0fnu)
            value=(weight.float().reshape(8,32)*scale.float()[:,None]).flatten().bfloat16()
            assert torch.isfinite(value).all()
            self.row_cache[prefix,row]=value
            with (self.output/'engram-rows.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(prefix=prefix,row=row,weight_sha256=hashlib.sha256(parts[row,'weight']).hexdigest(),
                    scale_sha256=hashlib.sha256(parts[row,'scale']).hexdigest(),native_bytes=264))+'\n')
        self.network_seconds+=time.perf_counter()-began
        return torch.stack([self.row_cache[prefix,r] for r in flat]).reshape(*ids.shape,256)


def meta_model():
    m=cpu.load_official_model('deepseek_v41_released_cpu')
    args=m.ModelArgs(**json.loads((cpu.OFFICIAL/'config.json').read_text()))
    args.max_batch_size=1;args.max_seq_len=256;args.temperature=0
    args.vision_n_layers=0;args.dspark_block_size=0;args.n_mtp_layers=0;args.dspark_target_layer_ids=()
    tokenizer=TokenizerAdapter()
    with torch.device('meta'):model=m.Transformer(args,tokenizer).eval()
    # Recreate derived caches from the original equations, never materialize meta values.
    model.engram_hash=m.NgramHashState(args,model.engram_layout,tokenizer)
    m.precompute_freqs_cis.cache_clear()
    for layer in model.layers:
        a=layer.attn
        a.freqs_cis=m.precompute_freqs_cis(args.rope_head_dim,args.max_seq_len,
            args.original_seq_len if a.compress_ratio else 0,
            args.compress_rope_theta if a.compress_ratio else args.rope_theta,
            args.rope_factor,args.beta_fast,args.beta_slow)
    for module in model.modules():
        for name,buffer in list(module._buffers.items()):
            if buffer is not None and buffer.is_meta:
                module._buffers[name]=torch.full(buffer.shape,-torch.inf if name=='score_state' else 0,
                    dtype=buffer.dtype,device='cpu')
    assert all(not b.is_meta for b in model.buffers())
    return m,args,tokenizer,model


def load_parameters(module,prefix,store):
    parameters=list(module.named_parameters())
    names=[prefix+n for n,_ in parameters]
    names += [prefix+n.replace('.weight','.scale') for n,_ in parameters if n.endswith('wo_a.weight')]
    store.ensure(names)
    for local,parameter in parameters:
        name=prefix+local;data=store.tensor(name)
        assert tuple(data.shape)==tuple(parameter.shape),(name,data.shape,parameter.shape)
        if local.endswith('wo_a.weight'):
            scales=store.tensor(name[:-6]+'scale')
            converted=torch.empty(data.shape,dtype=torch.bfloat16)
            for start in range(0,data.shape[0],256):
                block=data[start:start+256].float().unflatten(0,(-1,32)).unflatten(-1,(-1,32))
                converted[start:start+256]=(block*scales[start//32:(start+256)//32,None,:,None].float()).flatten(2,3).flatten(0,1).bfloat16()
            data=converted
        elif data.dtype!=parameter.dtype:
            assert data.dtype==torch.bfloat16 and parameter.dtype==torch.float32,(name,data.dtype,parameter.dtype)
            data=data.float()
        assert data.dtype==parameter.dtype
        owner_name,_,attr=local.rpartition('.')
        owner=module.get_submodule(owner_name) if owner_name else module
        owner._parameters[attr]=torch.nn.Parameter(data,requires_grad=False)
    for sub in module.modules():
        if getattr(sub,'scale',None) is not None and getattr(sub,'weight',None) is not None:
            sub.weight.scale=sub.scale


def bind_checkpoint(model,store,progress=print,dry_run=False):
    # Validate the complete declared text graph against the released inventory first.
    catalog=store.catalog.tensors
    parameters=dict(model.named_parameters())
    for name,p in parameters.items():
        assert name in catalog and list(p.shape)==catalog[name]['shape'],('Checkpoint shape mismatch',name)
    extra=set(catalog)-set(parameters)
    assert all(n.startswith(('vision.','aligner.','image_','mtp.')) or n.endswith(('.attn.wo_a.scale','.ffn.gate.bias_vl')) for n in extra),sorted(extra)[:10]
    if dry_run:
        return dict(passed=True,checkpoint_shape_validation=True,parameters=len(parameters),unused_tensors=len(extra),
            layers=len(model.layers),engram_rows=sum(model.engram_layout.num_embeddings),
            full_checkpoint_loaded=False,revision=REVISION)
    expected_common=0;expert_modules=0
    for layer in model.layers:
        if layer.engram is not None:
            engram=layer.engram
            class RemoteRows(torch.nn.Module):
                def __init__(self,prefix):super().__init__();self.prefix=prefix
                def forward(self,ids):return store.rows(self.prefix,ids)
            engram.embed=RemoteRows(f'layers.{layer.layer_id}.engram.embed')
        for i,expert in enumerate(layer.ffn.experts):
            prefix=f'layers.{layer.layer_id}.ffn.experts.{i}.'
            original=expert.forward
            def demand(this,x,weights=None,_prefix=prefix,_forward=original):
                load_parameters(this,_prefix,store)
                try:return _forward(x,weights)
                finally:
                    for sub in this.modules():
                        for name,p in list(sub._parameters.items()):
                            if p is not None:sub._parameters[name]=torch.nn.Parameter(torch.empty(p.shape,dtype=p.dtype,device='meta'),requires_grad=False)
                        if getattr(sub,'scale',None) is not None:sub.weight.scale=sub.scale
            expert.forward=types.MethodType(demand,expert);expert_modules+=1
    # Load common weights per layer to bound descriptors and download buffers.
    load_parameters(model.embed,'embed.',store);load_parameters(model.norm,'norm.',store)
    load_parameters(model.head,'head.',store)
    for layer in model.layers:
        experts=layer.ffn.experts
        layer.ffn.experts=torch.nn.ModuleList()
        try:
            expected_common+=sum(p.numel()*p.element_size() for p in layer.parameters())
            load_parameters(layer,f'layers.{layer.layer_id}.',store)
        finally:layer.ffn.experts=experts
        progress(json.dumps(dict(common_layer_loaded=layer.layer_id,cache_bytes=store.stored_bytes,downloaded_bytes=store.downloaded_bytes)),flush=True)
    assert expert_modules==40*384
    unresolved=[n for n,p in model.named_parameters() if p.is_meta and '.ffn.experts.' not in n]
    assert not unresolved,unresolved
    return dict(revision=REVISION,common_parameters_loaded=True,lazy_experts=expert_modules,
        native_experts=True,engram_rows_on_demand=True,vision=False,dspark=False,
        unused_tensors=len(extra),checkpoint_shape_validation=True)
