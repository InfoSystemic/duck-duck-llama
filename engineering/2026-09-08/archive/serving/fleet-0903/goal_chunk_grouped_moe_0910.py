"""Exact expert routing with two variable-token grouped calls for verifier chunks."""
import torch
import torch.nn.functional as F

from goal_grouped_moe_0910 import GroupedMoE


class ChunkGroupedMoE(GroupedMoE):
    def __init__(self,runtime,backend,active):
        self.active=active
        super().__init__(runtime,backend)

    def forward(self,moe,x,image_mask,original,supported,prefix):
        store=self.runtime.store
        if (not self.enabled or not self.active() or not supported or image_mask is not None
                or torch.is_grad_enabled() or x.requires_grad or x.device.type!='cpu'
                or x.dtype!=torch.bfloat16 or x.ndim!=3 or x.shape[0]!=1
                or not 2<=x.shape[1]<=8 or x.shape[2]!=moe.dim or not x.is_contiguous()
                or not getattr(store,'resident_enabled',False)):
            self.fallback_calls+=1
            return original(x,image_mask)
        shape=x.shape;x=x.view(-1,moe.dim)
        weights,indices=moe.gate(x)
        selected=sorted(set(indices.flatten().tolist()))
        assert len(selected)<=moe.n_activated_experts*x.shape[0]
        routed=[]
        for index in selected:
            idx,top=torch.where(indices==index)
            assert len(idx)==len(set(idx.tolist())), 'An expert appears twice for the same token'
            routed.append((moe.experts[index],prefix+str(index)+'.',idx,weights[idx,top,None]))
        if any(p not in store.residents for _,p,_,_ in routed):
            names=[p+name for expert,p,_,_ in routed for name,_ in expert.named_parameters()]
            store.ensure(names)
        for expert,p,_,_ in routed:store.activate(expert,p)
        m=self.module
        q,qs=m.act_quant(x,m.fp8_block_size,m.scale_fmt,m.scale_dtype)
        experts=[];up_tasks=[];routing=[];token_rows=[]
        for expert,_,idx,route in routed:
            # Index uint8 storage so the gather preserves every native FP8 byte.
            a=q.view(torch.uint8).index_select(0,idx).view(q.dtype)
            asc=qs.view(torch.uint8).index_select(0,idx).view(qs.dtype)
            experts.append((expert,4));routing.append(route);token_rows.append(idx)
            up_tasks.extend((4,a,asc,p.weight,p.weight.scale) for p in (expert.w1,expert.w3))
        shared=moe.shared_experts
        experts.append((shared,8));token_rows.append(torch.arange(x.shape[0],device='cpu'))
        up_tasks.extend((8,q,qs,p.weight,p.weight.scale) for p in (shared.w1,shared.w3))
        self.backend.workers=self.runtime.native.workers
        projected=self.backend.apply(up_tasks)
        gates=torch.cat(projected[0::2],dim=0).float()
        ups=torch.cat(projected[1::2],dim=0).float()
        limit=shared.swiglu_limit
        if limit>0:
            gates=torch.clamp(gates,max=limit)
            ups=torch.clamp(ups,min=-limit,max=limit)
        activated=F.silu(gates)*ups
        route=torch.cat(routing,dim=0)
        activated[:route.shape[0]]=route*activated[:route.shape[0]]
        intermediate=activated.to(x.dtype)
        aq,asc=m.act_quant(intermediate,m.fp8_block_size,m.scale_fmt,m.scale_dtype)
        down_tasks=[];at=0
        for (expert,mode),idx in zip(experts,token_rows):
            end=at+idx.numel()
            down_tasks.append((mode,aq[at:end],asc[at:end],expert.w2.weight,expert.w2.weight.scale))
            at=end
        outputs=self.backend.apply(down_tasks)
        y=torch.zeros_like(x,dtype=torch.float32)
        for idx,value in zip(token_rows[:-1],outputs[:-1]):y[idx]+=value
        y+=outputs[-1]
        self.calls+=1
        return y.to(x.dtype).view(shape)

    def uninstall(self):
        for moe,original in self.bindings:moe.forward=original
        self.bindings.clear()
        del self.runtime._goal_chunk_grouped_moe_0910


def install(runtime,backend,active):
    if hasattr(runtime,'_goal_chunk_grouped_moe_0910'):
        raise RuntimeError('Chunk grouped MoE is already installed')
    candidate=ChunkGroupedMoE(runtime,backend,active)
    runtime._goal_chunk_grouped_moe_0910=candidate
    return candidate
