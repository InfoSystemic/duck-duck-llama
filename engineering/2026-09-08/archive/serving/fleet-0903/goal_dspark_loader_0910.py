"""Optional native DSpark preparation/attachment, with no automatic execution.

prepare_dspark is metadata-only. install_dspark explicitly loads common tensors
through a separate MTPResidentStore; that store disables downloads by default.
No draft token is accepted or served by this loader.
"""
import copy
from dataclasses import dataclass
import json
import types

import torch

from deepseek_v41_checkpoint_0910 import REVISION, load_parameters
from goal_dspark_store_0910 import MTPResidentStore


@dataclass
class DSparkPlan:
    stages: torch.nn.ModuleList
    args: object
    common_names: list
    report: dict


def prepare_dspark(module, runtime_args, catalog):
    """Construct only meta parameters and validate them against pinned metadata."""
    args = copy.deepcopy(runtime_args)
    args.n_mtp_layers = 3
    args.dspark_block_size = 5
    args.dspark_target_layer_ids = (37, 38, 39)
    if args.n_layers != 40 or getattr(module, 'world_size', 1) != 1:
        raise ValueError('This attachment targets the pinned single-rank 40+3-stage model')
    if args.vision_enabled or args.dtype != 'fp8' or args.expert_dtype != 'fp4':
        raise ValueError('Expected the existing text-only native FP8/FP4 runtime')
    if torch.get_default_dtype() != torch.bfloat16:
        raise ValueError('Prepare under the existing runtime BF16 default dtype')
    with torch.device('meta'):
        stages = torch.nn.ModuleList([module.DSparkBlock(args.n_layers + stage, args)
                                      for stage in range(args.n_mtp_layers)])
    assert all(parameter.is_meta for parameter in stages.parameters())
    tensors = catalog.tensors
    common, expert_names, runtime_common_bytes = [], [], 0
    for local, parameter in stages.named_parameters():
        name = 'mtp.' + local
        assert name in tensors and list(parameter.shape) == tensors[name]['shape'], ('DSpark shape mismatch', name)
        expected = tensors[name]['dtype']
        compatible = ((parameter.dtype == torch.float4_e2m1fn_x2 and expected == 'I8')
                      or (parameter.dtype == torch.float8_e8m0fnu and expected == 'F8_E8M0')
                      or (parameter.dtype == torch.float8_e4m3fn and expected == 'F8_E4M3')
                      or (parameter.dtype == torch.bfloat16 and expected == 'BF16')
                      or (parameter.dtype == torch.float32 and expected in ('BF16', 'F32'))
                      or (name.endswith('attn.wo_a.weight') and parameter.dtype == torch.bfloat16
                          and expected == 'F8_E4M3'))
        assert compatible, ('DSpark dtype mismatch', name, parameter.dtype, expected)
        if '.ffn.experts.' in name:
            expert_names.append(name)
        else:
            common.append(name)
            runtime_common_bytes += parameter.numel() * parameter.element_size()
            if name.endswith('attn.wo_a.weight'):
                scale = name[:-6] + 'scale'
                assert scale in tensors
                common.append(scale)
    requested = set(common) | set(expert_names)
    unused = sorted(name for name in tensors if name.startswith('mtp.') and name not in requested)
    assert all(name.endswith('.ffn.gate.bias_vl') for name in unused), unused
    report = dict(revision=REVISION, backbone_layers=40, mtp_stages=3, stage_layer_ids=[40, 41, 42],
                  block_size=5, target_layer_ids=[37, 38, 39],
                  routed_experts=sum(len(stage.ffn.experts) for stage in stages),
                  activated_experts=[stage.ffn.n_activated_experts for stage in stages],
                  common_tensor_count=len(set(common)), expert_tensor_count=len(expert_names),
                  common_native_bytes=sum(tensors[name]['bytes'] for name in set(common)),
                  common_runtime_parameter_bytes=runtime_common_bytes,
                  all_expert_native_bytes=sum(tensors[name]['bytes'] for name in expert_names),
                  unused_text_only_tensors=unused, native_parameter_shapes_validated=True,
                  parameters_all_meta=True, model_executed=False, downloaded_bytes=0)
    return DSparkPlan(stages, args, sorted(set(common)), report)


def install_dspark(model, module, plan, store, *, progress=None):
    """Load common MTP weights, attach aliases and install lazy real experts.

    Call only while the runtime's normal inference lock is held. The store must
    be a separate MTPResidentStore. Its default allow_download=False prevents
    an unrequested fetch. Missing common tensors are checked before any loading.
    """
    if not isinstance(store, MTPResidentStore):
        raise TypeError('DSpark requires the isolated MTPResidentStore')
    if len(model.layers) != 40 or len(model.mtp) != 0:
        raise ValueError('Expected a 40-layer model with DSpark currently disabled')
    if plan.report['revision'] != REVISION:
        raise ValueError('Pinned revision mismatch')
    absent = [name for name in plan.common_names if name not in store.records]
    if absent and not store.allow_download:
        raise RuntimeError(f'DSpark common weights absent; downloads disabled ({len(absent)} tensors)')
    # Common loads cannot exceed the separate cache cap even when no expert is
    # cached. The existing store also enforces the 100-GiB physical RAM reserve.
    common_bytes = sum(store.catalog.tensors[name]['bytes'] for name in plan.common_names)
    if common_bytes > store.cap_bytes:
        raise ValueError('MTP cache cap is smaller than required common native tensors')
    stages = plan.stages
    for stage_id, stage in enumerate(stages):
        assert stage.embed is None and stage.head is None
        experts = stage.ffn.experts
        stage.ffn.experts = torch.nn.ModuleList()
        try:
            load_parameters(stage, f'mtp.{stage_id}.', store)
        finally:
            stage.ffn.experts = experts
        # Meta construction may retain a cached CPU RoPE buffer or create a meta
        # one. Compute a fresh CPU buffer without touching the shared LRU cache.
        with torch.device('cpu'):
            stage.attn.freqs_cis = module.precompute_freqs_cis.__wrapped__(
                plan.args.rope_head_dim, plan.args.max_seq_len, 0, plan.args.rope_theta,
                plan.args.rope_factor, plan.args.beta_fast, plan.args.beta_slow)
            for owner in stage.modules():
                for name, buffer in list(owner._buffers.items()):
                    if buffer is not None and buffer.is_meta:
                        owner._buffers[name] = torch.full(buffer.shape, -torch.inf if name == 'score_state' else 0,
                                                           dtype=buffer.dtype, device='cpu')
        if progress is not None:
            progress(dict(mtp_common_stage_loaded=stage_id, cache_bytes=store.stored_bytes,
                          downloaded_bytes=store.downloaded_bytes))
    assert all(not buffer.is_meta for buffer in stages.buffers())
    unresolved = [name for name, parameter in stages.named_parameters()
                  if parameter.is_meta and '.ffn.experts.' not in name]
    assert not unresolved, unresolved
    # Add hooks only after every common load succeeds, avoiding duplicate hooks
    # when a failed/partial common load is retried with the same prepared plan.
    for stage_id, stage in enumerate(stages):
        for index, expert in enumerate(stage.ffn.experts):
            if expert is None:
                continue
            prefix = f'mtp.{stage_id}.ffn.experts.{index}.'
            original = type(expert).forward
            def forward(this, x, weights=None, _prefix=prefix, _original=original):
                store.activate(this, _prefix)
                return _original(this, x, weights)
            expert.forward = types.MethodType(forward, expert)
        def prefetch(_owner, _inputs, output, _stage=stage_id):
            selected = sorted(set(output[1].flatten().tolist()))
            names = [f'mtp.{_stage}.ffn.experts.{index}.{weight}.{kind}' for index in selected
                     for weight in ('w1', 'w2', 'w3') for kind in ('weight', 'scale')]
            store.ensure(names)
        stage.ffn.gate.register_forward_hook(prefetch)
    # Preserve shared embedding/head objects by identity; loading occurs before
    # these aliases are attached so checkpoint enumeration never reloads them.
    stages.train(model.training)
    for stage in stages:
        stage.embed, stage.head = model.embed, model.head
    model.mtp = stages
    model.target_layer_ids = plan.args.dspark_target_layer_ids
    return dict(plan.report, parameters_all_meta=False, common_loaded=True,
                experts_lazy=True, shared_embed_head_identity_preserved=True,
                mtp_cache_root=str(store.root), cache_bytes=store.stored_bytes,
                downloaded_bytes=store.downloaded_bytes, draft_execution_enabled=False)
