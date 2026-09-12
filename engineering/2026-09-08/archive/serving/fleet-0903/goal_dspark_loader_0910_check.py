#!/usr/bin/env python3
"""Meta-only real catalog planning plus tiny synthetic loader/cache fixtures."""
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
from unittest.mock import patch

import torch

import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_checkpoint_0910 import Catalog, DTYPES, REVISION
from deepseek_v41_checkpoint_transport_0910b import CoalescedStore
from goal_dspark_loader_0910 import prepare_dspark, install_dspark
from goal_dspark_store_0910 import initialize_mtp_cache, MTPResidentStore, finish_mtp_eviction

BASE = Path(__file__).resolve().parent


def main():
    assert os.sched_getaffinity(0) == {0}
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    m = cpu.load_official_model('goal_dspark_loader_fixture_0910')
    args = m.ModelArgs(**json.loads((cpu.OFFICIAL / 'config.json').read_text()))
    args.vision_n_layers = 0
    args.max_batch_size, args.max_seq_len = 1, 256
    args.n_mtp_layers, args.dspark_block_size, args.dspark_target_layer_ids = 0, 0, ()
    real_catalog = Catalog()
    full = prepare_dspark(m, args, real_catalog)
    assert full.report['routed_experts'] == 384 and full.report['common_tensor_count'] == 94
    assert full.report['expert_tensor_count'] == 2304
    assert all(p.is_meta for p in full.stages.parameters())
    production_report = dict(full.report)
    del full, real_catalog
    gc.collect()

    verified = {}
    with tempfile.TemporaryDirectory(prefix='goal_dspark_0910_') as directory:
        root = Path(directory)
        backbone = root / 'backbone'
        backbone.mkdir()
        (backbone / 'untouched.txt').write_text('backbone sentinel\n')
        baseline_listing = sorted(str(p.relative_to(backbone)) for p in backbone.iterdir())
        try:
            initialize_mtp_cache(backbone, backbone)
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted the backbone cache as MTP cache')
        mtp = root / 'mtp'
        output = root / 'output'
        output.mkdir()
        initialize_mtp_cache(mtp, backbone)
        old = 'mtp.0.ffn.experts.0.w1.weight'
        new = 'mtp.0.ffn.experts.1.w1.weight'
        common = 'mtp.0.attn.attn_sink'
        raw = bytes(range(64))
        records = []
        for name in [old, common]:
            (mtp / (name + '.bin')).write_bytes(raw)
            records.append(dict(name=name, bytes=64, sha256=hashlib.sha256(raw).hexdigest(), revision=REVISION))
        (mtp / 'tensors.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        store = MTPResidentStore(mtp, output, cap_bytes=128)
        try:
            store.catalog.tensors = {name: dict(bytes=64, shape=[4, 8], dtype='BF16') for name in [old, new, common]}
            try:
                store.ensure([new])
            except RuntimeError as error:
                assert 'downloads disabled' in str(error)
            else:
                raise AssertionError('Missing native tensor fetched without opt-in')
            try:
                store.ensure(['layers.0.ffn.experts.0.w1.weight'])
            except AssertionError:
                pass
            else:
                raise AssertionError('Backbone namespace entered MTP store')
            expert = torch.nn.Module()
            expert.w1 = torch.nn.Module()
            expert.w1.weight = torch.nn.Parameter(torch.empty(4, 8, dtype=torch.bfloat16, device='meta'), requires_grad=False)
            prefix = 'mtp.0.ffn.experts.0.'
            store.activate(expert, prefix)
            assert not expert.w1.weight.is_meta
            assert expert.w1.weight.view(torch.uint8).numpy().tobytes() == raw
            store.allow_download = True
            # Synthetic transport writes 64 known bytes; it performs no network.
            def synthetic_ensure(this, names):
                for name in names:
                    if name not in this.records:
                        (this.root / (name + '.bin')).write_bytes(raw)
                        record = dict(name=name, bytes=64, sha256=hashlib.sha256(raw).hexdigest(), revision=REVISION)
                        with this.manifest.open('a') as handle:
                            handle.write(json.dumps(record) + '\n')
                        this.records[name] = record
                        this.stored_bytes += 64
            with patch.object(CoalescedStore, 'ensure', synthetic_ensure):
                store.ensure([new])
            assert expert.w1.weight.is_meta and not store.residents
            assert old not in store.records and common in store.records and new in store.records
            assert not (mtp / (old + '.bin')).exists() and (mtp / (common + '.bin')).exists()
            assert not (mtp / 'mtp-eviction.json').exists()
            assert store.stored_bytes == 128 and store.evicted_bytes == 64
            # Exercise restart recovery after a journal was written but not applied.
            (mtp / 'mtp-eviction.json').write_text(json.dumps(dict(victims=[store.records[new]])))
            finish_mtp_eviction(mtp)
            assert not (mtp / (new + '.bin')).exists() and (mtp / (common + '.bin')).exists()
            assert store.downloaded_bytes == 0 and store.requests == 0
        finally:
            store.close()
        assert sorted(str(p.relative_to(backbone)) for p in backbone.iterdir()) == baseline_listing
        assert (backbone / 'untouched.txt').read_text() == 'backbone sentinel\n'
        verified['separate_root_and_real_mtp_namespace'] = True
        verified['missing_native_tensor_rejected_by_default'] = True
        verified['bounded_expert_eviction_releases_resident_and_pins_common'] = True
        verified['mtp_journal_restart_recovery'] = True
        verified['backbone_cache_untouched'] = True

        # Tiny synthetic attachment uses the real official DSpark classes and
        # real load_parameters conversions, but never executes a model forward.
        args.dim, args.moe_inter_dim, args.vocab_size = 64, 64, 64
        args.n_heads, args.head_dim, args.rope_head_dim = 4, 32, 16
        args.q_lora_rank, args.o_lora_rank, args.o_groups = 32, 32, 1
        args.dspark_n_routed_experts, args.dspark_n_activated_experts = 2, 1
        args.dspark_markov_rank = 16
        args.max_seq_len, args.window_size, args.original_seq_len = 16, 8, 0
        args.n_mtp_layers, args.dspark_block_size, args.dspark_target_layer_ids = 3, 5, (37, 38, 39)
        with torch.device('meta'):
            prototypes = torch.nn.ModuleList([m.DSparkBlock(40 + i, args) for i in range(3)])
        mapping = {torch.float4_e2m1fn_x2: 'I8', torch.float8_e8m0fnu: 'F8_E8M0',
                   torch.float8_e4m3fn: 'F8_E4M3', torch.bfloat16: 'BF16', torch.float32: 'BF16'}
        catalog = types.SimpleNamespace(tensors={})
        for local, p in prototypes.named_parameters():
            name = 'mtp.' + local
            dtype = 'F8_E4M3' if name.endswith('attn.wo_a.weight') else mapping[p.dtype]
            size = 1 if dtype in ('I8', 'F8_E8M0', 'F8_E4M3') else 2
            catalog.tensors[name] = dict(shape=list(p.shape), dtype=dtype, bytes=p.numel() * size)
            if name.endswith('attn.wo_a.weight'):
                shape = [p.shape[0] // 32, p.shape[1] // 32]
                catalog.tensors[name[:-6] + 'scale'] = dict(shape=shape, dtype='F8_E8M0', bytes=shape[0] * shape[1])
        del prototypes
        plan = prepare_dspark(m, args, catalog)
        fake = object.__new__(MTPResidentStore)
        fake.catalog, fake.root, fake.allow_download = catalog, mtp, False
        fake.records = {name: {} for name in plan.common_names}
        fake.cap_bytes, fake.stored_bytes, fake.downloaded_bytes = 1 << 20, 0, 0
        requested = []
        def ensure(this, names):
            assert all(name in this.records for name in names)
        def tensor(this, name):
            requested.append(name)
            meta = this.catalog.tensors[name]
            dtype = DTYPES[meta['dtype']]
            return torch.ones(meta['shape'], dtype=dtype, device='cpu')
        fake.ensure = types.MethodType(ensure, fake)
        fake.tensor = types.MethodType(tensor, fake)
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(40)])
        model.mtp = torch.nn.ModuleList()
        model.embed = torch.nn.Embedding(64, 64)
        model.head = torch.nn.Linear(64, 64, bias=False)
        model.target_layer_ids = ()
        model.eval()
        embed, head = model.embed, model.head
        result = install_dspark(model, m, plan, fake)
        assert len(model.layers) == 40 and len(model.mtp) == 3
        assert model.target_layer_ids == (37, 38, 39)
        assert all(stage.embed is embed and stage.head is head for stage in model.mtp)
        assert all(not stage.training for stage in model.mtp)
        assert set(requested) == set(plan.common_names)
        assert not any('.ffn.experts.' in name for name in requested)
        assert not any(name.startswith(('embed.', 'head.')) or name in
                       {f'mtp.{i}.{kind}.weight' for i in range(3) for kind in ('embed', 'head')}
                       for name in requested)
        assert all(p.is_meta for stage in model.mtp for expert in stage.ffn.experts for p in expert.parameters())
        assert all(not b.is_meta for b in model.mtp.buffers())
        assert result['downloaded_bytes'] == 0 and not result['draft_execution_enabled']
        verified['tiny_official_stage_attachment_without_forward'] = True
        verified['shared_embed_head_not_reloaded_and_alias_identity_preserved'] = True
        verified['common_native_load_conversions_and_lazy_meta_experts'] = True
        verified['cpu_buffers_recreated_and_training_state_preserved'] = True
    paths = [Path(__file__), BASE / 'goal_dspark_loader_0910.py', BASE / 'goal_dspark_store_0910.py']
    report = dict(passed=True, real_catalog_plan=production_report, verified=verified,
                  cpu_affinity=[0], torch_threads=1, real_weights_loaded=False,
                  model_forward_executed=False, downloaded_bytes=0, performance_trial=False,
                  sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    out = BASE / 'results/goal_dspark_loader_0910'
    out.mkdir(exist_ok=True)
    (out / 'fixture-check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('real_catalog_plan', 'sha256')}))


if __name__ == '__main__':
    main()
