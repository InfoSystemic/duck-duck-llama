#!/usr/bin/env python3
"""One-thread synthetic attention parity, not a model or timing benchmark."""
import json
import types

import torch

import deepseek_v41_cpu_reference_0910 as cpu
from goal_sparse_script_0910 import install


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(4100926)
    module = types.SimpleNamespace(sparse_attn=cpu.sparse_attn)
    candidate = install(module)
    cases = []
    with torch.inference_mode():
        for seq, heads, dim in [(1, 4, 32), (3, 4, 32), (1, 64, 512)]:
            for count in [0, 1, 63, 64, 65, 128, 193]:
                batch, positions = 2 if dim == 32 else 1, 211
                q = (torch.randn(batch, seq, heads, dim) * .8).bfloat16()
                kv = (torch.randn(batch, positions, dim) * .7).bfloat16()
                sink = torch.linspace(-12, 12, heads)
                # Deliberately unsorted and duplicated positions, with -1 entries
                # within and across 64-position softmax boundaries.
                ids = torch.randint(-1, positions, (batch, seq, count), dtype=torch.int32)
                if count:
                    ids[..., ::7] = -1
                scale = dim ** -.5
                expected = cpu.sparse_attn(q, kv, sink, ids, scale)
                for repeat in range(3):
                    actual = module.sparse_attn(q, kv, sink, ids, scale)
                    assert torch.equal(actual, expected), (seq, heads, dim, count, repeat,
                        int((actual != expected).sum()), float((actual.float() - expected.float()).abs().max()))
                assert torch.isfinite(actual).all()
                cases.append(dict(seq=seq, heads=heads, dim=dim, positions=count, exact=True))
        # No reachable keys, and a sink dominating otherwise reachable keys.
        q = torch.randn(1, 2, 4, 32).bfloat16()
        kv = torch.randn(1, 70, 32).bfloat16()
        ids = torch.full((1, 2, 65), -1, dtype=torch.int32)
        sink = torch.tensor([-8., 0., 8., 90.])
        assert torch.equal(module.sparse_attn(q, kv, sink, ids, .125), torch.zeros_like(q))
        ids = torch.arange(65, dtype=torch.int32).view(1, 1, 65).expand(1, 2, 65)
        expected = cpu.sparse_attn(q, kv, sink, ids, .125)
        assert torch.equal(module.sparse_attn(q, kv, sink, ids, .125), expected)
        # The published caller can hand the kernel strided q and kv tensors.
        qs = q.repeat_interleave(2, -1)[..., ::2]
        kvs = kv.repeat_interleave(2, -1)[..., ::2]
        assert not qs.is_contiguous() and not kvs.is_contiguous()
        assert torch.equal(module.sparse_attn(qs, kvs, sink, ids, .125),
                           cpu.sparse_attn(qs, kvs, sink, ids, .125))
        candidate.enabled = False
        assert torch.equal(module.sparse_attn(q, kv, sink, ids, .125), expected)
        candidate.enabled = True
        assert torch.equal(module.sparse_attn(q, kv, sink, ids, .125), expected)
        candidate.uninstall()
        assert module.sparse_attn is cpu.sparse_attn
    print(json.dumps(dict(passed=True, cases=cases, repeated_script_execution=True,
                          all_invalid_indices=True, sink_extremes=True, strided_inputs=True,
                          toggle_and_uninstall=True, full_checkpoint_loaded=False,
                          model_speed_measured=False), indent=2))


if __name__ == '__main__':
    main()
