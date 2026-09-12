#!/usr/bin/env python3
"""Export bounded CPU oracles from the unchanged publisher V4.1 Compressor.

Synthetic projection weights isolate pooling and cache cadence from checkpoint
availability and quantized GEMM errors. These fixtures are not model inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

FLEET_0903 = Path(__file__).resolve().parents[2] / 'fleet-0903'
sys.path.insert(0, str(FLEET_0903))
import deepseek_v41_cpu_reference_0910 as cpu


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(False)
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.bfloat16)
    module = cpu.load_official_model('deepseek_v41_compressor_oracle_0912')
    arrays, cases = {}, []
    expected_values = 0

    for ratio in (1, 2):
        for width in (32, 512):
            config = module.ModelArgs(dim=width, head_dim=width, max_batch_size=2,
                                      max_seq_len=16, compress_ratios=(ratio,), norm_eps=1e-20)

            def create():
                compressor = module.Compressor(config, 0).eval()
                # Diagonal powers of two make projection exact across GEMM batch
                # shapes, leaving the publisher's pooling and BF16 round points.
                diagonal = torch.arange(width).remainder(5).float().sub(2).exp2()
                compressor.wkv.weight.copy_(torch.diag(diagonal).to(compressor.wkv.weight.dtype))
                compressor.norm.weight.copy_((1 + torch.arange(width).remainder(7).float() / 8).bfloat16())
                if ratio > 1:
                    gate = torch.arange(width).remainder(7).float().sub(3) / 4
                    compressor.wgate.weight.copy_(torch.diag(gate))
                return compressor

            generator = torch.Generator().manual_seed(410912 + width + ratio)
            x = torch.randn(2, 16, width, dtype=torch.float32, generator=generator).bfloat16()
            x[:, 0, ::7] = 0
            x[:, 1, ::9] = -0.
            # Highly unequal gates also exercise softmax stability and pooling.
            x[:, 6, ::8] = 64
            x[:, 7, ::8] = -64
            source = create()
            prefix = f'ratio{ratio}_width{width}'
            arrays[prefix + '_input'] = x.float().numpy()
            for name, value in source.state_dict().items():
                arrays[prefix + '_weight_' + name.replace('.', '_')] = value.float().numpy()
            full = source(x, 0)
            assert full is not None and full.shape == (2, 16 // ratio, width)
            assert torch.isfinite(full).all()
            arrays[prefix + '_full'] = full.float().numpy()

            for prefill in (1, 2, 3, 6, 7):
                compressor = create()
                pieces, emissions = [], []
                first = compressor(x[:, :prefill], 0)
                if first is not None:
                    pieces.append(first.clone())
                assert (first is not None) == (prefill >= ratio)
                for pos in range(prefill, 16):
                    output = compressor(x[:, pos:pos + 1], pos)
                    should_emit = (pos + 1) % ratio == 0
                    assert (output is not None) == should_emit, (ratio, width, prefill, pos)
                    emissions.append(dict(position=pos, emits=should_emit,
                                          compressed_index=pos // ratio if should_emit else None,
                                          rope_position=(pos + 1 - ratio) if should_emit else None))
                    if output is not None:
                        pieces.append(output.clone())
                joined = torch.cat(pieces, dim=1)
                assert torch.equal(joined, full), (ratio, width, prefill, 'streaming vs full')
                expected_values += joined.numel()

                # Reset after a completed request and after an uncompleted group.
                for stale_end in (15, 16):
                    reset = create()
                    reset(x[:, :stale_end], 0)
                    fresh = reset(x[:, :prefill], 0)
                    assert (fresh is None) == (first is None)
                    if fresh is not None:
                        assert torch.equal(fresh, first)
                    for pos in range(prefill, 16):
                        value = reset(x[:, pos:pos + 1], pos)
                        if value is not None:
                            assert torch.equal(value, full[:, pos // ratio:pos // ratio + 1])
                            expected_values += value.numel()
                cases.append(dict(ratio=ratio, width=width, batch=2, prefill_tokens=prefill,
                                  exact_streaming_vs_full=True, exact_request_reset=True,
                                  decode_emissions=emissions))

    args.output.mkdir(parents=True, exist_ok=True)
    arrays_path = args.output / 'compressor-oracle.npz'
    np.savez_compressed(arrays_path, **arrays)
    paths = [Path(__file__), Path(cpu.__file__), cpu.OFFICIAL / 'model.py',
             cpu.OFFICIAL / 'config.json', cpu.OFFICIAL / 'engram.py', cpu.OFFICIAL / 'vision.py',
             cpu.EXTRA / 'image_processor.py', arrays_path]
    result = dict(passed=True, official_model_revision='fb2764a5cf321eaa5070ca8f9e892818f477c16d',
                  component_only=True, full_checkpoint_loaded=False, gpu_reference_used=False,
                  torch_version=torch.__version__, torch_threads=1, synthetic_weights=True,
                  batch_sizes=[2], head_widths=[32, 512], compression_ratios=[1, 2],
                  exact_checked_values=expected_values, cases=cases,
                  npz=str(arrays_path.resolve()),
                  array_encoding='float32 arrays containing exactly represented BF16 input/output values; weights retain F32 or BF16 values',
                  arithmetic='Publisher Compressor.forward including BF16 rounding before RMSNorm, learned softmax gate, and per-group decode state',
                  ratio1='BF16 projection and normalization; no gate or state',
                  ratio2='F32 projection and gate; emit only on complete pairs; RoPE uses the first position of the pair',
                  limitations=['No checkpoint weights, Engram, sparse-index candidates, GGML graph, or full-model logits are checked.',
                               'Compressed KV FP4 and window KV FP8 quantization happen after Compressor.forward and are not covered.',
                               'Only fresh prefill at start_pos=0 followed by single-token decode is covered, matching this reference.'],
                  source_sha256={str(path.resolve()): digest(path) for path in paths})
    (args.output / 'compressor-oracle.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('passed', 'exact_checked_values', 'head_widths',
                                           'compression_ratios', 'full_checkpoint_loaded', 'npz')}))


if __name__ == '__main__':
    with torch.inference_mode():
        main()
