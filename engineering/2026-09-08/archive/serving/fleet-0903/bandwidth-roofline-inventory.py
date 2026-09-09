#!/usr/bin/env python3
"""Count one logical weight read per raw token; this is not a DRAM measurement."""
import importlib.util
import json
from pathlib import Path
import re

base = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('header_parser', base.parent / 'glm-sr950/remote-gguf-bytes-per-token.py')
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
models = {
    'GLM-5.3-Flash': '/models/gguf/GLM-5.3-Flash/UD-IQ2_XXS',
    'Qwen3.8-Flash-Next': '/models/gguf/Qwen3.8-Flash-Next/UD-Q2_K_XL',
    'GLM-5.3-Full': '/models/GLM-5.3-GGUF/UD-Q4_K_XL',
}
results = []
for name, directory in models.items():
    tensors = []
    metadata = {}
    for path in sorted(Path(directory).glob('*.gguf')):
        with path.open('rb') as file:
            data = file.read(24 * 1024 * 1024)
        _, meta, entries = parser.parse_header(data)
        if not metadata:
            metadata = meta
        tensors.extend(entries)
    assert tensors, name
    arch = metadata['general.architecture']
    used = metadata[arch + '.expert_used_count']
    layers = metadata[arch + '.block_count'] - metadata.get(arch + '.nextn_predict_layers', 0)
    records = []
    for tensor_name, dims, kind in tensors:
        layer = re.match(r'blk\.(\d+)\.', tensor_name)
        if layer and int(layer[1]) >= layers:
            continue
        if not tensor_name.endswith('.weight'):
            continue
        size = parser.tensor_bytes(dims, kind)
        indexed = 'token_embd' in tensor_name
        routed = '_exps.' in tensor_name and len(dims) > 2
        active = size * used / dims[2] if routed else size
        if indexed:
            active = 0  # indexed row reads are small; no full embedding-table scan
        # Approximate physical layout. This does not count replicated matrices,
        # repeated loads, activation/KV traffic, or cache reuse.
        physical = active
        layout = parser.TYPE_INFO[kind][0]
        if not indexed and dims[0] % 256 == 0 and len(dims) >= 2:
            if name != 'GLM-5.3-Full' and kind in (16, 17, 18) and dims[1] % 16 == 0:
                physical *= 2208 / (16 * parser.TYPE_INFO[kind][1])
                layout = 'IQ r16 (4.3125 bits/weight)'
            elif kind in (12, 13, 14) and dims[1] % 16 == 0:
                physical *= {12: 2368, 13: 2880, 14: 3360}[kind] / (16 * parser.TYPE_INFO[kind][1])
                layout += ' x16'
            elif name == 'GLM-5.3-Full' and kind == 8 and dims[1] % 16 == 0:
                output = len(dims) == 2 and tensor_name == 'output.weight'
                shexp = len(dims) == 2 and '_shexp.weight' in tensor_name
                attention = len(dims) == 2 and '.attn_' in tensor_name
                if output or shexp or attention:
                    physical *= (3360 if output else 2880) / (16 * 256) / (34 / 32)
                    layout = 'Q6_K x16' if output else 'Q5_K x16'
        records.append(dict(name=tensor_name, shape=dims, type=parser.TYPE_INFO[kind][0],
                            active_bytes=active, layout_estimate_bytes=physical,
                            layout=layout, routed=routed, indexed=indexed))
    canonical = sum(r['active_bytes'] for r in records) / 1e9
    packed = sum(r['layout_estimate_bytes'] for r in records) / 1e9
    row = dict(model=name, directory=directory, active_experts=used, target_layers=layers,
               canonical_weight_gb_per_raw_token=canonical,
               layout_estimate_weight_gb_per_raw_token=packed,
               canonical_tok_s={str(p): 380*p/100/canonical for p in (80, 85, 95, 100)},
               layout_estimate_tok_s={str(p): 380*p/100/packed for p in (80, 85, 95, 100)},
               records=records)
    results.append(row)
    print(json.dumps({k:v for k,v in row.items() if k != 'records'}), flush=True)
out = dict(bandwidth_gb_s=380, models=results,
           scope='Raw batch-one logical weight traffic, one model using the entire server.',
           limitations=['Not measured DRAM bytes.', 'Excludes MTP and KV/activation traffic.',
                        'Excludes replication, repeated reads, and cache reuse.',
                        'Packed layout estimates do not simulate per-NUMA tensor shapes or every runtime trait.'])
(base / 'results/bandwidth-roofline-inventory.json').write_text(json.dumps(out, indent=2) + '\n')
