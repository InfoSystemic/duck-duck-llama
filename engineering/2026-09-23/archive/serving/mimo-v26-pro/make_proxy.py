#!/usr/bin/env python3
# Truncated MiMo-V2.6-Pro proxy for engine parity tests: a few trunk layers + the 3 nextn (MTP) layers,
# renumbered, tensors copied raw (MXFP4 / Q8_0 bytes untouched). Default keeps trunk layers 0 (global
# attention + dense FFN), 1-2 (SWA + MoE) and 7 (global + MoE): every kind of layer the model has.
#   make_proxy.py OUT.gguf [comma-separated trunk layers]
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "convert" / "gguf-py"))
import gguf  # noqa: E402

out = Path(sys.argv[1])
keep = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 7]
readers = [gguf.GGUFReader(p) for p in sorted(Path("/models/mimo-v26-pro/gguf").glob("MiMo-V2.6-Pro-RL-MXFP4_MOE-*-of-*.gguf"))]
r0 = readers[0]
arch = r0.fields["general.architecture"].contents()
n_all = r0.fields[f"{arch}.block_count"].contents()
n_nextn = r0.fields[f"{arch}.nextn_predict_layers"].contents()
old_ids = keep + list(range(n_all - n_nextn, n_all))
remap = {o: i for i, o in enumerate(old_ids)}

writer = gguf.GGUFWriter(out, arch)
for field in r0.fields.values():
    if field.name == "general.architecture" or field.name.startswith(("GGUF.", "split.")):
        continue
    vtype = field.types[0]
    sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
    val = field.contents()
    if field.name == f"{arch}.block_count":
        val = len(old_ids)
    elif sub is not None and field.name.startswith(arch + ".") and isinstance(val, list) and len(val) == n_all:
        val = [val[o] for o in old_ids]  # per-layer arrays: head_count_kv, sliding_window_pattern
    writer.add_key_value(field.name, val, vtype, sub_type=sub)

tensors = []
for rd in readers:
    for t in rd.tensors:
        m = re.match(r"^blk\.(\d+)\.(.*)$", t.name)
        if m and int(m.group(1)) not in remap:
            continue
        tensors.append((f"blk.{remap[int(m.group(1))]}.{m.group(2)}" if m else t.name, t))
for name, t in tensors:
    writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for name, t in tensors:
    writer.write_tensor_data(t.data)
writer.close()
print(f"wrote {out}: {len(tensors)} tensors, layers {old_ids} -> 0..{len(old_ids) - 1} ({len(keep)} trunk + {n_nextn} nextn)")
