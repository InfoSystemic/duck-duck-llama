#!/usr/bin/env python3
"""Aggregate CPU_OP_PROFILE lines from a llama-server log into per-category time for one device.
Usage: agg_profile.py <log> [cpu_filter_node]  (node = NUMA node 0..3, picks lines whose cpu is on that node)
"""
import sys, re, collections
log = sys.argv[1]
node = int(sys.argv[2]) if len(sys.argv) > 2 else 0
def node_of(cpu):
    return (cpu % 64) // 16
op_re = re.compile(r"CPU_OP_PROFILE cpu=(\d+) graph=(0x[0-9a-f]+) node=(\d+) op=(\S+) time=([0-9.]+) ms name='([^']*)' src0_type=(\S+) src0_ne=\[(\d+),(\d+),(\d+)\] src0_name='([^']*)'")
hdr_re = re.compile(r"CPU_OP_PROFILE index=(\d+) cpu=(\d+) graph=(0x[0-9a-f]+) nodes=(\d+) total=([0-9.]+) ms first='([^']*)' last='([^']*)'")
by_cat = collections.Counter(); by_cat_n = collections.Counter()
by_op = collections.Counter(); by_tensor = collections.Counter(); by_tensor_n = collections.Counter()
graphs = []; total = 0.0; nlines = 0
def cat(op, name, src0):
    s = re.sub(r'blk\.\d+\.', 'blk.N.', src0); n = re.sub(r'-\d+$', '', name)
    if 'exps' in s: return 'routed_experts:' + s
    if 'shexp' in s: return 'shared_expert:' + s
    if '.attn_' in s or 'indexer' in s: return 'attention_mm:' + s
    if s.startswith('blk.N.ffn_') and 'inp' not in s: return 'dense_ffn:' + s
    if s == 'output.weight': return 'lm_head'
    return 'other:' + op + ':' + n
for line in open(log, errors='replace'):
    m = hdr_re.search(line)
    if m:
        cpu = int(m.group(2))
        if node_of(cpu) == node:
            graphs.append((int(m.group(1)), int(m.group(4)), float(m.group(5)), m.group(6), m.group(7)))
        continue
    m = op_re.search(line)
    if not m: continue
    cpu = int(m.group(1))
    if node_of(cpu) != node: continue
    op, t, name, s0type, src0 = m.group(4), float(m.group(5)), m.group(6), m.group(7), m.group(11)
    nlines += 1; total += t
    c = cat(op, name, src0)
    by_cat[c] += t; by_cat_n[c] += 1
    by_op[op] += t
    key = (re.sub(r'blk\.\d+\.', 'blk.N.', src0) or re.sub(r'-\d+$', '', name), s0type, op)
    by_tensor[key] += t; by_tensor_n[key] += 1
print(f"node {node}: {len(graphs)} subgraphs, {nlines} op lines, sum op time {total:.1f} ms, sum graph totals {sum(g[2] for g in graphs):.1f} ms")
print(f"first graph idx {graphs[0][0] if graphs else None}, last {graphs[-1][0] if graphs else None}")
print("\n== by category ==")
for c, t in by_cat.most_common(40):
    print(f"{t:9.2f} ms {100*t/total:5.1f}%  n={by_cat_n[c]:5d}  {c}")
print("\n== by op ==")
for c, t in by_op.most_common(20):
    print(f"{t:9.2f} ms {100*t/total:5.1f}%  {c}")
print("\n== by tensor/op (top 30) ==")
for k, t in by_tensor.most_common(30):
    print(f"{t:9.2f} ms {100*t/total:5.1f}%  n={by_tensor_n[k]:5d}  {k}")
