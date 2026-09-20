#!/usr/bin/env python3
"""Attribute operation traces per graph; never add different graph executions together."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

HEADER=re.compile(r'CPU_OP_PROFILE index=(\d+) cpu=(\d+) graph=(0x[0-9a-f]+) nodes=(\d+) total=([\d.]+) ms barrier=([\d.]+) ms')
ROW=re.compile(r"CPU_OP_PROFILE cpu=(\d+) graph=(0x[0-9a-f]+) node=(\d+) op=(\w+) time=([\d.]+) ms bar=([\d.]+) ms name='([^']*)'(?: src0_type=(\w+) src0_ne=\[([^]]+)\] src0_name='([^']*)')?")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('log',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    current={};graphs=[]
    for line in args.log.read_text(errors='replace').splitlines():
        if m:=HEADER.search(line):
            ix,cpu,ptr,nodes,total,bar=m.groups()
            graph={'index':int(ix),'cpu':int(cpu),'graph':ptr,'declared_nodes':int(nodes),
                   'total_ms':float(total),'reported_barrier_ms':float(bar),'rows':[]}
            graphs.append(graph);current[(cpu,ptr)]=graph
        elif m:=ROW.search(line):
            cpu,ptr,node,op,ms,bar,name,typ,shape,src=m.groups()
            if (cpu,ptr) in current:
                current[(cpu,ptr)]['rows'].append({'node':int(node),'op':op,'ms':float(ms),'barrier_ms':float(bar),
                                                  'name':name,'src0_type':typ,'src0_ne':shape,'src0_name':src})
    summaries=[]
    for graph in graphs:
        ops=defaultdict(float);dense=defaultdict(float);families=defaultdict(float)
        for row in graph['rows']:
            ops[row['op']]+=row['ms']
            family=re.sub(r'blk\.\d+\.', 'blk.#.',row['src0_name'] or row['name'])
            if row['op']=='MUL_MAT': dense[family]+=row['ms']
            families[(row['op'],re.sub(r'-\d+$','-#',row['name']))]+=row['ms']
        summary={k:v for k,v in graph.items() if k!='rows'}
        summary['profiled_rows']=len(graph['rows'])
        summary['operation_ms']=dict(sorted(ops.items(),key=lambda kv:-kv[1]))
        summary['dense_family_ms']=dict(sorted(dense.items(),key=lambda kv:-kv[1]))
        summary['top_operations']=[{'op':k[0],'name':k[1],'ms':v} for k,v in sorted(families.items(),key=lambda kv:-kv[1])[:20]]
        summary['input_shape_evidence']=[r for r in graph['rows'] if r['name']=='hc_init' or (r['op']=='RMS_NORM' and r['src0_ne'] and r['src0_ne'].startswith('16384,'))][:2]
        summaries.append(summary)
    assert summaries, 'No operation traces found'
    args.output.write_text(json.dumps({'source':str(args.log),'graphs':summaries},indent=2)+'\n')
    for s in summaries:
        print(f"index={s['index']} cpu={s['cpu']} nodes={s['declared_nodes']} total={s['total_ms']:.3f}ms",list(s['operation_ms'].items())[:5])

if __name__=='__main__': main()
