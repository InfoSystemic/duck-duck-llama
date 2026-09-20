#!/usr/bin/env python3
import json,statistics
from pathlib import Path
from validate import run,ARMS,D
report={'rows':[],'complete':False}
for pools in [48,1026,8194,25002]:
    reference=None
    for order,idx in enumerate([1,4,4,1]):
        row,data=run(ARMS[idx],15,pools=pools,suffix=f'-p{pools}-order{order}')
        if reference is None:reference=data
        assert data==reference
        row['bit_exact']=True;row['order']=order
        report['rows'].append(row)
        (D/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
        print(pools,row['arm'],[round(x,4) for x in row['milliseconds']],flush=True)
report['summary']=[]
for pools in [48,1026,8194,25002]:
    values={arm:statistics.median([v for r in report['rows'] if r['pools']==pools and r['arm']==arm for v in r['milliseconds']]) for arm in ['deployed','candidate-wide']}
    values.update(pools=pools,speedup=values['deployed']/values['candidate-wide'])
    report['summary'].append(values)
report['complete']=True
(D/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['summary'],indent=2),flush=True)
