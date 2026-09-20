#!/usr/bin/env python3
import hashlib,json,os,struct
from validate import run,ARMS,D
report={'candidate_sha256':hashlib.sha256((D/'build/libggml-cpu.so.0.22.0').read_bytes()).hexdigest(),'rows':[],'passed':False}
control=D/'validation/control-probe.u32'
control.write_bytes(struct.pack('<I',0))
try:
    os.environ['GGML_CPU_GLM_POOL_WIDE_CONTROL_FILE']=str(control)
    os.environ['POOL_CONTROL_PROBE']='1'
    for threads in [3,15]:
        row,data=run(ARMS[4],threads,suffix='-control')
        assert data==(D/f'validation/pool-t{threads}-deployed.bin').read_bytes()
        row.update(bit_exact=True,toggle_each_execution=True)
        report['rows'].append(row)
        print('PASS dynamic switch',threads,flush=True)
finally:
    os.environ.pop('GGML_CPU_GLM_POOL_WIDE_CONTROL_FILE',None)
    os.environ.pop('POOL_CONTROL_PROBE',None)
ref=None
for arm in [ARMS[0],ARMS[1],ARMS[4]]:
    row,data=run(arm,3,kind='copy')
    if ref is None:ref=data
    assert data==ref
    row['bit_exact']=True;report['rows'].append(row)
    print('PASS copy regression',arm[0],flush=True)
report['passed']=True
(D/'validation-control-copy.json').write_text(json.dumps(report,indent=2)+'\n')
