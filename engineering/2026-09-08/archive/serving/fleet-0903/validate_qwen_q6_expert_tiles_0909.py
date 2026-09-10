"""Check Q6 expert tiles in the private scheduling build's lifecycle lock."""
import json
import math
from pathlib import Path
import statistics

from qwen_split_trial import sha256


def validate(result, save, run, out, private, parent, library, env, runtime_dir, flags, links):
    source_path = Path(__file__).resolve().parent / 'check-qwen-q6-expert-tiles-0909.cpp'
    source = out / source_path.name
    source.write_bytes(source_path.read_bytes())
    result['private_source_sha256'][str(source)] = sha256(source)
    binary = out / 'q6-expert-tiles-check'
    run([*flags,str(source),*links,'-o',str(binary)],'compile-expert-tiles')
    result['expert_fixture_sha256'] = sha256(binary)
    result['expert_checks'], result['expert_timings'] = [], []

    def execute(name, tile, threads=15, timing=False, is_parent=False):
        expected = Path(parent['library']) if is_parent else library
        trial = dict(env,LD_LIBRARY_PATH=str(expected.parent)+':'+str(runtime_dir),
                     GGML_CPU_QWEN_HC_ORDERED_K='0',GGML_CPU_QWEN_HC_ROW_SPLIT='0',
                     GGML_CPU_QWEN_Q6_MOE_TILE_ROWS=str(tile),GGML_CPU_QWEN_Q6_MOE_TILE_AUDIT='0' if timing else '1')
        values = out/(name+'.bin')
        log = run(['taskset','-c','0-14',str(binary),'--timing' if timing else str(values),str(threads)],name,trial)
        maps = [line.removeprefix('CPU_LIBRARY ') for line in log.splitlines() if line.startswith('CPU_LIBRARY ')]
        assert len(maps)==1 and Path(maps[0]).resolve()==expected.resolve()
        assert 'MOE_COUNTER_PRESENT '+str(int(not is_parent))+'\n' in log
        rows = [json.loads(line.removeprefix('MOE_CASE ')) for line in log.splitlines() if line.startswith('MOE_CASE ')]
        summary = [json.loads(line.removeprefix('MOE_SUMMARY ')) for line in log.splitlines() if line.startswith('MOE_SUMMARY ')]
        assert summary == [dict(cases=6 if timing else 28,failures=0,weights_preserved=True)]
        assert len(rows)==(6 if timing else 28) and all(row['passed'] for row in rows)
        for row in rows:
            active = not timing and not is_parent and threads>1 and tile!=64 and row['nr']<=8
            assert (row['selected_calls']>0)==active,(name,row)
            assert row['samples']==(128 if timing else 3)
            if timing:
                assert row['active_experts']==(512 if row['rotating'] else 8 if row['nr']==1 else 40 if row['disjoint'] else 20)
        record = dict(name=name,tile=tile,threads=threads,parent=is_parent,cpu_sha256=sha256(expected),rows=rows)
        if not timing:
            record.update(output_bytes=values.stat().st_size,output_sha256=sha256(values))
        return record

    reference = None
    for name,tile,threads,is_parent in [('expert-parent',64,15,True),('expert-off',64,15,False),
                                      ('expert-tile16',16,15,False),('expert-tile32',32,15,False),('expert-tile48',48,15,False),
                                      ('expert-one-worker',32,1,False),('expert-four-workers',32,4,False)]:
        record = execute(name,tile,threads,is_parent=is_parent)
        if reference is None:
            reference = record
        assert record['output_sha256']==reference['output_sha256'] and record['output_bytes']==reference['output_bytes']
        result['expert_checks'].append(record)
        save()
    for index,tile in enumerate((None,64,16,32,48,48,32,16,64,None)):
        record = execute(f'expert-timing-{index}',64 if tile is None else tile,timing=True,is_parent=tile is None)
        result['expert_timings'].append(record)
        save()
    comparisons = []
    for row in result['expert_timings'][0]['rows']:
        key = {name:row[name] for name in ('nr','rotating','disjoint','active_experts')}
        matching = [(trial,other) for trial in result['expert_timings'] for other in trial['rows'] if all(other[k]==v for k,v in key.items())]
        assert len(matching)==10 and len({other['hash'] for _,other in matching})==1
        before = statistics.mean(other['median_us'] for trial,other in matching if trial['parent'])
        off = statistics.mean(other['median_us'] for trial,other in matching if not trial['parent'] and trial['tile']==64)
        for tile in (16,32,48):
            times = [other['median_us'] for trial,other in matching if trial['tile']==tile]
            after = statistics.mean(times)
            comparisons.append(dict(**key,tile=tile,parent_us=before,private_off_us=off,candidate_us=after,
                                    samples_us=times,speedup=before/after,change_percent=100*(after/before-1)))
    result['expert_comparisons'] = comparisons
    eligibility = []
    for tile in (16,32,48):
        rows = [row for row in comparisons if row['tile']==tile and row['rotating']]
        assert len(rows)==3
        ratio = math.exp(statistics.mean(math.log(row['speedup']) for row in rows))
        eligibility.append(dict(tile=tile,rotating_geomean_speedup=ratio,
                                model_test_eligible=ratio>=1.03 and all(row['speedup']>=1/1.05 for row in rows)))
    result.update(expert_bit_exact=True,expert_eligibility=eligibility)
    save()
