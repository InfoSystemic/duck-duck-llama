#!/usr/bin/env python3
"""Compare the two pooling runs with both saved 15-worker clamp controls."""
import json
from pathlib import Path

from compare_flash_q8_clamp_0908 import sha, stream_text

BASE = Path(__file__).resolve().parent
LABELS = [
    'glm-flash-q8-clamp-raw-0908',
    'glm-flash-q8-clamp-t15-restored-raw-0908',
    'glm-flash-q8-pool-raw-0908',
    'glm-flash-q8-pool-raw-repeat-0908',
]


def main():
    paths = [BASE/'results'/label/'result.json' for label in LABELS]
    runs = [json.loads(path.read_text()) for path in paths]
    for run in runs:
        assert run.get('finished') and not run.get('error') and run['input_integrity_verified']
        assert all(check['pass_check'] for check in run['checks'])
        assert run['server_command'][1:] == runs[0]['server_command'][1:]
        assert sha(run['server_command'][0]) == sha(runs[0]['server_command'][0])
        for key in ('port','alias','drafts','tokens','request_timeout_seconds','bandwidth_capacity_gb_s',
                    'bandwidth_target_gb_s','chat_template_kwargs','reasoning_budget_tokens','check_reasoning_budget_tokens'):
            assert run['config'][key] == runs[0]['config'][key], key
    assert runs[0]['runtime_env'] == runs[1]['runtime_env']
    assert runs[2]['runtime_env'] == runs[3]['runtime_env']
    assert runs[2]['target_pid'] == runs[3]['target_pid']
    changes = {key:[runs[0]['runtime_env'].get(key),runs[2]['runtime_env'].get(key)]
        for key in runs[0]['runtime_env'].keys() | runs[2]['runtime_env'].keys()
        if runs[0]['runtime_env'].get(key) != runs[2]['runtime_env'].get(key)}
    assert set(changes) == {'LD_LIBRARY_PATH','GGML_CPU_SOFTMAX_POOL_FUSION'}
    assert changes['GGML_CPU_SOFTMAX_POOL_FUSION'] == [None,'1']
    rows, streams = [], {}
    for kind in ('prose','code'):
        values = [next(x for x in run['measurements'] if x['kind']==kind) for run in runs]
        assert all(x['prompt']==values[0]['prompt'] and x['draft_n']==0 for x in values)
        texts = []
        for label,path,value in zip(LABELS,paths,values):
            chunks = path.parent/(kind+'-draft0/chunks.json')
            texts.append(stream_text(chunks))
            streams[str(chunks)] = sha(chunks)
            rows.append(dict(label=label,kind=kind,tok_s=value['timings']['predicted_per_second'],
                adjusted_gb_s=value['background_subtracted_gb_s'],
                utilization=value['background_subtracted_utilization'],completed_answer=value['completed_answer']))
        assert all(text==texts[0] for text in texts), kind+' complete streamed text changed'
    result = dict(passed=True,all_full_streams_equal=True,rows=rows,environment_changes=changes,
        source_sha256={str(p):sha(p) for p in [Path(__file__),*paths]},stream_sha256=streams,
        target_reached=all(row['adjusted_gb_s']>=285 for row in rows if 'pool-' in row['label']),
        note='Two saved controls followed by two candidate runs; not an interleaved or same-binary on/off experiment. These measurements do not certify source-model quality.')
    out = BASE/'results/glm-flash-q8-pool-repeat-comparison-0908.json'
    with out.open('x') as stream: json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=True,all_full_streams_equal=True,rows=rows,target_reached=result['target_reached']),indent=2))


if __name__ == '__main__':
    main()
