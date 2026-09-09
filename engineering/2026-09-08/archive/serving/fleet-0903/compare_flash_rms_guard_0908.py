#!/usr/bin/env python3
"""Compare the guarded RMS run with the fresh and retained sum16 controls."""
import argparse
import json
from pathlib import Path

from compare_flash_pool_0908 import sha, stream_text

BASE = Path(__file__).resolve().parent
REFERENCES = [BASE/f'results/{label}/result.json' for label in
              ('glm-flash-rms-guard-control-raw-0908','glm-flash-q8-sum16-raw-0908','glm-flash-q8-sum16-raw-repeat-0908')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--parity-label',required=True)
    args = parser.parse_args()
    for label in (args.label,args.parity_label):
        assert label and set(label) <= set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')
    candidate_path = BASE/f'results/{args.label}/result.json'
    parity_path = BASE/f'results/{args.parity_label}/result.json'
    b = json.loads(candidate_path.read_text())
    parity = json.loads(parity_path.read_text())
    assert parity['passed'] and all(x['tokens_equal'] for x in parity['responses'])
    assert len(parity['cache_checks']) == 4 and all(x['passed'] for x in parity['cache_checks'])
    assert b['target_pid'] == parity['current']['pid'] and b['runtime_env'] == parity['current']['runtime_env']
    assert parity['current']['dense_chunk'] == 16 and parity['current']['sum16'] and parity['current']['rms_guard']
    preflight = json.loads((BASE/'results/glm-flash-rms-guard-control-0908/preflight.json').read_text())
    assert preflight['passed'] and all(parity['current'][k] == v for k,v in preflight['candidate'].items())
    rows = []
    for reference_path in REFERENCES:
        a = json.loads(reference_path.read_text())
        for result in (a,b):
            assert result.get('finished') and not result.get('error') and result['input_integrity_verified']
            assert all(x['pass_check'] for x in result['checks'])
        assert a['server_command'][1:] == b['server_command'][1:]
        assert sha(a['server_command'][0]) == sha(b['server_command'][0])
        differences = {key:[a['runtime_env'].get(key),b['runtime_env'].get(key)]
                       for key in a['runtime_env'].keys() | b['runtime_env'].keys()
                       if a['runtime_env'].get(key) != b['runtime_env'].get(key)}
        assert differences == preflight['runtime_env_changes']
        assert a['runtime_env']['GGML_CPU_X16_CHUNK_MAX'] == b['runtime_env']['GGML_CPU_X16_CHUNK_MAX'] == '16'
        for key in ('port','alias','drafts','tokens','request_timeout_seconds','bandwidth_capacity_gb_s',
                    'bandwidth_target_gb_s','chat_template_kwargs','reasoning_budget_tokens','check_reasoning_budget_tokens'):
            assert a['config'][key] == b['config'][key],key
        for newer in b['measurements']:
            older = next(x for x in a['measurements'] if x['kind'] == newer['kind'])
            assert older['prompt'] == newer['prompt'] and older['draft_n'] == newer['draft_n'] == 0
            before_path = reference_path.parent/f"{newer['kind']}-draft0/chunks.json"
            after_path = candidate_path.parent/f"{newer['kind']}-draft0/chunks.json"
            before,after = stream_text(before_path),stream_text(after_path)
            rows.append(dict(reference=str(reference_path),kind=newer['kind'],
                baseline_tok_s=older['timings']['predicted_per_second'],candidate_tok_s=newer['timings']['predicted_per_second'],
                speed_change_percent=100*(newer['timings']['predicted_per_second']/older['timings']['predicted_per_second']-1),
                baseline_adjusted_gb_s=older['background_subtracted_gb_s'],candidate_adjusted_gb_s=newer['background_subtracted_gb_s'],
                candidate_utilization=newer['background_subtracted_utilization'],completed_answer=newer['completed_answer'],
                stream_fields_equal={key:before[key] == after[key] for key in before},
                stream_sha256={str(p):sha(p) for p in (before_path,after_path)}))
    result = dict(input_comparison_passed=True,rows=rows,runtime_env_changes=preflight['runtime_env_changes'],
                  server_binary_sha256=sha(b['server_command'][0]),
                  all_streams_equal=all(all(x['stream_fields_equal'].values()) for x in rows),
                  target_reached=all(x['candidate_adjusted_gb_s'] >= 285 for x in rows),
                  source_sha256={str(p):sha(p) for p in [Path(__file__),BASE/'compare_flash_pool_0908.py',*REFERENCES,candidate_path,parity_path]},
                  note='The fresh control immediately precedes this candidate configuration; older controls are also shown. Controls are not interleaved. Stream text equality does not establish all token IDs or source-model quality.')
    output = BASE/f'results/{args.label}-comparison.json'
    assert not output.exists()
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
