#!/usr/bin/env python3
"""Compare pinned Q6 XL and Q8 target tensor inventories before changing precision."""
import collections
import json
from pathlib import Path
import time

from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent


def main():
    source = BASE / 'results/qwen-higher-quant-bandwidth-inventory-0907.json'
    selection_path = BASE / 'results/qwen-higher-quant-selection-0907.json'
    inventory, selection = [json.loads(path.read_text()) for path in (source, selection_path)]
    assert inventory['revision'] == selection['revision'] == '38bb39ee97821de2c9009abb7e93950eec396e66'
    assert inventory['repo'] == selection['repo'] == 'unsloth/Qwen3.8-Flash-Next-GGUF'
    parent, candidate = [next(row for row in inventory['models'] if row['quant'] == quant) for quant in ('UD-Q6_K_XL', 'Q8_0')]
    before, after = [{row['name']: row for row in model['records']} for model in (parent, candidate)]
    assert set(before) == set(after) and len(before) == len(parent['records']) == len(candidate['records']) == 1152
    assert all(before[name]['shape'] == after[name]['shape'] for name in before)
    changes = []
    for name, old in before.items():
        new = after[name]
        if old['type'] != new['type']:
            changes.append(dict(name=name, shape=old['shape'], before=old['type'], after=new['type'],
                                before_active_bytes=old['active_bytes'], after_active_bytes=new['active_bytes']))
    counts = collections.Counter((row['before'], row['after']) for row in changes)
    assert counts == {('Q6_K', 'Q8_0'): 94, ('F32', 'Q8_0'): 168}
    experts = [row for row in changes if row['before'] == 'Q6_K']
    assert all(row['name'].endswith(('ffn_gate_exps.weight', 'ffn_up_exps.weight')) for row in experts)
    assert all(row['shape'] == [2560, 640, 512] for row in experts)
    active_delta = sum(row['after_active_bytes']-row['before_active_bytes'] for row in experts)
    # Routed tensors select ten of 512 experts; this excludes dense and state traffic.
    stored_delta = active_delta*512//10
    assert active_delta*512 % 10 == 0
    out = BASE / 'results/qwen-q8-precision-assessment-0910.json'
    result = dict(time=time.time(), passed=True, input_sha256={str(path): sha256(path) for path in (Path(__file__).resolve(), source, selection_path)},
                  revision=inventory['revision'], repo=inventory['repo'], inventoried_target_tensors=1152,
                  quantization_changes=changes, expert_upgrade_tensors=94, fp32_to_q8_tensors=168,
                  q6_file_bytes=selection['candidates']['UD-Q6_K_XL']['bytes'],
                  q8_file_bytes=selection['candidates']['Q8_0']['bytes'],
                  expert_only_upgrade_additional_stored_bytes=stored_delta,
                  expert_only_upgrade_additional_active_bytes_per_raw_token=active_delta,
                  expert_only_upgrade_weight_estimate_gb_per_raw_token=parent['layout_estimate_weight_gb_per_raw_token']+active_delta/1e9,
                  q8_file_is_uniform_precision_upgrade=False, payload_identity_checked=False,
                  candidate_downloaded=False, model_built=False, model_tested=False, quality_gain_established=False,
                  scope='Static comparison of previously fetched, pinned target inventories. Q8 raises 94 expert tensors from Q6 but also quantizes 168 currently FP32 HC-injection/SSM tensors. An expert-only Q8 mixture would preserve those FP32 tensors; it has not been built. Weight-traffic estimates exclude activation, state and speculative traffic and establish neither speed nor IMC utilization.')
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('quantization_changes', 'input_sha256')}, indent=2))


if __name__ == '__main__':
    main()
