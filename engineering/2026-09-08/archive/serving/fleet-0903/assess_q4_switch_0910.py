#!/usr/bin/env python3
"""Record quantization-specific byte estimates and the superseded Q8 trial."""
import json
from pathlib import Path
import time

from analyze_flash_bandwidth250_0910 import validate_counters
from flash_hugepages250_trial_0910 import measurement_rows
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/q4-switch-assessment-0910.json'


def main():
    assert not OUT.exists()
    flash4_path = BASE / 'results/glm-flash-q4-bandwidth-inventory-0907.json'
    flash8_path = BASE / 'results/glm-flash-q8-inventory-0908/result.json'
    qwen_path = BASE / 'results/qwen-higher-quant-bandwidth-inventory-0907.json'
    f4, f8, q = [json.loads(p.read_text()) for p in (flash4_path, flash8_path, qwen_path)]
    qw = {row['quant']: row for row in q['models']}
    pairs = [
        ('GLM-5.3-Flash', 'Q8_0', f8['canonical_active_weight_gb_per_raw_token'], f4['layout_estimate_weight_gb_per_raw_token']),
        ('Qwen3.8-Flash-Next', 'UD-Q6_K_XL', qw['UD-Q6_K_XL']['layout_estimate_weight_gb_per_raw_token'], qw['UD-Q4_K_XL']['layout_estimate_weight_gb_per_raw_token']),
    ]
    estimates = [dict(model=name, from_quant=quant, to_quant='UD-Q4_K_XL', before_gb_per_raw_token=before,
                      after_gb_per_raw_token=after, byte_reduction_percent=100*(1-after/before),
                      equal_bandwidth_weight_only_speedup_percent=100*(before/after-1),
                      raw_weight_only_tok_s_at_250_gb_s=250/after)
                 for name, quant, before, after in pairs]
    root = BASE / 'results/flash-hugepages250-comparison-0910'
    result, plan = [json.loads(p.read_text()) for p in (root/'result.json', root/'plan.json')]
    assert result['finished'] and result['qwen_restored'] and not result.get('restoration_error')
    assert 'InterruptedError' in result['error'] and not result['passed']
    assert result['plan_sha256'] == sha256(root/'plan.json')
    complete = [run for run in result['runs'] if run.get('finished')]
    assert len(complete) == 1 and complete[0]['configuration'] == 'raw_control'
    completed = complete[0]
    path = Path(completed['measurement'])
    assert sha256(path) == completed['measurement_sha256']
    rows = measurement_rows(path, plan['configurations']['raw_control'])
    assert rows == completed['rows']
    data = json.loads(path.read_text())
    counters = {row['kind']: validate_counters(path.parent/f"{row['kind']}-draft{row['draft_n']}", row)
                for row in data['measurements']}
    paths = [Path(__file__).resolve(), flash4_path, flash8_path, qwen_path, path, root/'plan.json', root/'result.json']
    assessment = dict(time=time.time(), passed=True, estimates=estimates,
                      flash_user_selected_q4=True, qwen_q6_retained=True,
                      superseded_q8_trial=dict(reason='User explicitly selected Q4 during the candidate load.',
                          comparison_complete=False, candidate_measurements=0, control_rows=rows,
                          control_counters=counters, exact_qwen_restore_verified_by_controller=True,
                          restored_qwen_pid=result['restored_qwen_pid'], restored_qwen_start=result['restored_qwen']['start']),
                      source_sha256={str(p):sha256(p) for p in paths},
                      scope='Metadata-based, raw target-weight planning at equal bandwidth. It excludes KV/state, '
                            'other runtime traffic, kernel-efficiency changes, and speculation. It does not establish '
                            'an end-to-end speedup or certify quality. Full already uses UD-Q4_K_XL with mixed runtime requants.')
    OUT.write_text(json.dumps(assessment, indent=2)+'\n')
    print(json.dumps(dict(passed=True, estimates=estimates, superseded_q8_trial_has_no_candidate_result=True), indent=2))


if __name__ == '__main__':
    main()
