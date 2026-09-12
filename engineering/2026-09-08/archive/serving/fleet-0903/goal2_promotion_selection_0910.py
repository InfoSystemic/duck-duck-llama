"""Pure evidence selection for the goal2 promoter; no processes or model loading."""
import math


FLAG_EXACT16 = 'DEEPSEEK_GOAL2_EXACT16'
FLAG_SPARSE = 'DEEPSEEK_GOAL2_NATIVE_SPARSE'
FLAG_CAP = 'DEEPSEEK_GOAL2_PACKED_CAP_GIB'
FLAGS = (FLAG_EXACT16, FLAG_SPARSE, FLAG_CAP)


def select_candidate(evidence):
    for key in ('passed', 'all_exact_golden', 'zero_measured_downloads',
                'zero_measured_packing', 'selected_sources_preserved'):
        assert evidence.get(key) is True, ('Incomplete exact quiet evidence', key)
    assert evidence['torch_workers'] == evidence['native_workers'] == 16
    assert evidence['omp_wait_policy'] == 'PASSIVE'
    cap = evidence['packed_cap_bytes']
    assert type(cap) is int and cap % (1 << 30) == 0 and 0 < cap <= 128 << 30
    rows = {row['label']: row for row in evidence['runs']}
    assert len(rows) == len(evidence['runs']), 'Duplicate measurement label'
    for row in rows.values():
        assert row['exact_golden'] and row['tokens_match'], row['label']
        if row['measured']:
            assert row['timings']['downloaded_bytes'] == 0, row['label']
            assert row['exact16_metrics_delta']['pack_count'] == 0, row['label']

    def rate(labels, exact16, sparse):
        selected = [rows[label] for label in labels]
        for row in selected:
            assert row['measured'] is True and row['exact16'] is exact16 and row['native_sparse'] is sparse
            assert len(row['token_ids']) > 1
            seconds = row['timings']['decode_seconds']
            assert math.isfinite(seconds) and seconds > 0
            if exact16:
                assert row['exact16_metrics_delta']['cap_fallbacks'] == 0
                assert row['exact16_metrics_delta']['grouped_native_calls'] > 0
            assert (row['sparse_calls_delta']['native'] > 0) is sparse
        return sum(len(row['token_ids']) - 1 for row in selected) / sum(
            row['timings']['decode_seconds'] for row in selected)

    baseline = rate(('baseline_before', 'baseline_after'), False, False)
    candidates = []
    for name, exact16, sparse, labels in (
        ('exact16', True, False, ('exact16_1', 'exact16_2')),
        ('sparse', False, True, ('sparse_1', 'sparse_2')),
        ('exact16_sparse', True, True, ('exact16_sparse_1', 'exact16_sparse_2')),
    ):
        speed = rate(labels, exact16, sparse)
        candidates.append(dict(name=name, decode_tok_s=speed, speedup=speed / baseline,
                               labels=list(labels), environment={FLAG_EXACT16: str(int(exact16)),
                                   FLAG_SPARSE: str(int(sparse)), FLAG_CAP: str(cap >> 30)}))
    winner = max(candidates, key=lambda row: row['decode_tok_s'])
    assert winner['speedup'] > 1.05, 'No exact candidate exceeds the matched baseline by 5%'
    return dict(baseline_decode_tok_s=baseline, candidates=candidates, winner=winner)
