"""Describe the completed quantization pair without replacing whole-run metrics."""
import hashlib
import json
from pathlib import Path
import time

from dram_bandwidth import summarize_samples

BASE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = BASE / 'results/qwen-quantize-blocks-interval-analysis-0909.json'
    assert not output.exists(), 'Preserve the completed analysis'
    comparison_path = BASE / 'results/qwen-private-comparison-quantize-blocks-0909.json'
    comparison = json.loads(comparison_path.read_text())
    assert comparison['all_outputs_match'] and not comparison['candidate_repeat_available']
    assert all(digest(path) == value for path, value in comparison['evidence'].items())
    inputs = {str(path): digest(path) for path in (Path(__file__), comparison_path, BASE / 'dram_bandwidth.py')}
    rows = []
    for arm in ('off', 'on'):
        label = 'qwen-private-quantize-blocks-' + arm + '-0909-decode'
        measurement_path = BASE / 'results' / label / 'result.json'
        measured = json.loads(measurement_path.read_text())
        inputs[str(measurement_path)] = digest(measurement_path)
        for row in measured['measurements']:
            samples_path = measurement_path.parent / (row['kind'] + '-draft4') / 'samples.json'
            sample_file = json.loads(samples_path.read_text())
            assert sample_file['metadata']['valid'] and sample_file['metadata']['exit_code'] == 0
            samples = sample_file['samples']
            first, last = row['first_content_monotonic'], row['last_content_monotonic']
            assert summarize_samples(samples, first + .5, last - .5) == row['decode']
            chosen = [sample for sample in samples if sample['valid'] and sample['start'] >= first + .5 and sample['end'] <= last - .5]
            low = [sample for sample in chosen if sample['total_gb_s'] < 100]
            intervals = [dict(start_seconds=sample['start'] - first, duration=sample['duration'], raw_gb_s=sample['total_gb_s']) for sample in low]
            rows.append(dict(arm=arm, workload=row['kind'], tok_s=row['timings']['predicted_per_second'],
                             adjusted_gb_s=row['background_subtracted_gb_s'], sampled_seconds=row['decode']['sampled_seconds'],
                             low_rate_threshold_raw_gb_s=100, low_rate_intervals=intervals,
                             low_rate_sampled_seconds=sum(sample['duration'] for sample in low)))
            inputs[str(samples_path)] = digest(samples_path)
    result = dict(time=time.time(), rows=rows, input_sha256=inputs,
                  root_cause_established=False,
                  scope='Intervals below 100 raw GB/s describe a phase change only. They remain included in whole-run throughput and bandwidth. Per-token arrival timestamps were not retained. This does not establish a kernel cause or remove interference from the result.')
    assert all(digest(path) == value for path, value in inputs.items())
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({**result, 'input_sha256': 'Recorded in result file'}, indent=2))


if __name__ == '__main__':
    main()
