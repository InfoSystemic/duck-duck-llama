#!/usr/bin/env python3
"""Continue the existing gather validation into serial, reversible model trials."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

from compare_qwen_get_rows_bandwidth_0909 import compare, read_run
from isolate_qwen_get_rows_runtime_0909 import package
from model_measurement_guard import ModelMeasurementGuard
from qwen_private_get_rows_trial_0909 import execute, prepare
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent


def main(args):
    assert os.sched_getaffinity(0) == {127}
    validation_path = BASE / 'results/qwen-get-rows-columns-validation-0909/result.json'
    peer_guard = ModelMeasurementGuard(args.peer_pid, {args.peer_pid: 18095}, inference_snapshot)
    assert process_info(args.peer_pid)['start'] == args.start_ticks
    out = BASE / 'results' / args.label
    out.mkdir(exist_ok=False)
    files = [Path(__file__).resolve(), BASE / 'isolate_qwen_get_rows_runtime_0909.py',
             BASE / 'qwen_private_get_rows_trial_0909.py', BASE / 'compare_qwen_get_rows_bandwidth_0909.py']
    result = dict(started=time.time(), passed=False, controller_pid=os.getpid(),
                  source_sha256={str(path): sha256(path) for path in files}, trials=[], target_gb_s=250,
                  peer_pid=args.peer_pid, peer_start=args.start_ticks,
                  predecessor_pid=args.wait_pid, predecessor_start=args.wait_start,
                  scope='Wait for the already running validation; hold the lifecycle lock through runtime loading and parent/corrected/corrected/parent model trials. Preserve the identified resident peer.')

    def save(stage):
        result['stage'] = stage
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    save('waiting for existing component validation')
    try:
        while True:
            validation = json.loads(validation_path.read_text())
            if validation.get('finished'):
                assert validation['passed'] and not validation.get('error'), 'Predecessor validation failed'
                break
            assert process_info(args.wait_pid)['start'] == args.wait_start, 'Predecessor identity changed'
            peer_guard.assert_idle()
            time.sleep(5)
        with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    peer_guard.assert_idle()
                    time.sleep(5)
            result['validation_sha256'] = sha256(validation_path)
            assert all(sha256(path) == digest for path, digest in result['source_sha256'].items())
            save('checking corrected runtime loader')
            package(args.peer_pid, args.start_ticks)
            labels = []
            baseline = None
            for index, gather in enumerate(('parent', 'columns', 'columns', 'parent')):
                label = 'qwen-private-get-rows-250-0909-' + str(index) + '-' + gather
                options = SimpleNamespace(label=label, peer_pid=args.peer_pid, start_ticks=args.start_ticks,
                                          shared_dispatch='on', drafts=4, profile=False, gather=gather)
                trial = BASE / 'results' / label
                save('preparing ' + label)
                prepare(options, trial)
                save('running ' + label)
                execute(trial, lock.fileno())
                run = read_run(label)
                if baseline is None:
                    baseline = run
                for kind in ('prose', 'code'):
                    assert all(run['rows'][kind][key] == baseline['rows'][kind][key] for key in
                               ('output_sha256', 'generated_tokens', 'draft_tokens', 'accepted_draft_tokens')), kind
                labels.append(label)
                result['trials'].append(dict(label=label, gather=gather, rows=run['rows'], evidence=run['evidence']))
                save('completed ' + label)
                print(json.dumps(dict(completed_model_trial=label, rows=run['rows'])), flush=True)
            comparison = compare(labels)
            assert comparison['all_outputs_match']
            comparison_path = out / 'comparison.json'
            comparison_path.write_text(json.dumps(comparison, indent=2) + '\n')
            assert all(sha256(path) == digest for path, digest in result['source_sha256'].items())
            result.update(passed=True, comparison=str(comparison_path), comparison_sha256=sha256(comparison_path),
                          all_candidate_bandwidth_targets_met=comparison['all_candidate_bandwidth_targets_met'])
            print(json.dumps(dict(completed=True, summaries=comparison['summaries'],
                                  all_candidate_bandwidth_targets_met=result['all_candidate_bandwidth_targets_met'])), flush=True)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        save('finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--wait-start', required=True)
    parser.add_argument('--peer-pid', type=int, default=1219506)
    parser.add_argument('--start-ticks', default='103969952')
    args = parser.parse_args()
    assert re.fullmatch(r'qwen-get-rows-model-comparison-[A-Za-z0-9_-]+', args.label)
    os.umask(0o077)
    main(args)
